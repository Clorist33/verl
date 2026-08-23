# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HF -> Megatron weight mapping for the EAGLE3 draft (Step 6).

Ported from the reference megatron-eagle draft_utils._map_hf_to_megatron_eagle,
adapted to MegatronEagle3DraftModel's key layout. The draft ckpt is HF-style
(midlayer.self_attn.q_proj / gate_proj / ...); Megatron fuses q/k/v into one
per-group-interleaved linear_qkv and gate/up into one linear_fc1, and shards
column/row-parallel weights across the TP group.

Divergence from the reference: our t2d/d2t are TOP-LEVEL buffers (the patch reads
draft.t2d), not eagle_module.d2t -- both are mapped here, unsharded.
"""

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Draft TP group for the current load. Module-level so the many small shard helpers
# don't each need it threaded through; set by load_megatron_draft_weights, reset after.
_ACTIVE_TP_GROUP = None


def _tp_rank_world():
    """(rank, world) of the active draft TP group -> the global TP group if unset."""
    g = _ACTIVE_TP_GROUP
    if g is not None:
        return g.rank(), g.size()
    from megatron.core import parallel_state as mpu
    return mpu.get_tensor_model_parallel_rank(), mpu.get_tensor_model_parallel_world_size()


def _shard_tp_weight(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Chunk ``tensor`` along ``dim`` and keep this TP rank's slice."""
    rank, world = _tp_rank_world()
    if world == 1:
        return tensor
    return torch.chunk(tensor, world, dim=dim)[rank].contiguous()


def _fuse_qkv_for_megatron(q, k, v, num_heads, num_groups, head_dim) -> torch.Tensor:
    """Fuse separate q/k/v into Megatron's per-query-group interleaved linear_qkv.

    Megatron reads linear_qkv as [ng, (np//ng + 2)*hd, in] per group, so rows must
    be interleaved [q_grp0, k0, v0, q_grp1, k1, v1, ...]. A flat cat([q,k,v]) is WRONG
    under GQA (loads silently, corrupts attention). Returns FULL (unsharded) weight.
    in_dim is preserved as-is (here it is 2H for the EAGLE3 layer-0 qkv).
    """
    if num_heads % num_groups != 0:
        raise RuntimeError(f"[eagle] heads={num_heads} not divisible by groups={num_groups}")
    r = num_heads // num_groups
    in_dim = q.shape[1]
    q_g = q.reshape(num_groups, r * head_dim, in_dim)
    k_g = k.reshape(num_groups, head_dim, in_dim)
    v_g = v.reshape(num_groups, head_dim, in_dim)
    fused = torch.cat([q_g, k_g, v_g], dim=1)  # [ng, (r+2)*hd, in]
    return fused.reshape(num_groups * (r + 2) * head_dim, in_dim).contiguous()


def _fuse_and_shard_qkv(q, k, v, num_heads, num_groups, head_dim, target) -> torch.Tensor:
    """Interleave-fuse q/k/v then column-shard along dim 0 for this TP rank.

    The interleaved layout keeps whole groups contiguous, so the col-parallel shard is
    a plain chunk on dim 0 -- valid only when num_groups % TP == 0 (fail loudly else)."""
    _, world = _tp_rank_world()
    if world > 1 and num_groups % world != 0:
        raise RuntimeError(
            f"[eagle] num_query_groups={num_groups} not divisible by TP={world}; "
            "kv-head replication is not supported for the draft QKV fusion."
        )
    fused = _fuse_qkv_for_megatron(q, k, v, num_heads, num_groups, head_dim)
    fused = _shard_tp_weight(fused, dim=0)
    return fused.to(dtype=target.dtype) if target is not None else fused


def _shard_gate_up(gate: torch.Tensor, up: torch.Tensor, target) -> torch.Tensor:
    """Build this TP rank's linear_fc1 as [gate_local; up_local].

    Megatron's MLP applies ``torch.chunk(x, 2, dim=-1)`` to the LOCAL fc1 output when
    gated_linear_unit is set (see megatron/core/transformer/mlp.py), so each rank must
    hold its own slice of gate stacked on its own slice of up. Sharding a whole
    cat([gate, up]) block instead hands the low ranks nothing but gate rows and the high
    ranks nothing but up rows, so every rank computes silu(gate_a) * gate_b or
    silu(up_a) * up_b and the MLP is silently wrong. Mirrors the layout built by
    verl/models/mcore/loader.py::_broadcast_tp_shard_tensor_gate_up.
    """
    rank, world = _tp_rank_world()
    if world == 1:
        fused = torch.cat([gate, up], dim=0)
        return fused.to(dtype=target.dtype) if target is not None else fused
    if gate.shape[0] % world != 0:
        raise RuntimeError(f"[eagle] ffn_hidden_size={gate.shape[0]} not divisible by TP={world}")
    ffn_local = gate.shape[0] // world
    fused = torch.cat(
        [gate[rank * ffn_local:(rank + 1) * ffn_local], up[rank * ffn_local:(rank + 1) * ffn_local]],
        dim=0,
    ).contiguous()
    return fused.to(dtype=target.dtype) if target is not None else fused


_NO_TP_KEYS = {"eagle_module.fc.weight", "eagle_module.enorm.weight"}
_COL_PARALLEL_RE = re.compile(r"(self_attention\.linear_qkv|mlp\.linear_fc1|eagle_output_layer)\.weight$")
_ROW_PARALLEL_RE = re.compile(r"(self_attention\.linear_proj|mlp\.linear_fc2)\.weight$")


def _shard_for_tp(megatron_key: str, tensor: torch.Tensor, model_state: dict) -> torch.Tensor:
    """Shard a single already-fused weight by its role (col=dim0, row=dim1), else pass through."""
    target = model_state.get(megatron_key)
    if target is None or tensor.shape == target.shape or megatron_key in _NO_TP_KEYS:
        return tensor.to(dtype=target.dtype) if target is not None else tensor
    if _COL_PARALLEL_RE.search(megatron_key):
        return _shard_tp_weight(tensor, dim=0).to(dtype=target.dtype)
    if _ROW_PARALLEL_RE.search(megatron_key):
        return _shard_tp_weight(tensor, dim=1).to(dtype=target.dtype)
    return tensor


_L0 = "eagle_module.decoder.layers.0"  # draft is a single layer


def map_hf_to_megatron_draft(hf_state, model_state, num_heads, num_groups, head_dim) -> dict:
    """Map an HF EAGLE3 draft ckpt to MegatronEagle3DraftModel's state dict.

    HF layer-norm naming (per vLLM llama_eagle3): 
    input_layernorm normalizes the EMBEDDING -> our eagle_module.enorm; 
    hidden_norm normalizes the HIDDEN state -> the decoder layer-0 input_layernorm; 
    post_attention_layernorm -> pre_mlp_layernorm.
    q/k/v -> fused interleaved linear_qkv; 
    gate/up -> fused linear_fc1. 
    t2d/d2t are top-level buffers (unsharded). 
    embed_tokens is intentionally absent (injected + frozen from the policy).
    """
    mapped = {}
    for key, tensor in hf_state.items():
        if key == "fc.weight":
            mapped["eagle_module.fc.weight"] = _shard_for_tp("eagle_module.fc.weight", tensor, model_state)
        elif key == "norm.weight":
            mapped["eagle_module.decoder.final_layernorm.weight"] = tensor
        elif key in ("lm_head.weight", "eagle_output_layer.weight"):
            mk = "eagle_module.eagle_output_layer.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state)
        elif key in ("t2d", "d2t"):
            if key in model_state:
                mapped[key] = tensor  # top-level buffer, no shard
        elif key == "midlayer.input_layernorm.weight":
            mapped["eagle_module.enorm.weight"] = tensor            # normalizes EMBEDDING
        elif key in ("midlayer.hidden_norm.weight",):
            mapped[f"{_L0}.input_layernorm.weight"] = tensor        # normalizes HIDDEN
        elif key == "midlayer.post_attention_layernorm.weight":
            mapped[f"{_L0}.pre_mlp_layernorm.weight"] = tensor
        elif key == "midlayer.self_attn.o_proj.weight":
            mk = f"{_L0}.self_attention.linear_proj.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state)
        elif key == "midlayer.mlp.down_proj.weight":
            mk = f"{_L0}.mlp.linear_fc2.weight"
            mapped[mk] = _shard_for_tp(mk, tensor, model_state)
        # q/k/v and gate/up are fused after the loop (need all parts)

    # fuse q/k/v -> interleaved linear_qkv
    q = hf_state.get("midlayer.self_attn.q_proj.weight")
    k = hf_state.get("midlayer.self_attn.k_proj.weight")
    v = hf_state.get("midlayer.self_attn.v_proj.weight")
    if q is not None and k is not None and v is not None:
        # DRAFT-WEIGHT-DEBUG: Print QKV fusion details
        print("-" * 80)
        print(f"DRAFT-WEIGHT-DEBUG: QKV fusion START")
        print(f"  num_heads={num_heads}, num_groups={num_groups}, head_dim={head_dim}")
        print(f"  Q shape: {tuple(q.shape)}, mean={q.float().mean():.6f}, std={q.float().std():.6f}")
        print(f"  K shape: {tuple(k.shape)}, mean={k.float().mean():.6f}, std={k.float().std():.6f}")
        print(f"  V shape: {tuple(v.shape)}, mean={v.float().mean():.6f}, std={v.float().std():.6f}")

        mk = f"{_L0}.self_attention.linear_qkv.weight"
        target = model_state.get(mk)
        print(f"  Target (Megatron) shape: {tuple(target.shape) if target is not None else 'None'}")

        mapped[mk] = _fuse_and_shard_qkv(q, k, v, num_heads, num_groups, head_dim, target)

        print(f"  Fused+Sharded shape: {tuple(mapped[mk].shape)}, "
              f"mean={mapped[mk].float().mean():.6f}, std={mapped[mk].float().std():.6f}")
        print(f"DRAFT-WEIGHT-DEBUG: QKV fusion END")
        print("-" * 80)
    elif any(x is not None for x in (q, k, v)):
        logger.warning("[eagle] incomplete q/k/v in draft ckpt; skipping linear_qkv")

    # fuse gate/up -> linear_fc1
    gate = hf_state.get("midlayer.mlp.gate_proj.weight")
    up = hf_state.get("midlayer.mlp.up_proj.weight")
    if gate is not None and up is not None:
        # DRAFT-WEIGHT-DEBUG: Print Gate-Up fusion details
        print("-" * 80)
        print(f"DRAFT-WEIGHT-DEBUG: Gate-Up fusion START")
        print(f"  Gate shape: {tuple(gate.shape)}, mean={gate.float().mean():.6f}, std={gate.float().std():.6f}")
        print(f"  Up shape: {tuple(up.shape)}, mean={up.float().mean():.6f}, std={up.float().std():.6f}")

        mk = f"{_L0}.mlp.linear_fc1.weight"
        target = model_state.get(mk)
        print(f"  Target (Megatron) shape: {tuple(target.shape) if target is not None else 'None'}")

        # Per-rank [gate_local; up_local]; do NOT cat([gate, up]) then chunk (see _shard_gate_up).
        mapped[mk] = _shard_gate_up(gate, up, target)

        print(f"  Sharded shape: {tuple(mapped[mk].shape)}, "
              f"mean={mapped[mk].float().mean():.6f}, std={mapped[mk].float().std():.6f}")
        print(f"DRAFT-WEIGHT-DEBUG: Gate-Up fusion END")
        print("-" * 80)
    elif gate is not None or up is not None:
        logger.warning("[eagle] incomplete gate/up in draft ckpt; skipping linear_fc1")

    return mapped


def _export_tp_world(group=None) -> int:
    """TP world size on the export path, where _ACTIVE_TP_GROUP is not set."""
    if group is not None:
        return group.size()
    from megatron.core import parallel_state as mpu

    return mpu.get_tensor_model_parallel_world_size()


def _gather_tp_weight(tensor: torch.Tensor, dim: int, group=None) -> torch.Tensor:
    """All-gather a TP-sharded weight along ``dim`` across the TP group."""
    if group is not None:
        world = group.size()
    else:
        from megatron.core import parallel_state as mpu
        world = mpu.get_tensor_model_parallel_world_size()
        group = mpu.get_tensor_model_parallel_group()

    if world == 1:
        return tensor

    # All-gather
    from megatron.core.parallel_state import get_tensor_model_parallel_rank
    gathered_list = [torch.empty_like(tensor) for _ in range(world)]
    torch.distributed.all_gather(gathered_list, tensor, group=group)

    # Concatenate along the split dimension
    return torch.cat(gathered_list, dim=dim).contiguous()


def _unfuse_qkv_from_megatron(fused_qkv: torch.Tensor, num_heads: int, num_groups: int, head_dim: int):
    """Unfuse Megatron's interleaved linear_qkv back to separate q/k/v weights.

    Inverse of _fuse_qkv_for_megatron. Returns (q, k, v) as separate tensors.
    """
    if num_heads % num_groups != 0:
        raise RuntimeError(f"[eagle] heads={num_heads} not divisible by groups={num_groups}")
    r = num_heads // num_groups
    in_dim = fused_qkv.shape[1]

    # Reshape to [ng, (r+2)*hd, in]
    fused_grouped = fused_qkv.reshape(num_groups, (r + 2) * head_dim, in_dim)

    # Split each group into q/k/v
    q_g = fused_grouped[:, :r * head_dim, :]      # [ng, r*hd, in]
    k_g = fused_grouped[:, r * head_dim:(r + 1) * head_dim, :]  # [ng, hd, in]
    v_g = fused_grouped[:, (r + 1) * head_dim:, :]  # [ng, hd, in]

    # Reshape back to full dimensions
    q = q_g.reshape(num_heads * head_dim, in_dim)
    k = k_g.reshape(num_groups * head_dim, in_dim)
    v = v_g.reshape(num_groups * head_dim, in_dim)

    return q.contiguous(), k.contiguous(), v.contiguous()


def _unfuse_gate_up(fused_weight: torch.Tensor, world: int = 1) -> tuple:
    """Unfuse an ALL-GATHERED linear_fc1 back to separate full gate and up weights.

    Inverse of _shard_gate_up. Each rank contributes [gate_local; up_local], so the
    gathered tensor is [gate_r0; up_r0; gate_r1; up_r1; ...] and must be de-interleaved
    by taking every other block -- a plain half/half split would return gate rows mixed
    with up rows once TP > 1.
    """
    if world <= 1:
        ffn = fused_weight.shape[0] // 2
        return fused_weight[:ffn].contiguous(), fused_weight[ffn:].contiguous()
    if fused_weight.shape[0] % (2 * world) != 0:
        raise RuntimeError(f"[eagle] fc1 rows={fused_weight.shape[0]} not divisible by 2*TP={2 * world}")
    blocks = torch.chunk(fused_weight, 2 * world, dim=0)
    gate = torch.cat([blocks[2 * i] for i in range(world)], dim=0)
    up = torch.cat([blocks[2 * i + 1] for i in range(world)], dim=0)
    return gate.contiguous(), up.contiguous()


def map_megatron_to_hf_draft(megatron_state: dict, num_heads: int, num_groups: int, head_dim: int, tp_group=None) -> dict:
    """Map MegatronEagle3DraftModel's state dict back to HF EAGLE3 draft format.

    Inverse of map_hf_to_megatron_draft. Performs:
    1. TP all-gather for sharded weights (linear_qkv, linear_fc1, linear_proj, linear_fc2, eagle_output_layer)
    2. Unfuse linear_qkv -> q/k/v, linear_fc1 -> gate/up
    3. Rename Megatron keys to HF keys

    Args:
        megatron_state: State dict from MegatronEagle3DraftModel (TP-sharded)
        num_heads: Number of attention heads
        num_groups: Number of KV groups (for GQA)
        head_dim: Dimension per head
        tp_group: TP group for all-gather (None = use global TP group)

    Returns:
        HF-format state dict with full (gathered, unfused) weights
    """
    hf_state = {}

    # 🔥 DEBUG: Track which keys are processed and which branch they take
    print("=" * 80)
    print(f"🔥 map_megatron_to_hf_draft: Processing {len(megatron_state)} Megatron keys")
    print(f"   _L0 = {_L0}")
    print("=" * 80)

    # Process each Megatron key
    for key, tensor in megatron_state.items():
        print(f"🔥 Processing key: {key}")
        # Top-level buffers (t2d/d2t) - pass through unchanged
        if key in ("t2d", "d2t"):
            print(f"   → buffer: {key}")
            hf_state[key] = tensor
            continue

        # eagle_module.fc.weight -> fc.weight (REPLICATED, not TP; pass-through).
        # fc is a plain nn.Linear (draft_megatron.py:101), in _NO_TP_KEYS, so the load
        # side keeps it full on every rank. Gathering it here would concat TP identical
        # copies -> [tp*out, in] and crash the vLLM receiver. Matches lilac export
        # (draft_utils.py:805) and mirrors our own load side (_shard_for_tp -> _NO_TP_KEYS).
        if key == "eagle_module.fc.weight":
            print(f"   → fc.weight")
            hf_state["fc.weight"] = tensor

        # eagle_module.decoder.final_layernorm.weight -> norm.weight
        elif key == "eagle_module.decoder.final_layernorm.weight":
            print(f"   → norm.weight")
            hf_state["norm.weight"] = tensor

        # eagle_module.eagle_output_layer.weight -> lm_head.weight (column-parallel, needs gather)
        elif key == "eagle_module.eagle_output_layer.weight":
            print(f"   → lm_head.weight (gathering)")
            gathered = _gather_tp_weight(tensor, dim=0, group=tp_group)
            hf_state["lm_head.weight"] = gathered

        # eagle_module.enorm.weight -> midlayer.input_layernorm.weight
        elif key == "eagle_module.enorm.weight":
            print(f"   → midlayer.input_layernorm.weight")
            hf_state["midlayer.input_layernorm.weight"] = tensor

        # Layer-0 decoder layer mappings
        elif key == f"{_L0}.input_layernorm.weight":
            print(f"   → midlayer.hidden_norm.weight")
            hf_state["midlayer.hidden_norm.weight"] = tensor

        elif key == f"{_L0}.pre_mlp_layernorm.weight":
            print(f"   → midlayer.post_attention_layernorm.weight")
            hf_state["midlayer.post_attention_layernorm.weight"] = tensor

        # linear_proj (row-parallel, needs gather on dim=1) -> o_proj
        elif key == f"{_L0}.self_attention.linear_proj.weight":
            print(f"   → midlayer.self_attn.o_proj.weight (gathering)")
            gathered = _gather_tp_weight(tensor, dim=1, group=tp_group)
            hf_state["midlayer.self_attn.o_proj.weight"] = gathered

        # linear_fc2 (row-parallel, needs gather on dim=1) -> down_proj
        elif key == f"{_L0}.mlp.linear_fc2.weight":
            print(f"   → midlayer.mlp.down_proj.weight (gathering)")
            gathered = _gather_tp_weight(tensor, dim=1, group=tp_group)
            hf_state["midlayer.mlp.down_proj.weight"] = gathered

        # linear_qkv (column-parallel, fused) -> q/k/v (needs gather + unfuse)
        elif key == f"{_L0}.self_attention.linear_qkv.weight":
            print(f"   → unfusing to q/k/v (gathering + unfusing)")
            gathered = _gather_tp_weight(tensor, dim=0, group=tp_group)
            q, k, v = _unfuse_qkv_from_megatron(gathered, num_heads, num_groups, head_dim)
            hf_state["midlayer.self_attn.q_proj.weight"] = q
            hf_state["midlayer.self_attn.k_proj.weight"] = k
            hf_state["midlayer.self_attn.v_proj.weight"] = v

        # linear_fc1 (column-parallel, fused) -> gate/up (needs gather + unfuse)
        elif key == f"{_L0}.mlp.linear_fc1.weight":
            print(f"   → unfusing to gate/up (gathering + unfusing)")
            gathered = _gather_tp_weight(tensor, dim=0, group=tp_group)
            gate, up = _unfuse_gate_up(gathered, world=_export_tp_world(tp_group))
            hf_state["midlayer.mlp.gate_proj.weight"] = gate
            hf_state["midlayer.mlp.up_proj.weight"] = up

        # embed_tokens is frozen from policy, skip
        elif key == "embed_tokens.weight":
            print(f"   → SKIP (embed_tokens, frozen from policy)")
            pass  # Not exported

        else:
            print(f"   → ⚠️ UNRECOGNIZED KEY!")
            logger.warning(f"[eagle] Unrecognized Megatron draft key during export: {key}")

    print("=" * 80)
    print(f"🔥 map_megatron_to_hf_draft: Conversion complete")
    print(f"   Input: {len(megatron_state)} Megatron keys")
    print(f"   Output: {len(hf_state)} HF keys")
    print(f"   HF keys: {sorted(hf_state.keys())}")
    print("=" * 80)

    # ⚠️ DO NOT add "model." prefix here!
    # vLLM's Eagle3LlamaForCausalLM.load_weights() automatically adds "model." prefix
    # to all weights except lm_head, d2t, and mask_hidden (see line 452-453)
    # If we add it here, it becomes "model.model.fc.weight" which causes KeyError

    return hf_state


def _load_hf_draft_state(draft_path: str) -> dict:
    """Load the HF draft ckpt state dict (safetensors preferred, then .bin)."""
    import glob

    state = {}
    if not draft_path or not os.path.isdir(draft_path):
        raise FileNotFoundError(f"[eagle] draft_model_path {draft_path!r} is not a local dir")
    try:
        from safetensors.torch import load_file

        for f in sorted(glob.glob(os.path.join(draft_path, "*.safetensors"))):
            state.update(load_file(f))
    except Exception:
        pass
    if not state:
        for f in sorted(glob.glob(os.path.join(draft_path, "*.bin"))):
            state.update(torch.load(f, map_location="cpu", weights_only=True))
    if not state:
        raise FileNotFoundError(f"[eagle] no weight files (*.safetensors/*.bin) in {draft_path!r}")
    return state



def _target_state(draft) -> dict:
    """Build {name: tensor} from params + buffers (avoids Megatron _get_extra_state)."""
    st = {name: p for name, p in draft.named_parameters()}
    for name, buf in draft.named_buffers():
        if buf is not None:
            st[name] = buf
    return st


def load_megatron_draft_weights(draft, draft_path: str, tp_group=None) -> None:
    """Load + TP-shard the HF EAGLE3 draft ckpt into a MegatronEagle3DraftModel (in place).

    ``tp_group`` is the draft's TP group (None -> the global TP group). All col/row shard
    math uses its rank/size, so the shards match however the draft's layers were built.

    embed_tokens is intentionally NOT loaded here -- it is injected from the policy and
    frozen by engine_support._inject_and_freeze_draft_embed AFTER this call, so it is
    expected in `missing`. The draft's own trained lm_head (eagle_output_layer) IS loaded
    (EAGLE3 drafts ship a real head over the draft vocab; do not overwrite from policy).

    draft:实例化 MegatronEagle3DraftModel
    draft_path:磁盘保存的draft权重路径
    tp_group:draft的tp（在训练脚本里由“DRAFT_TP”传递）
    """
    global _ACTIVE_TP_GROUP
    cfg = draft.config
    num_heads = cfg.num_attention_heads
    num_groups = getattr(cfg, "num_query_groups", num_heads)
    head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // num_heads)

    hf_state = _load_hf_draft_state(draft_path)

    # DRAFT-WEIGHT-DEBUG: Print loading parameters and HF weight statistics
    print("=" * 80)
    print(f"DRAFT-WEIGHT-DEBUG: Starting HF→Megatron draft weight loading")
    print(f"  draft_path: {draft_path}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_groups: {num_groups}")
    print(f"  head_dim: {head_dim}")
    print(f"  TP group: {tp_group} (size={tp_group.size() if tp_group else 1})")
    print("-" * 80)
    print(f"HF state keys loaded: {len(hf_state)}")
    for key in sorted(hf_state.keys()):
        tensor = hf_state[key]
        print(f"  [HF] {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
              f"mean={tensor.float().mean().item():.6f} std={tensor.float().std().item():.6f}")
    print("=" * 80)

    model_state = _target_state(draft)  # 拿目标格式
    # breakpoint()


    _ACTIVE_TP_GROUP = tp_group
    try:
        mapped = map_hf_to_megatron_draft(hf_state, model_state, num_heads, num_groups, head_dim)    # 把 HF→Megatron(融合 QKV/gate-up + TP 切分),得到 mapped
    finally:
        _ACTIVE_TP_GROUP = None

    # ===== 权重对比检查_REVERT_20260821 (调试用,回退时删除本块) =====
    # 逐层打印 HF 原始权重与 mapped 切分后权重的实际数值,并排对比。
    # 期望值全部直接从 hf_state 手工切片,不复用本文件的融合/切分函数。
    tp_rank, tp_world = _tp_rank_world()
    N = 8  # 每行打印前 N 个数

    def head(t):
        """取张量第 0 行前 N 个数,格式化成一行。"""
        if t is None:
            return "无"
        f = t.reshape(-1)[:N] if t.dim() == 1 else t[0, :N]
        return " ".join(f"{v:+.6f}" for v in f.float().tolist())

    def row(t, i):
        """取张量第 i 行前 N 个数。"""
        if t is None or i >= t.shape[0]:
            return "无"
        return " ".join(f"{v:+.6f}" for v in t[i, :N].float().tolist())

    def check(name, mg_key, expect, src):
        """打印 mapped[mg_key] 与手工重建期望值的首行数值及是否相等。"""
        got = mapped.get(mg_key)
        print(f"  [{name}]")
        print(f"    HF  来源 {src}")
        print(f"    HF  首行 {head(expect)}")
        print(f"    切分首行 {head(got)}")
        if got is None or expect is None:
            print("    结论 缺失,无法对比")
            return
        if got.shape != expect.shape:
            print(f"    结论 形状不符 切分={tuple(got.shape)} 期望={tuple(expect.shape)}")
            return
        g, e = got.cpu().float(), expect.cpu().float()
        bad = int((g != e).sum())
        if bad == 0:
            print(f"    结论 完全一致 形状={tuple(got.shape)}")
        else:
            idx = (g != e).nonzero()[0].tolist()
            print(f"    结论 不一致 形状={tuple(got.shape)} 不等元素={bad}/{g.numel()}")
            print(f"         首个不等位置={idx} 切分值={g[tuple(idx)].item():+.6f} 期望值={e[tuple(idx)].item():+.6f}")

    print("=" * 78)
    print(f"权重对比检查 当前 rank={tp_rank} 共 {tp_world} 张卡")
    print(f"参数 num_heads={num_heads} num_groups={num_groups} head_dim={head_dim}")
    print("=" * 78)
    print("一、不切分的层(每张卡都持有完整权重,应完全一致)")
    for hf_k, mg_k in [
        ("fc.weight", "eagle_module.fc.weight"),
        ("norm.weight", "eagle_module.decoder.final_layernorm.weight"),
        ("midlayer.input_layernorm.weight", "eagle_module.enorm.weight"),
        ("midlayer.hidden_norm.weight", f"{_L0}.input_layernorm.weight"),
        ("midlayer.post_attention_layernorm.weight", f"{_L0}.pre_mlp_layernorm.weight"),
    ]:
        if mg_k in mapped:
            check(mg_k.replace("eagle_module.", ""), mg_k, hf_state.get(hf_k), hf_k)

    print("-" * 78)
    print("二、按行切分的层(切 dim=0,rank0 取最前面的行)")
    hf_lm = hf_state.get("lm_head.weight", hf_state.get("eagle_output_layer.weight"))
    if hf_lm is not None:
        s = hf_lm.shape[0] // tp_world
        check("eagle_output_layer", "eagle_module.eagle_output_layer.weight",
              hf_lm[tp_rank * s:(tp_rank + 1) * s, :],
              f"lm_head.weight 第 {tp_rank * s}~{(tp_rank + 1) * s} 行")
        if tp_rank == 0:
            print(f"    对照 lm_head 原始首行 {head(hf_lm)}")

    print("-" * 78)
    print("三、按列切分的层(切 dim=1,每张卡取一段列)")
    for hf_k, mg_k in [
        ("midlayer.self_attn.o_proj.weight", f"{_L0}.self_attention.linear_proj.weight"),
        ("midlayer.mlp.down_proj.weight", f"{_L0}.mlp.linear_fc2.weight"),
    ]:
        t = hf_state.get(hf_k)
        if t is not None:
            s = t.shape[1] // tp_world
            check(mg_k.split(".")[-2], mg_k, t[:, tp_rank * s:(tp_rank + 1) * s],
                  f"{hf_k} 第 {tp_rank * s}~{(tp_rank + 1) * s} 列")
            if tp_rank == 0:
                print(f"    对照 {hf_k} 原始首行 {head(t)}")
    print("-" * 78)
    print("四、qkv 融合层")
    q = hf_state.get("midlayer.self_attn.q_proj.weight")
    k = hf_state.get("midlayer.self_attn.k_proj.weight")
    v = hf_state.get("midlayer.self_attn.v_proj.weight")
    mg_qkv = mapped.get(f"{_L0}.self_attention.linear_qkv.weight")
    if q is not None and k is not None and v is not None:
        ng_local = num_groups // tp_world
        rpg = num_heads // num_groups
        g0 = tp_rank * ng_local
        print(f"    HF  q 形状={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
        print(f"    本卡负责 query 组 {g0}~{g0 + ng_local - 1},每组 {rpg} 个 q 头 + 1 个 k 头 + 1 个 v 头")
        print(f"    HF  q_proj 第 {g0 * rpg * head_dim} 行 {row(q, g0 * rpg * head_dim)}")
        print(f"    HF  k_proj 第 {g0 * head_dim} 行 {row(k, g0 * head_dim)}")
        print(f"    HF  v_proj 第 {g0 * head_dim} 行 {row(v, g0 * head_dim)}")
        if mg_qkv is not None:
            print(f"    切分 linear_qkv 形状={tuple(mg_qkv.shape)}")
            print(f"    切分 第 0 行(应等于上面 q_proj 行) {row(mg_qkv, 0)}")
            print(f"    切分 第 {rpg * head_dim} 行(应等于上面 k_proj 行) {row(mg_qkv, rpg * head_dim)}")
            print(f"    切分 第 {(rpg + 1) * head_dim} 行(应等于上面 v_proj 行) {row(mg_qkv, (rpg + 1) * head_dim)}")
        rows = []
        for g in range(g0, g0 + ng_local):
            rows.append(q[g * rpg * head_dim:(g + 1) * rpg * head_dim, :])
            rows.append(k[g * head_dim:(g + 1) * head_dim, :])
            rows.append(v[g * head_dim:(g + 1) * head_dim, :])
        check("linear_qkv", f"{_L0}.self_attention.linear_qkv.weight",
              torch.cat(rows, dim=0), f"q/k/v 按组 {g0}~{g0 + ng_local - 1} 交错拼接")
        if tp_rank == 0:
            print(f"    对照 q_proj 原始首行 {head(q)}")
            print(f"    对照 k_proj 原始首行 {head(k)}")
            print(f"    对照 v_proj 原始首行 {head(v)}")
    print("-" * 78)
    print("五、gate/up 融合层")
    gate = hf_state.get("midlayer.mlp.gate_proj.weight")
    up = hf_state.get("midlayer.mlp.up_proj.weight")
    mg_fc1 = mapped.get(f"{_L0}.mlp.linear_fc1.weight")
    if gate is not None and up is not None:
        fl = gate.shape[0] // tp_world
        gate_local = gate[tp_rank * fl:(tp_rank + 1) * fl, :]
        up_local = up[tp_rank * fl:(tp_rank + 1) * fl, :]
        print(f"    HF  gate 形状={tuple(gate.shape)} up={tuple(up.shape)}")
        print(f"    本卡应持有 gate 第 {tp_rank * fl}~{(tp_rank + 1) * fl} 行 + up 同段")
        print(f"    HF  gate 第 {tp_rank * fl} 行 {head(gate_local)}")
        print(f"    HF  up   第 {tp_rank * fl} 行 {head(up_local)}")
        if mg_fc1 is not None:
            print(f"    切分 linear_fc1 形状={tuple(mg_fc1.shape)}")
            print(f"    切分 第 0 行(应等于上面 gate 行) {row(mg_fc1, 0)}")
            print(f"    切分 第 {fl} 行(应等于上面 up 行) {row(mg_fc1, fl)}")
        check("linear_fc1", f"{_L0}.mlp.linear_fc1.weight",
              torch.cat([gate_local, up_local], dim=0),
              f"gate 与 up 各取第 {tp_rank * fl}~{(tp_rank + 1) * fl} 行后拼接")
        if tp_rank == 0:
            print(f"    对照 gate 原始首行 {head(gate)}")
            print(f"    对照 up   原始首行 {head(up)}")

    print("-" * 78)
    print("六、词表映射缓冲区")
    for key in ("d2t", "t2d"):
        if key in mapped:
            g, e = mapped[key], hf_state.get(key)
            same = "完全一致" if (e is not None and bool((g == e).all())) else "不一致"
            print(f"  [{key}] 形状={tuple(g.shape)} 前 {N} 个={g[:N].tolist()} 结论 {same}")
    print("=" * 78)
    # ===== END 权重对比检查_REVERT_20260821 =====

    missing, unexpected = draft.load_state_dict(mapped, strict=False)    # load megatron格式的draft权重在这里实现，真正的加载启动点  
    # embed_tokens is supplied by the policy later -> drop it from the missing report.
    missing = [k for k in missing if not k.startswith("embed_tokens")]
    world = tp_group.size() if tp_group is not None else _tp_rank_world()[1]

    # DRAFT-WEIGHT-DEBUG: Print loaded Megatron weight statistics
    print("=" * 80)
    print(f"DRAFT-WEIGHT-DEBUG: Megatron draft weights loaded")
    print(f"  Mapped keys: {len(mapped)}")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    print("-" * 80)
    print("Mapped weights statistics:")
    for key in sorted(mapped.keys()):
        tensor = mapped[key]
        is_float = tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]
        print(f"  [MEGATRON] {key}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
              f"mean={tensor.float().mean().item() if is_float else 0:.6f} "
              f"std={tensor.float().std().item() if is_float else 0:.6f}")
    print("=" * 80)

    # DRAFT-WEIGHT-DEBUG: verify d2t/t2d actually landed in the draft buffers.
    # int64/bool tensors show mean/std=0 above only because that print hard-codes
    # 0 for non-float dtypes; this block prints raw values to confirm the load.
    print("=" * 80)
    print("DRAFT-WEIGHT-DEBUG: d2t/t2d value-level verification (first 5 elems)")
    loaded_buffers = dict(draft.named_buffers())
    for key in ("d2t", "t2d"):
        disk_t = hf_state.get(key)
        mapped_t = mapped.get(key)
        buf_t = loaded_buffers.get(key)
        disk_v = disk_t[:5].tolist() if disk_t is not None else "ABSENT"
        mapped_v = mapped_t[:5].tolist() if mapped_t is not None else "ABSENT"
        buf_v = buf_t[:5].tolist() if buf_t is not None else "ABSENT"
        match = (buf_t is not None and disk_t is not None
                 and torch.equal(buf_t.cpu(), disk_t.cpu()))
        print(f"  [{key}] disk(HF)   : {disk_v}")
        print(f"  [{key}] mapped     : {mapped_v}")
        print(f"  [{key}] draft.buf  : {buf_v}")
        print(f"  [{key}] FULL EQUAL disk==buf: {match}")
    print("=" * 80)

    # DRAFT-WEIGHT-DEBUG: Compare HF original vs Megatron loaded weights (global stats)
    # For TP-sharded weights, all-gather first; for fused weights, unfuse first
    print("=" * 80)
    print("DRAFT-WEIGHT-DEBUG: HF vs Megatron global statistics comparison")
    print("  (Megatron weights are all-gathered across TP ranks before computing stats)")
    print("-" * 80)

    def _compute_stats(t):
        """Compute mean, std, min, max, abs_max for a tensor."""
        t_f = t.float()
        return {
            'mean': t_f.mean().item(),
            'std': t_f.std().item(),
            'min': t_f.min().item(),
            'max': t_f.max().item(),
            'abs_max': t_f.abs().max().item(),
        }

    # Get loaded Megatron state (already in draft model)
    loaded_state = _target_state(draft)

    # Layer 1: fc.weight (no TP shard, should be identical)
    hf_fc = hf_state.get("fc.weight")
    mg_fc = loaded_state.get("eagle_module.fc.weight")
    if hf_fc is not None and mg_fc is not None:
        hf_fc_stats = _compute_stats(hf_fc)
        mg_fc_stats = _compute_stats(mg_fc)
        print(f"fc.weight (NO TP shard, should be identical):")
        print(f"  HF:       mean={hf_fc_stats['mean']:.6f} std={hf_fc_stats['std']:.6f} "
              f"min={hf_fc_stats['min']:.6f} max={hf_fc_stats['max']:.6f} abs_max={hf_fc_stats['abs_max']:.6f}")
        print(f"  Megatron: mean={mg_fc_stats['mean']:.6f} std={mg_fc_stats['std']:.6f} "
              f"min={mg_fc_stats['min']:.6f} max={mg_fc_stats['max']:.6f} abs_max={mg_fc_stats['abs_max']:.6f}")
        print(f"  MATCH: {torch.allclose(hf_fc.cpu(), mg_fc.cpu(), rtol=1e-5, atol=1e-6)}")

    # Layer 2: linear_qkv (TP sharded, need gather + unfuse)
    mg_qkv_sharded = loaded_state.get(f"{_L0}.self_attention.linear_qkv.weight")
    if mg_qkv_sharded is not None:
        mg_qkv_gathered = _gather_tp_weight(mg_qkv_sharded, dim=0, group=tp_group)
        q_mg, k_mg, v_mg = _unfuse_qkv_from_megatron(mg_qkv_gathered, num_heads, num_groups, head_dim)

        q_hf = hf_state.get("midlayer.self_attn.q_proj.weight")
        k_hf = hf_state.get("midlayer.self_attn.k_proj.weight")
        v_hf = hf_state.get("midlayer.self_attn.v_proj.weight")

        if q_hf is not None and q_mg is not None:
            q_hf_stats = _compute_stats(q_hf)
            q_mg_stats = _compute_stats(q_mg)
            print(f"q_proj.weight (TP sharded, gathered):")
            print(f"  HF:       mean={q_hf_stats['mean']:.6f} std={q_hf_stats['std']:.6f} "
                  f"min={q_hf_stats['min']:.6f} max={q_hf_stats['max']:.6f} abs_max={q_hf_stats['abs_max']:.6f}")
            print(f"  Megatron: mean={q_mg_stats['mean']:.6f} std={q_mg_stats['std']:.6f} "
                  f"min={q_mg_stats['min']:.6f} max={q_mg_stats['max']:.6f} abs_max={q_mg_stats['abs_max']:.6f}")
            print(f"  MATCH: {torch.allclose(q_hf.cpu(), q_mg.cpu(), rtol=1e-5, atol=1e-6)}")

        if k_hf is not None and k_mg is not None:
            k_hf_stats = _compute_stats(k_hf)
            k_mg_stats = _compute_stats(k_mg)
            print(f"k_proj.weight (TP sharded, gathered):")
            print(f"  HF:       mean={k_hf_stats['mean']:.6f} std={k_hf_stats['std']:.6f} "
                  f"min={k_hf_stats['min']:.6f} max={k_hf_stats['max']:.6f} abs_max={k_hf_stats['abs_max']:.6f}")
            print(f"  Megatron: mean={k_mg_stats['mean']:.6f} std={k_mg_stats['std']:.6f} "
                  f"min={k_mg_stats['min']:.6f} max={k_mg_stats['max']:.6f} abs_max={k_mg_stats['abs_max']:.6f}")
            print(f"  MATCH: {torch.allclose(k_hf.cpu(), k_mg.cpu(), rtol=1e-5, atol=1e-6)}")

        if v_hf is not None and v_mg is not None:
            v_hf_stats = _compute_stats(v_hf)
            v_mg_stats = _compute_stats(v_mg)
            print(f"v_proj.weight (TP sharded, gathered):")
            print(f"  HF:       mean={v_hf_stats['mean']:.6f} std={v_hf_stats['std']:.6f} "
                  f"min={v_hf_stats['min']:.6f} max={v_hf_stats['max']:.6f} abs_max={v_hf_stats['abs_max']:.6f}")
            print(f"  Megatron: mean={v_mg_stats['mean']:.6f} std={v_mg_stats['std']:.6f} "
                  f"min={v_mg_stats['min']:.6f} max={v_mg_stats['max']:.6f} abs_max={v_mg_stats['abs_max']:.6f}")
            print(f"  MATCH: {torch.allclose(v_hf.cpu(), v_mg.cpu(), rtol=1e-5, atol=1e-6)}")

    # Layer 3: linear_fc1 (gate+up fused, TP sharded)
    mg_fc1_sharded = loaded_state.get(f"{_L0}.mlp.linear_fc1.weight")
    if mg_fc1_sharded is not None:
        mg_fc1_gathered = _gather_tp_weight(mg_fc1_sharded, dim=0, group=tp_group)
        gate_mg, up_mg = _unfuse_gate_up(mg_fc1_gathered, world=_export_tp_world(tp_group))

        gate_hf = hf_state.get("midlayer.mlp.gate_proj.weight")
        up_hf = hf_state.get("midlayer.mlp.up_proj.weight")

        if gate_hf is not None and gate_mg is not None:
            gate_hf_stats = _compute_stats(gate_hf)
            gate_mg_stats = _compute_stats(gate_mg)
            print(f"gate_proj.weight (TP sharded, gathered):")
            print(f"  HF:       mean={gate_hf_stats['mean']:.6f} std={gate_hf_stats['std']:.6f} "
                  f"min={gate_hf_stats['min']:.6f} max={gate_hf_stats['max']:.6f} abs_max={gate_hf_stats['abs_max']:.6f}")
            print(f"  Megatron: mean={gate_mg_stats['mean']:.6f} std={gate_mg_stats['std']:.6f} "
                  f"min={gate_mg_stats['min']:.6f} max={gate_mg_stats['max']:.6f} abs_max={gate_mg_stats['abs_max']:.6f}")
            print(f"  MATCH: {torch.allclose(gate_hf.cpu(), gate_mg.cpu(), rtol=1e-5, atol=1e-6)}")

        if up_hf is not None and up_mg is not None:
            up_hf_stats = _compute_stats(up_hf)
            up_mg_stats = _compute_stats(up_mg)
            print(f"up_proj.weight (TP sharded, gathered):")
            print(f"  HF:       mean={up_hf_stats['mean']:.6f} std={up_hf_stats['std']:.6f} "
                  f"min={up_hf_stats['min']:.6f} max={up_hf_stats['max']:.6f} abs_max={up_hf_stats['abs_max']:.6f}")
            print(f"  Megatron: mean={up_mg_stats['mean']:.6f} std={up_mg_stats['std']:.6f} "
                  f"min={up_mg_stats['min']:.6f} max={up_mg_stats['max']:.6f} abs_max={up_mg_stats['abs_max']:.6f}")
            print(f"  MATCH: {torch.allclose(up_hf.cpu(), up_mg.cpu(), rtol=1e-5, atol=1e-6)}")

    # Layer 4: linear_proj (o_proj, TP sharded on dim=1 - row parallel)
    mg_proj_sharded = loaded_state.get(f"{_L0}.self_attention.linear_proj.weight")
    if mg_proj_sharded is not None:
        mg_proj_gathered = _gather_tp_weight(mg_proj_sharded, dim=1, group=tp_group)
        o_hf = hf_state.get("midlayer.self_attn.o_proj.weight")

        if o_hf is not None:
            o_hf_stats = _compute_stats(o_hf)
            o_mg_stats = _compute_stats(mg_proj_gathered)
            print(f"o_proj.weight (TP sharded dim=1, gathered):")
            print(f"  HF:       mean={o_hf_stats['mean']:.6f} std={o_hf_stats['std']:.6f} "
                  f"min={o_hf_stats['min']:.6f} max={o_hf_stats['max']:.6f} abs_max={o_hf_stats['abs_max']:.6f}")
            print(f"  Megatron: mean={o_mg_stats['mean']:.6f} std={o_mg_stats['std']:.6f} "
                  f"min={o_mg_stats['min']:.6f} max={o_mg_stats['max']:.6f} abs_max={o_mg_stats['abs_max']:.6f}")
            print(f"  MATCH: {torch.allclose(o_hf.cpu(), mg_proj_gathered.cpu(), rtol=1e-5, atol=1e-6)}")

    # Layer 5: linear_fc2 (down_proj, TP sharded on dim=1 - row parallel)
    mg_fc2_sharded = loaded_state.get(f"{_L0}.mlp.linear_fc2.weight")
    if mg_fc2_sharded is not None:
        mg_fc2_gathered = _gather_tp_weight(mg_fc2_sharded, dim=1, group=tp_group)
        down_hf = hf_state.get("midlayer.mlp.down_proj.weight")

        if down_hf is not None:
            down_hf_stats = _compute_stats(down_hf)
            down_mg_stats = _compute_stats(mg_fc2_gathered)
            print(f"down_proj.weight (TP sharded dim=1, gathered):")
            print(f"  HF:       mean={down_hf_stats['mean']:.6f} std={down_hf_stats['std']:.6f} "
                  f"min={down_hf_stats['min']:.6f} max={down_hf_stats['max']:.6f} abs_max={down_hf_stats['abs_max']:.6f}")
            print(f"  Megatron: mean={down_mg_stats['mean']:.6f} std={down_mg_stats['std']:.6f} "
                  f"min={down_mg_stats['min']:.6f} max={down_mg_stats['max']:.6f} abs_max={down_mg_stats['abs_max']:.6f}")
            print(f"  MATCH: {torch.allclose(down_hf.cpu(), mg_fc2_gathered.cpu(), rtol=1e-5, atol=1e-6)}")

    print("=" * 80)

    logger.warning(
        "eagle3: loaded MEGATRON draft weights (tp=%d, mapped=%d, missing=%d, unexpected=%d)",
        world, len(mapped), len(missing), len(unexpected),
    )
    if missing:
        logger.warning("eagle3: draft missing keys after load: %s", missing)
    if unexpected:
        logger.warning("eagle3: draft unexpected keys after load: %s", unexpected)

    # ===== DRAFT_LOAD_PROBE_REVERT_20260821 (调试用,回退时删除本块) =====
    # 主动抛异常:确认本函数确实被执行,并让 traceback 钉死调用链与真身文件路径。
    # raise RuntimeError(
    #     f"[DRAFT_LOAD_PROBE] load_megatron_draft_weights executed -> "
    #     f"draft_path={draft_path}, tp={world}, mapped={len(mapped)}, "
    #     f"missing={len(missing)} ({missing}), unexpected={len(unexpected)} ({unexpected}); "
    #     f"__file__={__file__}"
    # )
    # ===== END DRAFT_LOAD_PROBE_REVERT_20260821 =====
