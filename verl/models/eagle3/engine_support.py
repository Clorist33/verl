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
"""EAGLE3 engine-side wiring helpers (Megatron, P2a: PP=1 / TP=1).

Keeps the ``megatron/transformer_impl.py`` diff small: the engine calls
``setup_eagle3_training`` in ``initialize`` and ``eagle3_backward_step`` right
after the policy forward/backward. All EAGLE3-specific building lives here.

The draft is a plain replicated ``nn.Module`` (no Megatron TP) wrapped in DDP
for data parallelism, with its own optimizer -- independent of the policy.
"""

import logging
import os
from typing import Optional

import torch

from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _unwrap_gpt(model_list):
    """Return the GPTModel-ish object owning decoder.layers from a Megatron module list."""
    from verl.models.eagle3.hidden_capture_mcore import _resolve_gpt_model

    # module list is one entry per virtual PP stage; PP=1 -> single entry
    return _resolve_gpt_model(model_list[0])


def build_draft_hf_config(policy_hf_config, eagle3_cfg):
    """Build the draft's HF config from the draft ckpt, recording the policy
    hidden size as ``target_hidden_size`` (the aux hidden dim the draft fuses)."""
    from transformers import AutoConfig

    draft_path = eagle3_cfg.draft_model_path
    if not draft_path:
        raise ValueError("eagle3.draft_model_path is required for enable_train (draft ckpt config)")
    draft_config = AutoConfig.from_pretrained(draft_path, trust_remote_code=True)

    # policy hidden size is what the aux hidden states carry
    policy_hidden = getattr(policy_hf_config, "hidden_size", None)
    if policy_hidden is not None and getattr(draft_config, "target_hidden_size", None) is None:
        draft_config.target_hidden_size = policy_hidden

    if eagle3_cfg.enable_vocab_compression and eagle3_cfg.draft_vocab_size > 0:
        draft_config.draft_vocab_size = eagle3_cfg.draft_vocab_size
    return draft_config


def build_draft_module(policy_hf_config, eagle3_cfg, device, dtype=torch.bfloat16,
                       num_aux_hidden_states=None):
    """Construct the self-written draft, load ckpt weights + vocab mapping.

    Two backends, selected by ``eagle3_cfg.use_megatron_draft``:
      * False (default): plain ``nn.Module`` (LlamaForCausalLMEagle3), replicated,
        DDP for data parallelism. Stable path.
      * True: ``MegatronEagle3DraftModel`` whose TransformerBlock is TP-sharded like
        the policy (saves ~2.2GB/card at TP=4).

    Returns the draft module on ``device`` (NOT yet DDP-wrapped)."""
    if getattr(eagle3_cfg, "use_megatron_draft", False):
        return _build_megatron_draft(
            policy_hf_config, eagle3_cfg, device, dtype=dtype,
            num_aux_hidden_states=num_aux_hidden_states,
        )

    from verl.models.eagle3.draft_mcore import LlamaForCausalLMEagle3

    draft_config = build_draft_hf_config(policy_hf_config, eagle3_cfg)
    draft = LlamaForCausalLMEagle3(draft_config, attention_backend="sdpa")

    # load draft ckpt weights if present (best-effort; missing keys tolerated)
    _maybe_load_draft_weights(draft, eagle3_cfg.draft_model_path)

    # vocab mapping: explicit path overrides; else rely on ckpt-provided / identity
    if eagle3_cfg.vocab_mapping_path:
        draft.load_vocab_mapping(eagle3_cfg.vocab_mapping_path)

    draft = draft.to(device=device, dtype=dtype)
    # RoPE inv_freq must be rebuilt on the target device/dtype (esp. NPU)
    try:
        draft.reset_rope_buffers(dtype=torch.float32)
    except Exception as e:  # pragma: no cover
        logger.warning("eagle3: reset_rope_buffers failed: %r", e)
    return draft


def _get_rope_theta(hf_config, default: float = 10000.0) -> float:
    """Read rope_theta across transformers versions (flat vs nested rope_parameters)."""
    val = getattr(hf_config, "rope_theta", None)
    if val is not None:
        return val
    rope_params = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and rope_params.get("rope_theta") is not None:
        return rope_params["rope_theta"]
    return default


def _resolve_draft_tp_plan(requested_draft_tp, num_query_groups):
    """Decide the draft's TP degree + process group.

    Returns ``(draft_tp_size, tp_group, pg_collection)``.

    - ``requested_draft_tp <= 0``  -> auto: use policy TP, capped down to the largest
      divisor of BOTH policy_tp and num_query_groups (QKV fusion needs
      num_query_groups % draft_tp == 0; e.g. policy_tp=8, groups=4 -> draft_tp=4).
    - ``draft_tp == policy_tp``    -> reuse the GLOBAL TP group (validated default path,
      no sub-group), returns ``tp_group=None`` (draft layers fall back to the global TP).
    - ``draft_tp <  policy_tp``    -> build a dedicated draft TP SUB-GROUP that partitions
      each policy-TP group into ``policy_tp/draft_tp`` contiguous draft groups. Requires
      PP==1 and CP==1 (the EAGLE3 gate already enforces PP==1); otherwise falls back to
      the global group with a warning. NOTE: the sub-group path is not yet validated on
      real multi-process hardware (Step 7).
    """
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    policy_tp = mpu.get_tensor_model_parallel_world_size()

    def _largest_divisor_leq(n, cap):
        for d in range(min(n, cap), 0, -1):
            if n % d == 0 and cap % d == 0:
                return d
        return 1

    if requested_draft_tp and requested_draft_tp > 0:
        draft_tp = int(requested_draft_tp)
        if policy_tp % draft_tp != 0:
            raise ValueError(
                f"eagle3: draft_tensor_parallel_size={draft_tp} must divide policy TP={policy_tp}"
            )
        if num_query_groups % draft_tp != 0:
            raise ValueError(
                f"eagle3: draft_tensor_parallel_size={draft_tp} must divide num_query_groups="
                f"{num_query_groups} (QKV fusion constraint)"
            )
    else:
        draft_tp = _largest_divisor_leq(policy_tp, num_query_groups)
        if draft_tp != policy_tp:
            logger.warning(
                "eagle3: auto-capped draft TP to %d (policy_tp=%d, num_query_groups=%d) "
                "so the QKV fusion stays valid.", draft_tp, policy_tp, num_query_groups,
            )

    # draft_tp == policy_tp -> reuse the global group (no sub-group needed).
    if draft_tp == policy_tp:
        return draft_tp, None, None

    # draft_tp < policy_tp -> need a dedicated sub-group. Only safe at PP==1, CP==1.
    pp = mpu.get_pipeline_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    if pp != 1 or cp != 1 or not (dist.is_available() and dist.is_initialized()):
        logger.warning(
            "eagle3: draft TP sub-group needs PP==1 & CP==1 (got pp=%d cp=%d); "
            "falling back to draft_tp=policy_tp=%d on the global group.", pp, cp, policy_tp,
        )
        return policy_tp, None, None

    # Partition every policy-TP group [g*policy_tp, (g+1)*policy_tp) into policy_tp/draft_tp
    # contiguous draft groups. Build ALL groups collectively (dist.new_group requirement);
    # keep the one that owns this global rank.
    world = dist.get_world_size()
    my_rank = dist.get_rank()
    my_group = None
    for base in range(0, world, policy_tp):
        for sub in range(0, policy_tp, draft_tp):
            ranks = list(range(base + sub, base + sub + draft_tp))
            g = dist.new_group(ranks=ranks)
            if my_rank in ranks:
                my_group = g

    from megatron.core.process_groups_config import ProcessGroupCollection
    pgc = ProcessGroupCollection.use_mpu_process_groups()
    pgc.tp = my_group  # override TP; keep cp/pp/dp from the global mpu state
    logger.warning(
        "eagle3: built dedicated draft TP sub-group (draft_tp=%d, policy_tp=%d) -- "
        "NOT yet validated on real multi-process HW (Step 7).", draft_tp, policy_tp,
    )
    return draft_tp, my_group, pgc


def _build_draft_transformer_config(draft_hf_config, policy_hf_config, param_dtype,
                                    draft_tp_size=None):
    """Build the draft's Megatron ``TransformerConfig`` from its HF config.

    Mirrors the reference recipe (lilac draft_utils.load_eagle_draft_model): a fresh
    1-layer TransformerConfig sized to the DRAFT's own dims. ``draft_tp_size`` sets the
    TP degree (defaults to the live policy TP). Recompute is disabled (1 layer + the
    layer-0 attention pre-hook is stateful) and flash-attn is forced on NPU (MindSpeed
    mask path requires it).
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer.transformer_config import TransformerConfig
    from verl.utils.device import is_npu_available

    if draft_tp_size is None:
        draft_tp_size = mpu.get_tensor_model_parallel_world_size()

    hidden = draft_hf_config.hidden_size
    n_heads = draft_hf_config.num_attention_heads
    config = TransformerConfig(
        num_layers=getattr(draft_hf_config, "num_hidden_layers", 1),
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_query_groups=getattr(draft_hf_config, "num_key_value_heads", n_heads),
        kv_channels=getattr(draft_hf_config, "head_dim", hidden // n_heads),
        ffn_hidden_size=getattr(draft_hf_config, "intermediate_size", 4 * hidden),
        normalization="RMSNorm",
        layernorm_epsilon=getattr(draft_hf_config, "rms_norm_eps", 1e-5),
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        # draft TP degree (defaults to policy TP; may be capped/independent via DRAFT_TP).
        tensor_model_parallel_size=draft_tp_size,
        # capture requires SP off; keep the draft consistent.
        sequence_parallel=False,
        params_dtype=param_dtype,
        pipeline_dtype=param_dtype,
        bf16=(param_dtype == torch.bfloat16),
    )
    # attrs that are not TransformerConfig ctor kwargs in mcore 0.16
    config.seq_length = getattr(draft_hf_config, "max_position_embeddings", 4096)
    config.vocab_size = draft_hf_config.vocab_size
    config.rotary_base = _get_rope_theta(draft_hf_config, default=10000.0)
    config.rope_scaling = getattr(draft_hf_config, "rope_scaling", False) or False
    config.rope_scaling_factor = getattr(draft_hf_config, "rope_scaling_factor", 8.0)
    # draft is 1 layer: recompute saves nothing AND breaks the stateful attn pre-hook.
    config.recompute_granularity = None
    config.recompute_method = None
    config.recompute_num_layers = None
    if is_npu_available:
        config.use_flash_attn = True
    return config


def _build_megatron_draft(policy_hf_config, eagle3_cfg, device, dtype=torch.bfloat16,
                          num_aux_hidden_states=None):
    """Construct the MegatronModule draft (TP-sharded backbone).

    NOTE (Step 4): this builds + places the module. HF->Megatron weight mapping is
    Step 6; until then the backbone starts from random init (a warning is logged).
    The (B,S,H)<->(S,B,H) + dict-return forward adaptation is Step 5.
    """
    logger.warning("-" * 50)
    logger.warning("DRAFT-TRAIN: _build_megatron_draft called, draft_path=%r", eagle3_cfg.draft_model_path)
    logger.warning("-" * 50)
    from verl.models.eagle3.draft_megatron import MegatronEagle3DraftModel

    # ===============================================
    # 第一步:建 draft 的 HF config
    draft_hf_config = build_draft_hf_config(policy_hf_config, eagle3_cfg)

    # ===============================================
    # 第二步:提取关键尺寸
    num_aux = num_aux_hidden_states or getattr(draft_hf_config, "num_aux_hidden_states", 3)
    target_hidden = getattr(draft_hf_config, "target_hidden_size", policy_hf_config.hidden_size)
    draft_vocab = getattr(draft_hf_config, "draft_vocab_size", draft_hf_config.vocab_size)
    num_groups = getattr(draft_hf_config, "num_key_value_heads", draft_hf_config.num_attention_heads)

    # ===============================================
    # 第三步:决定 draft TP 度数并建专用并行组
    # Resolve the draft TP degree + (optional) dedicated sub-group. DRAFT_TP=0 -> auto
    # (policy TP, capped to a divisor of num_query_groups); reuses the global group when
    # draft_tp == policy_tp.
    requested_tp = getattr(eagle3_cfg, "draft_tensor_parallel_size", 0)
    draft_tp_size, tp_group, pg_collection = _resolve_draft_tp_plan(requested_tp, num_groups)

    # ===============================================
    # 第四步:建 draft 的 TransformerConfig, HF config 转成 Megatron 的 TransformerConfig
    tf_config = _build_draft_transformer_config(
        draft_hf_config, policy_hf_config, dtype, draft_tp_size=draft_tp_size
    )

    # ===============================================
    # 第五步:实例化 MegatronEagle3DraftModel
    draft = MegatronEagle3DraftModel(
        config=tf_config,
        vocab_size=draft_hf_config.vocab_size,
        draft_vocab_size=draft_vocab,
        target_hidden_size=target_hidden,
        num_aux_hidden_states=num_aux,
        pre_process=True,
        post_process=True,
        tp_group=tp_group,
        pg_collection=pg_collection,
    )
    draft = draft.to(device=device, dtype=dtype)

    
    # 打印结构（可注释）
    print("=" * 80)
    print("EAGLE3 Draft Model Structure (with shapes):")
    print("=" * 80)
    for name, param in draft.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")
    print("-" * 80)
    for name, buf in draft.named_buffers():
        print(f"{name:60s} (buffer) {tuple(buf.shape)}")
    print("=" * 80)


    # ===============================================
    # 第六步:加载 HF 权重并 TP-shard
    # 从你 DRAFT_PATH=/home/weight/Qwen3-a3B_eagle3 读 HF 格式的 safetensors 权重,做 QKV/gate-up fusion,按 TP 切片,灌进 Megatron 模块

    # Step 6: load + TP-shard the HF draft ckpt (fuses q/k/v + gate/up, shards col/row).
    # embed_tokens is injected from the policy + frozen AFTER this (missing is expected).
    # tp_group=None -> the loader uses the global TP group (draft_tp == policy_tp).
    # A load failure MUST stop training: silently falling back to a random-init draft
    # produces a model that trains on garbage draft weights with no error, which is
    # far harder to diagnose than a hard failure here.
    from verl.models.eagle3.draft_megatron_weights import load_megatron_draft_weights
    load_megatron_draft_weights(draft, eagle3_cfg.draft_model_path, tp_group=tp_group)
    logger.info(
        "eagle3: MEGATRON draft load weight (TP=%d, num_aux=%d, draft_vocab=%d)",
        tf_config.tensor_model_parallel_size, num_aux, draft_vocab,
    )
    logger.warning("-" * 50)
    logger.warning("DRAFT-TRAIN: draft module built successfully, num_params=%d",
                   sum(p.numel() for p in draft.parameters()))
    logger.warning("-" * 50)
    return draft


def _maybe_load_draft_weights(draft, draft_path):
    """Load draft ckpt weights by name intersection (best-effort)."""
    import glob

    if not draft_path or not os.path.isdir(draft_path):
        logger.warning("eagle3: draft_model_path %r not a local dir; skip weight load", draft_path)
        return
    state = {}
    # safetensors first
    try:
        from safetensors.torch import load_file

        for f in sorted(glob.glob(os.path.join(draft_path, "*.safetensors"))):
            state.update(load_file(f))
    except Exception:
        pass
    if not state:
        for f in sorted(glob.glob(os.path.join(draft_path, "*.bin"))):
            state.update(torch.load(f, map_location="cpu"))
    if not state:
        logger.warning("eagle3: no weight files found in %r; draft uses random init", draft_path)
        return
    missing, unexpected = draft.load_state_dict(state, strict=False)
    logger.info("eagle3: loaded draft weights (missing=%d, unexpected=%d)", len(missing), len(unexpected))


class Eagle3TrainingState:
    """Holds the draft module, its optimizer, and the hidden-capture handle.

    One instance lives on the engine as ``self._eagle3``. All fields are None
    when EAGLE3 training is disabled.
    """

    def __init__(self):
        self.draft_module = None       # DDP-wrapped draft
        self.draft_raw = None          # underlying nn.Module (for patch/save)
        self.draft_optimizer = None
        self.capture = None
        self.enabled = False
        self.optim_offload = False     # keep draft AdamW state on CPU between steps


def _inject_and_freeze_draft_embed(draft_raw, gpt) -> bool:
    """Copy the policy's token embedding into the draft and FREEZE it.

    EAGLE3 shares the target model's embedding: the draft ckpt deliberately omits
    ``embed_tokens.weight`` (hence the ``missing=1`` on load), expecting the loader
    to supply it from the policy. Without this the draft trains a random 311M-param
    embedding from scratch -- wrong (noisy token reps) AND expensive (its AdamW state
    is several GB, which was tipping TP=4 over the OOM edge).

    Policy embedding under Megatron is a VocabParallelEmbedding: weight is
    ``(vocab/TP, hidden)`` sharded on the vocab dim. All-gather across the TP group on
    dim 0 to reconstruct the full ``(vocab, hidden)`` table, then copy into the draft's
    plain ``nn.Embedding`` and set ``requires_grad=False`` so the optimizer drops it.

    Returns True on success (embed injected + frozen), False if it had to skip.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    draft = unwrap_draft(draft_raw)
    if not hasattr(draft, "embed_tokens") or draft.embed_tokens.weight is None:
        logger.warning("eagle3: draft has no embed_tokens; skip embed injection")
        return False

    # locate the policy VocabParallelEmbedding weight (vocab/TP, hidden)
    embedding = getattr(gpt, "embedding", None)
    word_emb = getattr(embedding, "word_embeddings", None) if embedding is not None else None
    if word_emb is None or getattr(word_emb, "weight", None) is None:
        logger.warning("eagle3: policy embedding not found; draft embed stays random (unfrozen)")
        return False

    tp_size = mpu.get_tensor_model_parallel_world_size()
    shard = word_emb.weight.detach()  # (vocab/TP, hidden) on this rank
    if tp_size > 1:
        # all-gather the vocab-sharded rows across the TP group -> full (vocab, hidden)
        gathered = [torch.empty_like(shard) for _ in range(tp_size)]
        dist.all_gather(gathered, shard.contiguous(), group=mpu.get_tensor_model_parallel_group())
        full = torch.cat(gathered, dim=0)  # (vocab, hidden)
    else:
        full = shard

    d = draft.embed_tokens.weight
    if full.shape != d.shape:
        # e.g. padded vocab on the policy side; copy the overlapping [:V] rows only.
        v = min(full.shape[0], d.shape[0])
        if full.shape[1] != d.shape[1]:
            logger.warning(
                "eagle3: embed hidden dim mismatch policy=%s draft=%s; skip injection",
                tuple(full.shape), tuple(d.shape),
            )
            return False
        with torch.no_grad():
            d[:v].copy_(full[:v].to(device=d.device, dtype=d.dtype))
        logger.warning(
            "eagle3: embed vocab mismatch policy=%d draft=%d; copied overlapping %d rows",
            full.shape[0], d.shape[0], v,
        )
    else:
        with torch.no_grad():
            d.copy_(full.to(device=d.device, dtype=d.dtype))

    draft.embed_tokens.weight.requires_grad = False
    logger.info(
        "eagle3: injected policy embed into draft + froze it (shape=%s, tp_size=%d)",
        tuple(d.shape), tp_size,
    )
    return True


def _assert_no_sp_cp(gpt) -> None:
    """Hard-forbid Context Parallel for EAGLE3 (SP is now supported).

    EAGLE3 hidden capture records the policy decoder activations and reconstructs
    full ``[B, S, *]`` sequences.

    - SP (sequence parallel): NOW SUPPORTED. eagle3_patch._postprocess gathers the
      SP-sharded ``(S/TP, B, H)`` capture back to full ``(S, B, H)`` via
      ``gather_from_sequence_parallel_region`` before feeding the draft. Megatron MoE
      training also REQUIRES SP when TP>1 (moe_layer raises otherwise), so SP=True is
      the correct setting for MoE policies. [changed 2026-08-11]
    - CP (context parallel): STILL FORBIDDEN. _postprocess has no CP-dim gather, so
      CP>1 would train the draft on mis-sliced hidden. Keep CP=1.

    Fix at the launch script:
      actor_rollout_ref.actor.megatron.context_parallel_size=1
    """
    from megatron.core import parallel_state as mpu

    pcfg = getattr(gpt, "config", None)
    sp = bool(getattr(pcfg, "sequence_parallel", False)) if pcfg is not None else False
    cp = mpu.get_context_parallel_world_size()
    tp = mpu.get_tensor_model_parallel_world_size()

    problems = []
    # ---- [DEPRECATED 2026-08-11] SP check removed ----
    # 旧逻辑：禁止 SP。但 eagle3_patch._postprocess 已实现 SP gather
    # (gather_from_sequence_parallel_region 把 (S/TP,B,H) -> (S,B,H))，且 Megatron MoE
    # 训练要求 TP>1 时必须开 SP，否则 allgather dispatcher 数值错。故放开 SP。
    # SP 冲突根因见 开发过程记录/debug梳理/SP冲突根因与放开.md
    # if sp:
    #     problems.append(
    #         f"sequence_parallel=True (TP={tp}) scatters captured activations across TP "
    #         "ranks on the sequence dim. Set actor_rollout_ref.actor.megatron.sequence_parallel=False."
    #     )
    _ = sp  # SP 现已支持（见上），保留变量避免未使用告警
    # CP 仍禁用：_postprocess 只处理了 SP 的序列 gather，没有处理 CP 的序列切分。
    if cp > 1:
        problems.append(
            f"context_parallel_size={cp} splits the sequence across CP ranks. EAGLE3 hidden "
            "capture requires CP=1. Set actor_rollout_ref.actor.megatron.context_parallel_size=1."
        )
    if problems:
        raise ValueError(
            "EAGLE3 forbids context parallelism (hidden capture needs full sequences): "
            + " ".join(problems)
        )


def setup_eagle3_training(engine, policy_module_list) -> Optional[Eagle3TrainingState]:
    """Build draft + independent optimizer + capture + patch. Call in engine.initialize()
    AFTER the policy module/optimizer are built. Returns the state (or None if disabled).

    Enforces the PP=1 first-version gate here (config can't see PP size).
    """
    from megatron.core import parallel_state as mpu

    eagle3_cfg = engine.model_config.eagle3
    logger.warning(
        "EAGLE3-DIAG[setup]: called. eagle3_is_none=%s enable_train=%s",
        eagle3_cfg is None, getattr(eagle3_cfg, "enable_train", "N/A"),
    )
    if not (eagle3_cfg is not None and eagle3_cfg.enable_train):
        logger.warning("EAGLE3-DIAG[setup]: EARLY RETURN (enable_train falsy) -> draft NOT set up")
        return None

    # ---- PP=1 gate (deferred from config __post_init__) ----
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if pp_size > 1 and not eagle3_cfg.allow_pp_gt_1:
        raise ValueError(
            f"EAGLE3 first version supports PP=1 only (got pipeline_model_parallel_size={pp_size}). "
            "Set model.eagle3.allow_pp_gt_1=True once P4 cross-stage hidden transfer lands."
        )
    logger.warning("EAGLE3-DIAG[setup]: passed PP gate, building draft now (pp=%d)", pp_size)

    from verl.models.eagle3.hidden_capture_mcore import Eagle3HiddenCapture, resolve_capture_layer_ids
    from verl.models.mcore.eagle3_patch import patch_eagle3_postprocess

    # Place the draft on the SAME accelerator as the policy. Use verl's
    # platform-agnostic device utils so this works on NPU (torch.cuda.is_available()
    # is False on NPU, which would otherwise wrongly drop the draft onto CPU and
    # crash on the policy/draft device mismatch).
    device_name = get_device_name()  # 'npu' | 'cuda' | 'cpu'
    device = torch.device(device_name if device_name == "cpu" else f"{device_name}:{get_device_id()}")
    param_dtype = getattr(engine, "param_dtype", torch.bfloat16)

    gpt = _unwrap_gpt(policy_module_list)

    # ---- [DEPRECATED 2026-08-11] EAGLE3 workaround: force sequence_parallel=False ----
    # 旧逻辑：强制关闭 SP。原因见 _assert_no_sp_cp 旧注释（capture 抓到 SP 切片会训练错误）。
    # 问题：与 Megatron MoE 训练守卫冲突（moe_layer.py: training+TP>1+SP=False -> raise），
    # 且 eagle3_patch._postprocess 其实已实现 SP gather（gather_from_sequence_parallel_region），
    # 本可支持 SP=True。故取消强制关 SP。详见 开发过程记录/debug梳理/SP冲突根因与放开.md
    # pcfg = getattr(gpt, "config", None)
    # if pcfg is not None and hasattr(pcfg, "sequence_parallel"):
    #     pcfg.sequence_parallel = False

    # ---- hard gate: forbid CP (SP 现已放开；CP 仍无对应 gather，保持禁用) ----
    _assert_no_sp_cp(gpt)

    num_layers = len(gpt.decoder.layers)

    # =================================================
    # 确定抽取policy模型的哪几层hiddenstate
    # capture layer ids: ckpt 内嵌 > 配置文件 > 默认公式
    ckpt_ids = _draft_ckpt_layer_ids(engine.model_config.eagle3)                                          # 从 draft ckpt 读
    config_ids = list(eagle3_cfg.capture_layer_ids) if eagle3_cfg.capture_layer_ids else None             # 从配置读：eagle3_cfg = engine.model_config.eagle3
    layer_ids = resolve_capture_layer_ids(num_layers, config_ids=config_ids, ckpt_ids=ckpt_ids)           # 实际上，config_ids=None，ckpt_ids=None，走的resolve_capture_layer_ids里面的默认公式计算
    # breakpoint()

    # =================================================
    # 建 draft 模块，重点看这里！！！！！！！！！！！！！！！！
    # build draft (num_aux = #capture layers, needed by the Megatron draft's fc dim)
    draft_raw = build_draft_module(
        engine.model_config.hf_config, eagle3_cfg, device, dtype=param_dtype,
        num_aux_hidden_states=len(layer_ids),
    )

    # =================================================
    # 共享并冻结 policy embedding
    # EAGLE3 shares the target's token embedding: inject the policy embedding into the
    # draft and FREEZE it. Must run BEFORE _build_draft_optimizer so the frozen embed is
    # excluded by its requires_grad filter (drops ~311M params + their AdamW state).
    _inject_and_freeze_draft_embed(draft_raw, gpt)


    # =================================================
    # 可选激活重计算（省 draft 前向的激活显存）
    # Optional: activation-checkpoint the draft backbone to cut the draft-path
    # activation peak (P5-09). Default off (unchanged behavior); flag lives on the
    # raw module and is honored inside draft_mcore.forward (ttt==1 + training only).
    if getattr(eagle3_cfg, "draft_forward_checkpoint", False):    # draft_forward_checkpoint=True的时候会开启draft的重计算，默认false，不开启draft的重计算
        draft_raw._use_forward_checkpoint = True
        logger.info("eagle3: draft backbone activation-checkpointing ENABLED")

    
    # =================================================
    # DDP 包装（为 data parallelism 包 DDP，用 PyTorch DDP 同步 draft 的梯度）
    # DDP wrap for data parallelism (draft is small; replicated, no TP)
    draft_module = _wrap_draft_ddp(draft_raw)

    
    # =================================================
    # 建draft 独立优化器
    # independent optimizer
    draft_optimizer = _build_draft_optimizer(draft_module, eagle3_cfg)

    # =================================================
    # 挂 hidden capture + patch 前向，loss的计算逻辑也在这里，重点看这里！！！！！！！！！！！！！！！！
    # hidden capture on the policy decoder layers + patch _postprocess
    capture = Eagle3HiddenCapture(gpt, capture_layer_ids=layer_ids).register()
    ## 
    patch_eagle3_postprocess(
        gpt, draft_raw, capture,
        ttt_length=eagle3_cfg.ttt_length,
        gamma=0.8,
        temperature=1.0,
    )

    # =================================================
    # 包进 Eagle3TrainingState 并返回
    state = Eagle3TrainingState()
    state.draft_module = draft_module
    state.draft_raw = draft_raw
    state.draft_optimizer = draft_optimizer
    state.capture = capture
    state.enabled = True
    state.optim_offload = bool(getattr(eagle3_cfg, "draft_optim_offload", False))
    logger.info("eagle3: training set up (layers=%s, ttt_length=%d, num_layers=%d, optim_offload=%s)",
                layer_ids, eagle3_cfg.ttt_length, num_layers, state.optim_offload)
    return state


def _draft_ckpt_layer_ids(eagle3_cfg):
    """Read eagle_aux_hidden_state_layer_ids from the draft ckpt config, if present."""
    try:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(eagle3_cfg.draft_model_path, trust_remote_code=True)
        for key in ("eagle_aux_hidden_state_layer_ids", "target_hidden_layer_ids"):
            ids = getattr(c, key, None)
            if ids:
                return [int(i) for i in ids]
    except Exception:
        pass
    return None


def _wrap_draft_ddp(draft_raw):
    """Wrap the draft in torch DDP over the data-parallel group (if distributed)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return draft_raw
    from megatron.core import parallel_state as mpu
    try:
        dp_group = mpu.get_data_parallel_group()
    except Exception:
        dp_group = None
    if dp_group is None or torch.distributed.get_world_size(dp_group) == 1:
        return draft_raw
    # device_ids must reference the local accelerator (NPU/CUDA); None on CPU.
    device_ids = None if get_device_name() == "cpu" else [get_device_id()]
    return torch.nn.parallel.DistributedDataParallel(
        draft_raw, device_ids=device_ids, process_group=dp_group, find_unused_parameters=False
    )


def _build_draft_optimizer(draft_module, eagle3_cfg):
    """Plain torch AdamW for the draft (independent of the Megatron optimizer)."""
    params = [p for p in draft_module.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=eagle3_cfg.draft_optim_lr,
        weight_decay=eagle3_cfg.draft_optim_weight_decay,
    )


def unwrap_draft(draft_module):
    """Return the underlying draft nn.Module (strip DDP ``module.`` wrapping)."""
    m = draft_module
    while hasattr(m, "module") and not hasattr(m, "lm_head"):
        m = m.module
    return m


def export_draft_weights(state, dtype=None):
    """Yield ``("draft." + name, tensor)`` for every draft parameter + buffer.

    Called by the engine's ``get_per_tensor_param`` to append the trained draft
    to the policy weight stream. The P1 receiver
    (``vllm_rollout/utils.py:_split_weights_by_draft_prefix``) strips the
    ``draft.`` prefix and routes these to the vLLM/SGLang drafter.

    For Megatron draft (MegatronEagle3DraftModel):
      - Performs TP all-gather for sharded weights (linear_qkv, linear_fc1, etc.)
      - Converts Megatron format to HF format (unfuse qkv/gate_up, rename keys)

    For HF draft (plain nn.Module):
      - No TP, weights are already full tensors; exports directly

    NOTE: parameter *names* here are HF-style after conversion (``embed_tokens.weight``,
    ``midlayer.self_attn.q_proj.weight``, ``fc.weight``, ``lm_head.weight``,
    ``t2d``/``d2t`` buffers, ...). Whether these line up with the vLLM EAGLE3
    drafter's expected names is handled by the receiver's
    ``_adapt_weight_names_for_model`` and can only be fully verified on real
    hardware with a real drafter. Vocab-mapping buffers (t2d/d2t) are included so
    the drafter's compression map stays in sync.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("🔥 export_draft_weights called")

    if state is None or not state.enabled:
        logger.warning("🔥 export_draft_weights: state is None or not enabled, returning early")
        return

    draft = unwrap_draft(state.draft_module)

    # Log draft parameter device locations
    logger.warning("🔥 Draft parameters device locations:")
    for name, param in list(draft.named_parameters())[:5]:
        logger.warning(f"   {name}: device={param.device}, shape={tuple(param.shape)}, requires_grad={param.requires_grad}")

    # Check if this is a Megatron draft (needs TP gather + format conversion)
    from verl.models.eagle3.draft_megatron import MegatronEagle3DraftModel
    is_megatron_draft = isinstance(draft, MegatronEagle3DraftModel)

    if is_megatron_draft:
        # Megatron draft: gather TP shards + convert to HF format
        from verl.models.eagle3.draft_megatron_weights import map_megatron_to_hf_draft

        # Collect Megatron state dict
        megatron_state = {}
        for name, param in draft.named_parameters():
            megatron_state[name] = param.detach()
        for name, buf in draft.named_buffers():
            if name.endswith(("t2d", "d2t")):
                megatron_state[name] = buf.detach()

        # 🔥 DEBUG: Print all collected Megatron keys
        print("=" * 80)
        print(f"🔥 DRAFT-EXPORT-DEBUG: Collected {len(megatron_state)} keys from draft.named_parameters/buffers:")
        for i, key in enumerate(sorted(megatron_state.keys()), 1):
            tensor = megatron_state[key]
            print(f"  [{i:2d}] {key:60s} shape={str(tuple(tensor.shape)):30s}")
        print("=" * 80)

        # Get draft config for unfusing
        # Assume draft uses same head config as policy's first layer
        config = draft.config
        num_heads = config.num_attention_heads
        num_groups = getattr(config, "num_query_groups", num_heads)
        # head_dim = config.hidden_size // num_heads
        head_dim = getattr(config, "kv_channels", config.hidden_size // config.num_attention_heads)
        tp_group = draft.tp_group  # Draft's own TP group (may differ from policy)

        # Convert Megatron -> HF
        hf_state = map_megatron_to_hf_draft(
            megatron_state,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            tp_group=tp_group
        )

        # DRAFT-WEIGHT-DEBUG: Print export statistics
        print("=" * 80)
        print(f"DRAFT-WEIGHT-DEBUG: Exporting Megatron draft to HF format")
        print(f"  num_heads={num_heads}, num_groups={num_groups}, head_dim={head_dim}")
        print("-" * 80)
        print("Megatron state (before export) - showing first 10:")
        for name in sorted(megatron_state.keys())[:10]:
            tensor = megatron_state[name]
            if tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                print(f"  [MEGATRON-PRE] {name}: shape={tuple(tensor.shape)} "
                      f"mean={tensor.float().mean():.6f} std={tensor.float().std():.6f}")
        print("-" * 80)
        print("HF state (after export) - showing first 10:")
        for name in sorted(hf_state.keys())[:10]:
            tensor = hf_state[name]
            print(f"  [HF-EXPORTED] {name}: shape={tuple(tensor.shape)} "
                  f"mean={tensor.float().mean():.6f} std={tensor.float().std():.6f}")
        print("=" * 80)

        # 🔥 DEBUG: Save exported weights for comparison
        import os
        from megatron.core import parallel_state as mpu
        rank = mpu.get_tensor_model_parallel_rank()
        if rank == 0:  # Only save on TP rank 0
            save_dir = "/home/t00972278/draft_weight_debug"
            os.makedirs(save_dir, exist_ok=True)
            # Get global step from environment or use timestamp
            import time
            timestamp = int(time.time())
            save_path = f"{save_dir}/exported_step_{timestamp}.pt"
            # Save all weights
            save_dict = {name: tensor.cpu() for name, tensor in hf_state.items()}
            torch.save(save_dict, save_path)
            print(f"🔥 DEBUG: Saved exported weights to {save_path}")
            print(f"   Total weights: {len(save_dict)}")

        # Yield HF-format weights
        for name, tensor in hf_state.items():
            t = tensor
            if dtype is not None:
                t = t.to(dtype)
            yield f"draft.{name}", t
    else:
        # HF draft: plain nn.Module, no TP, export directly
        for name, param in draft.named_parameters():
            t = param.detach()
            if dtype is not None:
                t = t.to(dtype)
            yield f"draft.{name}", t
        for name, buf in draft.named_buffers():
            # include t2d/d2t vocab-mapping buffers (skip transient rope caches)
            if name.endswith(("t2d", "d2t")):
                yield f"draft.{name}", buf.detach()


@torch.no_grad()
def _offload_draft_optimizer(optimizer):
    """Move the draft AdamW state (exp_avg / exp_avg_sq) to CPU to free device HBM.

    The draft optimizer is a plain torch.optim.AdamW; its per-parameter state tensors
    (first/second moments) live on the accelerator and are ~2x the trainable-param
    size. On a tight TP=4 card these few hundred MB can be the difference between
    fitting and OOM, so park them on CPU between steps. Mirrors verl's Megatron
    optimizer offload (megatron_utils.py:offload_megatron_optimizer).
    """
    if optimizer is None:
        return
    for v in optimizer.state.values():
        if "exp_avg" in v and v["exp_avg"] is not None:
            v["exp_avg"] = v["exp_avg"].to("cpu", non_blocking=True)
        if "exp_avg_sq" in v and v["exp_avg_sq"] is not None:
            v["exp_avg_sq"] = v["exp_avg_sq"].to("cpu", non_blocking=True)


@torch.no_grad()
def _load_draft_optimizer(optimizer):
    """Bring the draft AdamW state back onto the accelerator right before ``step()``."""
    if optimizer is None:
        return
    dev = get_device_id()
    for v in optimizer.state.values():
        if "exp_avg" in v and v["exp_avg"] is not None:
            v["exp_avg"] = v["exp_avg"].to(dev, non_blocking=True)
        if "exp_avg_sq" in v and v["exp_avg_sq"] is not None:
            v["exp_avg_sq"] = v["exp_avg_sq"].to(dev, non_blocking=True)


def eagle3_backward_step(engine) -> Optional[float]:
    """Backward the stashed L_draft and step the draft optimizer. Call right after
    the policy forward_backward_func. Returns the draft loss value (or None).

    The draft graph is rooted at the DETACHED aux hidden, so it is independent of
    the policy graph and survives the policy backward.
    """
    state = getattr(engine, "_eagle3", None)
    if state is None or not state.enabled:
        return None

    from verl.models.mcore.eagle3_patch import drain_draft_losses

    gpt = _unwrap_gpt(engine.module)
    losses = drain_draft_losses(gpt)
    if not losses:
        return None

    draft_loss = torch.stack([l for l in losses]).mean()
    state.draft_optimizer.zero_grad(set_to_none=True)

    # ========== 🔬 EXPERIMENT: Draft training disabled ==========
    # Testing whether sync path corrupts weights (e.g., missing lm_head).
    # If acceptance rate still drops at step 2 with frozen weights → sync bug.
    # If it stays stable → training updates too large.
    # TODO: Remove the next 2 lines and uncomment the block below to restore.
    # print("🔬 EXPERIMENT: Draft training DISABLED (weights frozen at init)")
    # return float(draft_loss.detach().item())
    # ========== END EXPERIMENT ==========

    # ========== ORIGINAL CODE (commented out for experiment) ==========
    draft_loss.backward()
    logger.warning("-" * 50)
    logger.warning("DRAFT-TRAIN: draft backward done, draft_loss=%f", draft_loss.item())
    logger.warning("-" * 50)
    
    clip = engine.model_config.eagle3.draft_optim_clip_grad
    if clip and clip > 0:
        torch.nn.utils.clip_grad_norm_(state.draft_module.parameters(), max_norm=clip)
    
    # optim state must be on-device for step(); park it back on CPU afterwards.
    if getattr(state, "optim_offload", False):
        _load_draft_optimizer(state.draft_optimizer)
    state.draft_optimizer.step()
    if getattr(state, "optim_offload", False):
        _offload_draft_optimizer(state.draft_optimizer)
    
    logger.warning("-" * 50)
    logger.warning("DRAFT-METRIC: draft_loss calculated = %f", draft_loss.item())
    logger.warning("-" * 50)
    return float(draft_loss.detach().item())
    # ========== END ORIGINAL CODE ==========

