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
"""EAGLE3 real-machine engine-level smoke test (PP=1 / TP=1).

This is the FIRST real-hardware check for the EAGLE3 draft-training wiring. It
exercises the real code path -- ``hf_to_mcore_config`` -> ``init_mcore_model``
-> real ``GPTModel`` -> real ``setup_eagle3_training`` (hidden capture +
_postprocess patch + draft + independent optimizer) -> real
``eagle3_backward_step`` -- on one device, with NO full RL trainer.

What it proves (Tier-1, weights-light):
  1. Megatron imports and a real GPTModel builds on this box (NPU/CUDA/CPU).
  2. The forward hooks capture aux hidden of the RIGHT shape (if wrong, the
     draft's project_hidden_states dim assert fires and the loss list stays
     empty -> this test FAILS loudly).
  3. L_draft is computed, backward runs, the draft optimizer steps.
  4. Overfitting one fixed batch drives draft loss DOWN (learning works).
  5. Gradient isolation: policy params get NO grad; draft params DO.
  6. refit export yields ``draft.*`` names.

What it does NOT prove (needs a matching EAGLE3 draft ckpt + full pipeline):
  - draft weight-name alignment with the vLLM/SGLang drafter (P3 refit),
  - real acceptance-rate lift end-to-end.

------------------------------------------------------------------------------
Run (single card):
    cd /home/t00972278/verl
    POLICY_PATH=/path/to/policy_hf  python tests/eagle3/real_machine_smoke.py

Optional env:
    DRAFT_PATH   real EAGLE3 draft ckpt dir (with config.json [+ weights, t2d/d2t]).
                 If unset, a 1-layer draft config is synthesized from the policy
                 and the draft runs from RANDOM init (fine for Tier-1).
    TTT_LENGTH   1 (default, P2a single-step) or >1 (P2b multi-step).
    STEPS        overfit iterations on the fixed batch (default 30).
    DRAFT_LR     draft AdamW lr (default 1e-3, higher than prod so the loss
                 visibly drops within STEPS on random init).
    LOAD_POLICY  1 (default) load real policy HF weights; 0 = random policy
                 (still valid: draft overfits whatever teacher it sees).
    SEQ_LEN / BATCH   fixed-batch shape (default 32 / 2).
------------------------------------------------------------------------------
"""

import os
import sys
import tempfile

import torch

BANNER = "=" * 78


def log(msg=""):
    print(msg, flush=True)


def env(key, default):
    return os.environ.get(key, default)


def pick_device():
    """NPU-aware device pick (torch.cuda.is_available() is False on Ascend)."""
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(0)
            return torch.device("npu:0"), "npu"
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0"), "cuda"
    return torch.device("cpu"), "cpu"


def init_distributed(device_type):
    """Single-process (world=1) group, so mpu.get_*_world_size() works."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "hccl" if device_type == "npu" else ("nccl" if device_type == "cuda" else "gloo")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, world_size=1, rank=0)
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    try:
        model_parallel_cuda_manual_seed(1234)
    except Exception:
        pass


def synthesize_draft_config(policy_hf_config, dst_dir, draft_vocab_size=0):
    """Write a minimal 1-layer draft config.json derived from the policy.

    Used only when DRAFT_PATH is not given: lets the draft run from random init
    for the Tier-1 wiring/learning check. num_aux defaults to 3 downstream.
    """
    import json

    cfg = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "hidden_size": policy_hf_config.hidden_size,
        "intermediate_size": getattr(policy_hf_config, "intermediate_size", 4 * policy_hf_config.hidden_size),
        "num_hidden_layers": 1,
        "num_attention_heads": policy_hf_config.num_attention_heads,
        "num_key_value_heads": getattr(
            policy_hf_config, "num_key_value_heads", policy_hf_config.num_attention_heads
        ),
        "vocab_size": policy_hf_config.vocab_size,
        "max_position_embeddings": getattr(policy_hf_config, "max_position_embeddings", 4096),
        "rms_norm_eps": getattr(policy_hf_config, "rms_norm_eps", 1e-5),
        "rope_theta": getattr(policy_hf_config, "rope_theta", 10000.0),
        "target_hidden_size": policy_hf_config.hidden_size,
    }
    if draft_vocab_size and draft_vocab_size > 0:
        cfg["draft_vocab_size"] = draft_vocab_size
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return dst_dir


class _EngineShim:
    """Minimal stand-in for MegatronEngine that setup_eagle3_training +
    eagle3_backward_step read. We keep it tiny on purpose so the test drives
    the REAL engine_support code, not a reimplementation."""

    def __init__(self, module_list, model_config, param_dtype):
        self.module = module_list          # list[GPTModel] (PP=1 -> len 1)
        self.model_config = model_config   # exposes .eagle3 and .hf_config
        self.param_dtype = param_dtype
        self._eagle3 = None


class _ModelConfigShim:
    def __init__(self, hf_config, eagle3_cfg):
        self.hf_config = hf_config
        self.eagle3 = eagle3_cfg


def build_eagle3_config(draft_path, ttt_length, draft_lr, draft_vocab_size):
    from verl.workers.config.model import Eagle3Config

    return Eagle3Config(
        enable_train=True,
        backend="megatron",
        draft_model_path=draft_path,
        ttt_length=ttt_length,
        draft_optim_lr=draft_lr,
        enable_vocab_compression=bool(draft_vocab_size and draft_vocab_size > 0),
        draft_vocab_size=draft_vocab_size or 0,
    )


def build_policy_gptmodel(policy_path, dtype, load_weights):
    """Real path: HF config -> mcore tf_config -> real GPTModel (+ HF weights)."""
    from transformers import AutoConfig

    from verl.models.mcore import hf_to_mcore_config, init_mcore_model

    hf_config = AutoConfig.from_pretrained(policy_path, trust_remote_code=True)
    tf_config = hf_to_mcore_config(hf_config, dtype)
    tie = bool(getattr(hf_config, "tie_word_embeddings", False))
    gpt = init_mcore_model(
        tf_config, hf_config,
        pre_process=True, post_process=True,
        share_embeddings_and_output_weights=tie,
    )
    if load_weights:
        try:
            from verl.utils.model import load_megatron_gptmodel_weights

            cfg_shim = type("C", (), {"model": type("M", (), {"path": policy_path})()})()
            load_megatron_gptmodel_weights(
                cfg_shim, hf_config, [gpt], params_dtype=dtype, is_value_model=False
            )
            log("  policy HF weights loaded.")
        except Exception as e:
            log(f"  [warn] policy weight load failed ({e!r}); continuing with RANDOM policy "
                f"(teacher is still a valid target for the draft).")
    return gpt, hf_config


def realign_draft_device(state, policy_device, eagle3_cfg):
    """Move draft to the policy's device if setup put it elsewhere (the NPU
    cuda.is_available()==False case) and rebuild its optimizer on that device."""
    from verl.models.eagle3.engine_support import _build_draft_optimizer, unwrap_draft

    draft = unwrap_draft(state.draft_module)
    dparam = next(draft.parameters())
    if dparam.device != policy_device:
        log(f"  [finding] draft was on {dparam.device} but policy is on {policy_device}; "
            f"moving draft (engine_support device pick is not NPU-aware -- flag for fix).")
        draft.to(policy_device)
        try:
            draft.reset_rope_buffers(dtype=torch.float32)
        except Exception:
            pass
        state.draft_module = draft
        state.draft_raw = draft
        state.draft_optimizer = _build_draft_optimizer(draft, eagle3_cfg)
    return draft


def make_fixed_batch(batch, seq_len, vocab_size, device):
    """One fixed batch, reused every step so loss MUST drop if learning works."""
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    loss_mask = torch.ones(batch, seq_len, dtype=torch.int64, device=device)
    # causal mask [b, 1, s, s], True = masked (megatron/local-attn convention)
    tri = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    attn = tri.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len).contiguous()
    return input_ids, position_ids, loss_mask, attn


def main():
    policy_path = env("POLICY_PATH", "")
    if not policy_path:
        log("ERROR: set POLICY_PATH=/path/to/policy_hf (a HF model dir).")
        return 2
    draft_path = env("DRAFT_PATH", "")
    ttt_length = int(env("TTT_LENGTH", "1"))
    steps = int(env("STEPS", "30"))
    draft_lr = float(env("DRAFT_LR", "1e-3"))
    load_policy = env("LOAD_POLICY", "1") == "1"
    seq_len = int(env("SEQ_LEN", "32"))
    batch = int(env("BATCH", "2"))
    draft_vocab_size = int(env("DRAFT_VOCAB_SIZE", "0"))

    log(BANNER)
    log("EAGLE3 real-machine engine smoke test (PP=1 / TP=1)")
    log(BANNER)

    try:
        import megatron  # noqa: F401
    except Exception as e:
        log(f"FAIL: megatron.core not importable on this box: {e!r}")
        log("      (this test needs a working Megatron-core install.)")
        return 1

    device, dtype_name = pick_device()
    dtype = torch.bfloat16 if dtype_name != "cpu" else torch.float32
    log(f"device={device} ({dtype_name}), dtype={dtype}, ttt_length={ttt_length}, steps={steps}")

    init_distributed(dtype_name)

    log("\n[1/5] building real policy GPTModel ...")
    gpt, hf_config = build_policy_gptmodel(policy_path, dtype, load_policy)
    gpt = gpt.to(device)
    gpt.train()
    n_layers = len(gpt.decoder.layers)
    log(f"  GPTModel built: {n_layers} decoder layers, hidden={hf_config.hidden_size}, "
        f"vocab={hf_config.vocab_size}")

    tmp = None
    if not draft_path:
        tmp = tempfile.mkdtemp(prefix="eagle3_draft_")
        draft_path = synthesize_draft_config(hf_config, tmp, draft_vocab_size)
        log(f"  no DRAFT_PATH -> synthesized 1-layer draft config at {draft_path} (RANDOM init)")

    eagle3_cfg = build_eagle3_config(draft_path, ttt_length, draft_lr, draft_vocab_size)
    model_config = _ModelConfigShim(hf_config, eagle3_cfg)
    engine = _EngineShim([gpt], model_config, dtype)

    log("\n[2/5] setup_eagle3_training (real capture + patch + draft + optimizer) ...")
    from verl.models.eagle3.engine_support import eagle3_backward_step, export_draft_weights, setup_eagle3_training

    engine._eagle3 = setup_eagle3_training(engine, engine.module)
    if engine._eagle3 is None or not engine._eagle3.enabled:
        log("FAIL: setup_eagle3_training returned disabled state.")
        return 1
    realign_draft_device(engine._eagle3, device, eagle3_cfg)
    log("  draft + capture + patch installed.")

    # snapshot a policy param to prove it never moves (frozen / no draft grad leak)
    p_ref_name, p_ref = next(iter(gpt.named_parameters()))
    p_ref_before = p_ref.detach().clone()

    log(f"\n[3/5] overfitting one fixed batch for {steps} steps ...")
    input_ids, position_ids, loss_mask, attn = make_fixed_batch(
        batch, seq_len, hf_config.vocab_size, device
    )
    losses = []
    for it in range(steps):
        with torch.no_grad():
            pass  # policy fwd graph not needed for draft; draft roots at detached hidden
        # policy forward triggers the patched _postprocess -> stashes L_draft
        _ = gpt(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attn,
            loss_mask=loss_mask,
        )
        dl = eagle3_backward_step(engine)
        if dl is None:
            log(f"FAIL: no draft loss stashed at step {it} -- capture/patch/shape path broke. "
                f"(check aux-hidden dim vs draft fc expectation.)")
            return 1
        losses.append(dl)
        if it < 3 or it % max(1, steps // 6) == 0 or it == steps - 1:
            log(f"    step {it:3d}  L_draft = {dl:.5f}")

    log("\n[4/5] checks ...")
    ok = True

    # (a) loss went down (overfit): mean of last 3 < mean of first 3
    head = sum(losses[:3]) / min(3, len(losses))
    tail = sum(losses[-3:]) / min(3, len(losses))
    drop = head - tail
    if drop > 0:
        log(f"  PASS  L_draft decreased: first3={head:.4f} -> last3={tail:.4f} (drop {drop:.4f})")
    else:
        ok = False
        log(f"  FAIL  L_draft did NOT decrease: first3={head:.4f} -> last3={tail:.4f}. "
            f"(learning broken, or lr too low -- try DRAFT_LR=3e-3.)")

    # (b) gradient isolation: policy params carry NO grad; draft params DO
    from verl.models.eagle3.engine_support import unwrap_draft

    policy_with_grad = [n for n, p in gpt.named_parameters() if p.grad is not None]
    draft = unwrap_draft(engine._eagle3.draft_module)
    draft_with_grad = [n for n, p in draft.named_parameters() if p.grad is not None]
    if not policy_with_grad:
        log(f"  PASS  policy has 0 params with grad (draft backward stayed off the policy graph)")
    else:
        ok = False
        log(f"  FAIL  {len(policy_with_grad)} policy params have grad (leak!): {policy_with_grad[:3]} ...")
    if draft_with_grad:
        log(f"  PASS  draft has {len(draft_with_grad)} params with grad (it is training)")
    else:
        ok = False
        log(f"  FAIL  draft has 0 params with grad (optimizer saw nothing)")

    # (c) policy weights unchanged (no in-place teacher corruption)
    if torch.equal(p_ref.detach().to(p_ref_before.device), p_ref_before):
        log(f"  PASS  policy param '{p_ref_name}' unchanged after {steps} draft steps")
    else:
        ok = False
        log(f"  FAIL  policy param '{p_ref_name}' CHANGED -- teacher was mutated")

    # (d) refit export yields draft.* names + t2d/d2t
    names = [n for n, _ in export_draft_weights(engine._eagle3, dtype=dtype)]
    has_prefix = names and all(n.startswith("draft.") for n in names)
    has_maps = any(n.endswith("t2d") for n in names) and any(n.endswith("d2t") for n in names)
    if has_prefix and has_maps:
        log(f"  PASS  refit export: {len(names)} tensors, all 'draft.*', incl t2d/d2t")
    else:
        ok = False
        log(f"  FAIL  refit export malformed (prefix_ok={has_prefix}, maps_ok={has_maps})")

    log("\n[5/5] verdict")
    log(BANNER)
    if ok:
        log("RESULT: PASS -- EAGLE3 engine wiring works end-to-end on this hardware.")
        log("        Next: swap in a matching EAGLE3 draft ckpt (DRAFT_PATH) and run the")
        log("        full GRPO loop to validate P3 refit + acceptance-rate lift.")
    else:
        log("RESULT: FAIL -- see the FAIL line(s) above.")
    log(BANNER)

    if tmp:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
