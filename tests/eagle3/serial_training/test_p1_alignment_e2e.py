# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""P1-1 end-to-end alignment regression against the real initial draft ckpt.

Guards the (token, hidden) pairing through the ACTUAL v3 collection path:
collect_draft_features -> DraftFeatureRecord -> stack_records -> draft forward
-> FrozenTeacherHead -> compute_draft_loss. The initial ckpt was trained with
the EAGLE pairing (f[p], x[p+1]); if any link in our chain regresses to the old
(f[p], x[p]) pairing, the loss here jumps from <1.5 to ~9 (measured:
0.87 vs 9.28, see 开发过程记录/串行训练/串行版本2/P1-1对齐核实结论-20260829.md).

Heavy (loads Qwen3-8B): opt-in via
  EAGLE3_ALIGNMENT_E2E=1 ASCEND_RT_VISIBLE_DEVICES=<free> TORCHDYNAMO_DISABLE=1 \
  python3 -m pytest tests/eagle3/serial_training/test_p1_alignment_e2e.py -q
(TORCHDYNAMO_DISABLE is required: the SpeCo draft model's @torch.compile RMSNorm
triggers a triton-on-NPU vector core exception on this machine.)
"""

import os
import sys

import pytest
import torch

TARGET = os.getenv("EAGLE3_TARGET_PATH", "/home/weight/Qwen3-8B")
DRAFT = os.getenv("EAGLE3_DRAFT_PATH", "/home/weight/qwen3_8b_eagle3")
AUX_HS_INDICES = (2, 18, 33)  # = decoder layers 1/17/32 outputs = verl capture formula

pytestmark = pytest.mark.skipif(
    not os.getenv("EAGLE3_ALIGNMENT_E2E") or not os.path.isdir(TARGET) or not os.path.isdir(DRAFT),
    reason="opt-in e2e test: set EAGLE3_ALIGNMENT_E2E=1 with model paths available",
)


def _device():
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def test_collection_pipeline_matches_ckpt_alignment():
    sys.path.insert(0, "/home/t00972278/verl-SpeCo")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from verl_speco.models.eagle.llama_eagle import LlamaForCausalLMEagle3

    from verl.models.eagle3.collect_plan import build_collect_plan
    from verl.models.eagle3.deferred_training import stack_records
    from verl.models.eagle3.feature_store import DraftFeatureStore, collect_draft_features
    from verl.models.eagle3.frozen_teacher import build_frozen_teacher_head
    from verl.models.eagle3.loss_mcore import compute_draft_loss

    dev = _device()
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = (
        AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        .to(dev)
        .eval()
    )
    cfg = AutoConfig.from_pretrained(DRAFT)
    draft = LlamaForCausalLMEagle3(cfg, attention_backend="sdpa")
    draft.load_state_dict(
        torch.load(f"{DRAFT}/pytorch_model.bin", map_location="cpu", weights_only=True), strict=False
    )
    draft.load_embedding(TARGET)
    draft = draft.to(dev, torch.bfloat16).eval()

    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )
    answer = (
        "In April, Natalia sold 48 clips. In May, she sold half as many, which is 48 / 2 = 24 clips. "
        "Altogether she sold 48 + 24 = 72 clips. The answer is 72."
    )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    response_ids = tok(answer + tok.eos_token, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([prompt_ids, response_ids], dim=1).to(dev)
    prompt_len, seq_len = prompt_ids.shape[1], ids.shape[1]
    response_len = seq_len - prompt_len

    with torch.no_grad():
        out = target(ids, output_hidden_states=True)
    hs = out.hidden_states
    # seq-first (S, B, D), the layout _eagle3_collect_features_step hands over
    aux_sf = torch.cat([hs[i] for i in AUX_HS_INDICES], dim=-1).transpose(0, 1)
    final_sf = hs[-1].transpose(0, 1)  # post-final-norm, same as megatron postprocess input
    loss_mask = torch.zeros(1, seq_len, device=dev)
    loss_mask[:, prompt_len:] = 1.0

    # ---- the real v3 collection path ----
    window = min(64, response_len - 1)
    plan = build_collect_plan(
        prompt_lens=[prompt_len], response_lens=[response_len], global_step=1, window_train_rows=window
    )
    assert plan is not None
    store = DraftFeatureStore()
    store.begin_step(1)
    stored = collect_draft_features(
        store=store, aux_hidden=aux_sf, final_hidden=final_sf, input_ids=ids,
        position_ids=None, loss_mask=loss_mask, plan=plan, global_step=1,
    )
    assert stored == 1

    head = build_frozen_teacher_head(
        target.lm_head.weight.detach(), draft.t2d.to("cpu"), tp_size=1, global_step=1
    )

    batch = stack_records(store.drain(), device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        teacher = head(batch["final_hidden"]).float().cpu()  # draft-vocab teacher
        d_out = draft(
            input_ids=batch["input_ids"], hidden_states=batch["aux_hidden"],
            loss_mask=batch["loss_mask"].float(), position_ids=batch["position_ids"], ttt_length=1,
        )
        good = compute_draft_loss(
            student_logits_per_step=[d_out["logits"][0].float().cpu()],
            teacher_logits=teacher, t2d=torch.ones(teacher.shape[-1], dtype=torch.bool),
            loss_mask=batch["loss_mask"].float().cpu(),
            position_masks_per_step=[m.float().cpu() for m in d_out["position_masks"]],
        )

        # control: the OLD (f[p], x[p]) pairing on the same window must stay bad
        wrong_ids = ids[0].index_select(0, plan.hidden_positions[0].to(dev)).unsqueeze(0)
        d_wrong = draft(
            input_ids=wrong_ids, hidden_states=batch["aux_hidden"],
            loss_mask=batch["loss_mask"].float(), position_ids=batch["position_ids"], ttt_length=1,
        )
        bad = compute_draft_loss(
            student_logits_per_step=[d_wrong["logits"][0].float().cpu()],
            teacher_logits=teacher, t2d=torch.ones(teacher.shape[-1], dtype=torch.bool),
            loss_mask=batch["loss_mask"].float().cpu(),
            position_masks_per_step=[m.float().cpu() for m in d_wrong["position_masks"]],
        )

    good_loss, bad_loss = float(good["loss"]), float(bad["loss"])
    print(f"\n[alignment-e2e] pipeline loss={good_loss:.4f}  old-pairing loss={bad_loss:.4f}")
    # 量纲说明（2026-08-29 实测）：整条序列喂 draft 时正确配对 loss=0.87；窗口式
    # 采集只含 response 行、draft 注意力看不到 prompt 段（SpeCo 窗口设计固有近似，
    # 推理时 draft 是能看到全前缀的），loss 抬到 ~2.7。配对回归（退回 (f[p],x[p])）
    # 的量级是 ~8-9，与窗口效应（~2.7）间隔清晰，故主断言用相对差，绝对界放宽。
    assert bad_loss - good_loss > 3.0, (
        f"pipeline pairing (loss {good_loss:.4f}) no longer clearly beats the old "
        f"(f[p], x[p]) pairing (loss {bad_loss:.4f}) -- the (token, hidden) pairing "
        "has regressed from the EAGLE alignment somewhere in the collection chain"
    )
    assert good_loss < 4.0, (
        f"collection pipeline produced loss {good_loss:.4f} on the initial ckpt; "
        "expected ~2.7 for windowed collection (0.87 full-sequence). Something beyond "
        "the window approximation is off."
    )
