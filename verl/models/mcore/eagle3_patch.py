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
"""EAGLE3 ``_postprocess`` patch (Megatron), mirroring ``mtp_patch.py``.

Under Megatron there is no central forward call -- the pipeline scheduler runs
microbatches, so hidden capture/use goes into ``GPTModel._postprocess`` just
like verl's MTP. This module patches ``_postprocess`` to, per microbatch:

1. compute the policy logits normally (unchanged policy path -> teacher);
2. pull the 3 aux hidden states captured in-flight (Eagle3HiddenCapture, already
   detached), feed the self-written draft, get per-step student logits;
3. compute ``L_draft`` (dense soft CE, loss_mcore) and **stash** it on the model.

Crucial difference from MTP: we do NOT call ``MTPLossAutoScaler`` -- the draft
gradient must not flow into the policy backbone. The stashed ``L_draft`` is
backward-ed separately by the engine with the draft's own optimizer (the draft
graph is rooted at the detached aux hidden, so it is independent of the policy
graph and survives the policy backward).

First version: PP=1 (draft on the post_process rank, all aux hidden local).
"""

import logging
import os
from typing import List, Optional

import torch

from verl.models.eagle3.loss_mcore import compute_draft_loss

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from megatron.core.models.gpt.gpt_model import GPTModel
except ImportError:  # pragma: no cover
    GPTModel = None

try:
    from megatron.core.utils import unwrap_model
except ImportError:  # pragma: no cover
    try:
        from verl.utils.megatron_utils import unwrap_model
    except ImportError:
        unwrap_model = None


class _NeverRaised(Exception):
    """占位异常：torch 未提供 OutOfMemoryError 时使用，保证 except 子句语法合法且永不命中。"""


# ---- OOM 识别（两处 draft 异常处理共用）----
# torch.OutOfMemoryError 在 torch 2.x 提供；torch_npu 的 NPU OOM 也抛这个类
# （2026-08-26 日志实证: OutOfMemoryError('NPU out of memory. Tried to allocate ...')）。
# 必须用异常类型判断，不要用 "out of memory" in str(e) —— 后者依赖错误文案，
# 换版本或换后端就失效，是之前 82 次静默失败没被及时发现的帮凶之一。
_OOM_ERRORS: tuple = tuple(
    e
    for e in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None))
    if isinstance(e, type) and issubclass(e, BaseException)
)
if not _OOM_ERRORS:  # pragma: no cover - 极老版本 torch 兜底，交给通用分支处理
    _OOM_ERRORS = (_NeverRaised,)


def _eagle3_empty_cache() -> None:
    """清 NPU/CUDA 碎片显存，阻断 OOM 累积溢出到 policy 路径。自身异常不外抛。"""
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as cache_err:
        logger.warning("eagle3_patch: empty_cache() 失败: %r", cache_err)


def _eagle3_strict_draft() -> bool:
    """EAGLE3_STRICT_DRAFT=1 时，并行路径的 draft 失败也直接抛出（调试用）。"""
    return os.getenv("EAGLE3_STRICT_DRAFT", "0") == "1"


def _get_patching_model(model: torch.nn.Module):
    """Unwrap to the GPTModel that owns ``_postprocess`` (mirrors mtp_patch)."""
    m = unwrap_model(model) if unwrap_model is not None else model
    if GPTModel is not None and isinstance(m, GPTModel):
        return m
    if hasattr(m, "language_model") and (GPTModel is None or isinstance(m.language_model, GPTModel)):
        return m.language_model
    if hasattr(m, "_postprocess"):
        return m
    logger.warning("eagle3_patch: %s is not a supported model for _postprocess patch", type(model).__name__)
    return None

# ========================================draft前向的入口========================================
def patch_eagle3_postprocess(
    model: torch.nn.Module,
    draft_module: torch.nn.Module,
    capture,
    ttt_length: int = 1,
    gamma: float = 0.8,
    temperature: float = 1.0,
):
    """Attach the draft/capture/config to the GPTModel and swap in the EAGLE3
    ``_postprocess``. Idempotent (re-patching updates the attached objects)."""

    # 取 policy 的 GPTModel
    m = _get_patching_model(model)
    if m is None:
        return

    
    # Wrap the draft in a single-element list so nn.Module.__setattr__ does NOT
    # auto-register it as a submodule. If it were a submodule, its params would
    # leak into the policy's named_parameters()/state_dict(), and mbridge's
    # load_weights/save_weights walk would map the draft's `_eagle3_draft.*` keys
    # -> _weight_name_mapping_other does name.split(".")[2] -> IndexError. The
    # draft has its OWN load path (setup_eagle3_training) + optimizer + checkpoint,
    # so it must stay out of the policy submodule tree.
    m._eagle3_draft = [draft_module]  # 把 draft 用单元素列表包起来挂上去。目的是防止 nn.Module.__setattr__ 把它自动注册成 policy 的子模块。
    m._eagle3_capture = capture       # 挂 hidden capture 句柄
    m._eagle3_ttt_length = int(ttt_length)      # 挂 TTT 长度(默认是1)
    m._eagle3_gamma = float(gamma)      # TTT 时间衰减(0.8)
    m._eagle3_temperature = float(temperature)      # softmax 温度(1.0)
    if not hasattr(m, "_eagle3_draft_losses"):
        m._eagle3_draft_losses = []     # 初始化 loss 暂存列表
    if not hasattr(m, "_postprocess_backup_eagle3"):
        m._postprocess_backup_eagle3 = m._postprocess       # 备份原方法

    # 替换！！！！！！这里重点看
    # 用 patch_eagle3_postprocess → 替换 gpt._postprocess,实现每次policy前向后跑 draft
    # 换的效果：原本 policy 前向结束后调 self._postprocess(hidden_states, ...) 只算 policy logits;打完补丁后,_postprocess 指向 _megatron_gptmodel_postprocess_eagle3(:156),每次 policy 前向都会顺带跑 draft 前向 + 算 loss 并暂存。
    m._postprocess = _megatron_gptmodel_postprocess_eagle3.__get__(m, m.__class__)

    # 绑定串行训练相关的三个方法到模型实例
    m._eagle3_collect_features_step = _eagle3_collect_features_step.__get__(m, m.__class__)
    m._eagle3_parallel_training = _eagle3_parallel_training.__get__(m, m.__class__)
    m._eagle3_actor_only_step = _eagle3_actor_only_step.__get__(m, m.__class__)
    m._eagle3_draft_training_step = _eagle3_draft_training_step.__get__(m, m.__class__)

    logger.warning("-" * 50)
    logger.warning("DRAFT-TRAIN: _postprocess hook registered on GPTModel, ttt_length=%d", ttt_length)
    logger.warning("-" * 50)


def unpatch_eagle3_postprocess(model: torch.nn.Module):
    """Restore the original ``_postprocess`` and drop attached EAGLE3 state. 恢复原 _postprocess,删除挂的所有 _eagle3_* 属性。训练结束/切换模式时用"""
    m = _get_patching_model(model)
    if m is None:
        return
    if hasattr(m, "_postprocess_backup_eagle3"):
        m._postprocess = m._postprocess_backup_eagle3
        del m._postprocess_backup_eagle3
    for attr in ("_eagle3_draft", "_eagle3_capture", "_eagle3_ttt_length",
                 "_eagle3_gamma", "_eagle3_temperature", "_eagle3_draft_losses",
                 "_eagle3_parallel_training", "_eagle3_actor_only_step", "_eagle3_draft_training_step",
                 "_eagle3_collect_features_step", "_eagle3_collect_only", "_eagle3_feature_store",
                 "_eagle3_collect_config"):
        if hasattr(m, attr):
            delattr(m, attr)

# ========================================draft-loss的入口========================================
def stash_draft_loss(model: torch.nn.Module, loss: torch.Tensor) -> bool:
    """Stash one draft loss for ``drain_draft_losses`` to collect. Returns success.

    Pairs with :func:`drain_draft_losses` and resolves the owning module the same
    way, so a producer cannot stash onto a different object than the drain reads
    from. The in-forward path appends to ``self._eagle3_draft_losses`` directly --
    ``self`` is already the patched model, so it cannot get this wrong. Deferred
    training starts from the engine instead and reaches the model through
    ``_unwrap_gpt``, which resolves independently of ``_get_patching_model``;
    when the two disagree the loss is stashed where nobody drains it, the draft
    never backwards, and nothing raises.
    """
    m = _get_patching_model(model)
    if m is None:
        logger.error(
            "eagle3_patch: cannot stash draft loss -- %s did not resolve to a patched "
            "model, so the loss would be dropped and the draft would silently not train.",
            type(model).__name__,
        )
        return False
    if not hasattr(m, "_eagle3_draft_losses"):
        m._eagle3_draft_losses = []
    m._eagle3_draft_losses.append(loss)
    return True


def drain_draft_losses(model: torch.nn.Module) -> List[torch.Tensor]:
    """Pop all stashed per-microbatch draft losses (engine calls after forward)."""
    m = _get_patching_model(model)
    if m is None or not hasattr(m, "_eagle3_draft_losses"):
        return []
    losses = m._eagle3_draft_losses
    m._eagle3_draft_losses = []
    return losses


def peek_draft_losses(model: torch.nn.Module) -> List[torch.Tensor]:
    """Read stashed draft losses without clearing (for logging/inspection)."""
    m = _get_patching_model(model)
    if m is None or not hasattr(m, "_eagle3_draft_losses"):
        return []
    return list(m._eagle3_draft_losses)


def _eagle3_tp_world_size() -> int:
    """Tensor-model-parallel world size, or 1 if Megatron mpu is unavailable.

    EAGLE3's teacher logits must be the FULL policy vocab so ``t2d`` (length =
    full vocab) can select the draft's columns. Under TP>1 Megatron's
    ``output_layer`` (a vocab-parallel ColumnParallelLinear) returns only this
    rank's vocab shard unless we force a gather. So we detect TP>1 here and pass
    ``runtime_gather_output=True`` for the teacher logits computation.
    """
    try:
        from megatron.core import parallel_state as mpu

        return mpu.get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _megatron_gptmodel_postprocess_eagle3(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    padding_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
    output_processor=None,
    output_processor_context=None,
    is_spec_decode=None,
):
    """EAGLE3 postprocess: policy logits unchanged (teacher) + draft L_draft stash (路由入口).

    【路由方法】根据标志决定执行哪种训练模式。
    本方法不包含任何业务逻辑，只做路由判断。

    支持三种模式：
    1. 串行模式 - Actor 训练步：禁用 draft
    2. 串行模式 - Draft 训练步：只训练 draft
    3. 并行模式（原有）：actor 和 draft 同时训练
    """
    # 提取标志
    # 注意：标志由 transformer_impl.forward_step 挂到模型实例上（见 _eagle3_train_draft_only /
    # _eagle3_enable_draft_training）。不能从 extra_block_kwargs 读——那是 decoder 层的 kwargs，
    # 是个 dict 且从不携带这些自定义标志，getattr 永远拿不到值会导致串行模式静默失效。
    train_draft_only = getattr(self, "_eagle3_train_draft_only", False)
    enable_draft_training = getattr(self, "_eagle3_enable_draft_training", True)

    if getattr(self, "_eagle3_collect_only", False):
        # === v3 延后训练：只采集特征，不在本次前向里训 draft ===
        # 挂在 compute_log_prob 那次前向上（PPO 本来就要跑），把 draft 训练需要的
        # hidden 采下来存到 host，真正的训练在 update_actor 返回之后进行。
        return self._eagle3_collect_features_step(
            hidden_states, input_ids, position_ids, loss_mask, runtime_gather_output
        )
    elif train_draft_only:
        # === 串行模式：Draft 训练步（只训练 draft）===
        return self._eagle3_draft_training_step(
            hidden_states, input_ids, position_ids, labels, rotary_pos_emb,
            rotary_pos_cos, rotary_pos_sin, mtp_in_postprocess, loss_mask,
            decoder_input, attention_mask, padding_mask, inference_params,
            packed_seq_params, sequence_len_offset, runtime_gather_output,
            extra_block_kwargs, inference_context, output_processor,
            output_processor_context, is_spec_decode
        )
    elif not enable_draft_training:
        # === 串行模式：Actor 训练步（禁用 draft）===
        return self._eagle3_actor_only_step(
            hidden_states, runtime_gather_output
        )
    else:
        # === 并行模式：原有逻辑（完全不变）===
        return self._eagle3_parallel_training(
            hidden_states, input_ids, position_ids, labels, rotary_pos_emb,
            rotary_pos_cos, rotary_pos_sin, mtp_in_postprocess, loss_mask,
            decoder_input, attention_mask, padding_mask, inference_params,
            packed_seq_params, sequence_len_offset, runtime_gather_output,
            extra_block_kwargs, inference_context, output_processor,
            output_processor_context, is_spec_decode
        )


def _eagle3_actor_only_step(self, hidden_states, runtime_gather_output):
    """串行模式 - Actor 训练步：只计算 policy logits，禁用 draft（新增方法）"""
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if not self.post_process:
        return hidden_states

    # 计算 policy logits
    logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    if logits is not None:
        logits = logits.transpose(0, 1).contiguous()  # [s b v] -> [b s v]

    return logits


def _eagle3_draft_forward_and_stash_loss(
    self,
    hidden_states,
    input_ids,
    position_ids,
    loss_mask,
    output_weight,
    logits,
):
    """Draft 前向 + loss 计算 + 暂存：串行与并行**共用同一份实现**。

    这份实现直接来自并行路径（backup/before-serial-training 分支已验证正确的版本），
    抽出来成为唯一真源。串行路径过去手抄了一份，抄写中丢了 SP gather、抄错了
    compute_draft_loss 的参数名和 ttt_length 默认值、抄漏了 loss_mask 的维度判断，
    连续崩了三次。抽成公共函数后，draft 训练逻辑只有一处定义，不会再出现
    "改一处漏一处"。

    调用方各自负责异常策略（并行吞掉保 policy 存活；串行抛出，因为串行 Draft 步
    的唯一目的就是训 draft，吞掉会空转还伪装成成功）和 capture.clear()。

    Returns:
        compute_draft_loss 的返回 dict（loss 已 append 到 self._eagle3_draft_losses）
    """
    # aux hidden captured in-flight during the decoder forward (detached).
    # Under sequence_parallel (forced on when TP>1), the captured hidden is
    # SP-sharded on the sequence dim: (S/TP, B, H*num_aux). The draft is a
    # REPLICATED nn.Module (no TP), and its input_emb = embed_tokens(input_ids)
    # is FULL sequence, so cat(input_emb, hidden) needs full-seq aux too. Gather
    # across the SP region here (seqlen-first, gather dim 0), THEN transpose to
    # (B, S, H*num_aux) as draft.forward expects. aux is detached, so this gather
    # never feeds gradient back into the policy.
    capture = getattr(self, "_eagle3_capture", None)
    draft = self._eagle3_draft[0]

    aux_hidden = capture.get_captured(seqlen_first=True)  # (S/TP, B, H*num_aux)
    if getattr(self.config, "sequence_parallel", False) and _eagle3_tp_world_size() > 1:
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        aux_hidden = gather_from_sequence_parallel_region(aux_hidden)  # (S, B, H*num_aux)
    aux_hidden = aux_hidden.transpose(0, 1).contiguous()  # (B, S, H*num_aux)

    ttt_length = getattr(self, "_eagle3_ttt_length", 1)
    eagle_loss_mask = loss_mask
    if eagle_loss_mask.dim() == 1:
        eagle_loss_mask = eagle_loss_mask.unsqueeze(0)
    draft_input_ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)

    # ---- EAGLE 对齐：token 相对 hidden 左移一位（P1-1 修复，2026-08-29）----
    # 推理时（vllm-ascend llm_base_proposer.py:1434 "Shift the input ids by one
    # token"）draft 的一步输入是 (f[p], x[p+1])：hidden 停在 p，token 是刚采样出
    # 的下一个。训练必须复刻这个差位，否则 draft 学到的融合方式与推理喂法错开
    # 一格。初始 ckpt 实验实证：正确对齐 loss 0.87/top1 79%，不移位 loss 9.28/
    # top1 6.9%（开发过程记录/串行训练/串行版本2/P1-1对齐核实结论-20260829.md）。
    #
    # 行 p 的训练样本 = (f[p], x[p+1]) -> P(x[p+2])：
    #   input_ids 左移一位（末行重复自身，其 teacher 由 shift_teacher_left 的
    #   valid[:,-1]=False 掐掉，不产生 loss）；
    #   loss_mask 左移两位取【被预测 token x[p+2]】的掩码（SpeCo 同款语义，
    #   base_trainer.py:2938 的 mask[2:2+L]），尾部补 0。
    #
    # 已知限界：若一个 micro-batch 把多条序列打包进同一行（THD packing），移位在
    # 序列边界处会串到下一条的开头 —— 这与本函数里 teacher 左移的既有行为一致
    # （每条序列至多污染 1-2 行，且通常被 loss_mask 的 prompt 段挡住），当前
    # 配置（micro_batch_size_per_gpu=1，一行一序列）不触发。
    draft_input_ids = torch.cat([draft_input_ids[:, 1:], draft_input_ids[:, -1:]], dim=1)
    eagle_loss_mask = torch.cat(
        [eagle_loss_mask[:, 2:], eagle_loss_mask.new_zeros(eagle_loss_mask.shape[0], 2)], dim=1
    )

    draft_out = draft(
        input_ids=draft_input_ids,
        hidden_states=aux_hidden,
        loss_mask=eagle_loss_mask,
        attention_mask=None,
        position_ids=position_ids,
        ttt_length=ttt_length,
    )

    # teacher = FULL-vocab policy logits so t2d (length = full vocab)
    # indexes correctly. Under TP=1 the policy `logits` above is already
    # full vocab. Under TP>1 it is only this rank's vocab shard, so
    # recompute the teacher from the same hidden with a gather. Extra
    # cost is training-only + draft-only; keeps the policy return path
    # (its own gather semantics) untouched.
    if _eagle3_tp_world_size() > 1:
        teacher_logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=True
        )
        teacher_logits = teacher_logits.transpose(0, 1).contiguous()  # [s b v] -> [b s v]
    else:
        teacher_logits = logits

    # teacher detach happens inside the loss fn.
    t2d = draft.t2d
    logger.warning("-" * 50)
    logger.warning("DRAFT-TRAIN: hook triggered, captured hidden_states shape=%s, calling compute_draft_loss",
                   aux_hidden.shape if aux_hidden is not None else None)
    logger.warning("-" * 50)
    loss_out = compute_draft_loss(
        student_logits_per_step=draft_out["logits"],
        teacher_logits=teacher_logits,
        t2d=t2d,
        loss_mask=eagle_loss_mask,
        position_masks_per_step=draft_out.get("position_masks"),
        gamma=getattr(self, "_eagle3_gamma", 0.8),
        temperature=getattr(self, "_eagle3_temperature", 1.0),
    )
    if not hasattr(self, "_eagle3_draft_losses"):
        self._eagle3_draft_losses = []
    self._eagle3_draft_losses.append(loss_out["loss"])
    return loss_out


def lens_from_loss_mask(loss_mask: torch.Tensor):
    """Derive per-sample ``(prompt_len, response_len)`` from a ``(B, S)`` loss mask.

    The collect plan needs both, but neither is handed to ``_postprocess`` --
    only ``loss_mask``, which is True exactly on the response tokens the policy
    loss covers. That is enough: the first True is where the response starts, and
    the count of Trues is its length.

    ``prompt_len`` is returned as the first response position, so
    ``build_collect_plan``'s ``start = prompt_len - 1`` lands on the last prompt
    token -- the hidden that predicts response[0].

    Samples with an all-False row get ``(0, 0)`` and are dropped by the plan's
    length gate rather than producing a bogus window.
    """
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)
    mask = loss_mask.bool()
    response_lens = mask.sum(dim=1).to(torch.long)
    has_any = response_lens > 0
    # argmax on a bool row returns the first True; meaningless for all-False rows,
    # which has_any then zeroes out.
    first_true = mask.to(torch.uint8).argmax(dim=1).to(torch.long)
    prompt_lens = torch.where(has_any, first_true, torch.zeros_like(first_true))
    response_lens = torch.where(has_any, response_lens, torch.zeros_like(response_lens))
    return prompt_lens.cpu(), response_lens.cpu()


def _eagle3_collect_features_step(
    self,
    hidden_states,
    input_ids,
    position_ids,
    loss_mask,
    runtime_gather_output,
):
    """v3 deferred training: harvest draft features, train nothing here.

    Runs on the ``compute_log_prob`` forward, which PPO pays for anyway. The
    policy return path is byte-identical to the other branches -- the draft side
    only reads, and everything it reads is detached.

    Failures are swallowed with an ERROR, following the parallel path's stance:
    this branch rides a forward the policy needs, so breaking it would take down
    old_log_probs over a draft-side problem. A step that collects nothing is not
    silent either way -- ``train_draft_from_store`` warns on an empty store.
    """
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if not self.post_process:
        return hidden_states

    logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    logits = logits.transpose(0, 1).contiguous()  # [s b h] -> [b s h]

    store = getattr(self, "_eagle3_feature_store", None)
    capture = getattr(self, "_eagle3_capture", None)
    if store is None or capture is None or loss_mask is None:
        logger.warning(
            "eagle3_patch: collect-only step skipped (store=%s capture=%s loss_mask=%s)",
            store is not None, capture is not None, loss_mask is not None,
        )
        return logits

    try:
        from verl.models.eagle3.collect_plan import build_collect_plan
        from verl.models.eagle3.feature_store import collect_draft_features

        cfg = getattr(self, "_eagle3_collect_config", None) or {}
        # Quota is per STEP, not per micro-batch. _postprocess runs once per
        # micro-batch, so passing the configured quota straight through would
        # restart it on every call -- and at micro_batch_size_per_gpu=1 a quota
        # of 16 never binds on a batch of 1, so the step ends up collecting one
        # window per sequence. The budget object carries what is left.
        budget = getattr(self, "_eagle3_collect_budget", None)
        if budget is not None and budget.exhausted():
            return logits
        max_samples = budget.remaining_samples() if budget is not None else cfg.get("max_samples_per_replica", 16)
        max_tokens = budget.remaining_tokens() if budget is not None else cfg.get("max_tokens_per_replica", 16384)

        prompt_lens, response_lens = lens_from_loss_mask(loss_mask)
        plan = build_collect_plan(
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            global_step=cfg.get("global_step", 0),
            window_train_rows=cfg.get("window_train_rows", 512),
            window_mode=cfg.get("window_mode", "front"),
            sample_rate=cfg.get("sample_rate", 1.0),
            max_samples_per_replica=max_samples,
            max_tokens_per_replica=max_tokens,
        )
        if plan is None:
            return logits

        # aux comes from the capture hooks (already detached); final_hidden is the
        # pre-lm_head activation this function was handed. Both are still
        # sequence-parallel sharded here -- collect_draft_features gathers them,
        # because the plan addresses global positions.
        stored = collect_draft_features(
            store=store,
            aux_hidden=capture.get_captured(seqlen_first=True),
            final_hidden=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            plan=plan,
            global_step=cfg.get("global_step", 0),
            sequence_parallel=getattr(self.config, "sequence_parallel", False),
            tp_world_size=_eagle3_tp_world_size(),
        )
        if budget is not None and stored:
            budget.consume(stored, stored * plan.hidden_rows)
        logger.warning(
            "[DRAFT-COLLECT] step %s: stashed %d/%d window(s) (%d candidates); "
            "step budget left: %s samples / %s rows",
            cfg.get("global_step", 0), stored, plan.selected_count, plan.candidate_count,
            budget.remaining_samples() if budget is not None else "n/a",
            budget.remaining_tokens() if budget is not None else "n/a",
        )
    except Exception as e:
        logger.error(
            "eagle3_patch: feature collection failed (%r). old_log_probs is unaffected; "
            "the draft simply has nothing to train on this step.",
            e, exc_info=True,
        )
        if _eagle3_strict_draft():
            raise
    finally:
        if capture is not None:
            capture.clear()

    return logits


def _eagle3_parallel_training(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess,
    loss_mask,
    decoder_input,
    attention_mask,
    padding_mask,
    inference_params,
    packed_seq_params,
    sequence_len_offset,
    runtime_gather_output,
    extra_block_kwargs,
    inference_context,
    output_processor,
    output_processor_context,
    is_spec_decode,
):
    """并行模式：原有的 EAGLE3 postprocess 逻辑（完全不改动，只是重命名）

    【原有逻辑封装】这是原有 _megatron_gptmodel_postprocess_eagle3 方法的完整复制。
    所有逻辑完全不变，只是移动到这个独立方法中。

    关闭串行开关后，执行路径会进入此方法，代码 100% 是原有逻辑。
    """
    output_weight = None       # ===============================   原版逻辑，算 policy logits,返回给上层做 policy loss。完全不动,保证 policy 训练不受影响。
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if not self.post_process:
        # non-final PP stage: nothing to do for the draft here (PP=1 first version)
        return hidden_states            

    # ---- policy logits (unchanged; policy return path keeps caller's gather) ----
    logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    logits = logits.transpose(0, 1).contiguous()  # [s b h] -> [b s h]      # ===============================

    # ---- draft path (training only, when we have labels/loss_mask) ----                #=============================================  验证 draft 四重门禁
    # _eagle3_draft is a single-element list (see patch_eagle3_postprocess) to keep
    # it out of the policy submodule registry; unwrap it here.
    _draft_holder = getattr(self, "_eagle3_draft", None)
    draft = _draft_holder[0] if _draft_holder else None
    capture = getattr(self, "_eagle3_capture", None)
    # Diagnostic: log the gate state once per forward-mode (train vs eval) so we can
    # tell whether the TRAINING forward opens the gate. The old one-shot burned during
    # the eval log_prob forward, hiding the training-forward state. Key by self.training.
    _gate_seen = getattr(self, "_eagle3_gate_logged_modes", None)
    if _gate_seen is None:
        _gate_seen = set()
        self._eagle3_gate_logged_modes = _gate_seen
    if bool(self.training) not in _gate_seen:
        _gate_seen.add(bool(self.training))
        logger.warning(
            "eagle3_patch: draft-gate check [training=%s] -> draft=%s capture=%s loss_mask=%s",
            bool(self.training), draft is not None, capture is not None, loss_mask is not None,
        )
    if draft is not None and capture is not None and self.training and loss_mask is not None:     #=============================================
        try:
            # draft 前向 + loss 计算 + 暂存：与串行路径共用 _eagle3_draft_forward_and_stash_loss，
            # 唯一真源。本函数只负责"失败了也不能拖死 policy 训练"的异常策略。
            _eagle3_draft_forward_and_stash_loss(
                self,
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                loss_mask=loss_mask,
                output_weight=output_weight,
                logits=logits,
            )
        except _OOM_ERRORS as e:
            # OOM 单独成支：这是最常见的 draft 失败原因，必须计数 + ERROR 级别可见。
            # 仍然吞掉（并行模式的设计意图是 draft 挂了也不能拖死 policy 训练），
            # 但绝不能像以前那样只打一条 WARNING 就算完 —— 2026-08-26 那次
            # 82 次 OOM 全被 WARNING 淹没，导致"draft 根本没训练"整轮没被发现。
            self._eagle3_draft_oom_count = getattr(self, "_eagle3_draft_oom_count", 0) + 1
            self._eagle3_draft_fail_count = getattr(self, "_eagle3_draft_fail_count", 0) + 1
            logger.error(
                "eagle3_patch: draft 路径 OOM（本 rank 累计 %d 次，本 microbatch 的 draft loss 被丢弃）: %r",
                self._eagle3_draft_oom_count,
                e,
            )
            if self._eagle3_draft_oom_count in (1, 10, 100) or self._eagle3_draft_oom_count % 500 == 0:
                logger.error(
                    "eagle3_patch: draft OOM 已累计 %d 次。draft 正在被静默跳过、几乎没有真正训练。"
                    "请检查 (1) 串行模式下 Actor 步是否漏关 draft（enable_draft_training 是否真的传到了 "
                    "forward_step）(2) draft 的 micro_batch_size 是否过大。"
                    "设 EAGLE3_STRICT_DRAFT=1 可让 draft 失败直接抛出以便定位。",
                    self._eagle3_draft_oom_count,
                )
            _eagle3_empty_cache()
            if _eagle3_strict_draft():
                raise
        except Exception as e:  # keep policy training alive if draft path fails                                                        #============================================= 异常兜底 + 清 capture
            # 非 OOM 失败（形状不匹配、d2t/t2d 越界等）几乎都是真 bug，不该被降级成 WARNING。
            self._eagle3_draft_fail_count = getattr(self, "_eagle3_draft_fail_count", 0) + 1
            logger.error(
                "eagle3_patch: draft 路径失败（非 OOM，本 rank 累计 %d 次失败）: %r",
                self._eagle3_draft_fail_count,
                e,
                exc_info=True,  # 打完整 traceback：非 OOM 异常没有栈根本无法定位
            )
            if _eagle3_strict_draft():
                raise
        finally:
            # free captured tensors for the next microbatch (hooks stay registered)
            if capture is not None:
                capture.clear()                                                                                                         #=============================================

    return logits             #返回 policy logits，算的 policy logits 给上层,draft loss 已暂存、走独立通道。


def _eagle3_draft_training_step(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess,
    loss_mask,
    decoder_input,
    attention_mask,
    padding_mask,
    inference_params,
    packed_seq_params,
    sequence_len_offset,
    runtime_gather_output,
    extra_block_kwargs,
    inference_context,
    output_processor,
    output_processor_context,
    is_spec_decode,
):
    """串行模式 - Draft 训练步：只训练 draft，Actor 冻结（新增方法）

    执行流程：
    1. Actor forward（冻结参数，生成 teacher logits）
    2. Draft forward + loss 计算
    3. Draft backward + 参数更新

    与并行模式的区别：
    - Actor 参数冻结，只做前向传播生成 teacher
    - 只有 Draft 参与梯度计算和参数更新
    """
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if not self.post_process:
        return hidden_states

    # 1. 计算 policy logits（Actor 前向，用作 teacher）
    logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    logits = logits.transpose(0, 1).contiguous()  # [s b v] -> [b s v]

    # 2. Draft 训练流程
    draft_list = getattr(self, "_eagle3_draft", None)
    capture = getattr(self, "_eagle3_capture", None)

    # _eagle3_draft 是单元素列表 [draft_module]，需要解包
    if draft_list is None or not draft_list or capture is None:
        logger.warning("[DRAFT-TRAIN-SERIAL] Draft or capture not available, skipping draft training")
        return logits

    if not self.training:
        # 与并行路径的四重门禁对齐（并行见 _eagle3_parallel_training 的 gate）。
        # eval/推理前向下算出的 draft loss 没有 grad_fn，一旦被 stash，随后
        # eagle3_backward_step 调 backward() 会报 "element 0 does not require grad"。
        logger.warning("[DRAFT-TRAIN-SERIAL] model not in training mode, skipping draft training")
        return logits

    if loss_mask is None:
        # 串行 Draft 步没有 loss_mask 就无法算 loss，本步注定空转。
        # 与并行路径的四重门禁保持一致（并行也要求 loss_mask is not None）。
        logger.warning("[DRAFT-TRAIN-SERIAL] loss_mask is None, skipping draft training")
        return logits

    try:
        # 3-7. draft 前向 + loss 计算 + 暂存
        # 与并行路径共用 _eagle3_draft_forward_and_stash_loss，唯一真源。
        # 之前这里是手抄的一份，抄写中丢了 SP gather、抄错了 compute_draft_loss 的
        # 参数名（draft_logits → student_logits_per_step）和 ttt_length 默认值
        # （None → 1）、抄漏了 loss_mask 的维度判断，连续崩了三次。
        # 现在只保留串行特有的异常策略（抛出而非吞掉）。
        loss_out = _eagle3_draft_forward_and_stash_loss(
            self,
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            output_weight=output_weight,
            logits=logits,
        )

        logger.info(
            f"[DRAFT-TRAIN-SERIAL] Draft loss={loss_out['loss'].item():.4f}, "
            f"num_tokens={loss_out['num_tokens'].item()}"
        )

    except _OOM_ERRORS:
        # 串行 Draft 步与并行路径不同：这一步的**唯一目的**就是训练 draft。
        # 吞掉异常会让整步变成空转，却仍然上报"训练成功"——比直接崩溃更糟，
        # 因为它会安静地烧掉整轮训练时间（2026-08-26 就是这么浪费掉的）。
        # 所以先清显存，再原样抛出，让上层看到真实失败。
        logger.error(
            "[DRAFT-TRAIN-SERIAL] Draft 训练步 OOM —— 本步没有任何 draft 参数被更新。"
            "已抛出而非静默跳过（静默跳过会让整步空转且伪装成成功）。"
            "请调小 draft 的 ppo_micro_batch_size_per_gpu 后重跑。",
            exc_info=True,
        )
        _eagle3_empty_cache()
        raise
    except Exception:
        logger.error(
            "[DRAFT-TRAIN-SERIAL] Draft 训练步失败（非 OOM）—— 本步没有任何 draft 参数被更新，已抛出。",
            exc_info=True,
        )
        raise
    finally:
        # 清理 capture
        if capture is not None:
            capture.clear()

    return logits  # 返回 policy logits（draft 步不会被用于 policy loss）


