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
                 "_eagle3_gamma", "_eagle3_temperature", "_eagle3_draft_losses"):
        if hasattr(m, attr):
            delattr(m, attr)

# ========================================draft-loss的入口========================================
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
    """EAGLE3 postprocess: policy logits unchanged (teacher) + draft L_draft stash.

    Signature mirrors ``mtp_patch._megatron_gptmodel_postprocess`` so the swap is
    drop-in. The policy output path is untouched; the draft path is additive and
    kept off the policy autograd graph (aux hidden already detached in capture,
    and we never call MTPLossAutoScaler).
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
            # aux hidden captured in-flight during the decoder forward (detached).                         #=============================================取 detached aux hidden + SP gather(:230-243)，若开 SP,抓到的 hidden 是 SP 切片,draft 需要完整序列,先 gather。这是放开 SP 限制的关键修复。
            # Under sequence_parallel (forced on when TP>1), the captured hidden is
            # SP-sharded on the sequence dim: (S/TP, B, H*num_aux). The draft is a
            # REPLICATED nn.Module (no TP), and its input_emb = embed_tokens(input_ids)
            # is FULL sequence, so cat(input_emb, hidden) needs full-seq aux too. Gather
            # across the SP region here (seqlen-first, gather dim 0), THEN transpose to
            # (B, S, H*num_aux) as draft.forward expects. aux is detached, so this gather
            # never feeds gradient back into the policy.
            aux_hidden = capture.get_captured(seqlen_first=True)  # (S/TP, B, H*num_aux)
            if getattr(self.config, "sequence_parallel", False) and _eagle3_tp_world_size() > 1:
                from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

                aux_hidden = gather_from_sequence_parallel_region(aux_hidden)  # (S, B, H*num_aux)
            aux_hidden = aux_hidden.transpose(0, 1).contiguous()  # (B, S, H*num_aux)                       #=============================================

            ttt_length = getattr(self, "_eagle3_ttt_length", 1)             #============================================= 跑 draft 前向，返回draft_out["logits"]
            eagle_loss_mask = loss_mask
            if eagle_loss_mask.dim() == 1:
                eagle_loss_mask = eagle_loss_mask.unsqueeze(0)

            draft_out = draft(
                input_ids=input_ids,
                hidden_states=aux_hidden,
                loss_mask=eagle_loss_mask,
                attention_mask=None,
                position_ids=position_ids,
                ttt_length=ttt_length,
            )                   #============================================= 

            # teacher = FULL-vocab policy logits so t2d (length = full vocab)               #============================================= 准备 teacher logit
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
                teacher_logits = logits                                                                                         #============================================= 

            # teacher detach happens inside the loss fn.                                                                        #============================================= 算 draft loss 并暂存
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
            self._eagle3_draft_losses.append(loss_out["loss"])                                                                          #============================================= 
        except Exception as e:  # keep policy training alive if draft path fails                                                        #============================================= 异常兜底 + 清 capture
            logger.warning("eagle3_patch: draft path failed this microbatch: %r", e)
            # OOM 兜底：清碎片显存阻断累积，防止溢出到 policy/kernel
            if "out of memory" in str(e).lower() or (hasattr(e, "__class__") and "OutOfMemory" in e.__class__.__name__):
                logger.warning("eagle3_patch: OOM detected in draft path, calling empty_cache() to prevent accumulation")
                try:
                    import torch
                    if hasattr(torch, "npu") and torch.npu.is_available():
                        torch.npu.empty_cache()
                    elif hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as cache_err:
                    logger.warning("eagle3_patch: empty_cache() failed: %r", cache_err)
        finally:
            # free captured tensors for the next microbatch (hooks stay registered)
            if capture is not None:
                capture.clear()                                                                                                         #=============================================

    return logits             #返回 policy logits，算的 policy logits 给上层,draft loss 已暂存、走独立通道。

