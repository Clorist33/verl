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
"""EAGLE3 draft loss ``L_draft`` (P2a: dense full-vocab soft cross-entropy).

The draft is a distillation student of the policy. Teacher = the policy's own
next-token logits, **shifted left by one and detached**; student = the draft's
per-step logits. Loss = soft CE / forward-KL between them, masked.

Shift/alignment (the #1 error source -- read carefully):

* One shift op ``new[t] = old[t+1]``. The draft at position ``t`` is asked to
  predict the token at ``t+1``; the teacher answer for position ``t`` is
  therefore the policy's distribution at ``t+1``. So teacher logits are shifted
  LEFT by one (the last position has no answer -> masked out).
* Online vs offline: verl-SpeCo pre-shifts its data offline, so its loss uses a
  plain sliding window with no base shift. We train ONLINE and do not pre-shift,
  so the base left-shift is applied here explicitly (matching NeMo-RL
  ``prepare_loss_input``).

Vocab compression: teacher is full policy vocab (V); ``t2d`` (bool[V]) selects
the draft's small vocab columns so teacher aligns with the student's
``draft_vocab_size`` head. PP=1 / TP=1 first version -> no vocab-parallel gather.

TTT (``ttt_length > 1``): the draft returns one logits tensor per step; step
``idx`` targets a teacher shifted ``idx`` further forward, weighted ``gamma**idx``
(gamma=0.8, matching verl-SpeCo). Sparse restricted top-k CE is a P4+ path and
is intentionally NOT implemented here.
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# EAGLE3 per-step temporal decay for TTT multi-step (verl-SpeCo uses 0.8).
DEFAULT_TTT_GAMMA = 0.8


def shift_teacher_left(teacher_logits: torch.Tensor) -> tuple:
    """Shift teacher logits left by one along the sequence dim: ``new[t]=old[t+1]``.

    Args:
        teacher_logits: (B, S, V) policy next-token logits.

    Returns:
        (shifted, valid_mask) where ``shifted`` is (B, S, V) with the last
        position duplicated as a placeholder, and ``valid_mask`` is (B, S) with
        the last position (which has no ``t+1`` answer) set to 0.
    """
    B, S, V = teacher_logits.shape
    shifted = torch.cat([teacher_logits[:, 1:, :], teacher_logits[:, -1:, :]], dim=1)
    valid = torch.ones((B, S), dtype=torch.bool, device=teacher_logits.device)
    valid[:, -1] = False
    return shifted, valid


def filter_teacher_to_draft_vocab(teacher_logits: torch.Tensor, t2d: torch.Tensor) -> torch.Tensor:
    """Select the draft's small-vocab columns from full-vocab teacher logits.

    Args:
        teacher_logits: (B, S, V) full policy vocab, OR already (B, S, draft_vocab).
        t2d: bool tensor of shape (V,); True where a full-vocab id is in the draft vocab.

    Returns:
        (B, S, draft_vocab) teacher logits aligned with the draft head.
    """
    t2d_bool = t2d.to(device=teacher_logits.device, dtype=torch.bool)
    v = teacher_logits.size(-1)
    if v == t2d_bool.numel():
        selected = teacher_logits[..., t2d_bool]
    elif v == int(t2d_bool.sum().item()):
        # already draft-vocab sized (e.g. teacher pre-filtered upstream)
        selected = teacher_logits
    else:
        raise ValueError(
            f"EAGLE3 teacher vocab mismatch: teacher_logits last dim {v}, "
            f"full vocab {t2d_bool.numel()}, draft vocab {int(t2d_bool.sum().item())}"
        )
    if selected.size(-1) == 0:
        raise ValueError("EAGLE3 t2d selects zero draft-vocab columns")
    return selected


def masked_soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    position_mask: torch.Tensor,
) -> tuple:
    """Per-token soft CE between student logits and teacher probabilities.

    Args:
        student_logits: (B, S, draft_vocab) draft logits for this step.
        teacher_probs:  (B, S, draft_vocab) teacher probabilities (detached).
        position_mask:  (B, S) valid-position mask (1 keep / 0 drop).

    Returns:
        (per_token_loss (B, S), valid_position (B, S) bool). Non-finite logits or
        empty teacher rows are dropped (guards against NaN on masked positions).
    """
    student_logits = student_logits.float()
    teacher_probs = teacher_probs.float()

    finite_logits = torch.isfinite(student_logits).all(dim=-1)
    finite_target = torch.isfinite(teacher_probs).all(dim=-1) & (teacher_probs.sum(dim=-1) > 0)
    valid_position = (position_mask > 0) & finite_logits & finite_target

    safe_logits = torch.where(torch.isfinite(student_logits), student_logits, torch.zeros_like(student_logits))
    safe_target = torch.where(torch.isfinite(teacher_probs), teacher_probs, torch.zeros_like(teacher_probs))
    safe_target = torch.where(valid_position.unsqueeze(-1), safe_target, torch.zeros_like(safe_target))

    log_probs = F.log_softmax(safe_logits, dim=-1)
    per_token_loss = -(safe_target * log_probs).sum(dim=-1)
    per_token_loss = torch.where(valid_position, per_token_loss, torch.zeros_like(per_token_loss))
    return per_token_loss, valid_position


def compute_draft_loss(
    student_logits_per_step: List[torch.Tensor],
    teacher_logits: torch.Tensor,
    t2d: torch.Tensor,
    loss_mask: torch.Tensor,
    position_masks_per_step: Optional[List[torch.Tensor]] = None,
    gamma: float = DEFAULT_TTT_GAMMA,
    temperature: float = 1.0,
) -> dict:
    """Dense full-vocab soft-CE draft loss (P2a), TTT-aware.

    Args:
        student_logits_per_step: list of (B, S, draft_vocab); one per TTT step
            (length 1 when ttt_length == 1). This is draft.forward()["logits"].
        teacher_logits: (B, S, V) policy next-token logits (will be detached +
            left-shifted here).
        t2d: bool (V,) full-vocab -> draft-vocab selector.
        loss_mask: (B, S) valid tokens for the loss.
        position_masks_per_step: optional list of (B, S) (or (B,S,1)) per-step
            masks from draft.forward()["position_masks"]; combined with loss_mask.
        gamma: per-step temporal decay for TTT (weight = gamma ** idx).
        temperature: softmax temperature (T=1 -> plain soft CE / forward KL).

    Returns:
        dict with ``loss`` (scalar, token-mean over all steps), ``num_tokens``,
        and ``per_step_loss`` (list of detached scalars) for logging.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("^" * 100)
    logger.warning("DRAFT-TRAIN: compute_draft_loss called, student_logits len=%d", len(student_logits_per_step))
    logger.warning("^" * 100)
    if len(student_logits_per_step) == 0:
        raise ValueError("compute_draft_loss: empty student_logits_per_step")

    device = student_logits_per_step[0].device
    teacher_logits = teacher_logits.detach()
    n_steps = len(student_logits_per_step)
    S = student_logits_per_step[0].size(1)

    # Base teacher: policy distribution at t+1 (left-shift), filtered to draft vocab.
    teacher_shifted, shift_valid = shift_teacher_left(teacher_logits)
    teacher_draft = filter_teacher_to_draft_vocab(teacher_shifted, t2d)  # (B,S,draft_vocab)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_draft.float() / temperature, dim=-1)  # (B,S,draft_vocab)

    base_mask = loss_mask.to(device=device).float() * shift_valid.to(device=device).float()  # (B,S)

    # TTT (ttt_length>1): the draft right-shifts its own input each step, so step
    # idx predicts token t+1+idx. The teacher target must slide forward by idx too
    # (verl-SpeCo: target_p_padded[:, idx:idx+S]). Right-pad probs with uniform
    # (1/V, masked out) and the mask with 0 so the slice is always length S.
    Vd = teacher_probs.size(-1)
    if n_steps > 1:
        teacher_probs_padded = F.pad(teacher_probs, (0, 0, 0, n_steps), value=1.0 / Vd)  # (B,S+n,Vd)
        base_mask_padded = F.pad(base_mask, (0, n_steps), value=0.0)                      # (B,S+n)
    else:
        teacher_probs_padded = teacher_probs
        base_mask_padded = base_mask

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = torch.zeros((), device=device, dtype=torch.float32)
    per_step_loss = []

    for idx in range(n_steps):
        logits = student_logits_per_step[idx]
        if temperature != 1.0:
            logits = logits / temperature

        # slide teacher target forward by idx to align with this step's prediction
        if n_steps > 1:
            step_teacher = teacher_probs_padded[:, idx: idx + S, :].contiguous()
            step_mask = base_mask_padded[:, idx: idx + S].contiguous()
        else:
            step_teacher = teacher_probs
            step_mask = base_mask

        if position_masks_per_step is not None:
            pm = position_masks_per_step[idx]
            if pm.dim() == 3:
                pm = pm.squeeze(-1)
            step_mask = step_mask * pm.to(device=device).float()

        per_token_loss, valid_position = masked_soft_cross_entropy(logits, step_teacher, step_mask)
        step_loss_sum = per_token_loss.sum()
        step_tokens = valid_position.float().sum()

        total_loss = total_loss + (gamma ** idx) * step_loss_sum
        total_tokens = total_tokens + step_tokens
        per_step_loss.append(step_loss_sum.detach() / step_tokens.clamp_min(1.0))

    loss = total_loss / total_tokens.clamp_min(1.0)
    return {
        "loss": loss,
        "num_tokens": total_tokens.detach(),
        "per_step_loss": per_step_loss,
    }

