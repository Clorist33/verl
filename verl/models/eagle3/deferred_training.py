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
"""Draft training that runs after the policy is done with the step.

The point of stashing features during ``compute_log_prob`` is to be able to train
the draft *here* -- after ``update_actor`` has returned and the policy's
activations and gradients are gone. Draft cost then sits beside the policy peak
instead of on top of it, which is what the old per-step alternation bought at the
price of an entire extra rollout.

Relation to the in-forward path: ``eagle3_patch._eagle3_draft_forward_and_stash_loss``
stays exactly as it is and keeps serving parallel mode. It reads a live capture and
recomputes the teacher through the policy's ``output_layer``, neither of which is
available once the forward has been torn down. This module is the deferred
counterpart, not a replacement -- deliberately additive, because that function is
the single source of truth 优化14 consolidated after three rounds of drift, and
re-cutting it under a live parallel path is how that drift started.

What is shared: the backward. ``eagle3_backward_step`` already drains the stashed
losses, averages, clips, and steps the draft optimizer with offload handling, and
it is verified on hardware. So this module only produces losses and stashes them
on the GPT module the same way the in-forward path does; the optimizer half is
reused untouched.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def stack_records(records, device=None, dtype=None) -> dict:
    """Batch per-sample stashed windows into draft-forward inputs.

    Every record spans exactly ``hidden_rows`` positions -- the collect plan uses
    one fixed window size -- so this is a plain stack with no padding and no
    length mask. That is the main ergonomic payoff of the fixed-window design.

    Args:
        records: list of :class:`~verl.models.eagle3.feature_store.DraftFeatureRecord`.
        device: destination for the stacked tensors (host -> accelerator).
        dtype: float dtype for the hidden states; index/mask tensors keep theirs.

    Returns:
        dict with ``aux_hidden`` ``(N, rows, H*num_aux)``, ``final_hidden``
        ``(N, rows, H)``, ``input_ids``/``position_ids``/``loss_mask`` ``(N, rows)``.

    Raises:
        ValueError: if the batch is empty or the windows disagree in length,
            which would mean records from two different plans got mixed.
    """
    if not records:
        raise ValueError("stack_records: no records to stack")

    widths = {int(r.aux_hidden.shape[0]) for r in records}
    if len(widths) != 1:
        raise ValueError(
            f"stack_records: records disagree on window length {sorted(widths)}. "
            "All records in a step must come from one collect plan."
        )

    def _stack(attr, want_float):
        t = torch.stack([getattr(r, attr) for r in records], dim=0)
        if device is not None:
            t = t.to(device)
        if want_float and dtype is not None:
            t = t.to(dtype)
        return t

    return {
        "aux_hidden": _stack("aux_hidden", True),
        "final_hidden": _stack("final_hidden", True),
        "input_ids": _stack("input_ids", False),
        "position_ids": _stack("position_ids", False),
        "loss_mask": _stack("loss_mask", False),
    }


def _chunks(seq, size):
    if not size or size <= 0 or size >= len(seq):
        yield seq
        return
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _dp_all_ranks_ready(has_records: bool) -> bool:
    """MIN-reduce readiness over the draft's data-parallel group.

    The draft is DDP-wrapped over the DP group (engine_support._wrap_draft_ddp),
    so its backward runs a gradient all-reduce across those ranks. Each rank
    harvests its own windows, and nothing guarantees uniformity: one rank's
    samples can all fail the collect plan's length gate while a peer's do not.
    If the non-empty ranks backward alone, they block forever on that
    all-reduce -- the empty rank never joins.

    So every rank must make the same train-or-skip decision. This mirrors
    verl-SpeCo's ``DrafterBaseTrainer._sync_batch_readiness``
    (``base_trainer.py:3208``): reduce a readiness flag with MIN, and if any
    rank is empty, all ranks skip together.

    TP peers inside one DP rank hold identical stores (they see the same
    micro-batches and gather the same full-sequence hidden), so reducing over
    the DP group alone is sufficient. Single-process / uninitialized runs
    degrade to the local answer.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return has_records
    try:
        from megatron.core import parallel_state as mpu

        group = mpu.get_data_parallel_group()
    except Exception:  # pragma: no cover - no-megatron unit-test paths
        return has_records
    if group is None or dist.get_world_size(group) == 1:
        return has_records

    from verl.utils.device import get_device_id, get_device_name

    device_name = get_device_name()
    device = torch.device("cpu") if device_name == "cpu" else torch.device(f"{device_name}:{get_device_id()}")
    flag = torch.tensor([1 if has_records else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def refresh_frozen_teacher_head(engine, global_step: Optional[int] = None):
    """Re-snapshot the policy ``lm_head`` onto ``engine._eagle3.frozen_lm_head``.

    Must run **before** ``update_actor``, not after.

    The hidden states being trained on were captured during ``compute_log_prob``,
    which runs before any of this step's mini-batch updates. The teacher is
    rebuilt as ``lm_head @ stashed_hidden``, so the head has to come from the same
    weights as the body. Snapshotting after ``update_actor`` composes a
    post-update head with a pre-update hidden -- a pairing that never existed as a
    model, so the distribution the draft is distilled toward is not any policy's.

    Nothing raises when this is ordered wrongly. The draft trains against a
    slightly wrong target and the only symptom is an acceptance rate that will not
    climb. verl-SpeCo pins the same ordering at ``speco_ray_trainer.py:1891``
    (sync) ahead of ``:1895`` (update).

    A fresh snapshot every step rather than a cached one, because the head does
    move between steps; the cost is a parameter read, never a forward.
    """
    from verl.models.eagle3.engine_support import _unwrap_gpt
    from verl.models.eagle3.frozen_teacher import build_frozen_teacher_head

    state = getattr(engine, "_eagle3", None)
    if state is None or not state.enabled:
        return None

    gpt = _unwrap_gpt(engine.module)
    draft_holder = getattr(gpt, "_eagle3_draft", None)
    draft = draft_holder[0] if draft_holder else None
    if draft is None or not hasattr(draft, "t2d"):
        logger.warning("[DRAFT-TEACHER] no draft/t2d available; skipping snapshot")
        return None

    if getattr(gpt, "share_embeddings_and_output_weights", False):
        weight = gpt.shared_embedding_or_output_weight()
    else:
        weight = gpt.output_layer.weight

    tp_size = 1
    try:
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
    except Exception:  # pragma: no cover - single-process / no-megatron paths
        pass

    state.frozen_lm_head = build_frozen_teacher_head(
        weight, draft.t2d, tp_size=tp_size, global_step=global_step
    )
    return state.frozen_lm_head


def train_draft_from_store(
    engine,
    store,
    *,
    micro_batch_size: Optional[int] = None,
    global_step: Optional[int] = None,
    frozen_head=None,
    steps_per_trigger: int = 1,
    batch_size_per_gpu: Optional[int] = None,
) -> Optional[dict]:
    """Train the draft on this step's stashed features (SpeCo-style inner loop).

    Call once per draft-training step, after ``update_actor`` returns.

    Mirrors verl-SpeCo's trigger semantics (``speco_worker.py:887-891`` +
    ``base_trainer.py:3359``): one trigger runs ``steps_per_trigger`` independent
    optimizer steps; each step draws ``batch_size_per_gpu`` windows at random
    (without replacement, ``_sample_training_items`` style) from this step's
    pool, forwards them, and does a full backward + clip + optimizer.step via
    ``eagle3_backward_step``. Windows get revisited across the inner steps --
    that resampling is where SpeCo's drafter gets its learning throughput
    (P1-2: one pass over the pool is ~10x fewer updates than the reference).

    The defaults (1 step, whole pool) keep the old single-update behaviour for
    direct callers/tests; ``update_draft_deferred`` passes the SpeCo-aligned
    config values (10 steps of 4 windows).

    Within one inner step, ``micro_batch_size`` still chunks the step's batch
    for forward memory; the chunks' losses are averaged into that step's single
    backward, exactly as before.

    Args:
        engine: the Megatron engine carrying ``_eagle3``.
        store: :class:`~verl.models.eagle3.feature_store.DraftFeatureStore`;
            drained here, so a second call in the same step is a no-op.
        micro_batch_size: windows per forward chunk; ``None`` = one chunk.
        global_step: for logging, the snapshot staleness check, and the
            deterministic sampling seed.
        frozen_head: override the snapshot on ``state.frozen_lm_head``.
        steps_per_trigger: optimizer steps per trigger (SpeCo ``training.step``).
        batch_size_per_gpu: windows per inner step (SpeCo ``batch_size_per_gpu``);
            ``None`` = the whole pool every step.

    Returns:
        ``{"losses": [per-update loss...], "num_windows": N}``, or ``None`` when
        there was nothing to train on.

    Raises:
        RuntimeError: if features exist but no teacher snapshot does -- training
            on a missing teacher is not a recoverable state, and falling back to
            the policy head would silently reintroduce the extra forward this
            design removes.
    """
    import random

    from verl.models.eagle3.engine_support import _unwrap_gpt, eagle3_backward_step
    from verl.models.eagle3.loss_mcore import compute_draft_loss
    from verl.models.mcore.eagle3_patch import stash_draft_loss

    state = getattr(engine, "_eagle3", None)
    if state is None or not state.enabled:
        return None

    records = store.drain()

    # Train-or-skip must be decided identically on every DP rank BEFORE any
    # forward/backward: an uneven decision hangs the draft DDP all-reduce.
    if not _dp_all_ranks_ready(bool(records)):
        if records:
            logger.warning(
                "[DRAFT-TRAIN-V3] step %s: skipping draft training -- a peer DP rank "
                "collected no windows (its samples all failed the length gate), and "
                "training without it would hang the draft DDP gradient all-reduce. "
                "%d local window(s) dropped.",
                global_step,
                len(records),
            )
        else:
            logger.warning(
                "[DRAFT-TRAIN-V3] step %s: feature store is empty, nothing to train. "
                "Either no sample passed the collect plan's length gate, or collection "
                "did not run on this step.",
                global_step,
            )
        return None

    head = frozen_head if frozen_head is not None else getattr(state, "frozen_lm_head", None)
    if head is None:
        raise RuntimeError(
            f"[DRAFT-TRAIN-V3] step {global_step}: {len(records)} stashed window(s) but no "
            "frozen lm_head snapshot. Call refresh_frozen_teacher_head(engine, step) after "
            "update_actor and before this. Training without it is impossible, and rebuilding "
            "the teacher through the policy would reintroduce the extra forward this path exists "
            "to avoid."
        )
    if head.source_step is not None and global_step is not None and head.source_step != global_step:
        logger.warning(
            "[DRAFT-TRAIN-V3] teacher snapshot is from step %s but training step is %s. "
            "A stashed hidden is only valid against the lm_head that produced it.",
            head.source_step,
            global_step,
        )
    # NOTE: the check above compares step numbers, so it cannot see an
    # ordering mistake *within* a step -- taking the snapshot after
    # update_actor carries the right step number while holding the wrong
    # weights. That was a real bug here (fixed by moving the snapshot ahead
    # of update_actor); it is caught by placement, not by this warning.
    # Detecting it at runtime would mean hashing the head at collection time
    # and comparing, which costs a full-head read on every step.

    gpt = _unwrap_gpt(engine.module)
    draft_holder = getattr(gpt, "_eagle3_draft", None)
    draft = draft_holder[0] if draft_holder else None
    if draft is None:
        logger.warning("[DRAFT-TRAIN-V3] no draft module on the engine; skipping")
        return None

    device = next(draft.parameters()).device
    dtype = next(draft.parameters()).dtype
    ttt_length = getattr(gpt, "_eagle3_ttt_length", 1)
    gamma = getattr(gpt, "_eagle3_gamma", 0.8)
    temperature = getattr(gpt, "_eagle3_temperature", 1.0)

    def _forward_and_stash(batch_records) -> int:
        """Forward the given windows (chunked) and stash each chunk's loss."""
        n = 0
        for chunk in _chunks(batch_records, micro_batch_size):
            batch = stack_records(chunk, device=device, dtype=dtype)

            # Teacher rebuilt from the stashed pre-lm_head hidden, in draft-vocab
            # width. compute_draft_loss accepts a pre-filtered teacher unchanged
            # (loss_mcore.py:87-89), so no t2d selection is needed here.
            teacher_logits = head(batch["final_hidden"])

            draft_out = draft(
                input_ids=batch["input_ids"],
                hidden_states=batch["aux_hidden"],
                loss_mask=batch["loss_mask"],
                attention_mask=None,
                position_ids=batch["position_ids"],
                ttt_length=ttt_length,
            )
            loss_out = compute_draft_loss(
                student_logits_per_step=draft_out["logits"],
                teacher_logits=teacher_logits,
                t2d=draft.t2d,
                loss_mask=batch["loss_mask"],
                position_masks_per_step=draft_out.get("position_masks"),
                gamma=gamma,
                temperature=temperature,
            )
            # Stash through the paired helper rather than touching the attribute:
            # eagle3_backward_step drains via _get_patching_model, while `gpt` here
            # came from _unwrap_gpt. The two resolve independently, and a mismatch
            # would park the loss where nothing drains it -- draft never backwards,
            # nothing raises.
            if not stash_draft_loss(gpt, loss_out["loss"]):
                raise RuntimeError(
                    f"[DRAFT-TRAIN-V3] step {global_step}: could not stash the draft loss on "
                    f"{type(gpt).__name__}. eagle3_backward_step would find nothing to backward "
                    "and report success on a step that trained nothing."
                )
            n += 1
        return n

    # Deterministic sampling: same seed on every DP rank so all ranks run the
    # same number of (symmetric) backwards; the pools themselves are rank-local,
    # so identical seeds do not correlate the data.
    steps = max(int(steps_per_trigger), 1)
    rng = random.Random(1000003 * int(global_step or 0) + 7)
    losses: list[float] = []
    for _ in range(steps):
        if batch_size_per_gpu and len(records) > int(batch_size_per_gpu):
            batch_records = rng.sample(records, int(batch_size_per_gpu))
        else:
            batch_records = records
        _forward_and_stash(batch_records)
        # Reuse the verified optimizer half: drain -> mean -> backward -> clip -> step.
        loss_val = eagle3_backward_step(engine)
        if loss_val is None:
            raise RuntimeError(
                f"[DRAFT-TRAIN-V3] step {global_step}: eagle3_backward_step found no stashed "
                "loss right after stashing -- something drained it in between."
            )
        losses.append(float(loss_val))

    logger.warning(
        "[DRAFT-TRAIN-V3] step %s: %d window(s), %d optimizer step(s) of %s window(s) each; "
        "loss first=%.4f last=%.4f",
        global_step,
        len(records),
        steps,
        batch_size_per_gpu or len(records),
        losses[0],
        losses[-1],
    )
    return {"losses": losses, "num_windows": len(records)}
