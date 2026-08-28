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
"""Host-side stash of EAGLE3 draft-training features.

The deferred-training design harvests hidden states during a forward the RL loop
already pays for, then trains the draft *after* ``update_actor`` returns -- once
the policy's activations and gradients are gone. That handoff needs somewhere to
park the features, and it must not be device memory: the whole point of deferring
is to keep the draft's cost out of the policy's peak.

So records land on the host (decision D2 in
开发设计/串行训练/方案设计：SpeCo式采集与延后训练_v3.md §5).

Lifetime is one training step. EAGLE3 stashes hidden states rather than logits,
and a stashed hidden is only meaningful against the ``lm_head`` version that
produced it -- once the actor updates, the teacher those rows imply is wrong.
verl-SpeCo pins the same constraint at ``base_trainer.py:2588-2596`` (it forces
``buffer_steps = 0`` for EAGLE3 when ``use_logits=False``). This store therefore
refuses cross-step accumulation outright instead of silently serving stale rows.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DraftFeatureRecord:
    """One sample's harvested window, on the host.

    Row ``i`` of every field refers to the same absolute token position,
    ``positions[i]``. ``aux_hidden``/``final_hidden`` are already
    sequence-parallel-gathered, so positions are global sequence coordinates.

    Attributes:
        aux_hidden: ``(rows, H * num_aux)`` -- the draft's input features.
        final_hidden: ``(rows, H)`` -- pre-``lm_head`` hidden, replayed through a
            frozen ``lm_head`` copy at training time to rebuild teacher logits.
            Stored instead of the logits themselves: full-vocab logits are
            ``rows x 151936``, three orders of magnitude larger.
        input_ids / position_ids / loss_mask: ``(rows,)`` draft forward inputs.
        positions: ``(rows,)`` absolute sequence positions this window covers.
        global_step: step that produced the record; used to enforce the
            one-step lifetime.
    """

    aux_hidden: torch.Tensor
    final_hidden: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: torch.Tensor
    positions: torch.Tensor
    global_step: int

    @property
    def num_rows(self) -> int:
        return int(self.aux_hidden.shape[0])


class DraftFeatureStore:
    """Host-resident, single-step stash of :class:`DraftFeatureRecord`.

    Not thread-safe; one instance per worker process, driven by that process's
    own forward.

    Usage::

        store.begin_step(global_step)      # drops anything older
        store.put(record)                  # once per selected sample
        ...
        records = store.drain()            # hands over and empties
    """

    def __init__(self, max_records: Optional[int] = None):
        """
        Args:
            max_records: hard cap on retained records; further ``put`` calls are
                dropped with a warning. ``None`` disables the cap. This is a
                backstop against a mis-sized collect plan, not a sampling knob --
                sampling belongs in
                :func:`~verl.models.eagle3.collect_plan.build_collect_plan`.
        """
        self._records: list[DraftFeatureRecord] = []
        self._step: Optional[int] = None
        self._max_records = max_records
        self._dropped = 0

    # ---------------------------------------------------------------- lifecycle

    def begin_step(self, global_step: int) -> None:
        """Open a step, discarding any records left from an earlier one."""
        if self._step is not None and self._step != global_step and self._records:
            logger.warning(
                "[DRAFT-STORE] dropping %d unconsumed record(s) from step %s at the "
                "start of step %s -- draft training did not drain them, so those "
                "features were harvested for nothing",
                len(self._records),
                self._step,
                global_step,
            )
        self._records = []
        self._step = global_step
        self._dropped = 0

    def put(self, record: DraftFeatureRecord) -> bool:
        """Stash one record. Returns False if it was dropped by the cap."""
        if record.global_step != self._step:
            raise ValueError(
                f"[DRAFT-STORE] record is from step {record.global_step} but the store "
                f"is open on step {self._step}. Stashed hidden states are only valid "
                "against the lm_head that produced them; mixing steps would train the "
                "draft against a teacher that no longer exists. Call begin_step first."
            )
        if self._max_records is not None and len(self._records) >= self._max_records:
            self._dropped += 1
            logger.warning(
                "[DRAFT-STORE] record cap %d reached at step %s; dropped %d so far. "
                "The collect plan is handing over more samples than the store admits.",
                self._max_records,
                self._step,
                self._dropped,
            )
            return False
        self._records.append(record)
        return True

    def drain(self) -> list[DraftFeatureRecord]:
        """Hand over every record and empty the store."""
        records, self._records = self._records, []
        return records

    def clear(self) -> None:
        self._records = []
        self._dropped = 0

    # ---------------------------------------------------------------- introspection

    def __len__(self) -> int:
        return len(self._records)

    @property
    def step(self) -> Optional[int]:
        return self._step

    @property
    def dropped(self) -> int:
        return self._dropped

    def nbytes(self) -> int:
        """Bytes held across all records -- for the ``drafter/store_mb`` metric."""
        return sum(
            r.aux_hidden.numel() * r.aux_hidden.element_size()
            + r.final_hidden.numel() * r.final_hidden.element_size()
            for r in self._records
        )


def _maybe_gather_sequence_parallel(tensor: torch.Tensor, sequence_parallel: bool, tp_world_size: int):
    """Reconstitute the full sequence from sequence-parallel shards.

    Under SP each rank holds ``S/TP`` rows, but a
    :class:`~verl.models.eagle3.collect_plan.CollectPlan` addresses *global*
    positions -- so the sequence has to be whole before any row selection.

    Gathering first and slicing second (rather than translating the plan into
    rank-local coordinates and slicing inside the capture hook) costs a transient
    full-sequence tensor. That transient is not new: the parallel draft path
    already materializes it at ``eagle3_patch.py:325-330``. What stays afterwards
    is only the selected rows. The alternative would need a collective that
    gathers scattered rows, which is not a primitive Megatron offers.
    """
    if not sequence_parallel or tp_world_size <= 1:
        return tensor
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

    return gather_from_sequence_parallel_region(tensor)


def _row(tensor: Optional[torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
    """Pick sample ``batch_idx``'s row from a (B, S) or (S,) tensor."""
    if tensor is None:
        return None
    return tensor[batch_idx] if tensor.dim() >= 2 else tensor


def collect_draft_features(
    *,
    store: DraftFeatureStore,
    aux_hidden: torch.Tensor,
    final_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    plan,
    global_step: int,
    sequence_parallel: bool = False,
    tp_world_size: int = 1,
) -> int:
    """Gather, slice to the plan's rows, move to host, and stash.

    Call this from inside the policy forward, where the tensor-parallel group is
    still reachable -- the gather cannot be deferred to draft-training time.

    Args:
        store: destination; ``begin_step(global_step)`` must already have run.
        aux_hidden: ``(S_local, B, H*num_aux)`` sequence-first, from
            ``Eagle3HiddenCapture.get_captured(seqlen_first=True)``. Must be a
            full rank-local shard -- do **not** combine with
            ``Eagle3HiddenCapture.set_row_index`` when ``sequence_parallel`` is
            on, since the hook would slice rank-local rows against global indices.
        final_hidden: ``(S_local, B, H)`` sequence-first, pre-``lm_head``.
        input_ids: ``(B, S)`` global sequence (never SP-sharded).
        position_ids: ``(B, S)`` or ``(S,)``; ``None`` stores a zero row.
        loss_mask: ``(B, S)`` or ``(S,)``; ``None`` stores an all-ones row.
        plan: a :class:`~verl.models.eagle3.collect_plan.CollectPlan`.
        global_step: must match the store's open step.
        sequence_parallel / tp_world_size: gather control.

    Returns:
        Number of records stashed.
    """
    if plan is None:
        return 0

    aux_hidden = _maybe_gather_sequence_parallel(aux_hidden, sequence_parallel, tp_world_size)
    final_hidden = _maybe_gather_sequence_parallel(final_hidden, sequence_parallel, tp_world_size)

    # (S, B, *) -> (B, S, *): per-sample slicing reads much better this way, and
    # the draft forward wants batch-first anyway.
    aux_hidden = aux_hidden.transpose(0, 1)
    final_hidden = final_hidden.transpose(0, 1)

    seq_len = aux_hidden.shape[1]
    batch_size = aux_hidden.shape[0]
    stored = 0

    for batch_idx in range(batch_size):
        if not bool(plan.collect_mask[batch_idx]):
            continue
        positions = plan.hidden_positions[batch_idx]
        if int(positions.max()) >= seq_len:
            raise IndexError(
                f"[DRAFT-COLLECT] sample {batch_idx} wants position {int(positions.max())} "
                f"but the gathered sequence is only {seq_len} long. Either the collect "
                "plan was built from different lengths than this forward saw, or the "
                "sequence-parallel gather did not run (sequence_parallel="
                f"{sequence_parallel}, tp_world_size={tp_world_size})."
            )
        idx = positions.to(aux_hidden.device)

        ids_row = _row(input_ids, batch_idx)
        pos_row = _row(position_ids, batch_idx)
        mask_row = _row(loss_mask, batch_idx)

        record = DraftFeatureRecord(
            # .detach() guards the case where a caller passes a live activation:
            # a stashed tensor must never keep the policy graph alive.
            aux_hidden=aux_hidden[batch_idx].index_select(0, idx).detach().to("cpu"),
            final_hidden=final_hidden[batch_idx].index_select(0, idx).detach().to("cpu"),
            input_ids=ids_row.index_select(0, idx.to(ids_row.device)).detach().to("cpu"),
            position_ids=(
                pos_row.index_select(0, idx.to(pos_row.device)).detach().to("cpu")
                if pos_row is not None
                else torch.zeros(idx.numel(), dtype=torch.long)
            ),
            loss_mask=(
                mask_row.index_select(0, idx.to(mask_row.device)).detach().to("cpu")
                if mask_row is not None
                else torch.ones(idx.numel(), dtype=torch.bool)
            ),
            positions=positions.detach().to("cpu"),
            global_step=global_step,
        )
        if store.put(record):
            stored += 1

    return stored
