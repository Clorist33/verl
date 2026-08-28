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
"""Frozen, draft-vocab-restricted copy of the policy ``lm_head``.

Deferred draft training stashes hidden states, not teacher logits -- full-vocab
logits are ``rows x 151936``, roughly 1.8 GB per sample, against 4096-wide
hiddens they can be rebuilt from. So the teacher has to be reconstructed at
training time, and reconstructing it by running the policy would reintroduce
exactly the forward this design exists to avoid.

Instead we keep a read-only snapshot of ``lm_head`` and replay the stashed
hiddens through it. Two properties make the snapshot cheap:

* **Vocab compression.** ``t2d`` maps the policy's 151936 columns down to the
  draft's 32000 before the loss ever sees them
  (``loss_mcore.py:73`` ``filter_teacher_to_draft_vocab``), so only those rows
  are worth keeping: 32000 x 4096 x 2B = 262 MB instead of 1.2 GB.
* **Pre-filtered teachers are already accepted.** ``filter_teacher_to_draft_vocab``
  passes a teacher straight through when its last dim already equals
  ``t2d.sum()`` (``loss_mcore.py:87-89``), so producing draft-vocab logits
  directly needs no change to ``compute_draft_loss``.

The snapshot is refreshed once per draft-training step by *reading parameters*,
never by a forward pass -- mirroring verl-SpeCo
(``verl_speco/models/target/target_head.py:45-58``,
``verl_speco/integration/rollout_publish.py:271``).

Correctness note: a stale snapshot is the dangerous failure mode here. It does
not raise; it silently trains the draft against a teacher the policy no longer
implements, and only shows up as an acceptance rate that will not climb. The
project has already paid for that lesson once. Hence
``build_frozen_teacher_head`` is cheap enough to call every step, and the
matching test asserts elementwise parity against the live policy head rather
than merely checking shapes.
"""

import logging
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FrozenTeacherHead:
    """Applies a detached ``(draft_vocab, H)`` weight to stashed hidden states.

    Deliberately not an ``nn.Module``: it must never appear in
    ``named_parameters()``. The draft is kept out of the policy's module tree by
    a list wrapper for the same reason (``eagle3_patch.py:130``), and a stray
    teacher head in the optimizer or the weight-export walk would be a similar
    class of bug.
    """

    __slots__ = ("weight", "draft_vocab_size", "hidden_size", "source_step")

    def __init__(self, weight: torch.Tensor, source_step: Optional[int] = None):
        if weight.dim() != 2:
            raise ValueError(f"FrozenTeacherHead weight must be 2-D (draft_vocab, H), got {tuple(weight.shape)}")
        self.weight = weight.detach()
        self.draft_vocab_size, self.hidden_size = weight.shape
        self.source_step = source_step

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(..., H) -> (..., draft_vocab)``, always under ``no_grad``.

        The teacher is a target, never a path for gradient: ``compute_draft_loss``
        detaches it too, but blocking it here means a caller cannot accidentally
        build a graph through the policy's old weights.
        """
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"FrozenTeacherHead expects hidden size {self.hidden_size}, got {hidden.shape[-1]}"
            )
        with torch.no_grad():
            return F.linear(hidden, self.weight.to(device=hidden.device, dtype=hidden.dtype))

    def to(self, *args, **kwargs) -> "FrozenTeacherHead":
        self.weight = self.weight.to(*args, **kwargs)
        return self

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def __repr__(self) -> str:
        return (
            f"FrozenTeacherHead(draft_vocab={self.draft_vocab_size}, "
            f"hidden={self.hidden_size}, step={self.source_step}, "
            f"{self.nbytes() / 1024**2:.1f} MB)"
        )


def _default_gather(shard: torch.Tensor, tp_size: int) -> list:
    """All-gather ``shard`` across the tensor-model-parallel group."""
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    out = [torch.empty_like(shard) for _ in range(tp_size)]
    dist.all_gather(out, shard.contiguous(), group=mpu.get_tensor_model_parallel_group())
    return out


def select_draft_vocab_rows(
    weight_shard: torch.Tensor,
    t2d: torch.Tensor,
    *,
    tp_size: int = 1,
    gather_fn: Optional[Callable[[torch.Tensor, int], list]] = None,
) -> torch.Tensor:
    """Assemble the ``(draft_vocab, H)`` teacher weight from a vocab-parallel head.

    Under tensor parallelism Megatron's ``output_layer`` is a vocab-parallel
    ``ColumnParallelLinear``: rank ``r`` owns full-vocab rows
    ``[r*V/TP, (r+1)*V/TP)``. Each rank therefore holds a *different, contiguous*
    slice of the draft vocabulary.

    Rows are gathered before selection rather than selected before gathering.
    Selecting first yields a different row count per rank, which needs a
    variable-length collective (pad-to-max plus a count exchange); gathering
    first is one ``all_gather`` of the shard the rank already has. The transient
    is the full ``(V, H)`` head, and it exists only while this function runs --
    the snapshot that survives is the 262 MB draft-vocab slice. This mirrors
    ``_inject_and_freeze_draft_embed`` (``engine_support.py:415-422``), which
    reconstructs the embedding table the same way.

    Row order is ascending full-vocab id, matching what
    ``teacher_logits[..., t2d_bool]`` would have produced -- boolean masking
    preserves index order, and concatenating rank shards in rank order preserves
    it across the TP group. Getting this backwards would misalign every teacher
    column against the draft head without changing any shape.

    Args:
        weight_shard: ``(V/TP, H)`` this rank's ``lm_head`` rows, or ``(V, H)``
            when ``tp_size == 1``.
        t2d: ``(V,)`` bool -- True where a full-vocab id is in the draft vocab.
        tp_size: tensor-model-parallel world size.
        gather_fn: override for the collective, for tests.

    Returns:
        ``(draft_vocab, H)`` detached weight.
    """
    if weight_shard.dim() != 2:
        raise ValueError(f"weight_shard must be 2-D (vocab_shard, H), got {tuple(weight_shard.shape)}")

    shard = weight_shard.detach()
    if tp_size > 1:
        gather = gather_fn or _default_gather
        full = torch.cat(gather(shard, tp_size), dim=0)
    else:
        full = shard

    t2d_bool = t2d.to(device=full.device, dtype=torch.bool)
    vocab_rows, _ = full.shape
    if t2d_bool.numel() != vocab_rows:
        # Megatron pads the vocab up to a TP multiple; t2d is sized to the real
        # vocab. Trailing padded rows can never be draft-vocab ids, so dropping
        # them is safe -- but a t2d LONGER than the head means the caller paired
        # a mapping with the wrong model, which must not be papered over.
        if t2d_bool.numel() > vocab_rows:
            raise ValueError(
                f"t2d covers {t2d_bool.numel()} ids but the gathered lm_head only has "
                f"{vocab_rows} rows. The vocab mapping does not belong to this policy."
            )
        logger.warning(
            "eagle3: lm_head has %d rows vs t2d's %d ids (padded vocab); "
            "ignoring the %d trailing padded rows",
            vocab_rows,
            t2d_bool.numel(),
            vocab_rows - t2d_bool.numel(),
        )
        full = full[: t2d_bool.numel()]

    rows = full[t2d_bool]
    if rows.shape[0] == 0:
        raise ValueError("eagle3: t2d selects zero draft-vocab rows; the mapping is empty or misaligned")
    return rows.contiguous()


def build_frozen_teacher_head(
    weight_shard: torch.Tensor,
    t2d: torch.Tensor,
    *,
    tp_size: int = 1,
    global_step: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    gather_fn: Optional[Callable[[torch.Tensor, int], list]] = None,
) -> FrozenTeacherHead:
    """Snapshot the policy ``lm_head`` for teacher reconstruction.

    Call once per draft-training step, immediately before draft training, so the
    snapshot matches the ``lm_head`` that produced the stashed hiddens. See the
    module docstring on why staleness here fails silently.
    """
    rows = select_draft_vocab_rows(weight_shard, t2d, tp_size=tp_size, gather_fn=gather_fn)
    if dtype is not None:
        rows = rows.to(dtype)
    head = FrozenTeacherHead(rows, source_step=global_step)
    logger.warning(
        "[DRAFT-TEACHER] frozen lm_head snapshot at step %s: %s", global_step, head
    )
    return head
