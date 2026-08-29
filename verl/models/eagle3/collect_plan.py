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
"""EAGLE3 draft-training feature collection plan.

Decides which ``(sample, token_position)`` rows of the policy forward get
harvested for draft training, so the piggybacked forward only pays for the rows
actually used.

This is a line-for-line port of verl-SpeCo's
``SpecoRayPPOTrainer._speco_build_oldlogprob_collect_plan``
(``verl_speco/trainer/speco_ray_trainer.py:938-1042``); the defaults below mirror
``verl_speco/config/speco_base.yaml:58-66``. Ported deliberately without
deviation so the first run can be compared against a known-good reference --
see 开发设计/串行训练/方案设计：SpeCo式采集与延后训练_v3.md §5 D1.

One intentional simplification: SpeCo buckets samples across a separate
``SpecoWorker`` replica group, so it carries an ``owner_rank`` per sample. Our
draft is DDP-replicated inside the *same* workers (one draft per DP rank), so
each rank plans only for its own shard and ``owner_count`` defaults to 1. The
parameter is kept so the quota logic stays identical to the reference.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Defaults mirrored from verl_speco/config/speco_base.yaml:58-66
DEFAULT_WINDOW_TRAIN_ROWS = 512
DEFAULT_WINDOW_MODE = "front"
DEFAULT_SAMPLE_RATE = 1.0
DEFAULT_MAX_SAMPLES_PER_REPLICA = 16
DEFAULT_MAX_TOKENS_PER_REPLICA = 16384

_VALID_WINDOW_MODES = {"front", "random"}


@dataclass
class CollectPlan:
    """Which rows to harvest from the upcoming policy forward.

    Attributes:
        collect_mask: ``(B,)`` bool -- which samples were selected.
        hidden_positions: ``(B, hidden_rows)`` long -- absolute token positions
            to gather per sample. Rows of unselected samples are zero-filled and
            must be ignored via ``collect_mask``.
        hidden_position_mask: ``(B, hidden_rows)`` bool -- valid entries of
            ``hidden_positions``.
        owner_rank: ``(B,)`` long -- which draft replica owns each sample.
        hidden_rows: ``window_train_rows + 1``. The extra row supplies the
            next-token target for the last trained position.
        selected_count / candidate_count: for the ``drafter/collect_*`` metrics.
        owner_token_counts: rows charged to each owner's token quota.
    """

    collect_mask: torch.Tensor
    hidden_positions: torch.Tensor
    hidden_position_mask: torch.Tensor
    owner_rank: torch.Tensor
    hidden_rows: int
    owner_count: int
    selected_count: int
    candidate_count: int
    owner_token_counts: list[int]
    window_mode: str


@dataclass
class CollectBudget:
    """Step-level remaining quota, shared across micro-batches.

    The quotas in :func:`build_collect_plan` bound one call, and verl-SpeCo calls
    it once per step from the driver, where the whole batch is visible. We call it
    from ``_postprocess``, which Megatron invokes **per micro-batch** -- and at
    ``ppo_micro_batch_size_per_gpu=1`` that is one sequence per call. A per-call
    quota of 16 can never bind on a batch of 1, so every micro-batch collected its
    sample and the step ended up with one window per sequence (64 per rank at the
    current sizing) instead of the 16 the design budgets for.

    This carries what is left of the step's quota between those calls. Reset once
    per ``compute_log_prob`` in ``_eagle3_open_collection``, decremented after each
    micro-batch actually stashes.
    """

    max_samples: Optional[int] = DEFAULT_MAX_SAMPLES_PER_REPLICA
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS_PER_REPLICA
    used_samples: int = 0
    used_tokens: int = 0

    def remaining_samples(self) -> Optional[int]:
        return None if self.max_samples is None else max(self.max_samples - self.used_samples, 0)

    def remaining_tokens(self) -> Optional[int]:
        return None if self.max_tokens is None else max(self.max_tokens - self.used_tokens, 0)

    def exhausted(self) -> bool:
        return self.remaining_samples() == 0 or self.remaining_tokens() == 0

    def consume(self, n_samples: int, n_tokens: int) -> None:
        self.used_samples += int(n_samples)
        self.used_tokens += int(n_tokens)

    def reset(self) -> None:
        self.used_samples = 0
        self.used_tokens = 0


def _hash_fraction(key: str) -> float:
    """Deterministic [0, 1) draw. Port of speco_ray_trainer.py:927-929."""
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)


def _hash_int(key: str, inclusive_max: int) -> int:
    """Deterministic int in [0, inclusive_max]. Port of speco_ray_trainer.py:931-936."""
    if inclusive_max <= 0:
        return 0
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (inclusive_max + 1)


def _as_int_list(value: Union[torch.Tensor, Sequence[int]], name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.dim() != 1:
            raise ValueError(f"{name} must be 1-D, got shape {tuple(value.shape)}")
        return [int(v) for v in value.detach().cpu().tolist()]
    return [int(v) for v in value]


def build_collect_plan(
    *,
    prompt_lens: Union[torch.Tensor, Sequence[int]],
    response_lens: Union[torch.Tensor, Sequence[int]],
    global_step: int,
    window_train_rows: int = DEFAULT_WINDOW_TRAIN_ROWS,
    window_mode: str = DEFAULT_WINDOW_MODE,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    max_samples_per_replica: Optional[int] = DEFAULT_MAX_SAMPLES_PER_REPLICA,
    max_tokens_per_replica: Optional[int] = DEFAULT_MAX_TOKENS_PER_REPLICA,
    seed_by_step: bool = True,
    owner_count: int = 1,
) -> Optional[CollectPlan]:
    """Build the harvest plan for one step, or ``None`` if nothing qualifies.

    Selection runs three filters, in the reference's order:

    1. **Length gate** -- samples whose response is shorter than ``hidden_rows``
       are dropped whole (not padded, not truncated-and-kept).
       ``speco_ray_trainer.py:999``
    2. **Hash sampling** -- inactive at the default ``sample_rate=1.0``. Uses a
       hash rather than an RNG so a given ``(step, sample)`` always decides the
       same way. ``:1002``
    3. **Round-robin + quota** -- samples go to owners in turn; an owner stops
       accepting once it hits either the sample quota or the token quota. ``:1004``

    The window itself starts at ``prompt_len - 1`` (the last prompt position,
    whose hidden predicts the first response token), so prompt rows are never
    harvested -- the draft only trains on responses.

    Args:
        prompt_lens: ``(B,)`` real prompt length per sample (padding excluded).
        response_lens: ``(B,)`` real response length per sample.
        global_step: mixed into the hash when ``seed_by_step``, so the chosen
            windows move across steps instead of locking onto one slice.
        window_train_rows: trained positions per sample; one more row is
            harvested to supply the last position's target.
        window_mode: ``"front"`` always starts at the response head;
            ``"random"`` picks a deterministic hashed offset.
        sample_rate: keep-probability for filter 2; ``>= 1.0`` disables it.
        max_samples_per_replica: per-owner sample quota; ``None`` = unlimited.
        max_tokens_per_replica: per-owner row quota; ``None`` = unlimited.
        seed_by_step: include ``global_step`` in the hash key.
        owner_count: number of draft replicas to spread samples over.

    Returns:
        A :class:`CollectPlan`, or ``None`` when the step collects nothing
        (rate <= 0, non-positive ``window_train_rows``, or no sample passing all
        three filters). Callers must treat ``None`` as "run the plain forward".
    """
    if sample_rate <= 0:
        return None

    train_rows = int(window_train_rows or 0)
    if train_rows <= 0:
        return None
    hidden_rows = train_rows + 1

    mode = str(window_mode or DEFAULT_WINDOW_MODE).strip().lower()
    if mode not in _VALID_WINDOW_MODES:
        mode = DEFAULT_WINDOW_MODE

    prompt_lens = _as_int_list(prompt_lens, "prompt_lens")
    response_lens = _as_int_list(response_lens, "response_lens")
    if len(prompt_lens) != len(response_lens):
        raise ValueError(
            f"prompt_lens ({len(prompt_lens)}) and response_lens ({len(response_lens)}) "
            "must describe the same batch"
        )
    batch_size = len(prompt_lens)
    if batch_size == 0:
        return None

    owner_count = max(int(owner_count), 1)
    max_per_owner = batch_size if max_samples_per_replica is None else max(int(max_samples_per_replica), 0)
    max_tokens_per_owner = None if max_tokens_per_replica is None else max(int(max_tokens_per_replica), 0)

    collect_mask = torch.zeros(batch_size, dtype=torch.bool)
    hidden_positions = torch.zeros(batch_size, hidden_rows, dtype=torch.long)
    hidden_position_mask = torch.zeros(batch_size, hidden_rows, dtype=torch.bool)
    owner_rank = torch.zeros(batch_size, dtype=torch.long)

    owner_counts = [0 for _ in range(owner_count)]
    owner_token_counts = [0 for _ in range(owner_count)]
    step_key = global_step if seed_by_step else "request"

    candidate_count = 0
    selected_count = 0

    for batch_idx in range(batch_size):
        prompt_len = prompt_lens[batch_idx]
        response_len = response_lens[batch_idx]

        # Filter 1: too short to fill a window -> drop the sample entirely.
        if prompt_len <= 0 or response_len < hidden_rows:
            continue
        candidate_count += 1

        sample_key = f"{step_key}:{batch_idx}:{prompt_len}:{response_len}"

        # Filter 2: hash sampling (no-op at the default rate of 1.0).
        if sample_rate < 1.0 and _hash_fraction(sample_key) >= sample_rate:
            continue

        # Filter 3: round-robin owner assignment, then per-owner quotas.
        owner = selected_count % owner_count
        if owner_counts[owner] >= max_per_owner:
            continue
        if max_tokens_per_owner is not None and owner_token_counts[owner] + hidden_rows > max_tokens_per_owner:
            continue

        max_start_offset = max(response_len - hidden_rows, 0)
        random_offset = _hash_int(f"{sample_key}:window", max_start_offset) if mode == "random" else 0
        # prompt_len - 1: the last prompt position, whose hidden state predicts
        # the first response token. Starting at prompt_len would skip it.
        start = max(prompt_len - 1, 0) + random_offset

        collect_mask[batch_idx] = True
        hidden_positions[batch_idx, :] = torch.arange(start, start + hidden_rows, dtype=torch.long)
        hidden_position_mask[batch_idx, :] = True
        owner_rank[batch_idx] = owner
        owner_counts[owner] += 1
        owner_token_counts[owner] += hidden_rows
        selected_count += 1

    if selected_count <= 0:
        logger.warning(
            "[DRAFT-COLLECT] step %s collected nothing: %d/%d samples passed the "
            "length gate (need response_len >= %d)",
            global_step,
            candidate_count,
            batch_size,
            hidden_rows,
        )
        return None

    return CollectPlan(
        collect_mask=collect_mask,
        hidden_positions=hidden_positions,
        hidden_position_mask=hidden_position_mask,
        owner_rank=owner_rank,
        hidden_rows=hidden_rows,
        owner_count=owner_count,
        selected_count=selected_count,
        candidate_count=candidate_count,
        owner_token_counts=owner_token_counts,
        window_mode=mode,
    )
