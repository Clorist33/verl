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
"""P2 tests: frozen teacher head (A4) + teacher reconstruction parity (B3).

The parity tests are the point of this file. A frozen head that disagrees with
the live policy head does not raise -- it silently trains the draft against a
teacher the policy no longer implements, and surfaces only as an acceptance rate
that never climbs. So these compare values elementwise against the reference
path, not shapes.

Run with pytest:  python3 -m pytest tests/eagle3/serial_training/test_p2_frozen_teacher.py -q
"""

import pytest
import torch

from verl.models.eagle3.frozen_teacher import (
    FrozenTeacherHead,
    build_frozen_teacher_head,
    select_draft_vocab_rows,
)
from verl.models.eagle3.loss_mcore import filter_teacher_to_draft_vocab

V, DRAFT_V, H = 200, 32, 16


def _t2d(vocab=V, draft_vocab=DRAFT_V, seed=0):
    """Bool (V,) selecting `draft_vocab` ids, mimicking a compressed vocab map."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(vocab, generator=g)[:draft_vocab]
    mask = torch.zeros(vocab, dtype=torch.bool)
    mask[idx] = True
    return mask


def _lm_head(vocab=V, hidden=H, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, hidden, generator=g, dtype=torch.float32)


# ------------------------------------------------- B3: parity with the live head


def test_frozen_teacher_matches_the_full_vocab_path_exactly():
    """THE critical assertion: frozen head == policy head then t2d filter."""
    w, t2d = _lm_head(), _t2d()
    hidden = torch.randn(2, 7, H)

    # reference: what eagle3_patch.py:349-357 computes today
    full_logits = torch.nn.functional.linear(hidden, w)          # (B, S, V)
    reference = filter_teacher_to_draft_vocab(full_logits, t2d)  # (B, S, draft_vocab)

    head = build_frozen_teacher_head(w, t2d)
    got = head(hidden)

    assert got.shape == reference.shape == (2, 7, DRAFT_V)
    torch.testing.assert_close(got, reference, rtol=0, atol=0)


def test_parity_holds_in_bfloat16():
    """The real run is bf16; reduction order must not drift the two paths apart."""
    w, t2d = _lm_head(), _t2d()
    hidden = torch.randn(2, 7, H, dtype=torch.bfloat16)
    w_bf = w.to(torch.bfloat16)

    reference = filter_teacher_to_draft_vocab(torch.nn.functional.linear(hidden, w_bf), t2d)
    got = build_frozen_teacher_head(w_bf, t2d, dtype=torch.bfloat16)(hidden)

    torch.testing.assert_close(got, reference, rtol=0, atol=0)


def test_row_order_is_ascending_vocab_id():
    """Column k of the teacher must be full-vocab id `nonzero(t2d)[k]`.

    A permuted head keeps every shape intact while pairing each teacher column
    with the wrong draft column -- invisible except as a draft that will not learn.
    """
    w, t2d = _lm_head(), _t2d()
    rows = select_draft_vocab_rows(w, t2d)
    expected_ids = torch.nonzero(t2d, as_tuple=False).squeeze(-1)
    assert expected_ids.tolist() == sorted(expected_ids.tolist())
    for k, vocab_id in enumerate(expected_ids.tolist()):
        torch.testing.assert_close(rows[k], w[vocab_id], rtol=0, atol=0)


def test_compute_draft_loss_accepts_the_prefiltered_teacher():
    """loss_mcore.py:87-89 passes a draft-width teacher through untouched."""
    w, t2d = _lm_head(), _t2d()
    teacher = build_frozen_teacher_head(w, t2d)(torch.randn(2, 5, H))
    passed_through = filter_teacher_to_draft_vocab(teacher, t2d)
    torch.testing.assert_close(passed_through, teacher, rtol=0, atol=0)


# ------------------------------------------------- A4: tensor-parallel assembly


def _fake_gather(shards):
    """Stand-in for the TP all-gather: every rank sees the same shard list."""
    return lambda shard, tp_size: list(shards)


def test_tp_shards_reassemble_into_the_full_head():
    """Concatenating rank shards in rank order must reproduce the unsharded result."""
    w, t2d = _lm_head(), _t2d()
    tp = 4
    shards = list(w.chunk(tp, dim=0))

    single = select_draft_vocab_rows(w, t2d, tp_size=1)
    sharded = select_draft_vocab_rows(shards[0], t2d, tp_size=tp, gather_fn=_fake_gather(shards))
    torch.testing.assert_close(sharded, single, rtol=0, atol=0)


def test_tp_path_still_matches_the_live_head():
    """End-to-end parity under TP, not just self-consistency between the two paths."""
    w, t2d = _lm_head(), _t2d()
    tp = 4
    shards = list(w.chunk(tp, dim=0))
    hidden = torch.randn(3, 4, H)

    reference = filter_teacher_to_draft_vocab(torch.nn.functional.linear(hidden, w), t2d)
    head = build_frozen_teacher_head(shards[1], t2d, tp_size=tp, gather_fn=_fake_gather(shards))
    torch.testing.assert_close(head(hidden), reference, rtol=0, atol=0)


def test_padded_vocab_rows_are_dropped():
    """Megatron pads vocab to a TP multiple; the pad rows are not draft ids."""
    w, t2d = _lm_head(), _t2d()
    padded = torch.cat([w, torch.randn(8, H)], dim=0)  # 8 padding rows
    torch.testing.assert_close(
        select_draft_vocab_rows(padded, t2d), select_draft_vocab_rows(w, t2d), rtol=0, atol=0
    )


def test_oversized_t2d_is_rejected_not_papered_over():
    """A mapping longer than the head means it belongs to a different model."""
    w = _lm_head()
    with pytest.raises(ValueError, match="does not belong to this policy"):
        select_draft_vocab_rows(w, _t2d(vocab=V + 16))


def test_empty_mapping_is_rejected():
    w = _lm_head()
    with pytest.raises(ValueError, match="zero draft-vocab rows"):
        select_draft_vocab_rows(w, torch.zeros(V, dtype=torch.bool))


# ------------------------------------------------- A4: head behaviour


def test_head_output_carries_no_gradient():
    """The teacher is a target; it must never open a path back into old weights."""
    w, t2d = _lm_head().requires_grad_(True), _t2d()
    head = build_frozen_teacher_head(w, t2d)
    out = head(torch.randn(2, 3, H, requires_grad=True))
    assert out.grad_fn is None
    assert not out.requires_grad


def test_weight_is_detached_from_the_policy():
    w, t2d = _lm_head().requires_grad_(True), _t2d()
    head = build_frozen_teacher_head(w, t2d)
    assert not head.weight.requires_grad
    assert head.weight.grad_fn is None


def test_head_is_not_an_nn_module():
    """It must stay out of named_parameters() -- same reason the draft does."""
    head = build_frozen_teacher_head(_lm_head(), _t2d())
    assert not isinstance(head, torch.nn.Module)
    assert not hasattr(head, "named_parameters")


def test_hidden_size_mismatch_is_caught():
    head = build_frozen_teacher_head(_lm_head(), _t2d())
    with pytest.raises(ValueError, match="expects hidden size"):
        head(torch.randn(2, 3, H + 1))


def test_snapshot_records_its_step():
    head = build_frozen_teacher_head(_lm_head(), _t2d(), global_step=7)
    assert head.source_step == 7
    assert "step=7" in repr(head)


def test_nbytes_matches_the_262mb_budget():
    """32000 x 4096 bf16 should land on the design doc's 262 MB."""
    head = FrozenTeacherHead(torch.zeros(32000, 4096, dtype=torch.bfloat16))
    mb = head.nbytes() / 1024**2
    assert 245 < mb < 275, f"expected ~262 MB per the design doc, got {mb:.1f} MB"


def test_dtype_override_is_applied():
    head = build_frozen_teacher_head(_lm_head(), _t2d(), dtype=torch.bfloat16)
    assert head.weight.dtype == torch.bfloat16


def test_rejects_non_2d_weight():
    with pytest.raises(ValueError, match="2-D"):
        FrozenTeacherHead(torch.zeros(4))
    with pytest.raises(ValueError, match="2-D"):
        select_draft_vocab_rows(torch.zeros(4), torch.ones(4, dtype=torch.bool))


# ------------------------------------------------- staleness

def test_a_stale_snapshot_diverges_measurably():
    """Guards the premise behind refreshing every step.

    If a stale head happened to agree with an updated policy, per-step refresh
    would be pointless ceremony. It does not: this pins that the failure mode is
    real, so the refresh is load-bearing rather than defensive habit.
    """
    w, t2d = _lm_head(), _t2d()
    hidden = torch.randn(2, 5, H)
    stale = build_frozen_teacher_head(w, t2d, global_step=1)

    w_updated = w + 0.05 * torch.randn_like(w)  # one optimizer step later
    fresh_reference = filter_teacher_to_draft_vocab(
        torch.nn.functional.linear(hidden, w_updated), t2d
    )

    assert not torch.allclose(stale(hidden), fresh_reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        build_frozen_teacher_head(w_updated, t2d, global_step=2)(hidden),
        fresh_reference, rtol=0, atol=0,
    )


# ------------------------------------------------- production-scale parity

# The tests above run at H=16, where both paths almost certainly hit the same
# BLAS kernel, so their atol=0 says little about the real model. Output width
# drives kernel and tiling choice, and the two paths differ in exactly that
# (151936 vs 32000 columns). These use the production hidden size so the
# comparison is against genuinely different reduction orders.
BIG_H, BIG_V, BIG_DRAFT_V = 4096, 8192, 2048


def test_parity_at_production_hidden_size_fp32():
    w, t2d = _lm_head(BIG_V, BIG_H, seed=3), _t2d(BIG_V, BIG_DRAFT_V, seed=4)
    hidden = torch.randn(1, 8, BIG_H)

    reference = filter_teacher_to_draft_vocab(torch.nn.functional.linear(hidden, w), t2d)
    got = build_frozen_teacher_head(w, t2d)(hidden)

    assert got.shape == (1, 8, BIG_DRAFT_V)
    # Tolerance, not equality: a differing reduction order is acceptable, a
    # differing *row* is not -- and a misaligned row would be off by far more
    # than this. Bounded relative to the logit magnitude fp32 produces here.
    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-4)


def test_parity_at_production_hidden_size_bf16():
    """bf16 is what the real run uses; 8 mantissa bits make drift most visible."""
    w = _lm_head(BIG_V, BIG_H, seed=5).to(torch.bfloat16)
    t2d = _t2d(BIG_V, BIG_DRAFT_V, seed=6)
    hidden = torch.randn(1, 8, BIG_H, dtype=torch.bfloat16)

    reference = filter_teacher_to_draft_vocab(torch.nn.functional.linear(hidden, w), t2d)
    got = build_frozen_teacher_head(w, t2d, dtype=torch.bfloat16)(hidden)

    torch.testing.assert_close(got, reference, rtol=2e-2, atol=2e-1)


def test_a_single_swapped_row_is_caught_at_production_scale():
    """Confirms the tolerances above still reject a real misalignment.

    Without this, a loose enough tolerance would make the parity tests pass on a
    permuted head -- the exact silent failure this phase exists to rule out.
    """
    w, t2d = _lm_head(BIG_V, BIG_H, seed=7), _t2d(BIG_V, BIG_DRAFT_V, seed=8)
    hidden = torch.randn(1, 8, BIG_H)

    reference = filter_teacher_to_draft_vocab(torch.nn.functional.linear(hidden, w), t2d)
    rows = select_draft_vocab_rows(w, t2d).clone()
    rows[[0, 1]] = rows[[1, 0]]  # swap two adjacent draft-vocab rows

    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            FrozenTeacherHead(rows)(hidden), reference, rtol=1e-5, atol=1e-4
        )

