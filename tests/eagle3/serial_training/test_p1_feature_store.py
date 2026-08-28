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
"""P1 tests: feature store (A3) + gather-then-slice collection (B2).

Run with pytest:  python3 -m pytest tests/eagle3/serial_training/test_p1_feature_store.py -q
"""

import pytest
import torch

from verl.models.eagle3.collect_plan import build_collect_plan
from verl.models.eagle3.feature_store import (
    DraftFeatureRecord,
    DraftFeatureStore,
    collect_draft_features,
)

H, NUM_AUX = 8, 3
AUX_DIM = H * NUM_AUX


def _record(step=1, rows=4):
    return DraftFeatureRecord(
        aux_hidden=torch.zeros(rows, AUX_DIM),
        final_hidden=torch.zeros(rows, H),
        input_ids=torch.zeros(rows, dtype=torch.long),
        position_ids=torch.zeros(rows, dtype=torch.long),
        loss_mask=torch.ones(rows, dtype=torch.bool),
        positions=torch.arange(rows),
        global_step=step,
    )


# ---------------------------------------------------------------- A3: store


def test_put_and_drain_round_trip():
    store = DraftFeatureStore()
    store.begin_step(1)
    assert store.put(_record(1)) is True
    assert store.put(_record(1)) is True
    assert len(store) == 2

    drained = store.drain()
    assert len(drained) == 2
    assert len(store) == 0, "drain must empty the store"


def test_cross_step_record_is_rejected():
    """A stashed hidden is only valid against the lm_head that produced it."""
    store = DraftFeatureStore()
    store.begin_step(2)
    with pytest.raises(ValueError, match="lm_head that produced them"):
        store.put(_record(step=1))


def test_begin_step_drops_stale_records():
    store = DraftFeatureStore()
    store.begin_step(1)
    store.put(_record(1))
    store.begin_step(2)
    assert len(store) == 0
    assert store.step == 2


def test_max_records_caps_and_reports():
    store = DraftFeatureStore(max_records=2)
    store.begin_step(1)
    assert store.put(_record(1)) is True
    assert store.put(_record(1)) is True
    assert store.put(_record(1)) is False, "third put must be refused"
    assert len(store) == 2
    assert store.dropped == 1


def test_nbytes_counts_the_hidden_tensors():
    store = DraftFeatureStore()
    store.begin_step(1)
    store.put(_record(1, rows=4))
    expected = 4 * AUX_DIM * 4 + 4 * H * 4  # float32
    assert store.nbytes() == expected


def test_num_rows_reflects_the_window():
    assert _record(rows=513).num_rows == 513


# ---------------------------------------------------------------- B2: collection

S, B = 40, 3


def _hidden(seq=S, batch=B, dim=AUX_DIM):
    """Sequence-first (S, B, D) with a value encoding (position, sample, channel)."""
    base = torch.arange(seq, dtype=torch.float32).view(seq, 1, 1)
    off = torch.arange(batch, dtype=torch.float32).view(1, batch, 1) * 1000
    ch = torch.arange(dim, dtype=torch.float32).view(1, 1, dim) * 0.001
    return base + off + ch


def _plan_for(prompt_lens, response_lens, rows=6, step=1, **kw):
    return build_collect_plan(
        prompt_lens=prompt_lens, response_lens=response_lens,
        global_step=step, window_train_rows=rows, **kw,
    )


def test_collected_rows_match_the_plan_positions():
    """The stashed rows must be exactly the ones the plan asked for."""
    plan = _plan_for([3] * B, [30] * B, rows=6)
    aux, final = _hidden(), _hidden(dim=H)
    store = DraftFeatureStore()
    store.begin_step(1)

    n = collect_draft_features(
        store=store, aux_hidden=aux, final_hidden=final,
        input_ids=torch.arange(S).repeat(B, 1),
        position_ids=torch.arange(S).repeat(B, 1),
        loss_mask=torch.ones(B, S, dtype=torch.bool),
        plan=plan, global_step=1,
    )
    assert n == B

    for rec, b in zip(store.drain(), range(B)):
        pos = plan.hidden_positions[b]
        torch.testing.assert_close(rec.positions, pos)
        # value encoding lets us verify identity, not just shape
        torch.testing.assert_close(rec.aux_hidden, aux.transpose(0, 1)[b].index_select(0, pos))
        torch.testing.assert_close(rec.final_hidden, final.transpose(0, 1)[b].index_select(0, pos))
        torch.testing.assert_close(rec.input_ids, pos)


def test_unselected_samples_are_skipped():
    # sample 1's response is too short to fill a 7-row window
    plan = _plan_for([3, 3, 3], [30, 5, 30], rows=6)
    assert plan.collect_mask.tolist() == [True, False, True]

    store = DraftFeatureStore()
    store.begin_step(1)
    n = collect_draft_features(
        store=store, aux_hidden=_hidden(), final_hidden=_hidden(dim=H),
        input_ids=torch.arange(S).repeat(B, 1), position_ids=None, loss_mask=None,
        plan=plan, global_step=1,
    )
    assert n == 2


def test_records_land_on_cpu():
    plan = _plan_for([3] * B, [30] * B, rows=6)
    store = DraftFeatureStore()
    store.begin_step(1)
    collect_draft_features(
        store=store, aux_hidden=_hidden(), final_hidden=_hidden(dim=H),
        input_ids=torch.arange(S).repeat(B, 1), position_ids=None, loss_mask=None,
        plan=plan, global_step=1,
    )
    for rec in store.drain():
        assert rec.aux_hidden.device.type == "cpu"
        assert rec.final_hidden.device.type == "cpu"


def test_stashed_tensors_are_detached():
    """A stashed tensor must never keep the policy graph alive."""
    plan = _plan_for([3], [30], rows=6)
    aux = _hidden(batch=1).requires_grad_(True)
    live = aux * 2  # has grad_fn
    store = DraftFeatureStore()
    store.begin_step(1)
    collect_draft_features(
        store=store, aux_hidden=live, final_hidden=_hidden(batch=1, dim=H),
        input_ids=torch.arange(S).repeat(1, 1), position_ids=None, loss_mask=None,
        plan=plan, global_step=1,
    )
    rec = store.drain()[0]
    assert rec.aux_hidden.grad_fn is None
    assert not rec.aux_hidden.requires_grad


def test_none_plan_collects_nothing():
    store = DraftFeatureStore()
    store.begin_step(1)
    assert collect_draft_features(
        store=store, aux_hidden=_hidden(), final_hidden=_hidden(dim=H),
        input_ids=torch.arange(S).repeat(B, 1), position_ids=None, loss_mask=None,
        plan=None, global_step=1,
    ) == 0


def test_missing_optional_inputs_get_defaults():
    plan = _plan_for([3], [30], rows=6)
    store = DraftFeatureStore()
    store.begin_step(1)
    collect_draft_features(
        store=store, aux_hidden=_hidden(batch=1), final_hidden=_hidden(batch=1, dim=H),
        input_ids=torch.arange(S).repeat(1, 1), position_ids=None, loss_mask=None,
        plan=plan, global_step=1,
    )
    rec = store.drain()[0]
    assert torch.equal(rec.position_ids, torch.zeros(7, dtype=torch.long))
    assert rec.loss_mask.all()


def test_1d_position_and_mask_are_broadcast():
    """(S,) inputs are shared across the batch; (B, S) are per-sample."""
    plan = _plan_for([3] * B, [30] * B, rows=6)
    store = DraftFeatureStore()
    store.begin_step(1)
    collect_draft_features(
        store=store, aux_hidden=_hidden(), final_hidden=_hidden(dim=H),
        input_ids=torch.arange(S).repeat(B, 1),
        position_ids=torch.arange(S),                 # 1-D
        loss_mask=torch.ones(S, dtype=torch.bool),    # 1-D
        plan=plan, global_step=1,
    )
    for rec, b in zip(store.drain(), range(B)):
        torch.testing.assert_close(rec.position_ids, plan.hidden_positions[b])


def test_out_of_range_position_names_the_gather():
    """If the SP gather is skipped, positions overrun -- the error must say so."""
    # window spans positions 2..22, but only a 10-row rank-local shard arrives
    plan = _plan_for([3], [30], rows=20)
    assert int(plan.hidden_positions[0].max()) == 22
    short = _hidden(seq=10, batch=1)
    store = DraftFeatureStore()
    store.begin_step(1)
    with pytest.raises(IndexError, match="sequence-parallel gather did not run"):
        collect_draft_features(
            store=store, aux_hidden=short, final_hidden=_hidden(seq=10, batch=1, dim=H),
            input_ids=torch.arange(10).repeat(1, 1), position_ids=None, loss_mask=None,
            plan=plan, global_step=1,
        )


def test_gather_is_skipped_when_tp_is_one():
    """tp_world_size=1 must not import or call the Megatron collective."""
    plan = _plan_for([3], [30], rows=6)
    store = DraftFeatureStore()
    store.begin_step(1)
    n = collect_draft_features(
        store=store, aux_hidden=_hidden(batch=1), final_hidden=_hidden(batch=1, dim=H),
        input_ids=torch.arange(S).repeat(1, 1), position_ids=None, loss_mask=None,
        plan=plan, global_step=1,
        sequence_parallel=True, tp_world_size=1,   # SP flag on, but TP=1
    )
    assert n == 1


def test_step_mismatch_between_plan_and_store_is_caught():
    plan = _plan_for([3], [30], rows=6)
    store = DraftFeatureStore()
    store.begin_step(1)
    with pytest.raises(ValueError, match="lm_head"):
        collect_draft_features(
            store=store, aux_hidden=_hidden(batch=1), final_hidden=_hidden(batch=1, dim=H),
            input_ids=torch.arange(S).repeat(1, 1), position_ids=None, loss_mask=None,
            plan=plan, global_step=99,   # store is open on step 1
        )


def test_store_size_matches_the_design_budget():
    """16 samples x 513 rows should land near the ~269 MB the design predicts."""
    rows, samples = 513, 16
    h, aux_dim = 4096, 4096 * 3
    store = DraftFeatureStore()
    store.begin_step(1)
    for _ in range(samples):
        store.put(
            DraftFeatureRecord(
                aux_hidden=torch.zeros(rows, aux_dim, dtype=torch.bfloat16),
                final_hidden=torch.zeros(rows, h, dtype=torch.bfloat16),
                input_ids=torch.zeros(rows, dtype=torch.long),
                position_ids=torch.zeros(rows, dtype=torch.long),
                loss_mask=torch.ones(rows, dtype=torch.bool),
                positions=torch.arange(rows),
                global_step=1,
            )
        )
    mb = store.nbytes() / 1024**2
    assert 250 < mb < 290, f"expected ~269 MB per the design doc, got {mb:.1f} MB"
