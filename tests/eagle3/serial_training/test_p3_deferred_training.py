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
"""P3 tests: record batching (A5) + deferred training orchestration.

The orchestration is exercised against fakes rather than a real Megatron engine:
what is worth pinning here is the contract -- teacher rebuilt from the snapshot
and not the policy, losses stashed where eagle3_backward_step will find them,
store drained exactly once -- and none of that needs a GPU to be wrong.

Run with pytest:  python3 -m pytest tests/eagle3/serial_training/test_p3_deferred_training.py -q
"""

import sys
import types

import pytest
import torch

from verl.models.eagle3.deferred_training import _chunks, stack_records
from verl.models.eagle3.feature_store import DraftFeatureRecord, DraftFeatureStore
from verl.models.eagle3.frozen_teacher import FrozenTeacherHead

ROWS, H, NUM_AUX, DRAFT_V = 9, 8, 3, 6
AUX_DIM = H * NUM_AUX


def _record(step=1, rows=ROWS, fill=0.0):
    return DraftFeatureRecord(
        aux_hidden=torch.full((rows, AUX_DIM), fill),
        final_hidden=torch.full((rows, H), fill),
        input_ids=torch.arange(rows),
        position_ids=torch.arange(rows),
        loss_mask=torch.ones(rows, dtype=torch.bool),
        positions=torch.arange(rows),
        global_step=step,
    )


# ---------------------------------------------------------------- stack_records


def test_stack_batches_on_a_new_leading_dim():
    out = stack_records([_record(fill=1.0), _record(fill=2.0)])
    assert out["aux_hidden"].shape == (2, ROWS, AUX_DIM)
    assert out["final_hidden"].shape == (2, ROWS, H)
    assert out["input_ids"].shape == (2, ROWS)
    assert out["loss_mask"].shape == (2, ROWS)


def test_stack_preserves_record_order_and_values():
    out = stack_records([_record(fill=1.0), _record(fill=2.0)])
    assert out["aux_hidden"][0].unique().tolist() == [1.0]
    assert out["aux_hidden"][1].unique().tolist() == [2.0]


def test_stack_casts_only_the_float_tensors():
    out = stack_records([_record()], dtype=torch.bfloat16)
    assert out["aux_hidden"].dtype == torch.bfloat16
    assert out["final_hidden"].dtype == torch.bfloat16
    assert out["input_ids"].dtype == torch.int64, "index tensors must not be cast"
    assert out["loss_mask"].dtype == torch.bool


def test_stack_rejects_mixed_window_lengths():
    """Two window sizes means records from two plans got mixed."""
    with pytest.raises(ValueError, match="disagree on window length"):
        stack_records([_record(rows=9), _record(rows=7)])


def test_stack_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="no records"):
        stack_records([])


def test_chunking_covers_every_record_exactly_once():
    records = list(range(10))
    for size, expected in [(None, [10]), (0, [10]), (4, [4, 4, 2]), (10, [10]), (99, [10])]:
        chunks = list(_chunks(records, size))
        assert [len(c) for c in chunks] == expected
        assert [x for c in chunks for x in c] == records


# ---------------------------------------------------------------- orchestration


class _FakeDraft(torch.nn.Module):
    """Minimal stand-in for the EAGLE3 draft: real params so a backward works."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(AUX_DIM, DRAFT_V)
        self.register_buffer("t2d", torch.ones(DRAFT_V, dtype=torch.bool))
        self.seen = []

    def forward(self, *, input_ids, hidden_states, loss_mask, attention_mask, position_ids, ttt_length):
        self.seen.append(hidden_states.shape)
        return {"logits": [self.proj(hidden_states)], "position_masks": None}


class _FakeGPT:
    def __init__(self, draft):
        self._eagle3_draft = [draft]
        self.share_embeddings_and_output_weights = False
        # _get_patching_model (eagle3_patch.py:100) accepts anything exposing
        # _postprocess. A real patched model always does -- that swap is what
        # makes it a draft-carrying model in the first place -- so a fake without
        # it would resolve to None and quietly diverge from production.
        self._postprocess = lambda *a, **kw: None


class _FakeState:
    def __init__(self, draft):
        self.enabled = True
        self.draft_module = draft
        self.draft_optimizer = torch.optim.SGD(draft.parameters(), lr=0.0)
        self.frozen_lm_head = None
        self.optim_offload = False
        self.draft_lr_scheduler = None  # #1 LR scheduler optional
        self.last_trained_global_step = -1  # #2 weight-sync guard


class _FakeEngine:
    def __init__(self, draft):
        self.module = _FakeGPT(draft)
        self._eagle3 = _FakeState(draft)
        self.model_config = types.SimpleNamespace(
            eagle3=types.SimpleNamespace(draft_optim_clip_grad=0.0)
        )


@pytest.fixture
def patched(monkeypatch):
    """Point _unwrap_gpt at the fake and give compute_draft_loss a real graph."""
    import verl.models.eagle3.engine_support as es

    monkeypatch.setattr(es, "_unwrap_gpt", lambda module: module, raising=True)

    loss_mod = types.ModuleType("verl.models.eagle3.loss_mcore")

    def compute_draft_loss(*, student_logits_per_step, teacher_logits, **kw):
        # keeps a grad path to the draft so eagle3_backward_step can backward
        return {"loss": (student_logits_per_step[0] - teacher_logits).pow(2).mean()}

    loss_mod.compute_draft_loss = compute_draft_loss
    monkeypatch.setitem(sys.modules, "verl.models.eagle3.loss_mcore", loss_mod)
    yield


def _engine_with(records, head=True):
    draft = _FakeDraft()
    engine = _FakeEngine(draft)
    if head:
        engine._eagle3.frozen_lm_head = FrozenTeacherHead(torch.zeros(DRAFT_V, H), source_step=1)
    store = DraftFeatureStore()
    store.begin_step(1)
    for r in records:
        store.put(r)
    return engine, store, draft


def test_training_runs_and_returns_losses(patched):
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, draft = _engine_with([_record(fill=0.5) for _ in range(4)])
    result = train_draft_from_store(engine, store, global_step=1)

    assert isinstance(result, dict)
    assert len(result["losses"]) == 1 and isinstance(result["losses"][0], float)
    assert result["num_windows"] == 4
    assert len(store) == 0, "store must be drained"
    assert draft.seen == [(4, ROWS, AUX_DIM)], "one chunk by default"


def test_speco_inner_loop_multiple_updates_with_sampling(patched):
    """P1-2: steps_per_trigger independent updates, each on a random sub-batch.

    Mirrors SpeCo speco_worker.py:887-891 + base_trainer.py:2580: pool of 6,
    batch 4, 3 steps -> 3 optimizer steps, each forwarding exactly 4 windows.
    """
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, draft = _engine_with([_record() for _ in range(6)])
    result = train_draft_from_store(
        engine, store, global_step=1, steps_per_trigger=3, batch_size_per_gpu=4
    )
    assert len(result["losses"]) == 3
    assert draft.seen == [(4, ROWS, AUX_DIM)] * 3


def test_inner_loop_uses_whole_pool_when_batch_exceeds_it(patched):
    """batch >= pool degrades to full-pool steps (SpeCo _sample_training_items:2540)."""
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, draft = _engine_with([_record() for _ in range(3)])
    result = train_draft_from_store(
        engine, store, global_step=1, steps_per_trigger=2, batch_size_per_gpu=8
    )
    assert len(result["losses"]) == 2
    assert draft.seen == [(3, ROWS, AUX_DIM)] * 2


def test_micro_batch_size_splits_the_forward(patched):
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, draft = _engine_with([_record() for _ in range(5)])
    train_draft_from_store(engine, store, micro_batch_size=2, global_step=1)
    assert draft.seen == [(2, ROWS, AUX_DIM), (2, ROWS, AUX_DIM), (1, ROWS, AUX_DIM)]


def test_every_chunk_contributes_one_stashed_loss(patched):
    """eagle3_backward_step averages the stash, so chunks must all land in it."""
    from verl.models.eagle3.deferred_training import train_draft_from_store
    import verl.models.eagle3.engine_support as es

    engine, store, _ = _engine_with([_record() for _ in range(6)])
    seen = {}

    def spy(eng):
        seen["n"] = len(eng.module._eagle3_draft_losses)
        return 1.0

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(es, "eagle3_backward_step", spy, raising=True)
        train_draft_from_store(engine, store, micro_batch_size=2, global_step=1)
    assert seen["n"] == 3


def test_empty_store_is_a_no_op_not_a_crash(patched):
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, draft = _engine_with([])
    assert train_draft_from_store(engine, store, global_step=1) is None
    assert draft.seen == []


def test_missing_snapshot_raises_instead_of_falling_back(patched):
    """Silently rebuilding the teacher via the policy would undo the whole design."""
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, _ = _engine_with([_record()], head=False)
    with pytest.raises(RuntimeError, match="no\\s+frozen lm_head snapshot"):
        train_draft_from_store(engine, store, global_step=1)


def test_stale_snapshot_warns_but_proceeds(patched, caplog):
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, _ = _engine_with([_record()])
    engine._eagle3.frozen_lm_head.source_step = 99  # snapshot from another step
    with caplog.at_level("WARNING"):
        assert train_draft_from_store(engine, store, global_step=1) is not None
    assert any("only valid against the lm_head" in r.message for r in caplog.records)


def test_disabled_state_short_circuits(patched):
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, draft = _engine_with([_record()])
    engine._eagle3.enabled = False
    assert train_draft_from_store(engine, store, global_step=1) is None
    assert len(store) == 1, "a disabled run must not consume the stash"
    assert draft.seen == []


def test_teacher_comes_from_the_snapshot_not_the_policy(patched):
    """The defining property of this path: no policy forward is involved."""
    from verl.models.eagle3.deferred_training import train_draft_from_store

    engine, store, _ = _engine_with([_record(fill=1.0)])
    calls = []
    real = engine._eagle3.frozen_lm_head

    class _Spy(FrozenTeacherHead):
        def __call__(self, hidden):
            calls.append(hidden.shape)
            return super().__call__(hidden)

    engine._eagle3.frozen_lm_head = _Spy(real.weight, source_step=1)
    train_draft_from_store(engine, store, global_step=1)

    assert calls == [(1, ROWS, H)], "teacher must be built from stashed final_hidden"
    assert not hasattr(engine.module, "output_layer"), "fake GPT has no head to fall back to"
