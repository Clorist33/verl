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
"""P4 tests: length derivation from loss_mask + collect-only routing.

Run with pytest:  python3 -m pytest tests/eagle3/serial_training/test_p4_collect_step.py -q
"""

import pytest
import torch

from verl.models.eagle3.collect_plan import build_collect_plan
from verl.models.mcore.eagle3_patch import lens_from_loss_mask

# --------------------------------------------------- lens_from_loss_mask


def _mask(prompt_len, response_len, total):
    m = torch.zeros(total, dtype=torch.bool)
    m[prompt_len : prompt_len + response_len] = True
    return m


def test_prompt_and_response_lengths_come_out_of_the_mask():
    mask = torch.stack([_mask(3, 5, 20), _mask(7, 9, 20)])
    p, r = lens_from_loss_mask(mask)
    assert p.tolist() == [3, 7]
    assert r.tolist() == [5, 9]


def test_prompt_len_is_the_first_response_position():
    """build_collect_plan does start = prompt_len - 1, so this must be the
    first response index -- not the prompt's token count minus one."""
    mask = torch.stack([_mask(10, 6, 30)])
    p, _ = lens_from_loss_mask(mask)
    assert p.tolist() == [10]

    plan = build_collect_plan(
        prompt_lens=p, response_lens=torch.tensor([6]), global_step=1, window_train_rows=4
    )
    assert plan.hidden_positions[0][0].item() == 9, "window must open on the last prompt token"


def test_all_false_row_yields_zeros_and_is_dropped():
    mask = torch.stack([_mask(3, 5, 20), torch.zeros(20, dtype=torch.bool)])
    p, r = lens_from_loss_mask(mask)
    assert p.tolist() == [3, 0]
    assert r.tolist() == [5, 0]

    plan = build_collect_plan(prompt_lens=p, response_lens=r, global_step=1, window_train_rows=3)
    assert plan.collect_mask.tolist() == [True, False]


def test_1d_mask_is_treated_as_batch_of_one():
    p, r = lens_from_loss_mask(_mask(4, 6, 20))
    assert p.tolist() == [4]
    assert r.tolist() == [6]


def test_lengths_land_on_cpu_for_the_plan_builder():
    p, r = lens_from_loss_mask(torch.stack([_mask(3, 5, 20)]))
    assert p.device.type == "cpu" and r.device.type == "cpu"


def test_response_running_to_the_end_is_measured_correctly():
    mask = torch.stack([_mask(5, 15, 20)])
    p, r = lens_from_loss_mask(mask)
    assert (p.tolist(), r.tolist()) == ([5], [15])


def test_float_mask_is_accepted():
    """loss_mask arrives as float in some paths; truthiness must still work."""
    p, r = lens_from_loss_mask(torch.stack([_mask(3, 5, 20)]).float())
    assert (p.tolist(), r.tolist()) == ([3], [5])


# --------------------------------------------------- collect-only routing


class _Cfg:
    sequence_parallel = False


class _FakeCapture:
    def __init__(self, aux):
        self._aux = aux
        self.cleared = 0

    def get_captured(self, seqlen_first=True):
        return self._aux

    def clear(self):
        self.cleared += 1


S, B, H, NUM_AUX = 24, 2, 8, 3


class _Model:
    """Enough of a GPTModel for the collect-only branch."""

    share_embeddings_and_output_weights = False
    post_process = True

    def __init__(self, store, capture, cfg=None):
        self.config = _Cfg()
        self._eagle3_feature_store = store
        self._eagle3_capture = capture
        self._eagle3_collect_config = cfg or {"global_step": 1, "window_train_rows": 4}
        self.output_calls = 0

    def output_layer(self, hidden_states, weight=None, runtime_gather_output=None):
        self.output_calls += 1
        return hidden_states.new_zeros(hidden_states.shape[0], hidden_states.shape[1], 5), None


def _run(model, loss_mask):
    from verl.models.mcore.eagle3_patch import _eagle3_collect_features_step

    return _eagle3_collect_features_step(
        model,
        hidden_states=torch.randn(S, B, H),
        input_ids=torch.arange(S).repeat(B, 1),
        position_ids=torch.arange(S).repeat(B, 1),
        loss_mask=loss_mask,
        runtime_gather_output=True,
    )


def _store():
    from verl.models.eagle3.feature_store import DraftFeatureStore

    s = DraftFeatureStore()
    s.begin_step(1)
    return s


def test_collect_step_stashes_and_returns_policy_logits():
    store = _store()
    cap = _FakeCapture(torch.randn(S, B, H * NUM_AUX))
    model = _Model(store, cap)

    logits = _run(model, torch.stack([_mask(3, 15, S), _mask(3, 15, S)]))

    assert logits.shape == (B, S, 5), "policy return path must be unchanged"
    assert len(store) == 2, "both samples should have been stashed"
    assert cap.cleared == 1, "capture must be released for the next microbatch"


def test_collect_step_trains_nothing():
    """No optimizer, no backward -- that is the whole point of deferring."""
    store = _store()
    cap = _FakeCapture(torch.randn(S, B, H * NUM_AUX))
    model = _Model(store, cap)
    _run(model, torch.stack([_mask(3, 15, S), _mask(3, 15, S)]))
    assert not hasattr(model, "_eagle3_draft_losses")


def test_short_responses_leave_the_store_empty():
    store = _store()
    cap = _FakeCapture(torch.randn(S, B, H * NUM_AUX))
    model = _Model(store, cap, cfg={"global_step": 1, "window_train_rows": 20})
    _run(model, torch.stack([_mask(3, 5, S), _mask(3, 5, S)]))
    assert len(store) == 0
    assert cap.cleared == 1, "capture must be released even when nothing is collected"


def test_missing_store_is_skipped_not_fatal():
    cap = _FakeCapture(torch.randn(S, B, H * NUM_AUX))
    model = _Model(None, cap)
    logits = _run(model, torch.stack([_mask(3, 15, S), _mask(3, 15, S)]))
    assert logits.shape == (B, S, 5), "old_log_probs must survive a missing store"


def test_collection_failure_does_not_break_the_policy_path():
    """A draft-side error must not take down the forward PPO needs."""
    class _Broken(_FakeCapture):
        def get_captured(self, seqlen_first=True):
            raise RuntimeError("capture exploded")

    store = _store()
    cap = _Broken(None)
    model = _Model(store, cap)
    logits = _run(model, torch.stack([_mask(3, 15, S), _mask(3, 15, S)]))

    assert logits.shape == (B, S, 5)
    assert len(store) == 0
    assert cap.cleared == 1


def test_strict_mode_surfaces_the_failure(monkeypatch):
    import verl.models.mcore.eagle3_patch as ep

    class _Broken(_FakeCapture):
        def get_captured(self, seqlen_first=True):
            raise RuntimeError("capture exploded")

    monkeypatch.setattr(ep, "_eagle3_strict_draft", lambda: True)
    model = _Model(_store(), _Broken(None))
    with pytest.raises(RuntimeError, match="capture exploded"):
        _run(model, torch.stack([_mask(3, 15, S), _mask(3, 15, S)]))


def test_non_final_pipeline_stage_passes_hidden_through():
    from verl.models.mcore.eagle3_patch import _eagle3_collect_features_step

    model = _Model(_store(), _FakeCapture(torch.randn(S, B, H * NUM_AUX)))
    model.post_process = False
    hidden = torch.randn(S, B, H)
    out = _eagle3_collect_features_step(
        model, hidden_states=hidden, input_ids=None, position_ids=None,
        loss_mask=None, runtime_gather_output=True,
    )
    assert out is hidden
    assert model.output_calls == 0


# --------------------------------------------------- step-level quota

from verl.models.eagle3.collect_plan import CollectBudget  # noqa: E402


def _model_with_budget(store, budget, rows=4):
    cap = _FakeCapture(torch.randn(S, B, H * NUM_AUX))
    m = _Model(store, cap, cfg={"global_step": 1, "window_train_rows": rows})
    m._eagle3_collect_budget = budget
    return m, cap


def test_quota_spans_micro_batches():
    """_postprocess runs per micro-batch; the quota must not restart each time.

    At micro_batch_size_per_gpu=1 a per-call quota of 16 can never bind on a
    batch of 1, so every micro-batch collected its sample and a step ended up
    with one window per sequence instead of the budgeted 16.
    """
    store = _store()
    budget = CollectBudget(max_samples=3, max_tokens=None)
    mask = torch.stack([_mask(3, 15, S), _mask(3, 15, S)])  # B=2 per call

    for _ in range(5):  # five micro-batches, two candidates each
        model, _ = _model_with_budget(store, budget)
        _run(model, mask)

    assert len(store) == 3, "quota must cap the whole step, not each micro-batch"
    assert budget.remaining_samples() == 0


def test_token_quota_also_spans_micro_batches():
    store = _store()
    rows = 4
    budget = CollectBudget(max_samples=None, max_tokens=2 * (rows + 1))
    mask = torch.stack([_mask(3, 15, S), _mask(3, 15, S)])

    for _ in range(4):
        model, _ = _model_with_budget(store, budget, rows=rows)
        _run(model, mask)

    assert len(store) == 2
    assert budget.remaining_tokens() == 0


def test_exhausted_budget_short_circuits_before_planning():
    """Once spent, later micro-batches must not even build a plan."""
    store = _store()
    budget = CollectBudget(max_samples=0, max_tokens=0)
    model, cap = _model_with_budget(store, budget)
    logits = _run(model, torch.stack([_mask(3, 15, S), _mask(3, 15, S)]))

    assert logits.shape == (B, S, 5), "policy path still unaffected"
    assert len(store) == 0
    assert cap.cleared == 1, "capture still released"


def test_budget_only_charges_what_was_actually_stored():
    store = _store()
    budget = CollectBudget(max_samples=10, max_tokens=None)
    # sample 1's response is too short -> only one window stored
    model, _ = _model_with_budget(store, budget)
    _run(model, torch.stack([_mask(3, 15, S), _mask(3, 2, S)]))

    assert len(store) == 1
    assert budget.used_samples == 1


def test_absent_budget_falls_back_to_the_config_quota():
    """Keeps the branch usable without a budget (e.g. a one-shot call)."""
    store = _store()
    cap = _FakeCapture(torch.randn(S, B, H * NUM_AUX))
    model = _Model(store, cap, cfg={"global_step": 1, "window_train_rows": 4,
                                    "max_samples_per_replica": 1})
    _run(model, torch.stack([_mask(3, 15, S), _mask(3, 15, S)]))
    assert len(store) == 1
