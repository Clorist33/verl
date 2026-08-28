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
"""P0 tests: collect plan (A1) + capture row slicing (A2).

Run with pytest:  python3 -m pytest tests/eagle3/serial_training/test_p0_collect_plan.py -q
"""

import pytest
import torch

from verl.models.eagle3.collect_plan import CollectPlan, build_collect_plan
from verl.models.eagle3.hidden_capture_mcore import Eagle3HiddenCapture

# ---------------------------------------------------------------- A1: plan


def test_window_starts_at_last_prompt_position():
    """start = prompt_len - 1, so the first harvested hidden predicts response[0]."""
    plan = build_collect_plan(
        prompt_lens=[100], response_lens=[800], global_step=1, window_train_rows=512
    )
    assert plan is not None
    assert plan.hidden_rows == 513  # 512 trained rows + 1 target row
    pos = plan.hidden_positions[0]
    assert pos[0].item() == 99, "must start one before prompt_len to predict response[0]"
    assert pos[-1].item() == 99 + 512
    assert torch.equal(pos, torch.arange(99, 99 + 513))


def test_prompt_rows_are_never_harvested():
    plan = build_collect_plan(
        prompt_lens=[100], response_lens=[800], global_step=1, window_train_rows=512
    )
    # only position 99 (the last prompt token) is in range; 0..98 must be absent
    assert plan.hidden_positions[0].min().item() == 99


def test_short_responses_are_dropped_whole():
    """Filter 1: response_len < hidden_rows drops the sample -- no padding, no truncation."""
    plan = build_collect_plan(
        prompt_lens=[10, 10, 10],
        response_lens=[512, 513, 514],  # 512 < 513 -> dropped
        global_step=1,
        window_train_rows=512,
    )
    assert plan.collect_mask.tolist() == [False, True, True]
    assert plan.candidate_count == 2
    assert plan.selected_count == 2


def test_returns_none_when_nothing_qualifies():
    plan = build_collect_plan(
        prompt_lens=[10, 10], response_lens=[5, 5], global_step=1, window_train_rows=512
    )
    assert plan is None, "callers rely on None meaning 'run the plain forward'"


def test_sample_quota_caps_selection():
    """32 candidates, quota 16, one owner -> exactly 16 selected."""
    n = 32
    plan = build_collect_plan(
        prompt_lens=[10] * n,
        response_lens=[600] * n,
        global_step=1,
        window_train_rows=512,
        max_samples_per_replica=16,
        max_tokens_per_replica=None,
        owner_count=1,
    )
    assert plan.candidate_count == 32
    assert plan.selected_count == 16
    assert plan.collect_mask.sum().item() == 16


def test_token_quota_can_bind_before_sample_quota():
    """16384 rows / 513 = 31.9 -> 31 samples, even though the sample quota allows 40."""
    n = 40
    plan = build_collect_plan(
        prompt_lens=[10] * n,
        response_lens=[600] * n,
        global_step=1,
        window_train_rows=512,
        max_samples_per_replica=40,
        max_tokens_per_replica=16384,
        owner_count=1,
    )
    assert plan.selected_count == 16384 // 513
    assert plan.owner_token_counts[0] <= 16384


def test_owners_receive_round_robin():
    n = 10
    plan = build_collect_plan(
        prompt_lens=[10] * n,
        response_lens=[600] * n,
        global_step=1,
        window_train_rows=512,
        max_samples_per_replica=16,
        max_tokens_per_replica=None,
        owner_count=2,
    )
    owners = plan.owner_rank[plan.collect_mask].tolist()
    assert owners == [0, 1] * 5
    assert plan.owner_token_counts == [5 * 513, 5 * 513]


def test_front_mode_is_deterministic_and_ignores_length():
    """front (the SpeCo default) always starts at the response head."""
    a = build_collect_plan(
        prompt_lens=[50], response_lens=[600], global_step=1, window_train_rows=512, window_mode="front"
    )
    b = build_collect_plan(
        prompt_lens=[50], response_lens=[8000], global_step=7, window_train_rows=512, window_mode="front"
    )
    assert a.hidden_positions[0][0].item() == b.hidden_positions[0][0].item() == 49


def test_random_mode_is_hash_stable_but_step_dependent():
    kw = dict(prompt_lens=[50], response_lens=[8000], window_train_rows=512, window_mode="random")
    s1a = build_collect_plan(global_step=1, **kw).hidden_positions[0][0].item()
    s1b = build_collect_plan(global_step=1, **kw).hidden_positions[0][0].item()
    s2 = build_collect_plan(global_step=2, **kw).hidden_positions[0][0].item()
    assert s1a == s1b, "same step must reproduce the same window (hash, not RNG)"
    assert s1a != s2, "different steps should explore different windows"
    assert 49 <= s1a <= 49 + (8000 - 513)


def test_random_window_stays_in_bounds():
    for step in range(20):
        plan = build_collect_plan(
            prompt_lens=[10], response_lens=[513], global_step=step,
            window_train_rows=512, window_mode="random",
        )
        # response_len == hidden_rows -> offset must be pinned to 0
        assert plan.hidden_positions[0][0].item() == 9


def test_sample_rate_zero_disables_collection():
    assert build_collect_plan(
        prompt_lens=[10], response_lens=[600], global_step=1, sample_rate=0.0
    ) is None


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same batch"):
        build_collect_plan(prompt_lens=[1, 2], response_lens=[600], global_step=1)


def test_accepts_tensor_inputs():
    plan = build_collect_plan(
        prompt_lens=torch.tensor([100]), response_lens=torch.tensor([800]), global_step=1
    )
    assert isinstance(plan, CollectPlan)
    assert plan.hidden_positions[0][0].item() == 99


# ---------------------------------------------------------------- A2: capture


class _FakeLayer(torch.nn.Module):
    """Stands in for a Megatron TransformerLayer: returns (hidden, context)."""

    def __init__(self, tag: float):
        super().__init__()
        self.tag = tag

    def forward(self, hs):
        return hs + self.tag, None


class _FakeDecoder(torch.nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer(float(i)) for i in range(n_layers)])


class _FakeGPT(torch.nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.decoder = _FakeDecoder(n_layers)

    def forward(self, hs):
        for layer in self.decoder.layers:
            hs, _ = layer(hs)
        return hs


S, B, H = 20, 2, 8


def _run(capture, model):
    with torch.no_grad():
        model(torch.zeros(S, B, H))
    return capture


def test_full_capture_is_the_default():
    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1, 2, 3]).register()
    try:
        _run(cap, model)
        assert cap.row_index is None
        assert cap.get_captured(seqlen_first=True).shape == (S, B, H * 3)
    finally:
        cap.remove()


def test_row_index_slices_the_sequence_dim():
    model = _FakeGPT()
    rows = torch.tensor([0, 5, 19])
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1, 2, 3]).register()
    cap.set_row_index(rows)
    try:
        _run(cap, model)
        out = cap.get_captured(seqlen_first=True)
        assert out.shape == (3, B, H * 3), "sequence dim must shrink to len(rows)"
    finally:
        cap.remove()


def test_sliced_values_match_the_full_capture():
    """The rows we keep must be bit-identical to the same rows of a full capture."""
    rows = torch.tensor([0, 7, 13, 19])

    model = _FakeGPT()
    full = Eagle3HiddenCapture(model, capture_layer_ids=[1, 2, 3]).register()
    try:
        _run(full, model)
        full_out = full.get_captured(seqlen_first=True)
    finally:
        full.remove()

    model2 = _FakeGPT()
    sliced = Eagle3HiddenCapture(model2, capture_layer_ids=[1, 2, 3]).register()
    sliced.set_row_index(rows)
    try:
        _run(sliced, model2)
        sliced_out = sliced.get_captured(seqlen_first=True)
    finally:
        sliced.remove()

    torch.testing.assert_close(sliced_out, full_out.index_select(0, rows))


def test_batch_and_hidden_dims_survive_slicing():
    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1, 2]).register()
    cap.set_row_index(torch.tensor([2, 3]))
    try:
        _run(cap, model)
        assert cap.get_captured(seqlen_first=False).shape == (B, 2, H * 2)
    finally:
        cap.remove()


def test_out_of_range_index_raises_with_sp_hint():
    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1]).register()
    cap.set_row_index(torch.tensor([0, S]))  # S is one past the end
    try:
        with pytest.raises(IndexError, match="sequence_parallel"):
            _run(cap, model)
    finally:
        cap.remove()


def test_row_index_can_be_reset_to_none():
    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1]).register()
    try:
        cap.set_row_index(torch.tensor([1, 2]))
        _run(cap, model)
        assert cap.get_captured().shape[0] == 2
        cap.clear()

        cap.set_row_index(None)
        _run(cap, model)
        assert cap.get_captured().shape[0] == S
    finally:
        cap.remove()


def test_clear_keeps_the_row_index():
    """clear() drops data, not configuration -- the plan outlives a micro-batch."""
    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1]).register()
    try:
        cap.set_row_index(torch.tensor([1, 2]))
        cap.clear()
        assert cap.row_index is not None
    finally:
        cap.remove()


def test_set_row_index_rejects_2d():
    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1])
    with pytest.raises(ValueError, match="1-D"):
        cap.set_row_index(torch.zeros(2, 3, dtype=torch.long))


def test_set_row_index_accepts_a_plain_list():
    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1]).register()
    cap.set_row_index([1, 4, 9])
    try:
        _run(cap, model)
        assert cap.get_captured().shape[0] == 3
    finally:
        cap.remove()


# ------------------------------------------------------- A1 + A2 together


def test_plan_positions_drive_the_capture():
    """End-to-end P0 contract: plan picks rows, capture returns exactly those."""
    prompt_len, response_len, train_rows = 3, 12, 6
    plan = build_collect_plan(
        prompt_lens=[prompt_len], response_lens=[response_len],
        global_step=1, window_train_rows=train_rows,
    )
    assert plan.hidden_rows == train_rows + 1

    model = _FakeGPT()
    cap = Eagle3HiddenCapture(model, capture_layer_ids=[1, 2, 3]).register()
    cap.set_row_index(plan.hidden_positions[0])
    try:
        _run(cap, model)
        out = cap.get_captured(seqlen_first=True)
        assert out.shape == (train_rows + 1, B, H * 3)
    finally:
        cap.remove()
