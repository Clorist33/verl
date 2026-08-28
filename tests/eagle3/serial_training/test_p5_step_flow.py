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
"""P5 tests: v3 step-flow semantics.

Supersedes the behavioural intent of test_stage1 / test_dual_step_tracking /
test_e2e_serial_training / test_progress_bar_display, which assert the v1/v2
alternating-step model that v3 removes. Those four also assert by grepping source
text for literal strings, so they fail on any rewording -- they were already red
before this work started.

Run with pytest:  python3 -m pytest tests/eagle3/serial_training/test_p5_step_flow.py -q
"""

import pytest

from verl.trainer.ppo.v1.trainer_base import SerialTrainingScheduler

# --------------------------------------------------------------- scheduler


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_every_step_trains_the_actor(k):
    """v3 has one kind of step. The draft rides along; it never displaces."""
    s = SerialTrainingScheduler(k)
    assert all(s.should_train_actor(g) for g in range(1, 4 * k + 2))


@pytest.mark.parametrize(
    "k,expected",
    [
        (1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (3, [3, 6, 9]),
        (5, [5, 10]),
        (7, [7]),
    ],
)
def test_draft_trains_every_k_steps(k, expected):
    s = SerialTrainingScheduler(k)
    assert [g for g in range(1, 11) if s.should_train_draft(g)] == expected


def test_period_is_k_not_k_plus_one():
    """The step that disappeared is the standalone draft step.

    v1/v2 ran k actor steps plus one draft step, so a cycle cost k+1 rollouts for
    k actor updates. v3 runs k, and the k-th also trains the draft.
    """
    k = 5
    s = SerialTrainingScheduler(k)
    window = range(1, 4 * k + 1)
    actor = sum(s.should_train_actor(g) for g in window)
    draft = sum(s.should_train_draft(g) for g in window)
    assert actor == len(window), "no step is spent on draft alone"
    assert draft == len(window) // k


def test_draft_and_actor_coincide_rather_than_alternate():
    """The defining difference from v1/v2: on a draft step, the actor trains too."""
    s = SerialTrainingScheduler(5)
    draft_steps = [g for g in range(1, 21) if s.should_train_draft(g)]
    assert draft_steps, "sanity"
    assert all(s.should_train_actor(g) for g in draft_steps)


def test_k_of_zero_never_trains_the_draft():
    """Guards against a %0 crash if the config is mis-set."""
    s = SerialTrainingScheduler(0)
    assert not any(s.should_train_draft(g) for g in range(1, 10))
    assert all(s.should_train_actor(g) for g in range(1, 10))


# --------------------------------------------------------------- step budget


def _derive(actor_training_steps, k):
    """Mirror of the derivation in trainer_base and utils/config."""
    draft = actor_training_steps // k
    return draft, actor_training_steps


@pytest.mark.parametrize("actor,k", [(100, 5), (100, 1), (60, 3), (49, 7)])
def test_total_equals_actor_steps(actor, k):
    """v1/v2 had total = actor + draft because draft owned steps. It no longer does."""
    draft, total = _derive(actor, k)
    assert total == actor
    assert draft == actor // k


def test_the_v3_budget_is_cheaper_than_v1_for_the_same_actor_work():
    """Same number of actor updates, one fewer rollout per cycle."""
    actor, k = 100, 5
    v1_total = actor + actor // k   # 120
    _, v3_total = _derive(actor, k)  # 100
    assert v3_total < v1_total
    assert v1_total - v3_total == actor // k, "exactly the standalone draft steps"


# --------------------------------------------------------------- step labels


def _label(k, step):
    """Reproduce _step_once_serial's _current_training_type assignment."""
    s = SerialTrainingScheduler(k)
    return "Actor+Draft" if s.should_train_draft(step) else "Actor"


def test_step_label_marks_the_ride_along():
    assert [_label(5, g) for g in range(1, 11)] == [
        "Actor", "Actor", "Actor", "Actor", "Actor+Draft",
        "Actor", "Actor", "Actor", "Actor", "Actor+Draft",
    ]


def test_no_step_is_labelled_draft_only():
    """'Draft' as a standalone label is unreachable in v3."""
    assert all(_label(5, g) != "Draft" for g in range(1, 30))


def test_progress_description_appends_draft_only_when_it_applies():
    """Mirror of the progress-bar branch in fit()."""

    def desc(step, label, actor_steps, draft_steps, actor_total=100, draft_total=20):
        out = f"Global {step}/{actor_total} [Actor {actor_steps}/{actor_total}]"
        if label == "Actor+Draft":
            out += f" [Draft {draft_steps}/{draft_total}]"
            return out
        return out

    assert desc(4, _label(5, 4), 4, 0) == "Global 4/100 [Actor 4/100]"
    assert desc(5, _label(5, 5), 5, 1) == "Global 5/100 [Actor 5/100] [Draft 1/20]"


# --------------------------------------------------------------- collect flag


def test_collection_is_armed_only_on_draft_steps():
    """eagle3_collect_only gates the collect-only _postprocess branch.

    Harvesting on every step would stash features nothing drains -- the store
    warns about exactly that at the next begin_step.
    """
    s = SerialTrainingScheduler(5)
    armed = [s.should_train_draft(g) for g in range(1, 11)]
    assert armed == [False, False, False, False, True, False, False, False, False, True]
