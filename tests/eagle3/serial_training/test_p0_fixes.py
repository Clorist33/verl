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
"""P0 regression tests: v3 config validation + DP readiness sync.

P0-1: _validate_eagle3_serial_training_config must follow v3 semantics.
The v1/v2 version rejected any actor_training_steps whose total was not a
multiple of (k+1) -- but v3's cycle is k (SerialTrainingScheduler
.should_train_draft: global_steps % k == 0), so perfectly valid configs like
k=5, actor=100 were refused at startup (100 % 6 == 4 -> ValueError).

P0-2: train_draft_from_store must decide train-or-skip identically on every
DP rank. _dp_all_ranks_ready is the sync point; outside a distributed run it
must degrade to the local answer, never crash.

Run with pytest:
  python3 -m pytest tests/eagle3/serial_training/test_p0_fixes.py -q
"""

import logging

import pytest
from omegaconf import OmegaConf

from verl.utils.config import _validate_eagle3_serial_training_config


def _make_config(enable=True, actor_training_steps=100, k=5):
    trainer = {}
    if actor_training_steps is not None:
        trainer["actor_training_steps"] = actor_training_steps
    return OmegaConf.create(
        {
            "trainer": trainer,
            "algorithm": {
                "eagle3": {
                    "enable_serial_training": enable,
                    "actor_steps_per_draft_step": k,
                }
            },
        }
    )


# ------------------------------------------------- P0-1: config validation


def test_disabled_serial_training_skips_validation():
    # Even a broken config must pass when serial training is off.
    _validate_eagle3_serial_training_config(_make_config(enable=False, actor_training_steps=None))


def test_missing_actor_training_steps_raises():
    with pytest.raises(ValueError, match="actor_training_steps"):
        _validate_eagle3_serial_training_config(_make_config(actor_training_steps=None))


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(k):
    with pytest.raises(ValueError, match="actor_steps_per_draft_step"):
        _validate_eagle3_serial_training_config(_make_config(k=k))


@pytest.mark.parametrize(
    "actor_steps,k",
    [
        (100, 5),  # the exact config the v1/v2 (k+1) check refused: 100 % 6 == 4
        (200, 5),
        (30, 3),
        (10, 1),
        (7, 7),
    ],
)
def test_v3_multiples_of_k_pass(actor_steps, k):
    _validate_eagle3_serial_training_config(_make_config(actor_training_steps=actor_steps, k=k))


def test_non_divisible_warns_but_does_not_raise(caplog):
    # v3: a trailing partial cycle just trains the draft one time fewer.
    with caplog.at_level(logging.WARNING, logger="verl.utils.config"):
        _validate_eagle3_serial_training_config(_make_config(actor_training_steps=103, k=5))
    assert any("不是 k" in rec.getMessage() for rec in caplog.records)


def test_validation_period_matches_scheduler():
    # The property the v1/v2 check violated: every actor_training_steps that is
    # a multiple of the SCHEDULER's cycle must pass validation.
    from verl.trainer.ppo.v1.trainer_base import SerialTrainingScheduler

    k = 5
    scheduler = SerialTrainingScheduler(k)
    draft_steps_in_cycle = [g for g in range(1, k + 1) if scheduler.should_train_draft(g)]
    assert draft_steps_in_cycle == [k]  # cycle length is k, not k+1
    _validate_eagle3_serial_training_config(_make_config(actor_training_steps=4 * k, k=k))


# ------------------------------------------------- P0-2: DP readiness sync


def test_readiness_degrades_to_local_answer_without_distributed():
    # No torch.distributed initialized in unit tests: the helper must return the
    # local flag unchanged instead of touching any process group.
    from verl.models.eagle3.deferred_training import _dp_all_ranks_ready

    assert _dp_all_ranks_ready(True) is True
    assert _dp_all_ranks_ready(False) is False


def test_empty_store_skips_without_backward():
    # An empty store must return None (skip) rather than raise -- and must do so
    # AFTER the readiness sync, so a rank with data can also skip when a peer is
    # empty. Here (single process) empty simply means skip.
    from verl.models.eagle3.deferred_training import train_draft_from_store
    from verl.models.eagle3.feature_store import DraftFeatureStore

    class _State:
        enabled = True
        frozen_lm_head = None

    class _Engine:
        _eagle3 = _State()

    store = DraftFeatureStore()
    store.begin_step(1)
    assert train_draft_from_store(_Engine(), store, global_step=1) is None
