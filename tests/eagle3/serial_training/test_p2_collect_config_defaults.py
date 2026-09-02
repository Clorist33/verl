# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""采集面板的 fallback 必须跟随 collect_plan 的 DEFAULT_*，而不是复写字面量。

_eagle3_collect_config 原来把 512 / "front" / 1.0 / 16 / 16384 直接写在
调用里，于是 collect_plan.DEFAULT_* 成了死常量：改那边对任何"把 ActorConfig
字段留空"的调用者都不生效，而且没有任何提示。
"""

import pytest


def _config_with(**overrides):
    """构造一个只带指定字段的 actor_config，其余字段缺失（等价 None）。"""
    from verl.workers.engine_workers import ActorRolloutRefWorker

    worker = object.__new__(ActorRolloutRefWorker)
    worker.actor_config = type("C", (), dict(overrides))()
    return ActorRolloutRefWorker._eagle3_collect_config(worker, 7)


def test_empty_config_falls_back_to_collect_plan_defaults():
    from verl.models.eagle3 import collect_plan as cp

    cfg = _config_with()
    assert cfg["window_train_rows"] == cp.DEFAULT_WINDOW_TRAIN_ROWS
    assert cfg["window_mode"] == cp.DEFAULT_WINDOW_MODE
    assert cfg["sample_rate"] == cp.DEFAULT_SAMPLE_RATE
    assert cfg["max_samples_per_replica"] == cp.DEFAULT_MAX_SAMPLES_PER_REPLICA
    assert cfg["max_tokens_per_replica"] == cp.DEFAULT_MAX_TOKENS_PER_REPLICA


@pytest.mark.parametrize(
    "const_name,cfg_key",
    [
        ("DEFAULT_WINDOW_TRAIN_ROWS", "window_train_rows"),
        ("DEFAULT_MAX_SAMPLES_PER_REPLICA", "max_samples_per_replica"),
        ("DEFAULT_MAX_TOKENS_PER_REPLICA", "max_tokens_per_replica"),
    ],
)
def test_fallback_tracks_the_constant(monkeypatch, const_name, cfg_key):
    """改 collect_plan 的常量，fallback 必须跟着变 —— 这正是硬编码字面量做不到的。"""
    from verl.models.eagle3 import collect_plan as cp

    sentinel = 4242
    monkeypatch.setattr(cp, const_name, sentinel)
    assert _config_with()[cfg_key] == sentinel, (
        f"{cfg_key} 没有跟随 collect_plan.{const_name}，说明 fallback 又被写成了字面量"
    )


def test_explicit_values_win_over_defaults():
    """字段有值时用字段值，fallback 不该插手。"""
    cfg = _config_with(
        draft_collect_window_train_rows=256,
        draft_collect_window_mode="random",
        draft_collect_sample_rate=0.5,
        draft_collect_max_samples_per_replica=64,
        draft_collect_max_tokens_per_replica=65536,
    )
    assert cfg["window_train_rows"] == 256
    assert cfg["window_mode"] == "random"
    assert cfg["sample_rate"] == 0.5
    assert cfg["max_samples_per_replica"] == 64
    assert cfg["max_tokens_per_replica"] == 65536


def test_explicit_none_still_falls_back():
    """字段显式为 None（hydra 传 null）时也要回退，而不是把 None 灌进 plan。"""
    from verl.models.eagle3 import collect_plan as cp

    cfg = _config_with(
        draft_collect_max_samples_per_replica=None,
        draft_collect_window_mode=None,
    )
    assert cfg["max_samples_per_replica"] == cp.DEFAULT_MAX_SAMPLES_PER_REPLICA
    assert cfg["window_mode"] == cp.DEFAULT_WINDOW_MODE


def test_no_literal_defaults_left_in_the_resolver():
    """静态守卫：解析函数里不得再出现那几个魔法数。"""
    import inspect

    from verl.workers.engine_workers import ActorRolloutRefWorker

    src = inspect.getsource(ActorRolloutRefWorker._eagle3_collect_config)
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    for literal in ("512", "16384", '"front"', "1.0"):
        assert literal not in code, (
            f"解析函数里出现了字面量 {literal}；应改用 collect_plan 的 DEFAULT_* 常量"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
