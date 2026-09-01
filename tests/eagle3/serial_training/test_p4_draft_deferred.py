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
"""P4：update_draft_deferred 的 engine 访问方式，以及"不进 mode 上下文"约束。

两个真机 bug 的回归：

1. 20260831 —— ``self.engine`` 不存在（该类只有 ``self.actor.engine``）：
       AttributeError: 'ActorRolloutRefWorker' object has no attribute 'engine'
   955/965 两处是 v3 P4 的既有 bug，此前所有运行都死在到达这里之前。

2. 20260901 07:58 —— 退出 ``train_mode`` 时清 policy 梯度：
       param_and_grad_buffer.py:962  self.grad_data.zero_()
       RuntimeError: The tensor has a non-zero number of elements, but its data
                     is not allocated yet.
   ``disable_auto_offload=True`` 让进入时不搬 grad buffer（这是对的：policy 此刻
   应留在 CPU，正是 v3 显存错峰的前提），但 ``zero_grad_on_exit`` 默认 True，
   退出时对已被 ``storage().resize_(0)`` 的 buffer 做 ``zero_()``。并行分支不炸，
   是因为它的 ``train_mode`` 是嵌套的 —— 外层已经把 buffer 搬上了设备。

修法是**根本不进 engine 的 mode 上下文**：draft 训练不需要 policy 的任何上下文，
理由详见 engine_workers.update_draft_deferred 的注释。所以本文件断言的是"两个 mode
上下文都没被进入"，而不是"进入后参数在设备上"。
"""

import inspect


class _FakeEngine:
    """记录 mode 上下文是否被进入 —— 被进入即视为回归，直接抛。"""

    def __init__(self, is_src_rank=True):
        self.mode_calls = []
        self._is_src_rank = is_src_rank
        self._eagle3 = None  # 由用例按需替换
        self.module = object()

    def train_mode(self, **kw):
        self.mode_calls.append(("train_mode", kw))
        raise AssertionError(
            "update_draft_deferred 不应进入 engine.train_mode：退出时会对已 offload 的 "
            "policy grad buffer 做 zero_()，真机以 RuntimeError 崩溃（20260901）。"
        )

    def eval_mode(self, **kw):
        self.mode_calls.append(("eval_mode", kw))
        raise AssertionError("update_draft_deferred 不应进入 engine.eval_mode")

    def is_mp_src_rank_with_outputs(self):
        return self._is_src_rank


def _make_worker(engine):
    """只挂真实类确实拥有的属性。

    刻意不设 ``worker.engine`` —— ActorRolloutRefWorker 没有这个属性，凭空加上会把
    ``self.engine`` 的 AttributeError 盖住（20260831 就是这么漏过去的）。
    """
    from verl.workers import engine_workers

    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    worker.actor = type("A", (), {"engine": engine})()
    worker.actor_config = type(
        "C",
        (),
        {
            "draft_ppo_micro_batch_size_per_gpu": 8,
            "draft_steps_per_trigger": 10,
            "draft_train_batch_size_per_gpu": 4,
        },
    )()
    return worker


def _unwrapped(name):
    """绕开 @register / @DistProfiler 装饰器，拿到底层函数。"""
    from verl.workers import engine_workers

    fn = getattr(engine_workers.ActorRolloutRefWorker, name)
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _batch(global_step):
    import torch
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    data = TensorDict({"x": torch.zeros(2, 2)}, batch_size=[2])
    tu.assign_non_tensor_data(data, "global_steps", global_step)
    return data


def _attach_store(engine):
    """update_draft_deferred 从 engine._eagle3.feature_store 取 store。"""
    from verl.models.eagle3.feature_store import DraftFeatureStore

    store = DraftFeatureStore(max_records=8)
    engine._eagle3 = type("S", (), {"enabled": True, "feature_store": store})()
    return store


def test_trains_without_entering_any_mode_context(monkeypatch):
    """核心：能跑通，且全程没进过 train_mode / eval_mode。"""
    engine = _FakeEngine(is_src_rank=True)
    _attach_store(engine)
    seen = {}

    def _fake_train(eng, store, **kw):
        seen["engine_id"] = id(eng)
        seen["global_step"] = kw.get("global_step")
        seen["steps_per_trigger"] = kw.get("steps_per_trigger")
        seen["batch_size_per_gpu"] = kw.get("batch_size_per_gpu")
        return {"losses": [0.1, 0.2], "num_windows": 3}

    monkeypatch.setattr(
        "verl.models.eagle3.deferred_training.train_draft_from_store", _fake_train
    )

    worker = _make_worker(engine)
    result = _unwrapped("update_draft_deferred")(worker, _batch(5))

    # _FakeEngine 的 train_mode/eval_mode 直接抛，所以能走到这里就说明没进过；
    # 仍显式断言一次，防止将来 fake 改成不抛时静默放过。
    assert engine.mode_calls == [], f"进入了 mode 上下文：{engine.mode_calls}"

    # engine 必须来自 self.actor.engine
    assert seen["engine_id"] == id(engine), "传入的 engine 不是 self.actor.engine"
    assert seen["global_step"] == 5
    # 配置面板要如实透传，否则 P1-2 的内循环调不动
    assert seen["steps_per_trigger"] == 10
    assert seen["batch_size_per_gpu"] == 4

    from verl.utils import tensordict_utils as tu

    metrics = tu.get_non_tensor_data(result, "metrics", {})
    assert metrics["draft_updates"] == [2.0]
    assert metrics["draft_windows"] == [3.0]


def test_global_step_recorded_on_engine_for_weight_sync_guard(monkeypatch):
    """engine._eagle3_last_global_step 要在训练前挂好。

    P2 的"只在训过的步导出 draft 权重"守卫靠它判断；漏了会导致每步都同步，
    或者反过来永远不同步。
    """
    engine = _FakeEngine()
    _attach_store(engine)
    monkeypatch.setattr(
        "verl.models.eagle3.deferred_training.train_draft_from_store",
        lambda eng, store, **kw: {"losses": [0.5], "num_windows": 1},
    )

    worker = _make_worker(engine)
    _unwrapped("update_draft_deferred")(worker, _batch(11))

    assert getattr(engine, "_eagle3_last_global_step", None) == 11


def test_returns_none_on_non_src_rank(monkeypatch):
    """非 src rank 返回 None。

    这些 None 会被 collect_mask 过滤掉（collect_nd_compute:245），所以不违反
    dispatch 的 concat 契约 —— 与 snapshot_draft_teacher 不同，后者所有 rank
    都返回，src rank 的 None 活到断言处，必须给可 concat 的值。
    """
    engine = _FakeEngine(is_src_rank=False)
    _attach_store(engine)
    called = []

    def _fake_train(eng, store, **kw):
        called.append(True)
        return {"losses": [0.1], "num_windows": 1}

    monkeypatch.setattr(
        "verl.models.eagle3.deferred_training.train_draft_from_store", _fake_train
    )

    worker = _make_worker(engine)
    result = _unwrapped("update_draft_deferred")(worker, _batch(5))

    assert called, "train_draft_from_store 应该被调用了"
    assert result is None, "非 src rank 应返回 None，而不是尝试构造 metrics"


def test_returns_none_when_eagle3_disabled():
    """eagle3 未启用时早退，且不碰 mode 上下文。"""
    engine = _FakeEngine()
    engine._eagle3 = None

    worker = _make_worker(engine)
    result = _unwrapped("update_draft_deferred")(worker, _batch(5))

    assert result is None
    assert engine.mode_calls == []


def test_source_avoids_self_engine_and_mode_context():
    """静态守卫：不得出现 self.engine，也不得重新引入 mode 上下文。"""
    import re

    from verl.workers import engine_workers

    src = inspect.getsource(engine_workers.ActorRolloutRefWorker.update_draft_deferred)
    code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))

    assert "engine = self.actor.engine" in code or "engine = getattr(self.actor" in code
    assert "engine.is_mp_src_rank_with_outputs()" in code
    assert not re.findall(r"\bself\.engine\b", code), (
        "self.engine 不存在于 ActorRolloutRefWorker；应使用局部变量 engine"
    )
    assert "train_mode(" not in code, (
        "重新引入了 train_mode：退出时会对已 offload 的 policy grad buffer 做 zero_()，"
        "而 zero_grad_on_exit=False 也挡不住异常路径"
    )
    assert "eval_mode(" not in code


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
