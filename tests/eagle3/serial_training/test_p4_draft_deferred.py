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
"""P4 回归：update_draft_deferred 必须通过 self.actor.engine 访问 engine。

20260831 真机第二次回归 (k=1 第一次 draft 触发):
    engine_workers.py:955   with self.engine.train_mode(...)
    AttributeError: 'ActorRolloutRefWorker' object has no attribute 'engine'

ActorRolloutRefWorker 从未给 self.engine 赋值，只有 self.actor.engine。
三处（955/965/1017）误写成 self.engine 的 bug 从 v3 P4 起就在 (92d8e375)，
只是所有运行在此之前全部因其他错误失败，从未走到第一次 draft 训练入口。
"""

import inspect


class _FakeEngineCtx:
    """记录进出、并在进入时把参数搬到 device、退出时根据 timer 决定是否回搬。"""

    def __init__(self, engine, disable_auto_offload=False):
        self.engine = engine
        self.disable_auto_offload = disable_auto_offload

    def __enter__(self):
        self.engine.events.append("enter:train")
        self.engine.param_device = "npu"
        return self

    def __exit__(self, *exc):
        self.engine.events.append("exit:train")
        if not self.disable_auto_offload:
            self.engine.param_device = "cpu"
        return False


class _FakeEngine:
    """最简化的 engine mock，能记录 mode 切换和 rank 判断。"""

    def __init__(self, is_src_rank=True):
        self.events = []
        self.param_device = "cpu"
        self._is_src_rank = is_src_rank
        self._eagle3 = type("S", (), {"enabled": True})()
        self.module = object()

    def train_mode(self, disable_auto_offload=False, **kw):
        return _FakeEngineCtx(self, disable_auto_offload=disable_auto_offload)

    def is_mp_src_rank_with_outputs(self):
        return self._is_src_rank


def test_update_draft_deferred_accesses_engine_via_actor(monkeypatch):
    """核心断言：必须用 self.actor.engine，不能用 self.engine (后者不存在)。"""
    from verl.workers import engine_workers

    engine = _FakeEngine(is_src_rank=True)
    seen = {}

    def _fake_train(eng, store, **kw):
        # 记录被调用时的状态
        seen["param_device"] = eng.param_device
        seen["global_step"] = kw.get("global_step")
        seen["engine_id"] = id(eng)
        return {"losses": [0.1, 0.2], "num_windows": 1}

    monkeypatch.setattr(
        "verl.models.eagle3.deferred_training.train_draft_from_store",
        _fake_train,
    )

    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    worker.actor = type("A", (), {"engine": engine})()
    # 注意不能给 worker.engine 赋值 —— ActorRolloutRefWorker 真实没有这个属性，
    # 凭空造一个会把 self.engine 的 AttributeError 盖住（20260831 真机回归）。

    import torch
    from tensordict import TensorDict

    from verl.models.eagle3.feature_store import DraftFeatureStore
    from verl.utils import tensordict_utils as tu

    store = DraftFeatureStore(max_records=8)
    # update_draft_deferred 从 engine._eagle3.feature_store 取 store，不是 worker._eagle3_state
    engine._eagle3 = type("S", (), {"enabled": True, "feature_store": store})()

    data = TensorDict({"x": torch.zeros(2, 2)}, batch_size=[2])
    tu.assign_non_tensor_data(data, "global_steps", 5)

    # 配置 eagle3，让 _draft_config_at_step 返回有效配置
    worker.actor_config = type(
        "C",
        (),
        {
            "eagle3_k": 1,
            "eagle3_draft_bsz": 4,
            "eagle3_draft_micro_bsz": 2,
            "eagle3_draft_steps_per_trigger": 10,
        },
    )()

    # 绕开 @register 装饰器，直接调用底层函数
    fn = engine_workers.ActorRolloutRefWorker.update_draft_deferred
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    result = fn(worker, data)

    # 断言确实调用了 train_draft_from_store，且传入的是通过 self.actor.engine 取到的 engine
    assert seen["engine_id"] == id(engine), "传入的 engine 对象不是预期的实例"
    assert seen["param_device"] == "npu", (
        "train_draft_from_store 被调用时参数应已在设备上 (train_mode 已进入)"
    )
    assert seen["global_step"] == 5
    assert result is not None
    from verl.utils import tensordict_utils as tu
    metrics = tu.get_non_tensor_data(result, "metrics", {})
    assert "draft_loss" in metrics


def test_update_draft_deferred_returns_none_on_non_src_rank(monkeypatch):
    """非 src rank 应返回 None，且不会走到 metrics 构造（会因 losses 不存在而失败）。"""
    from verl.workers import engine_workers

    engine = _FakeEngine(is_src_rank=False)  # ← 关键：非 src rank
    called = []

    def _fake_train(eng, store, **kw):
        called.append(True)
        return {"losses": [0.1], "num_windows": 1}

    monkeypatch.setattr(
        "verl.models.eagle3.deferred_training.train_draft_from_store",
        _fake_train,
    )

    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    worker.actor = type("A", (), {"engine": engine})()

    import torch
    from tensordict import TensorDict

    from verl.models.eagle3.feature_store import DraftFeatureStore
    from verl.utils import tensordict_utils as tu

    store = DraftFeatureStore(max_records=8)
    # update_draft_deferred 从 engine._eagle3.feature_store 取 store，不是 worker._eagle3_state
    engine._eagle3 = type("S", (), {"enabled": True, "feature_store": store})()

    data = TensorDict({"x": torch.zeros(2, 2)}, batch_size=[2])
    tu.assign_non_tensor_data(data, "global_steps", 5)

    worker.actor_config = type(
        "C",
        (),
        {
            "eagle3_k": 1,
            "eagle3_draft_bsz": 4,
            "eagle3_draft_micro_bsz": 2,
            "eagle3_draft_steps_per_trigger": 10,
        },
    )()

    fn = engine_workers.ActorRolloutRefWorker.update_draft_deferred
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    result = fn(worker, data)

    assert called, "train_draft_from_store 应该被调用了"
    assert result is None, "非 src rank 应返回 None，而不是尝试构造 metrics"


def test_code_uses_engine_not_self_engine():
    """防回归：update_draft_deferred 内不得出现 self.engine（应为局部变量 engine）。"""
    from verl.workers import engine_workers

    src = inspect.getsource(engine_workers.ActorRolloutRefWorker.update_draft_deferred)
    # 只看代码，剥掉注释和文档字符串
    lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    code = "\n".join(lines)

    # 断言用的是局部变量 engine（通过 self.actor.engine 取得）
    assert "engine = self.actor.engine" in code or "engine = getattr(self.actor" in code
    assert "engine.train_mode(" in code
    assert "engine.is_mp_src_rank_with_outputs()" in code

    # 断言不出现 self.engine（这是 AttributeError 的来源）
    import re

    # 排除 self.actor.engine，只抓 self.engine（后面不是 .actor）
    bad_pattern = re.compile(r"\bself\.engine\b(?!\s*=\s*self\.actor\.engine)")
    matches = bad_pattern.findall(code)
    assert not matches, (
        f"发现 {len(matches)} 处 self.engine（应为局部变量 engine 或 self.actor.engine）"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
