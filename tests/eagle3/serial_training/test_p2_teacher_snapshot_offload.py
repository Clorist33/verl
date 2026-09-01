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
"""P2 回归：snapshot_draft_teacher 必须在 engine 的 mode 上下文内读 lm_head。

真机现象（20260830，TP=4 + param_offload=True，k=1 的第一次 draft 触发）：
    frozen_teacher.py:187  rows = full[t2d_bool]
    RuntimeError: ... the current working operator name is HcclAllgather

根因：param_offload=True 时 policy 参数只在 train_mode/eval_mode 内被搬到设备
（BaseEngineCtx._context_switch），上下文之外 output_layer.weight 在 CPU；
refresh_frozen_teacher_head 要对它做 TP all_gather，CPU 张量进 HCCL 直接失败。
TP=1 不走 all_gather，所以这个 bug 只在 TP>1 的真机上暴露。

这里不去模拟 HCCL，而是断言那条**因果链的前提**：读权重时参数必须已在设备上。
只断言"调用没抛异常"是抓不到的 —— 原来的代码在单机 CPU 上同样不抛。
"""

import inspect

import pytest


class _FakeEngineCtx:
    """记录进出、并在进入时把参数搬到 device、退出时搬回 cpu。"""

    def __init__(self, engine, mode):
        self.engine = engine
        self.mode = mode

    def __enter__(self):
        self.engine.events.append(f"enter:{self.mode}")
        # 复刻 BaseEngineCtx._context_switch：eval 只搬模型，train 连优化器一起搬
        self.engine.param_device = "npu"
        if self.mode == "train":
            self.engine.optimizer_device = "npu"
        return self

    def __exit__(self, *exc):
        self.engine.events.append(f"exit:{self.mode}")
        self.engine.param_device = "cpu"
        self.engine.optimizer_device = "cpu"
        return False


class _FakeEngine:
    """param_offload=True 的引擎：上下文之外参数在 CPU。"""

    def __init__(self):
        self.events = []
        self.param_device = "cpu"       # 起始即离线，和真机一致
        self.optimizer_device = "cpu"
        self._eagle3 = type("S", (), {"enabled": True})()
        self.module = object()

    def eval_mode(self, **kw):
        return _FakeEngineCtx(self, "eval")

    def train_mode(self, **kw):
        return _FakeEngineCtx(self, "train")


def test_snapshot_reads_lm_head_while_params_are_on_device(monkeypatch):
    """核心断言：refresh 被调用的那一刻，参数必须已经在设备上。"""
    from verl.workers import engine_workers

    engine = _FakeEngine()
    seen = {}

    def _fake_refresh(eng, global_step=None):
        # 这一步等价于真机上的 output_layer.weight + TP all_gather
        seen["param_device"] = eng.param_device
        seen["global_step"] = global_step
        return None

    monkeypatch.setattr(
        "verl.models.eagle3.deferred_training.refresh_frozen_teacher_head",
        _fake_refresh,
    )

    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    worker.actor = type("A", (), {"engine": engine})()
    # 注意不能给 worker.engine 赋值 —— ActorRolloutRefWorker 真实没有这个属性，
    # 凭空造一个会把 self.engine 的 AttributeError 盖住（20260831 真机回归）。

    import torch
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    data = TensorDict({"x": torch.zeros(2, 2)}, batch_size=[2])
    tu.assign_non_tensor_data(data, "global_steps", 3)

    # 绕开 @register 装饰器，直接调用底层函数
    fn = engine_workers.ActorRolloutRefWorker.snapshot_draft_teacher
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    fn(worker, data)

    assert seen["param_device"] == "npu", (
        "读 lm_head 时参数还在 CPU —— 真机上这会让 TP all_gather 以 HcclAllgather 失败"
    )
    assert seen["global_step"] == 3
    assert engine.events == ["enter:eval", "exit:eval"]


def test_worker_has_no_engine_attribute_directly():
    """防回归：ActorRolloutRefWorker 不得有 self.engine 属性。

    该类只有 self.actor.engine。20260831 真机回归：三处误写成 self.engine，
    而测试里凭空给 fake worker 加了个 engine 属性盖住了 AttributeError。
    这个用例确保如果再有代码误写 self.engine，测试会红而不是过。
    """
    from verl.workers import engine_workers

    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    assert not hasattr(worker, "engine"), (
        "ActorRolloutRefWorker 不应有 engine 属性。"
        "engine 通过 self.actor.engine 访问。见 engine_workers.py:940/1000/1142。"
    )


def test_snapshot_draft_teacher_uses_engine_not_self_engine():
    """防回归：snapshot_draft_teacher 不得出现 self.engine，应为局部变量 engine。"""
    from verl.workers import engine_workers
    import re

    src = inspect.getsource(engine_workers.ActorRolloutRefWorker.snapshot_draft_teacher)
    # 只看代码，剥掉注释
    lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    code = "\n".join(lines)

    # 断言用的是局部变量 engine（通过 getattr(self.actor, "engine") 取得）
    assert "engine = getattr(self.actor" in code or "engine = self.actor.engine" in code
    assert "engine.eval_mode(" in code

    # 断言不出现 self.engine（这是 AttributeError 的来源）
    bad_pattern = re.compile(r"\bself\.engine\b")
    matches = bad_pattern.findall(code)
    assert not matches, (
        f"发现 {len(matches)} 处 self.engine（应为局部变量 engine 或 self.actor.engine）"
    )


def test_uses_eval_mode_not_train_mode():
    """eval 只搬模型参数；train 还会搬 Adam state（8B 上是权重的两倍）并 zero_grad。"""
    from verl.workers import engine_workers

    src = inspect.getsource(engine_workers.ActorRolloutRefWorker.snapshot_draft_teacher)
    # 只看代码，剥掉注释 —— 注释里会正当地提到 train_mode 来解释为什么不用它
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.strip().startswith("#")
    )
    # 修复后是 engine.eval_mode()，而 engine 通过 self.actor.engine 取得
    assert "with engine.eval_mode()" in code
    assert "train_mode" not in code


def test_refresh_is_not_called_bare():
    """防回归：refresh 不得脱离 mode 上下文直接调用。"""
    from verl.workers import engine_workers

    src = inspect.getsource(engine_workers.ActorRolloutRefWorker.snapshot_draft_teacher)
    body = [ln.strip() for ln in src.splitlines()]
    for i, line in enumerate(body):
        if line.startswith("refresh_frozen_teacher_head("):
            prev = " ".join(body[max(0, i - 3): i])
            assert "eval_mode" in prev or "train_mode" in prev, (
                "refresh_frozen_teacher_head 出现在 mode 上下文之外"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
