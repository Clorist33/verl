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
"""P2 回归：snapshot_draft_teacher 的返回值必须满足 dispatch 的 collect 契约。

真机现象（20260901 03:32，k=1 第一次 draft 触发）：
    decorator.py:260  assert BatchData(output).is_concatable()
    AssertionError: expecting concatable output, but got element type <class 'NoneType'>

快照工作本身已经成功（日志里 [DRAFT-TEACHER] frozen lm_head snapshot 已打印），
崩在返回值往回收的那一刻：make_nd_compute_dataproto_dispatch_fn 的 collect 阶段
要求各存活 rank 的返回值可 concat，而该方法三个 return 全是 None。

为什么 update_draft_deferred 用同一个 dispatch 却没炸：它只在**非 src rank**
返回 None，而 collect_mask 只保留 src rank 的输出（collect_nd_compute:245），
那些 None 恰好都被过滤掉了。snapshot_draft_teacher 是所有 rank 都返回 None，
src rank 的 None 活到了断言处。
"""

import inspect

import pytest


def test_empty_dispatch_output_survives_the_real_collect_path():
    """走真实的 collect 断言 + concat，而不是只检查类型。"""
    from verl.protocol import BatchData
    from verl.single_controller.base.decorator import _concat_data_proto_or_future
    from verl.workers.engine_workers import _empty_dispatch_output

    # 模拟 4 个 DP src rank 各返回一次
    outputs = [_empty_dispatch_output() for _ in range(4)]

    assert BatchData(outputs).is_concatable(), (
        "返回值过不了 decorator.py:260 的断言"
    )
    # 断言能真的 concat —— is_concatable 只查类型，不保证 concat 本身不抛
    merged = _concat_data_proto_or_future(outputs)
    assert merged is not None


def test_none_would_fail_the_collect_assert():
    """反向确认：None 确实过不了，即这个测试不是空转的。"""
    from verl.protocol import BatchData

    assert not BatchData([None, None]).is_concatable()


def test_snapshot_has_no_bare_none_return():
    """防回归：三个 return 路径都不得返回 None。

    早退分支（state 未启用）同样会被 collect，所以不能只修最后一个 return。
    """
    from verl.workers import engine_workers

    src = inspect.getsource(engine_workers.ActorRolloutRefWorker.snapshot_draft_teacher)
    code_lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]

    returns = [ln.strip() for ln in code_lines if ln.strip().startswith("return")]
    assert returns, "没找到 return 语句，测试假设已失效"
    for r in returns:
        assert r != "return None", (
            f"发现裸 return None：{r!r}。collect 阶段会以 "
            "'expecting concatable output' 失败。"
        )


def test_all_return_paths_are_reachable_and_concatable(monkeypatch):
    """两条真实路径（早退 / 正常完成）的返回值都要能过 collect。"""
    from verl.protocol import BatchData
    from verl.workers import engine_workers

    import torch
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    fn = engine_workers.ActorRolloutRefWorker.snapshot_draft_teacher
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    data = TensorDict({"x": torch.zeros(2, 2)}, batch_size=[2])
    tu.assign_non_tensor_data(data, "global_steps", 7)

    # --- 路径 1：eagle3 未启用，走早退 ---
    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    engine_off = type("E", (), {"_eagle3": None})()
    worker.actor = type("A", (), {"engine": engine_off})()
    out_early = fn(worker, data)
    assert BatchData([out_early]).is_concatable(), "早退路径的返回值不可 concat"

    # --- 路径 2：正常完成 ---
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    calls = []
    monkeypatch.setattr(
        "verl.models.eagle3.deferred_training.refresh_frozen_teacher_head",
        lambda eng, global_step=None: calls.append(global_step),
    )

    engine_on = type(
        "E",
        (),
        {
            "_eagle3": type("S", (), {"enabled": True})(),
            "eval_mode": lambda self, **kw: _Ctx(),
        },
    )()
    worker2 = object.__new__(engine_workers.ActorRolloutRefWorker)
    worker2.actor = type("A", (), {"engine": engine_on})()
    out_done = fn(worker2, data)

    assert calls == [7], "快照没被调用，或 global_step 传错"
    assert BatchData([out_done]).is_concatable(), "正常路径的返回值不可 concat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
