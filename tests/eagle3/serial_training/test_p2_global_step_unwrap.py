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
"""P2 回归：global_steps 以 NonTensorStack 形式到达 worker 时不能崩。

真机现象（20260829，k=5 的第一次 draft 触发）：
    RuntimeError: Converting a tensordict to boolean value is not permitted
根因：global_steps 是逐样本赋值的，批上取回来是 NonTensorStack，
`int(... or 0)` 里的 `or` 触发 __bool__。int() 同样会 TypeError。
"""

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import _eagle3_scalar_global_step


def _stacked_batch(step_value, rows=4):
    """逐样本带 global_steps 的批 —— 复现 dispatch 层交给 worker 的形状。"""
    rowtds = []
    for _ in range(rows):
        td = TensorDict({"x": torch.zeros(2)}, batch_size=[])
        tu.assign_non_tensor_data(td, "global_steps", step_value)
        rowtds.append(td)
    return torch.stack(rowtds)


def test_nontensorstack_is_unwrapped_to_int():
    batch = _stacked_batch(5)
    # 先确认这个批真的能触发原 bug，否则测试是空转的
    raw = tu.get_non_tensor_data(batch, "global_steps", default=0)
    with pytest.raises(RuntimeError):
        bool(raw)

    assert _eagle3_scalar_global_step(batch) == 5


def test_scalar_batch_still_works():
    td = TensorDict({"x": torch.zeros(4, 2)}, batch_size=[4])
    tu.assign_non_tensor_data(td, "global_steps", 7)
    assert _eagle3_scalar_global_step(td) == 7


def test_missing_key_defaults_to_zero():
    td = TensorDict({"x": torch.zeros(4, 2)}, batch_size=[4])
    assert _eagle3_scalar_global_step(td) == 0


def test_step_zero_is_not_swallowed():
    """`or 0` 语义下 0 和缺失无法区分；这里两者都必须得到 0 且不抛。"""
    assert _eagle3_scalar_global_step(_stacked_batch(0)) == 0


def test_all_three_call_sites_use_the_helper():
    """防回归：三个入口不得退回 `int(... or 0)` 写法。"""
    import inspect

    from verl.workers import engine_workers

    src = inspect.getsource(engine_workers)
    assert 'get_non_tensor_data(data, "global_steps", default=0) or 0' not in src, (
        "有调用点退回了会在 NonTensorStack 上抛 RuntimeError 的写法"
    )
    assert src.count("global_step = _eagle3_scalar_global_step(data)") == 3
