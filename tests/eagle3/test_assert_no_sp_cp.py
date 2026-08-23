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
"""Unit test for the EAGLE3 SP/CP gate `_assert_no_sp_cp` after the 2026-08-11 change.

Change under test (开发过程记录/debug梳理/SP冲突根因与放开.md):
  - SP (sequence_parallel) is now ALLOWED. eagle3_patch._postprocess gathers the
    SP-sharded capture back to full sequence, and Megatron MoE+TP>1 requires SP.
  - CP (context_parallel_size>1) is still FORBIDDEN (no CP-dim gather in _postprocess).

We mock `megatron.core.parallel_state` so the test needs NO distributed init / NPU.
Run:
    cd /home/t00972278/verl && python tests/eagle3/test_assert_no_sp_cp.py
    # or: pytest tests/eagle3/test_assert_no_sp_cp.py -v
"""

import sys
import types
from contextlib import contextmanager


class _FakeConfig:
    """Stand-in for gpt.config carrying only the sequence_parallel flag."""

    def __init__(self, sequence_parallel):
        self.sequence_parallel = sequence_parallel


class _FakeGPT:
    def __init__(self, sequence_parallel):
        self.config = _FakeConfig(sequence_parallel)


@contextmanager
def _mock_parallel_state(cp_size, tp_size):
    """Inject a fake megatron.core.parallel_state with fixed CP/TP world sizes.

    `_assert_no_sp_cp` does `from megatron.core import parallel_state as mpu`
    at call time, so patching sys.modules before the call is enough and avoids
    importing/initializing real Megatron distributed state.
    """
    import megatron.core as mcore

    real_ps = getattr(mcore, "parallel_state", None)
    fake_ps = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_size,
        get_tensor_model_parallel_world_size=lambda: tp_size,
    )
    mcore.parallel_state = fake_ps
    sys.modules["megatron.core.parallel_state"] = fake_ps
    try:
        yield
    finally:
        if real_ps is not None:
            mcore.parallel_state = real_ps
            sys.modules["megatron.core.parallel_state"] = real_ps


def _run_case(name, sp, cp, tp, expect_raise):
    from verl.models.eagle3.engine_support import _assert_no_sp_cp

    gpt = _FakeGPT(sequence_parallel=sp)
    raised = None
    with _mock_parallel_state(cp_size=cp, tp_size=tp):
        try:
            _assert_no_sp_cp(gpt)
        except ValueError as e:
            raised = e

    ok = (raised is not None) == expect_raise
    status = "PASS" if ok else "FAIL"
    detail = f"raised={raised!r}" if raised is not None else "no raise"
    print(f"  [{status}] {name}: SP={sp} CP={cp} TP={tp} -> {detail}")
    if not ok:
        raise AssertionError(
            f"{name}: expected {'raise' if expect_raise else 'no raise'} "
            f"for SP={sp} CP={cp} TP={tp}, got {detail}"
        )
    # Extra: when CP>1 raises, the message must mention context parallelism, not SP.
    if raised is not None:
        msg = str(raised)
        assert "context parallel" in msg.lower(), f"{name}: raise msg should cite CP, got: {msg}"
        assert "sequence_parallel=True" not in msg, (
            f"{name}: raise msg must NOT forbid SP anymore, got: {msg}"
        )
    return ok


def main():
    print("test_assert_no_sp_cp: EAGLE3 SP allowed / CP forbidden")
    # (name, sp, cp, tp, expect_raise)
    cases = [
        # SP now allowed at TP>1 (the whole point of the change) -> NO raise
        ("SP=True,  CP=1, TP=4  (MoE+TP normal case)", True, 1, 4, False),
        # SP off, CP=1 -> NO raise (still fine)
        ("SP=False, CP=1, TP=4", False, 1, 4, False),
        # SP off, TP=1 -> NO raise
        ("SP=False, CP=1, TP=1", False, 1, 1, False),
        # CP>1 still forbidden regardless of SP
        ("SP=True,  CP=2, TP=4  (CP must still fail)", True, 2, 4, True),
        ("SP=False, CP=2, TP=4  (CP must still fail)", False, 2, 4, True),
    ]
    all_ok = True
    for name, sp, cp, tp, expect_raise in cases:
        try:
            _run_case(name, sp, cp, tp, expect_raise)
        except AssertionError as e:
            all_ok = False
            print(f"    !! {e}")

    print("-" * 60)
    if all_ok:
        print("ALL PASS: SP is allowed (incl. TP>1); CP>1 still raises with a CP-only message.")
        return 0
    print("FAILURES above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
