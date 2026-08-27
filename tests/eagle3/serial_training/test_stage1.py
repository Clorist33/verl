#!/usr/bin/env python3
"""Stage 1 代码搬运测试：验证 SerialTrainingScheduler 和路由逻辑"""

import sys
sys.path.insert(0, '/home/t00972278/verl')

def test_serial_training_scheduler():
    """测试 SerialTrainingScheduler 调度逻辑"""
    from verl.trainer.ppo.v1.trainer_base import SerialTrainingScheduler

    print("=" * 60)
    print("测试 1: SerialTrainingScheduler 类是否存在")
    print("=" * 60)

    scheduler = SerialTrainingScheduler(k=5)
    print(f"✓ 成功创建 SerialTrainingScheduler(k=5)")

    print("\n" + "=" * 60)
    print("测试 2: 调度逻辑验证 (k=5, 周期=6)")
    print("=" * 60)
    print("预期: step 0-4 训练 Actor, step 5 训练 Draft, 然后循环")
    print()

    # 测试两个完整周期
    expected_pattern = [
        (0, True, False),   # Actor
        (1, True, False),   # Actor
        (2, True, False),   # Actor
        (3, True, False),   # Actor
        (4, True, False),   # Actor
        (5, False, True),   # Draft
        (6, True, False),   # Actor (新周期开始)
        (7, True, False),   # Actor
        (8, True, False),   # Actor
        (9, True, False),   # Actor
        (10, True, False),  # Actor
        (11, False, True),  # Draft
    ]

    all_passed = True
    for step, expected_actor, expected_draft in expected_pattern:
        actual_actor = scheduler.should_train_actor(step)
        actual_draft = scheduler.should_train_draft(step)

        status = "✓" if (actual_actor == expected_actor and actual_draft == expected_draft) else "✗"
        if status == "✗":
            all_passed = False

        mode = "Actor" if actual_actor else "Draft"
        print(f"{status} Step {step:2d}: train_actor={actual_actor}, train_draft={actual_draft} ({mode})")

    if all_passed:
        print("\n✓ 所有调度逻辑测试通过")
    else:
        print("\n✗ 调度逻辑测试失败")
        sys.exit(1)

def test_method_existence():
    """测试新增方法是否存在"""
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    print("\n" + "=" * 60)
    print("测试 3: 新增方法是否存在")
    print("=" * 60)

    methods = [
        '_is_serial_training_enabled',
        '_step_once_parallel',
        '_step_once_serial',
        '_update_draft',
    ]

    all_exist = True
    for method_name in methods:
        if hasattr(PPOTrainer, method_name):
            print(f"✓ PPOTrainer.{method_name} 存在")
        else:
            print(f"✗ PPOTrainer.{method_name} 不存在")
            all_exist = False

    if all_exist:
        print("\n✓ 所有新增方法都存在")
    else:
        print("\n✗ 某些方法缺失")
        sys.exit(1)

def test_routing_logic():
    """测试路由逻辑（检查方法签名）"""
    import inspect
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    print("\n" + "=" * 60)
    print("测试 4: _step_once 路由方法签名")
    print("=" * 60)

    sig = inspect.signature(PPOTrainer._step_once)
    params = list(sig.parameters.keys())
    expected_params = ['self', 'metrics', 'timing_raw', 'sample_batch_size']

    if params == expected_params:
        print(f"✓ _step_once 签名正确: {params}")
    else:
        print(f"✗ _step_once 签名错误")
        print(f"  预期: {expected_params}")
        print(f"  实际: {params}")
        sys.exit(1)

    # 检查 _step_once_parallel 和 _step_once_serial 签名
    for method_name in ['_step_once_parallel', '_step_once_serial']:
        method = getattr(PPOTrainer, method_name)
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        if params == expected_params:
            print(f"✓ {method_name} 签名正确: {params}")
        else:
            print(f"✗ {method_name} 签名错误")
            print(f"  预期: {expected_params}")
            print(f"  实际: {params}")
            sys.exit(1)

if __name__ == '__main__':
    try:
        test_serial_training_scheduler()
        test_method_existence()
        test_routing_logic()

        print("\n" + "=" * 60)
        print("Stage 1 所有测试通过 ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
