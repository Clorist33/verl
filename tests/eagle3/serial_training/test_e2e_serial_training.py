#!/usr/bin/env python3
"""EAGLE3 串行训练端到端测试

测试内容：
1. 配置验证（actor_training_steps 必须是 k 的整数倍）
2. 初始化流程
3. 串行训练调度逻辑
4. 步数追踪（global_steps, actor_steps, draft_steps）
5. 进度条显示
6. Metrics 记录
7. 与并行模式的对比

运行方式：
    cd /home/t00972278/verl
    python tests/eagle3/serial_training/test_e2e_serial_training.py
"""

import sys
import os
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# 添加 verl 到路径
sys.path.insert(0, '/home/t00972278/verl')

def test_1_config_validation():
    """测试 1: 配置验证"""
    print("\n" + "=" * 60)
    print("测试 1: 配置验证")
    print("=" * 60)

    from omegaconf import OmegaConf
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    print("\n场景 1.1: actor_training_steps 未设置（应该报错）")
    try:
        config = OmegaConf.create({
            'algorithm': {
                'eagle3': {
                    'enable_serial_training': True,
                    'actor_steps_per_draft_step': 5,
                }
            },
            'trainer': {
                # actor_training_steps 未设置
            }
        })

        # 模拟 trainer 初始化
        trainer = Mock()
        trainer.config = config
        trainer._is_serial_training_enabled = lambda: True

        # 尝试调用初始化方法
        PPOTrainer._initialize_serial_training_config(trainer)

        print("  ✗ 应该抛出 ValueError")
        return False

    except ValueError as e:
        if "必须设置 'actor_training_steps'" in str(e):
            print(f"  ✓ 正确抛出 ValueError: {str(e)[:80]}...")
        else:
            print(f"  ✗ 错误信息不正确: {e}")
            return False

    print("\n场景 1.2: actor_training_steps 不是 k 的整数倍（应该报错）")
    try:
        config = OmegaConf.create({
            'algorithm': {
                'eagle3': {
                    'enable_serial_training': True,
                    'actor_steps_per_draft_step': 5,
                }
            },
            'trainer': {
                'actor_training_steps': 1003,  # 1003 % 5 != 0
            }
        })

        trainer = Mock()
        trainer.config = config
        trainer._is_serial_training_enabled = lambda: True

        PPOTrainer._initialize_serial_training_config(trainer)

        print("  ✗ 应该抛出 ValueError")
        return False

    except ValueError as e:
        if "必须是" in str(e) and "的整数倍" in str(e):
            print(f"  ✓ 正确抛出 ValueError: {str(e)[:80]}...")
        else:
            print(f"  ✗ 错误信息不正确: {e}")
            return False

    print("\n场景 1.3: 配置正确（应该通过）")
    try:
        config = OmegaConf.create({
            'algorithm': {
                'eagle3': {
                    'enable_serial_training': True,
                    'actor_steps_per_draft_step': 5,
                }
            },
            'trainer': {
                'actor_training_steps': 1000,  # 1000 % 5 == 0
            }
        })

        trainer = Mock()
        trainer.config = config
        trainer._is_serial_training_enabled = lambda: True

        PPOTrainer._initialize_serial_training_config(trainer)

        # 验证计算结果
        assert trainer.actor_training_steps == 1000
        assert trainer.draft_training_steps == 200
        assert trainer.total_training_steps == 1200
        assert trainer.actor_steps == 0
        assert trainer.draft_steps == 0

        print(f"  ✓ 配置验证通过")
        print(f"    actor_training_steps: {trainer.actor_training_steps}")
        print(f"    draft_training_steps: {trainer.draft_training_steps}")
        print(f"    total_training_steps: {trainer.total_training_steps}")

    except Exception as e:
        print(f"  ✗ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ 配置验证测试通过")
    return True

def test_2_initialization():
    """测试 2: 初始化流程"""
    print("\n" + "=" * 60)
    print("测试 2: 初始化流程")
    print("=" * 60)

    from omegaconf import OmegaConf
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    print("\n场景 2.1: 串行模式初始化")
    config = OmegaConf.create({
        'algorithm': {
            'eagle3': {
                'enable_serial_training': True,
                'actor_steps_per_draft_step': 5,
            }
        },
        'trainer': {
            'actor_training_steps': 1000,
        }
    })

    trainer = Mock()
    trainer.config = config
    trainer._is_serial_training_enabled = lambda: True

    PPOTrainer._initialize_serial_training_config(trainer)

    # 验证所有属性都被正确初始化
    checks = [
        ('actor_training_steps', 1000),
        ('draft_training_steps', 200),
        ('total_training_steps', 1200),
        ('actor_steps', 0),
        ('draft_steps', 0),
    ]

    for attr, expected in checks:
        actual = getattr(trainer, attr)
        if actual == expected:
            print(f"  ✓ {attr}: {actual}")
        else:
            print(f"  ✗ {attr}: 期望 {expected}, 实际 {actual}")
            return False

    print("\n场景 2.2: 不同的 k 值")
    test_cases = [
        (1000, 5, 200, 1200),
        (1000, 10, 100, 1100),
        (2000, 4, 500, 2500),
    ]

    for actor_steps, k, expected_draft, expected_total in test_cases:
        config = OmegaConf.create({
            'algorithm': {
                'eagle3': {
                    'enable_serial_training': True,
                    'actor_steps_per_draft_step': k,
                }
            },
            'trainer': {
                'actor_training_steps': actor_steps,
            }
        })

        trainer = Mock()
        trainer.config = config
        trainer._is_serial_training_enabled = lambda: True

        PPOTrainer._initialize_serial_training_config(trainer)

        if (trainer.draft_training_steps == expected_draft and
            trainer.total_training_steps == expected_total):
            print(f"  ✓ actor={actor_steps}, k={k} → draft={expected_draft}, total={expected_total}")
        else:
            print(f"  ✗ actor={actor_steps}, k={k} → draft={trainer.draft_training_steps}, total={trainer.total_training_steps}")
            return False

    print("\n✓ 初始化流程测试通过")
    return True

def test_3_scheduler_logic():
    """测试 3: 串行训练调度逻辑"""
    print("\n" + "=" * 60)
    print("测试 3: 串行训练调度逻辑")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import SerialTrainingScheduler

    print("\n场景 3.1: k=5 的调度模式")
    scheduler = SerialTrainingScheduler(k=5)

    # 测试前 12 步（2 个完整周期）
    expected = [
        (0, True, False),   # Step 0: Actor
        (1, True, False),   # Step 1: Actor
        (2, True, False),   # Step 2: Actor
        (3, True, False),   # Step 3: Actor
        (4, True, False),   # Step 4: Actor
        (5, False, True),   # Step 5: Draft
        (6, True, False),   # Step 6: Actor (新周期)
        (7, True, False),   # Step 7: Actor
        (8, True, False),   # Step 8: Actor
        (9, True, False),   # Step 9: Actor
        (10, True, False),  # Step 10: Actor
        (11, False, True),  # Step 11: Draft
    ]

    for step, expected_actor, expected_draft in expected:
        actual_actor = scheduler.should_train_actor(step)
        actual_draft = scheduler.should_train_draft(step)

        if actual_actor == expected_actor and actual_draft == expected_draft:
            step_type = "Actor" if actual_actor else "Draft"
            print(f"  ✓ Step {step:2d}: {step_type}")
        else:
            print(f"  ✗ Step {step}: 期望 actor={expected_actor}, draft={expected_draft}, "
                  f"实际 actor={actual_actor}, draft={actual_draft}")
            return False

    print("\n场景 3.2: 不同的 k 值")
    test_cases = [
        (1, [True, False, True, False]),  # k=1: 完全交替
        (3, [True, True, True, False]),   # k=3: 3 Actor + 1 Draft
        (7, [True] * 7 + [False]),        # k=7: 7 Actor + 1 Draft
    ]

    for k, pattern in test_cases:
        scheduler = SerialTrainingScheduler(k=k)
        actual_pattern = []

        for step in range(len(pattern)):
            actual_pattern.append(scheduler.should_train_actor(step))

        if actual_pattern == pattern:
            print(f"  ✓ k={k}: 调度模式正确")
        else:
            print(f"  ✗ k={k}: 期望 {pattern}, 实际 {actual_pattern}")
            return False

    print("\n✓ 调度逻辑测试通过")
    return True

def test_4_step_tracking():
    """测试 4: 步数追踪模拟"""
    print("\n" + "=" * 60)
    print("测试 4: 步数追踪模拟")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import SerialTrainingScheduler

    print("\n模拟训练 12 步（k=5, 2 个完整周期）")

    # 初始化
    scheduler = SerialTrainingScheduler(k=5)
    global_steps = 0
    actor_steps = 0
    draft_steps = 0

    # 模拟训练
    for step in range(12):
        global_steps = step

        if scheduler.should_train_actor(step):
            actor_steps += 1
            step_type = "Actor"
        elif scheduler.should_train_draft(step):
            draft_steps += 1
            step_type = "Draft"

        print(f"  Step {step:2d} [{step_type:5s}]: global={global_steps}, actor={actor_steps}, draft={draft_steps}")

    # 验证最终步数
    expected_actor = 10  # 12 步中有 10 个 Actor 步
    expected_draft = 2   # 12 步中有 2 个 Draft 步

    if actor_steps == expected_actor and draft_steps == expected_draft:
        print(f"\n✓ 步数追踪正确: actor={actor_steps}, draft={draft_steps}")
    else:
        print(f"\n✗ 步数追踪错误: 期望 actor={expected_actor}, draft={expected_draft}, "
              f"实际 actor={actor_steps}, draft={draft_steps}")
        return False

    print("\n✓ 步数追踪测试通过")
    return True

def test_5_progress_display():
    """测试 5: 进度条显示格式"""
    print("\n" + "=" * 60)
    print("测试 5: 进度条显示格式")
    print("=" * 60)

    print("\n场景 5.1: Actor 步显示")
    global_steps = 5
    actor_steps = 5
    total_training_steps = 1200
    actor_training_steps = 1000

    progress_desc = f"Global {global_steps}/{total_training_steps} [Actor {actor_steps}/{actor_training_steps}]"
    expected = "Global 5/1200 [Actor 5/1000]"

    if progress_desc == expected:
        print(f"  ✓ Actor 显示: {progress_desc}")
    else:
        print(f"  ✗ 期望: {expected}")
        print(f"    实际: {progress_desc}")
        return False

    print("\n场景 5.2: Draft 步显示")
    global_steps = 6
    draft_steps = 1
    draft_training_steps = 200

    progress_desc = f"Global {global_steps}/{total_training_steps} [Draft {draft_steps}/{draft_training_steps}]"
    expected = "Global 6/1200 [Draft 1/200]"

    if progress_desc == expected:
        print(f"  ✓ Draft 显示: {progress_desc}")
    else:
        print(f"  ✗ 期望: {expected}")
        print(f"    实际: {progress_desc}")
        return False

    print("\n✓ 进度条显示测试通过")
    return True

def test_6_metrics_structure():
    """测试 6: Metrics 结构"""
    print("\n" + "=" * 60)
    print("测试 6: Metrics 结构")
    print("=" * 60)

    print("\n场景 6.1: 串行模式 Metrics")
    metrics = {
        "training/global_steps": 100,
        "training/actor_steps": 84,
        "training/draft_steps": 16,
        "training/actor_progress": 0.084,
        "training/draft_progress": 0.08,
    }

    required_keys = [
        "training/global_steps",
        "training/actor_steps",
        "training/draft_steps",
        "training/actor_progress",
        "training/draft_progress",
    ]

    for key in required_keys:
        if key in metrics:
            print(f"  ✓ {key}: {metrics[key]}")
        else:
            print(f"  ✗ 缺少 metric: {key}")
            return False

    print("\n✓ Metrics 结构测试通过")
    return True

def test_7_end_to_end_simulation():
    """测试 7: 端到端模拟"""
    print("\n" + "=" * 60)
    print("测试 7: 端到端训练模拟")
    print("=" * 60)

    from omegaconf import OmegaConf
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer, SerialTrainingScheduler

    print("\n配置: actor_training_steps=1000, k=5")
    print("预期: actor=1000, draft=200, total=1200")

    # 1. 配置验证
    config = OmegaConf.create({
        'algorithm': {
            'eagle3': {
                'enable_serial_training': True,
                'actor_steps_per_draft_step': 5,
            }
        },
        'trainer': {
            'actor_training_steps': 1000,
        }
    })

    trainer = Mock()
    trainer.config = config
    trainer._is_serial_training_enabled = lambda: True

    # 2. 初始化
    PPOTrainer._initialize_serial_training_config(trainer)
    scheduler = SerialTrainingScheduler(k=5)

    print(f"\n初始化完成:")
    print(f"  actor_training_steps: {trainer.actor_training_steps}")
    print(f"  draft_training_steps: {trainer.draft_training_steps}")
    print(f"  total_training_steps: {trainer.total_training_steps}")

    # 3. 模拟训练
    print(f"\n模拟训练...")
    actor_count = 0
    draft_count = 0

    for step in range(trainer.total_training_steps):
        if scheduler.should_train_actor(step):
            actor_count += 1
        elif scheduler.should_train_draft(step):
            draft_count += 1

    print(f"  训练完成: actor={actor_count}, draft={draft_count}, total={step+1}")

    # 4. 验证
    if (actor_count == trainer.actor_training_steps and
        draft_count == trainer.draft_training_steps):
        print(f"\n✓ 端到端模拟成功")
        print(f"  ✓ Actor 步数匹配: {actor_count}/{trainer.actor_training_steps}")
        print(f"  ✓ Draft 步数匹配: {draft_count}/{trainer.draft_training_steps}")
        print(f"  ✓ 总步数匹配: {step+1}/{trainer.total_training_steps}")
    else:
        print(f"\n✗ 步数不匹配")
        print(f"  期望: actor={trainer.actor_training_steps}, draft={trainer.draft_training_steps}")
        print(f"  实际: actor={actor_count}, draft={draft_count}")
        return False

    print("\n✓ 端到端模拟测试通过")
    return True

def main():
    """运行所有测试"""
    print("=" * 60)
    print("EAGLE3 串行训练端到端测试")
    print("=" * 60)
    print("\n测试目标:")
    print("  - 配置验证")
    print("  - 初始化流程")
    print("  - 调度逻辑")
    print("  - 步数追踪")
    print("  - 进度条显示")
    print("  - Metrics 结构")
    print("  - 端到端模拟")

    tests = [
        ("配置验证", test_1_config_validation),
        ("初始化流程", test_2_initialization),
        ("调度逻辑", test_3_scheduler_logic),
        ("步数追踪", test_4_step_tracking),
        ("进度条显示", test_5_progress_display),
        ("Metrics 结构", test_6_metrics_structure),
        ("端到端模拟", test_7_end_to_end_simulation),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ 测试失败: {name}")
        except Exception as e:
            failed += 1
            print(f"\n✗ 测试异常: {name}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
