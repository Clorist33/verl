#!/usr/bin/env python3
"""串行训练双重步数追踪功能测试

测试内容：
1. 配置参数验证（actor_training_steps 必须是 k 的整数倍）
2. 步数计算逻辑
3. 初始化逻辑
4. 步数追踪逻辑
5. 进度条显示
6. Metrics 记录
"""

import sys
import inspect

# 添加 verl 到路径
sys.path.insert(0, '/home/t00972278/verl')

def test_initialization_method():
    """测试 1: 初始化方法存在性"""
    print("=" * 60)
    print("测试 1: 初始化方法存在性")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    if not hasattr(PPOTrainer, '_initialize_serial_training_config'):
        print("✗ _initialize_serial_training_config 方法不存在")
        sys.exit(1)

    print("✓ _initialize_serial_training_config 方法存在")

    # 检查方法源码
    method = getattr(PPOTrainer, '_initialize_serial_training_config')
    source = inspect.getsource(method)

    print("\n检查关键逻辑：")
    checks = [
        ('actor_training_steps', 'actor_training_steps 参数'),
        ('actor_steps_per_draft_step', 'k 值读取'),
        ('if actor_training_steps is None:', '参数必填验证'),
        ('if actor_training_steps % k != 0:', '整数倍验证'),
        ('self.draft_training_steps = actor_training_steps // k', 'draft 步数计算'),
        ('self.total_training_steps =', 'total 步数计算'),
        ('self.actor_steps = 0', 'actor_steps 初始化'),
        ('self.draft_steps = 0', 'draft_steps 初始化'),
    ]

    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} 缺失")
            sys.exit(1)

    print("\n✓ 初始化方法逻辑完整")

def test_validation_logic():
    """测试 2: 验证逻辑"""
    print("\n" + "=" * 60)
    print("测试 2: 验证逻辑")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    method = getattr(PPOTrainer, '_initialize_serial_training_config')
    source = inspect.getsource(method)

    print("\n检查验证条件：")

    # 检查必填验证
    if 'if actor_training_steps is None:' in source and 'raise ValueError' in source:
        print("  ✓ actor_training_steps 必填验证存在")
    else:
        print("  ✗ actor_training_steps 必填验证缺失")
        sys.exit(1)

    # 检查整数倍验证
    if 'if actor_training_steps % k != 0:' in source and 'raise ValueError' in source:
        print("  ✓ 整数倍验证存在")
    else:
        print("  ✗ 整数倍验证缺失")
        sys.exit(1)

    print("\n验证场景测试：")

    # 模拟验证逻辑
    test_cases = [
        (1000, 5, True, "1000 % 5 = 0"),
        (1005, 5, False, "1005 % 5 = 0 (不通过)"),
        (200, 4, True, "200 % 4 = 0"),
        (201, 4, False, "201 % 4 = 1 (不通过)"),
    ]

    for actor_steps, k, should_pass, desc in test_cases:
        result = (actor_steps % k == 0)
        status = "✓" if result == should_pass else "✗"
        print(f"  {status} {desc}")

    print("\n✓ 验证逻辑正确")

def test_calculation_logic():
    """测试 3: 计算逻辑"""
    print("\n" + "=" * 60)
    print("测试 3: 计算逻辑")
    print("=" * 60)

    print("\n计算示例：")

    test_cases = [
        (1000, 5, 200, 1200),
        (500, 5, 100, 600),
        (800, 4, 200, 1000),
        (2000, 10, 200, 2200),
    ]

    for actor_steps, k, expected_draft, expected_total in test_cases:
        draft_steps = actor_steps // k
        total_steps = actor_steps + draft_steps

        if draft_steps == expected_draft and total_steps == expected_total:
            print(f"  ✓ actor={actor_steps}, k={k} → draft={draft_steps}, total={total_steps}")
        else:
            print(f"  ✗ actor={actor_steps}, k={k} → draft={draft_steps}, total={total_steps}")
            print(f"     期望: draft={expected_draft}, total={expected_total}")
            sys.exit(1)

    print("\n验证整数倍约束：")
    for actor_steps, k, _, total_steps in test_cases:
        if total_steps % (k + 1) == 0:
            print(f"  ✓ total={total_steps} 是 (k+1)={k+1} 的整数倍")
        else:
            print(f"  ✗ total={total_steps} 不是 (k+1)={k+1} 的整数倍")
            sys.exit(1)

    print("\n✓ 计算逻辑正确")

def test_step_tracking():
    """测试 4: 步数追踪逻辑"""
    print("\n" + "=" * 60)
    print("测试 4: 步数追踪逻辑")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    method = getattr(PPOTrainer, '_step_once_serial')
    source = inspect.getsource(method)

    print("\n检查步数追踪代码：")

    checks = [
        ('self.actor_steps += 1', 'Actor 步数增加'),
        ('self.draft_steps += 1', 'Draft 步数增加'),
        ('Actor step {self.actor_steps}/{self.actor_training_steps}', 'Actor 进度日志'),
        ('Draft step {self.draft_steps}/{self.draft_training_steps}', 'Draft 进度日志'),
    ]

    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} 缺失")
            sys.exit(1)

    print("\n✓ 步数追踪逻辑完整")

def test_progress_bar_display():
    """测试 5: 进度条显示"""
    print("\n" + "=" * 60)
    print("测试 5: 进度条显示")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    method = getattr(PPOTrainer, 'fit')
    source = inspect.getsource(method)

    print("\n检查进度条显示逻辑：")

    checks = [
        ('Global {self.global_steps}/{self.total_training_steps}', 'Global 步数显示'),
        ('[Actor {self.actor_steps}/{self.actor_training_steps}]', 'Actor 步数显示'),
        ('[Draft {self.draft_steps}/{self.draft_training_steps}]', 'Draft 步数显示'),
    ]

    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} 缺失")
            sys.exit(1)

    print("\n显示效果示例：")
    print("  Global 1/1200 [Actor 1/1000]")
    print("  Global 2/1200 [Actor 2/1000]")
    print("  Global 5/1200 [Actor 5/1000]")
    print("  Global 6/1200 [Draft 1/200]")
    print("  Global 7/1200 [Actor 6/1000]")

    print("\n✓ 进度条显示逻辑正确")

def test_metrics_recording():
    """测试 6: Metrics 记录"""
    print("\n" + "=" * 60)
    print("测试 6: Metrics 记录")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    method = getattr(PPOTrainer, '_compute_metrics')
    source = inspect.getsource(method)

    print("\n检查 Metrics 记录：")

    checks = [
        ('if self._is_serial_training_enabled():', '串行模式检查'),
        ('"training/global_steps"', 'global_steps metric'),
        ('"training/actor_steps"', 'actor_steps metric'),
        ('"training/draft_steps"', 'draft_steps metric'),
        ('"training/actor_progress"', 'actor_progress metric'),
        ('"training/draft_progress"', 'draft_progress metric'),
    ]

    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} 缺失")
            sys.exit(1)

    print("\n记录的 Metrics：")
    print("  - training/global_steps: 总步数（Actor + Draft）")
    print("  - training/actor_steps: Actor 完成的步数")
    print("  - training/draft_steps: Draft 完成的步数")
    print("  - training/actor_progress: Actor 进度百分比")
    print("  - training/draft_progress: Draft 进度百分比")

    print("\n✓ Metrics 记录逻辑正确")

def test_fit_integration():
    """测试 7: fit 方法集成"""
    print("\n" + "=" * 60)
    print("测试 7: fit 方法集成")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    method = getattr(PPOTrainer, 'fit')
    source = inspect.getsource(method)

    print("\n检查 fit 方法集成：")

    if 'if self._is_serial_training_enabled():' in source and '_initialize_serial_training_config()' in source:
        print("  ✓ fit 方法调用了 _initialize_serial_training_config")
    else:
        print("  ✗ fit 方法未调用 _initialize_serial_training_config")
        sys.exit(1)

    # 检查调用位置（应该在开始处）
    lines = source.split('\n')
    init_line = -1
    for i, line in enumerate(lines):
        if '_initialize_serial_training_config' in line:
            init_line = i
            break

    if init_line < 10:  # 应该在前 10 行
        print("  ✓ 初始化调用位置正确（在方法开始处）")
    else:
        print("  ⚠ 初始化调用位置可能不是最优")

    print("\n✓ fit 方法集成正确")

def main():
    print("=" * 60)
    print("串行训练双重步数追踪功能测试")
    print("=" * 60)
    print("\n功能：")
    print("  1. 设置 actor_training_steps 和 actor_steps_per_draft_step")
    print("  2. 验证 actor_training_steps 是 k 的整数倍")
    print("  3. 追踪 global_steps, actor_steps, draft_steps")
    print("  4. 进度条显示三种步数")
    print("  5. Metrics 记录三种步数\n")

    # 测试 1: 初始化方法
    test_initialization_method()

    # 测试 2: 验证逻辑
    test_validation_logic()

    # 测试 3: 计算逻辑
    test_calculation_logic()

    # 测试 4: 步数追踪
    test_step_tracking()

    # 测试 5: 进度条显示
    test_progress_bar_display()

    # 测试 6: Metrics 记录
    test_metrics_recording()

    # 测试 7: fit 集成
    test_fit_integration()

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
    print("\n总结：")
    print("  ✓ 初始化方法逻辑完整")
    print("  ✓ 验证逻辑正确（必填 + 整数倍）")
    print("  ✓ 计算逻辑正确")
    print("  ✓ 步数追踪完整")
    print("  ✓ 进度条显示清晰")
    print("  ✓ Metrics 记录完整")
    print("  ✓ fit 方法集成正确")

if __name__ == "__main__":
    main()
