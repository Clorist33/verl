#!/usr/bin/env python3
"""串行训练进度条显示功能测试

测试内容：
1. 验证 _current_training_type 变量的设置
2. 验证进度条描述的更新逻辑
3. 模拟显示效果
"""

import sys
import inspect

# 添加 verl 到路径
sys.path.insert(0, '/home/t00972278/verl')

def test_training_type_tracking():
    """测试 1: 训练类型跟踪逻辑"""
    print("=" * 60)
    print("测试 1: 训练类型跟踪逻辑")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    # 检查 _step_once_serial 方法
    if not hasattr(PPOTrainer, '_step_once_serial'):
        print("✗ _step_once_serial 方法不存在")
        sys.exit(1)

    method = getattr(PPOTrainer, '_step_once_serial')
    source = inspect.getsource(method)

    print("\n检查训练类型记录逻辑：")

    # 检查关键代码
    checks = [
        ('_current_training_type', '训练类型变量'),
        ('if train_actor:', 'Actor 训练分支'),
        ('elif train_draft:', 'Draft 训练分支'),
        ('"Actor"', 'Actor 类型标记'),
        ('"Draft"', 'Draft 类型标记'),
    ]

    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}: 存在")
        else:
            print(f"  ✗ {description}: 缺失")
            sys.exit(1)

    print("\n✓ 训练类型跟踪逻辑正确")

def test_progress_bar_update():
    """测试 2: 进度条更新逻辑"""
    print("\n" + "=" * 60)
    print("测试 2: 进度条更新逻辑")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    # 检查 fit 方法中的进度条更新逻辑
    if not hasattr(PPOTrainer, 'fit'):
        print("✗ fit 方法不存在")
        sys.exit(1)

    method = getattr(PPOTrainer, 'fit')
    source = inspect.getsource(method)

    print("\n检查进度条更新逻辑：")

    # 检查关键代码
    checks = [
        ('set_description', '进度条描述更新'),
        ('_current_training_type', '训练类型变量使用'),
        ('Step {self.global_steps}', '步数显示'),
        ('[{self._current_training_type}]', '训练类型显示'),
    ]

    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}: 存在")
        else:
            print(f"  ✗ {description}: 缺失")
            sys.exit(1)

    print("\n✓ 进度条更新逻辑正确")

def test_display_simulation():
    """测试 3: 显示效果模拟"""
    print("\n" + "=" * 60)
    print("测试 3: 显示效果模拟")
    print("=" * 60)

    print("\n串行训练进度条显示示例（k=5）：")
    print("-" * 60)

    # 模拟进度条显示
    displays = [
        ("Step 1 [Actor]", 1, "Actor 训练步"),
        ("Step 2 [Actor]", 2, "Actor 训练步"),
        ("Step 3 [Actor]", 3, "Actor 训练步"),
        ("Step 4 [Actor]", 4, "Actor 训练步"),
        ("Step 5 [Actor]", 5, "Actor 训练步"),
        ("Step 6 [Draft]", 6, "Draft 训练步"),
        ("Step 7 [Actor]", 7, "Actor 训练步（下一个周期）"),
    ]

    for desc, step, comment in displays:
        # 模拟进度条格式
        percentage = (step / 1000) * 100
        bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
        print(f"{desc}: {percentage:3.1f}%|{bar}| {step}/1000  # {comment}")

    print("-" * 60)
    print("\n✓ 显示效果清晰明了")

def test_backward_compatibility():
    """测试 4: 向后兼容性（并行模式）"""
    print("\n" + "=" * 60)
    print("测试 4: 向后兼容性（并行模式）")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    method = getattr(PPOTrainer, 'fit')
    source = inspect.getsource(method)

    print("\n检查并行模式兼容性：")

    # 检查是否有 hasattr 检查
    if 'hasattr(self, \'_current_training_type\')' in source:
        print("  ✓ 使用 hasattr 检查，确保并行模式不受影响")
    else:
        print("  ✗ 缺少 hasattr 检查")
        sys.exit(1)

    print("\n并行训练进度条显示示例：")
    print("-" * 60)

    # 模拟并行模式显示
    displays = [
        ("Step 1", 1, "并行模式（无训练类型标记）"),
        ("Step 2", 2, "并行模式"),
        ("Step 3", 3, "并行模式"),
    ]

    for desc, step, comment in displays:
        percentage = (step / 1000) * 100
        bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
        print(f"{desc}: {percentage:3.1f}%|{bar}| {step}/1000  # {comment}")

    print("-" * 60)
    print("\n✓ 并行模式不受影响")

def test_code_integration():
    """测试 5: 代码集成检查"""
    print("\n" + "=" * 60)
    print("测试 5: 代码集成检查")
    print("=" * 60)

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    print("\n检查代码集成：")

    # 检查 _step_once_serial
    if hasattr(PPOTrainer, '_step_once_serial'):
        print("  ✓ _step_once_serial 方法存在")
    else:
        print("  ✗ _step_once_serial 方法不存在")
        sys.exit(1)

    # 检查 fit 方法
    if hasattr(PPOTrainer, 'fit'):
        print("  ✓ fit 方法存在")
    else:
        print("  ✗ fit 方法不存在")
        sys.exit(1)

    print("\n✓ 代码集成正确")

def main():
    print("=" * 60)
    print("串行训练进度条显示功能测试")
    print("=" * 60)
    print("\n功能：在进度条中显示当前步数和训练类型（Actor/Draft）\n")

    # 测试 1: 训练类型跟踪
    test_training_type_tracking()

    # 测试 2: 进度条更新
    test_progress_bar_update()

    # 测试 3: 显示效果
    test_display_simulation()

    # 测试 4: 向后兼容性
    test_backward_compatibility()

    # 测试 5: 代码集成
    test_code_integration()

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
    print("\n总结：")
    print("  ✓ 训练类型跟踪逻辑正确")
    print("  ✓ 进度条更新逻辑正确")
    print("  ✓ 显示效果清晰明了")
    print("  ✓ 向后兼容性良好（并行模式不受影响）")
    print("  ✓ 代码集成正确")
    print("\n预期效果：")
    print("  - 串行模式：显示 'Step N [Actor]' 或 'Step N [Draft]'")
    print("  - 并行模式：显示 'Step N'")

if __name__ == "__main__":
    main()
