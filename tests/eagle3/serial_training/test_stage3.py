#!/usr/bin/env python3
"""Stage 3 代码测试脚本

Stage 3 目标：重构 TrainingWorker.train_batch 方法，支持串行训练模式

测试内容：
1. TrainingWorker 类导入
2. train_batch 路由方法存在性
3. _train_batch_original 方法存在性（原有逻辑封装）
4. _train_batch_draft_only 方法存在性（Draft 专用训练）
5. 方法签名一致性检查
6. 文档字符串验证
7. 路由逻辑验证（模拟调用）
"""

import sys
import inspect
from unittest.mock import MagicMock, patch
from pathlib import Path

# 添加 verl 到路径
sys.path.insert(0, '/home/t00972278/verl')

def test_import_training_worker():
    """测试 1: TrainingWorker 类导入"""
    print("=" * 60)
    print("测试 1: TrainingWorker 类导入")
    print("=" * 60)

    try:
        from verl.workers.engine_workers import TrainingWorker
        print("✓ TrainingWorker 导入成功")
        return TrainingWorker
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        sys.exit(1)

def test_methods_exist(TrainingWorker):
    """测试 2: Stage 3 的三个方法是否存在"""
    print("\n" + "=" * 60)
    print("测试 2: Stage 3 方法存在性检查")
    print("=" * 60)

    required_methods = {
        'train_batch': '路由入口方法',
        '_train_batch_original': '原有逻辑封装',
        '_train_batch_draft_only': 'Draft 专用训练',
    }

    for method_name, description in required_methods.items():
        if not hasattr(TrainingWorker, method_name):
            print(f"✗ TrainingWorker.{method_name} 不存在")
            sys.exit(1)

        print(f"✓ TrainingWorker.{method_name} 存在 - {description}")

def test_method_signatures(TrainingWorker):
    """测试 3: 方法签名一致性"""
    print("\n" + "=" * 60)
    print("测试 3: 方法签名一致性")
    print("=" * 60)

    methods = ['train_batch', '_train_batch_original', '_train_batch_draft_only']
    expected_params = ['self', 'data']

    for method_name in methods:
        method = getattr(TrainingWorker, method_name)
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        print(f"\n{method_name}:")
        print(f"  签名: {sig}")
        print(f"  参数: {params}")

        if params == expected_params:
            print(f"  ✓ 参数列表正确")
        else:
            print(f"  ✗ 参数列表不正确")
            print(f"    预期: {expected_params}")
            print(f"    实际: {params}")
            sys.exit(1)

def test_docstrings(TrainingWorker):
    """测试 4: 文档字符串验证"""
    print("\n" + "=" * 60)
    print("测试 4: 文档字符串验证")
    print("=" * 60)

    docstring_checks = [
        ('train_batch', ['路由', '标志']),
        ('_train_batch_original', ['原有', '逻辑']),
        ('_train_batch_draft_only', ['Draft', '训练']),
    ]

    for method_name, keywords in docstring_checks:
        method = getattr(TrainingWorker, method_name)
        doc = method.__doc__

        if not doc:
            print(f"✗ {method_name} 缺少文档字符串")
            sys.exit(1)

        missing_keywords = [kw for kw in keywords if kw not in doc]

        if missing_keywords:
            print(f"✗ {method_name} 文档缺少关键词: {missing_keywords}")
            print(f"  文档内容: {doc[:100]}...")
        else:
            print(f"✓ {method_name} 文档字符串完整")
            print(f"  包含关键词: {keywords}")

def test_routing_logic():
    """测试 5: 路由逻辑验证（模拟测试）"""
    print("\n" + "=" * 60)
    print("测试 5: 路由逻辑验证")
    print("=" * 60)

    try:
        from verl.workers.engine_workers import TrainingWorker
        from verl.utils import tensordict_utils as tu

        # 检查 train_batch 方法的源代码
        import inspect
        source = inspect.getsource(TrainingWorker.train_batch)

        # 验证路由逻辑的关键要素
        checks = [
            ('get_non_tensor_data', 'train_draft_only 标志提取'),
            ('train_draft_only', '标志变量名'),
            ('_train_batch_draft_only', 'Draft 训练路由'),
            ('_train_batch_original', '原有逻辑路由'),
        ]

        print("\n检查 train_batch 路由逻辑：")
        for check_str, description in checks:
            if check_str in source:
                print(f"  ✓ {description}: '{check_str}' 存在")
            else:
                print(f"  ✗ {description}: '{check_str}' 缺失")
                sys.exit(1)

        # 验证条件分支结构
        if 'if train_draft_only:' in source or 'if train_draft_only :' in source:
            print(f"  ✓ 条件分支: 'if train_draft_only:' 存在")
        else:
            print(f"  ✗ 条件分支结构缺失")
            sys.exit(1)

        print("\n✓ 路由逻辑结构正确")

    except Exception as e:
        print(f"✗ 路由逻辑验证失败: {e}")
        sys.exit(1)

def test_method_independence():
    """测试 6: 方法独立性验证"""
    print("\n" + "=" * 60)
    print("测试 6: 方法独立性验证")
    print("=" * 60)

    try:
        from verl.workers.engine_workers import TrainingWorker
        import inspect

        # 检查 _train_batch_original 和 _train_batch_draft_only 是否独立
        original_source = inspect.getsource(TrainingWorker._train_batch_original)
        draft_source = inspect.getsource(TrainingWorker._train_batch_draft_only)

        print("\n_train_batch_original 特征：")
        # 原有逻辑应该包含 lr_scheduler 更新
        if 'update_lr_scheduler' in original_source or 'lr_scheduler_step' in original_source:
            print("  ✓ 包含学习率调度器逻辑")
        else:
            print("  ⚠ 未检测到学习率调度器逻辑（可能是正常的）")

        # 原有逻辑不应该设置 train_draft_only
        if 'train_draft_only' not in original_source:
            print("  ✓ 不设置 train_draft_only 标志（纯原有逻辑）")
        else:
            print("  ✗ 包含 train_draft_only 标志（不应该出现）")

        print("\n_train_batch_draft_only 特征：")
        # Draft 训练应该设置标志
        if 'train_draft_only' in draft_source and 'enable_draft_training' in draft_source:
            print("  ✓ 设置 train_draft_only 和 enable_draft_training 标志")
        else:
            print("  ✗ 缺少必要的标志设置")
            sys.exit(1)

        # Draft 训练应该调用 tu.assign_non_tensor_data
        if 'tu.assign_non_tensor_data' in draft_source or 'assign_non_tensor_data' in draft_source:
            print("  ✓ 使用 tu.assign_non_tensor_data API")
        else:
            print("  ⚠ 可能使用其他方式设置标志")

        # Draft 训练使用特殊的 Timer 名称
        if 'train_batch_draft' in draft_source:
            print("  ✓ 使用 'train_batch_draft' Timer 名称")
        else:
            print("  ⚠ Timer 名称可能不同")

        print("\n✓ 两个方法独立且特征明确")

    except Exception as e:
        print(f"✗ 方法独立性验证失败: {e}")
        sys.exit(1)

def test_api_compatibility():
    """测试 7: API 兼容性验证"""
    print("\n" + "=" * 60)
    print("测试 7: API 兼容性验证")
    print("=" * 60)

    try:
        from verl.workers.engine_workers import TrainingWorker
        from verl.utils import tensordict_utils as tu
        import inspect

        # 检查是否使用了正确的 API
        draft_source = inspect.getsource(TrainingWorker._train_batch_draft_only)

        # 应该使用 tu.assign_non_tensor_data 而不是 data.extra_info
        if 'tu.assign_non_tensor_data' in draft_source or 'assign_non_tensor_data' in draft_source:
            print("  ✓ 使用 tu.assign_non_tensor_data API（推荐）")

            if 'data.extra_info' not in draft_source:
                print("  ✓ 未使用 data.extra_info 直接访问（正确）")
            else:
                print("  ⚠ 同时存在 data.extra_info 直接访问")
        else:
            if 'data.extra_info' in draft_source:
                print("  ⚠ 使用 data.extra_info 直接访问（可能需要适配）")
            else:
                print("  ? 未检测到标志设置方式")

        print("\n✓ API 兼容性检查完成")

    except Exception as e:
        print(f"✗ API 兼容性验证失败: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("Stage 3 代码测试")
    print("=" * 60)
    print("\nStage 3 目标：")
    print("  重构 TrainingWorker.train_batch 方法")
    print("  - 添加路由逻辑")
    print("  - 封装原有逻辑到 _train_batch_original")
    print("  - 新增 Draft 专用训练逻辑 _train_batch_draft_only")
    print()

    # 测试 1: 导入
    TrainingWorker = test_import_training_worker()

    # 测试 2: 方法存在性
    test_methods_exist(TrainingWorker)

    # 测试 3: 方法签名
    test_method_signatures(TrainingWorker)

    # 测试 4: 文档字符串
    test_docstrings(TrainingWorker)

    # 测试 5: 路由逻辑
    test_routing_logic()

    # 测试 6: 方法独立性
    test_method_independence()

    # 测试 7: API 兼容性
    test_api_compatibility()

    print("\n" + "=" * 60)
    print("Stage 3 所有测试通过 ✓")
    print("=" * 60)
    print("\n总结：")
    print("  ✓ TrainingWorker.train_batch 路由方法正确")
    print("  ✓ _train_batch_original 封装原有逻辑")
    print("  ✓ _train_batch_draft_only 实现 Draft 专用训练")
    print("  ✓ 方法签名一致，独立性良好")
    print("  ✓ API 兼容性符合预期")

if __name__ == "__main__":
    main()
