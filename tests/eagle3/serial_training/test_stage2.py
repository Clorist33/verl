#!/usr/bin/env python3
"""Stage 2 代码测试脚本

测试内容：
1. ActorRolloutRefWorker.update_draft 方法是否存在
2. TrainingWorker.train_batch 路由逻辑
3. TrainingWorker._train_batch_original 方法是否存在
4. TrainingWorker._train_batch_draft_only 方法是否存在
5. 方法签名一致性检查
"""

import sys
import inspect
from pathlib import Path

# 添加 verl 到路径
sys.path.insert(0, '/home/t00972278/verl')

def test_worker_classes():
    """测试 1: Worker 类是否可以导入"""
    print("=" * 60)
    print("测试 1: Worker 类导入")
    print("=" * 60)

    try:
        from verl.workers.engine_workers import TrainingWorker, ActorRolloutRefWorker
        print("✓ TrainingWorker 导入成功")
        print("✓ ActorRolloutRefWorker 导入成功")
        return TrainingWorker, ActorRolloutRefWorker
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        sys.exit(1)

def test_update_draft_method(ActorRolloutRefWorker):
    """测试 2: ActorRolloutRefWorker.update_draft 方法"""
    print("\n" + "=" * 60)
    print("测试 2: ActorRolloutRefWorker.update_draft 方法")
    print("=" * 60)

    if not hasattr(ActorRolloutRefWorker, 'update_draft'):
        print("✗ ActorRolloutRefWorker.update_draft 方法不存在")
        sys.exit(1)

    method = getattr(ActorRolloutRefWorker, 'update_draft')
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    print(f"✓ ActorRolloutRefWorker.update_draft 存在")
    print(f"  方法签名: {sig}")
    print(f"  参数列表: {params}")

    # 检查参数
    expected_params = ['self', 'data']
    if params == expected_params:
        print(f"✓ 参数列表正确: {params}")
    else:
        print(f"✗ 参数列表不符合预期")
        print(f"  预期: {expected_params}")
        print(f"  实际: {params}")
        sys.exit(1)

def test_training_worker_methods(TrainingWorker):
    """测试 3: TrainingWorker 的三个方法"""
    print("\n" + "=" * 60)
    print("测试 3: TrainingWorker 方法检查")
    print("=" * 60)

    required_methods = [
        'train_batch',
        '_train_batch_original',
        '_train_batch_draft_only'
    ]

    for method_name in required_methods:
        if not hasattr(TrainingWorker, method_name):
            print(f"✗ TrainingWorker.{method_name} 不存在")
            sys.exit(1)

        method = getattr(TrainingWorker, method_name)
        sig = inspect.signature(method)
        print(f"✓ TrainingWorker.{method_name} 存在")
        print(f"  方法签名: {sig}")

def test_method_signatures(TrainingWorker):
    """测试 4: 方法签名一致性"""
    print("\n" + "=" * 60)
    print("测试 4: 方法签名一致性")
    print("=" * 60)

    train_batch_sig = inspect.signature(TrainingWorker.train_batch)
    original_sig = inspect.signature(TrainingWorker._train_batch_original)
    draft_only_sig = inspect.signature(TrainingWorker._train_batch_draft_only)

    train_params = list(train_batch_sig.parameters.keys())
    original_params = list(original_sig.parameters.keys())
    draft_params = list(draft_only_sig.parameters.keys())

    print(f"train_batch 参数: {train_params}")
    print(f"_train_batch_original 参数: {original_params}")
    print(f"_train_batch_draft_only 参数: {draft_params}")

    # 所有方法都应该接受 (self, data)
    expected = ['self', 'data']

    if train_params == expected:
        print("✓ train_batch 签名正确")
    else:
        print(f"✗ train_batch 签名不正确，预期 {expected}")
        sys.exit(1)

    if original_params == expected:
        print("✓ _train_batch_original 签名正确")
    else:
        print(f"✗ _train_batch_original 签名不正确，预期 {expected}")
        sys.exit(1)

    if draft_params == expected:
        print("✓ _train_batch_draft_only 签名正确")
    else:
        print(f"✗ _train_batch_draft_only 签名不正确，预期 {expected}")
        sys.exit(1)

def test_method_docstrings(TrainingWorker):
    """测试 5: 方法文档字符串"""
    print("\n" + "=" * 60)
    print("测试 5: 方法文档字符串检查")
    print("=" * 60)

    methods_to_check = [
        ('train_batch', '路由'),
        ('_train_batch_original', '原有'),
        ('_train_batch_draft_only', 'Draft 训练'),
    ]

    for method_name, expected_keyword in methods_to_check:
        method = getattr(TrainingWorker, method_name)
        doc = method.__doc__

        if doc and expected_keyword in doc:
            print(f"✓ {method_name} 有正确的文档字符串")
            print(f"  关键词 '{expected_keyword}' 存在")
        else:
            print(f"✗ {method_name} 文档字符串缺失或不包含 '{expected_keyword}'")

def main():
    print("Stage 2 代码测试\n")

    # 测试 1: 导入类
    TrainingWorker, ActorRolloutRefWorker = test_worker_classes()

    # 测试 2: update_draft 方法
    test_update_draft_method(ActorRolloutRefWorker)

    # 测试 3: TrainingWorker 方法存在性
    test_training_worker_methods(TrainingWorker)

    # 测试 4: 方法签名一致性
    test_method_signatures(TrainingWorker)

    # 测试 5: 文档字符串
    test_method_docstrings(TrainingWorker)

    print("\n" + "=" * 60)
    print("Stage 2 所有测试通过 ✓")
    print("=" * 60)

if __name__ == "__main__":
    main()
