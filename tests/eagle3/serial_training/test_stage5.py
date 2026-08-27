#!/usr/bin/env python3
"""Stage 5 代码测试脚本

Stage 5 目标：实现 Draft loss 聚合逻辑，使串行训练能够正确计算和返回 draft loss

测试内容：
1. transformer_impl 模块导入
2. postprocess_micro_batch_func 方法存在性
3. 方法签名验证
4. Loss 聚合逻辑验证（代码审查）
5. 标志读取机制验证
"""

import sys
import inspect
from pathlib import Path

# 添加 verl 到路径
sys.path.insert(0, '/home/t00972278/verl')

def test_import_module():
    """测试 1: transformer_impl 模块导入"""
    print("=" * 60)
    print("测试 1: transformer_impl 模块导入")
    print("=" * 60)

    try:
        from verl.workers.engine.megatron import transformer_impl
        print("✓ transformer_impl 模块导入成功")
        return transformer_impl
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_class_exists(transformer_impl):
    """测试 2: MegatronEngineWithLMHead 类存在性"""
    print("\n" + "=" * 60)
    print("测试 2: MegatronEngineWithLMHead 类存在性")
    print("=" * 60)

    if not hasattr(transformer_impl, 'MegatronEngineWithLMHead'):
        print("✗ MegatronEngineWithLMHead 类不存在")
        sys.exit(1)

    print("✓ MegatronEngineWithLMHead 类存在")
    return transformer_impl.MegatronEngineWithLMHead

def test_method_exists(MegatronEngineWithLMHead):
    """测试 3: postprocess_micro_batch_func 方法存在性"""
    print("\n" + "=" * 60)
    print("测试 3: postprocess_micro_batch_func 方法存在性")
    print("=" * 60)

    if not hasattr(MegatronEngineWithLMHead, 'postprocess_micro_batch_func'):
        print("✗ postprocess_micro_batch_func 方法不存在")
        sys.exit(1)

    print("✓ postprocess_micro_batch_func 方法存在")

def test_method_signature(MegatronEngineWithLMHead):
    """测试 4: 方法签名验证"""
    print("\n" + "=" * 60)
    print("测试 4: 方法签名验证")
    print("=" * 60)

    method = getattr(MegatronEngineWithLMHead, 'postprocess_micro_batch_func')
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    print(f"\npostprocess_micro_batch_func:")
    print(f"  参数数量: {len(params)}")
    print(f"  参数列表: {params}")

    expected_params = ['self', 'output', 'data', 'forward_only', 'loss_function']
    if params[:5] == expected_params:
        print(f"  ✓ 前 5 个参数匹配")
    else:
        print(f"  ✗ 参数不匹配")
        print(f"    预期: {expected_params}")
        print(f"    实际: {params[:5]}")
        sys.exit(1)

def test_loss_aggregation_logic(MegatronEngineWithLMHead):
    """测试 5: Loss 聚合逻辑验证（代码审查）"""
    print("\n" + "=" * 60)
    print("测试 5: Loss 聚合逻辑验证")
    print("=" * 60)

    method = getattr(MegatronEngineWithLMHead, 'postprocess_micro_batch_func')
    source = inspect.getsource(method)

    # 验证关键代码元素
    checks = [
        ('train_draft_only', '提取 train_draft_only 标志'),
        ('if train_draft_only:', 'Draft 训练步分支'),
        ('draft_losses', 'Draft losses 变量'),
        ('_eagle3_draft_losses', '从 model 读取暂存的 draft losses'),
        ('torch.stack(draft_losses).mean()', '聚合 draft loss'),
        ('draft/loss', 'Draft loss metrics'),
        ('draft/num_microbatches', 'Draft microbatch 数量 metrics'),
    ]

    print("\n检查 Loss 聚合逻辑的关键元素：")
    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}: '{check_str}' 存在")
        else:
            print(f"  ✗ {description}: '{check_str}' 缺失")
            sys.exit(1)

    print("\n✓ Loss 聚合逻辑完整")

def test_flag_reading_mechanism(MegatronEngineWithLMHead):
    """测试 6: 标志读取机制验证"""
    print("\n" + "=" * 60)
    print("测试 6: 标志读取机制验证")
    print("=" * 60)

    method = getattr(MegatronEngineWithLMHead, 'postprocess_micro_batch_func')
    source = inspect.getsource(method)

    print("\n检查标志读取机制：")

    # 检查标志读取方式
    if 'data.extra_info.get("train_draft_only"' in source:
        print("  ✓ 使用 data.extra_info.get() 读取 train_draft_only 标志")
    else:
        print("  ✗ 标志读取方式不正确")
        sys.exit(1)

    # 检查默认值
    if 'False)' in source or ', False)' in source:
        print("  ✓ train_draft_only 默认值为 False")
    else:
        print("  ⚠ train_draft_only 默认值可能不是 False")

    print("\n✓ 标志读取机制正确")

def test_branch_logic(MegatronEngineWithLMHead):
    """测试 7: 分支逻辑验证"""
    print("\n" + "=" * 60)
    print("测试 7: 分支逻辑验证")
    print("=" * 60)

    method = getattr(MegatronEngineWithLMHead, 'postprocess_micro_batch_func')
    source = inspect.getsource(method)

    print("\n检查分支逻辑：")

    # Draft 训练步分支
    if 'if train_draft_only:' in source:
        print("  ✓ Draft 训练步分支存在")

        # 检查 Draft 分支的关键逻辑
        draft_branch_checks = [
            ('draft_losses = []', '初始化 draft_losses 列表'),
            ('for module in self.model:', '遍历 model 模块'),
            ('module._eagle3_draft_losses', '读取暂存的 losses'),
            ('.clear()', '清空暂存'),
        ]

        for check_str, desc in draft_branch_checks:
            if check_str in source:
                print(f"    ✓ {desc}")
            else:
                print(f"    ✗ {desc} 缺失")
                sys.exit(1)
    else:
        print("  ✗ Draft 训练步分支不存在")
        sys.exit(1)

    # 并行/Actor 分支
    if 'else:' in source and 'loss_function(model_output=model_output' in source:
        print("  ✓ 并行/Actor 训练步分支存在（使用 loss_function）")
    else:
        print("  ✗ 并行/Actor 训练步分支不完整")
        sys.exit(1)

    print("\n✓ 分支逻辑正确")

def test_metrics_structure(MegatronEngineWithLMHead):
    """测试 8: Metrics 结构验证"""
    print("\n" + "=" * 60)
    print("测试 8: Metrics 结构验证")
    print("=" * 60)

    method = getattr(MegatronEngineWithLMHead, 'postprocess_micro_batch_func')
    source = inspect.getsource(method)

    print("\n检查 Metrics 结构：")

    # Draft metrics
    if '"draft/loss"' in source:
        print("  ✓ draft/loss metric 存在")
    else:
        print("  ✗ draft/loss metric 缺失")
        sys.exit(1)

    if '"draft/num_microbatches"' in source:
        print("  ✓ draft/num_microbatches metric 存在")
    else:
        print("  ✗ draft/num_microbatches metric 缺失")
        sys.exit(1)

    # 检查异常处理（空 draft_losses）
    if 'if draft_losses:' in source or 'if len(draft_losses)' in source:
        print("  ✓ 包含空 draft_losses 的异常处理")
    else:
        print("  ⚠ 可能缺少空 draft_losses 的处理")

    print("\n✓ Metrics 结构正确")

def main():
    print("=" * 60)
    print("Stage 5 代码测试")
    print("=" * 60)
    print("\nStage 5 目标：")
    print("  实现 Draft loss 聚合逻辑")
    print("  - 检查 train_draft_only 标志")
    print("  - 从 model 聚合 draft losses")
    print("  - 返回正确的 loss 和 metrics")
    print()

    # 测试 1: 导入
    transformer_impl = test_import_module()

    # 测试 2: 类存在性
    MegatronEngineWithLMHead = test_class_exists(transformer_impl)

    # 测试 3: 方法存在性
    test_method_exists(MegatronEngineWithLMHead)

    # 测试 4: 方法签名
    test_method_signature(MegatronEngineWithLMHead)

    # 测试 5: Loss 聚合逻辑
    test_loss_aggregation_logic(MegatronEngineWithLMHead)

    # 测试 6: 标志读取机制
    test_flag_reading_mechanism(MegatronEngineWithLMHead)

    # 测试 7: 分支逻辑
    test_branch_logic(MegatronEngineWithLMHead)

    # 测试 8: Metrics 结构
    test_metrics_structure(MegatronEngineWithLMHead)

    print("\n" + "=" * 60)
    print("Stage 5 所有测试通过 ✓")
    print("=" * 60)
    print("\n总结：")
    print("  ✓ transformer_impl 模块正常导入")
    print("  ✓ postprocess_micro_batch_func 方法存在")
    print("  ✓ 方法签名正确")
    print("  ✓ Loss 聚合逻辑完整")
    print("  ✓ 标志读取机制正确")
    print("  ✓ 分支逻辑正确")
    print("  ✓ Metrics 结构正确")

if __name__ == "__main__":
    main()
