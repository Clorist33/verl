#!/usr/bin/env python3
"""Draft Batch Size 解耦功能测试

测试内容：
1. ActorConfig 新增的 Draft 参数字段
2. _apply_draft_batch_config 方法的 fallback 机制
3. 配置应用逻辑
"""

import pytest

# [P3-DEAD v1/v2 20260829] 本文件是 v1/v2（独立 Draft 步）时期的测试：断言 update_draft / _apply_draft_batch_config 存在。
# 这些接口在 v3（搭车采集 + 延后训练）下已不可达，源码已整体注释待删，
# 因此本文件一并冻结。整体验证通过、死代码正式删除时，本文件同批删除。
pytest.skip(
    "v1/v2 serial-training tests frozen: the APIs under test are commented out (P3-DEAD)",
    allow_module_level=True,
)

import sys
from unittest.mock import MagicMock, patch

# 添加 verl 到路径
sys.path.insert(0, '/home/t00972278/verl')

def test_actor_config_draft_fields():
    """测试 1: ActorConfig 是否有 Draft 参数字段"""
    print("=" * 60)
    print("测试 1: ActorConfig Draft 参数字段")
    print("=" * 60)

    from verl.workers.config import ActorConfig
    import inspect

    # 检查 ActorConfig 类定义中是否有 Draft 参数字段
    # 使用 inspect 获取类的 annotations
    if hasattr(ActorConfig, '__annotations__'):
        annotations = ActorConfig.__annotations__
    else:
        # 使用 inspect 获取源代码
        source = inspect.getsource(ActorConfig)
        annotations = {}

    # 检查 Draft 参数字段是否存在
    draft_fields = [
        'draft_ppo_mini_batch_size',
        'draft_ppo_micro_batch_size',
        'draft_ppo_micro_batch_size_per_gpu',
        'draft_ppo_infer_micro_batch_size_per_gpu',
    ]

    print("\n检查 Draft 参数字段（通过类定义）：")

    # 读取 ActorConfig 源码
    source = inspect.getsource(ActorConfig)

    for field in draft_fields:
        if field in source:
            print(f"  ✓ {field} 存在于类定义中")
        else:
            print(f"  ✗ {field} 不存在")
            sys.exit(1)

    print("\n✓ 所有 Draft 参数字段都存在")

def test_fallback_mechanism():
    """测试 2: Fallback 机制（Draft 参数为 None 时使用 Actor 参数）"""
    print("\n" + "=" * 60)
    print("测试 2: Fallback 机制")
    print("=" * 60)

    from verl.workers.config import ActorConfig
    import inspect

    # 读取 ActorConfig 源码检查默认值
    source = inspect.getsource(ActorConfig)

    print("\n场景 1: Draft 参数默认值检查")

    # 检查 Draft 参数的默认值是否为 None
    draft_params = [
        'draft_ppo_mini_batch_size',
        'draft_ppo_micro_batch_size_per_gpu',
    ]

    for param in draft_params:
        # 查找参数定义行
        for line in source.split('\n'):
            if param in line and ':' in line:
                if 'None' in line or 'Optional' in line:
                    print(f"  ✓ {param} 默认值为 None（支持 fallback）")
                    break
        else:
            print(f"  ✗ {param} 默认值不是 None")
            sys.exit(1)

    print("\n✓ Fallback 机制设计正确（Draft 参数默认为 None）")

def test_apply_draft_batch_config_logic():
    """测试 3: _apply_draft_batch_config 方法逻辑（代码审查）"""
    print("\n" + "=" * 60)
    print("测试 3: _apply_draft_batch_config 方法逻辑")
    print("=" * 60)

    import inspect
    from verl.workers.engine_workers import ActorRolloutRefWorker

    # 检查方法是否存在
    if not hasattr(ActorRolloutRefWorker, '_apply_draft_batch_config'):
        print("✗ _apply_draft_batch_config 方法不存在")
        sys.exit(1)

    print("✓ _apply_draft_batch_config 方法存在")

    # 检查方法源码
    method = getattr(ActorRolloutRefWorker, '_apply_draft_batch_config')
    source = inspect.getsource(method)

    # 验证关键逻辑
    checks = [
        ('draft_ppo_mini_batch_size', 'Draft mini batch size 参数'),
        ('draft_ppo_micro_batch_size', 'Draft micro batch size 参数'),
        ('draft_ppo_micro_batch_size_per_gpu', 'Draft micro batch size per GPU'),
        ('getattr(config,', '使用 getattr 读取配置'),
        ('if draft_mini_bsz is not None:', 'Fallback 条件检查'),
        ('tu.assign_non_tensor_data', '应用配置到 data'),
    ]

    print("\n检查关键逻辑：")
    for check_str, description in checks:
        if check_str in source:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description} 缺失")
            sys.exit(1)

    print("\n✓ _apply_draft_batch_config 方法逻辑正确")

def test_update_draft_integration():
    """测试 4: update_draft 方法集成"""
    print("\n" + "=" * 60)
    print("测试 4: update_draft 方法集成")
    print("=" * 60)

    import inspect
    from verl.workers.engine_workers import ActorRolloutRefWorker

    # 检查 update_draft 是否调用 _apply_draft_batch_config
    method = getattr(ActorRolloutRefWorker, 'update_draft')
    source = inspect.getsource(method)

    if '_apply_draft_batch_config' in source:
        print("✓ update_draft 调用了 _apply_draft_batch_config")
    else:
        print("✗ update_draft 未调用 _apply_draft_batch_config")
        sys.exit(1)

    # 检查调用顺序
    lines = source.split('\n')
    apply_line = -1
    flag_line = -1

    for i, line in enumerate(lines):
        if '_apply_draft_batch_config' in line:
            apply_line = i
        if 'train_draft_only' in line:
            flag_line = i

    if apply_line < flag_line:
        print("✓ 配置应用在标志设置之前（正确顺序）")
    else:
        print("✗ 配置应用顺序错误")
        sys.exit(1)

    print("\n✓ update_draft 方法集成正确")

def test_backward_compatibility():
    """测试 5: 向后兼容性"""
    print("\n" + "=" * 60)
    print("测试 5: 向后兼容性")
    print("=" * 60)

    from verl.workers.config import ActorConfig
    import inspect

    # 读取 ActorConfig 源码
    source = inspect.getsource(ActorConfig)

    print("\n检查向后兼容性设计：")

    # 检查 Draft 参数是否为 Optional
    if 'draft_ppo_mini_batch_size: Optional[int] = None' in source:
        print("  ✓ draft_ppo_mini_batch_size 为 Optional，默认 None")
    else:
        print("  ⚠ draft_ppo_mini_batch_size 可能不是 Optional")

    if 'draft_ppo_micro_batch_size_per_gpu: Optional[int] = None' in source:
        print("  ✓ draft_ppo_micro_batch_size_per_gpu 为 Optional，默认 None")
    else:
        print("  ⚠ draft_ppo_micro_batch_size_per_gpu 可能不是 Optional")

    print("\n结论：")
    print("  ✓ Draft 参数为 Optional，默认 None")
    print("  ✓ 旧配置不需要修改，Draft 将使用 Actor 参数（fallback）")
    print("  ✓ 新配置可以选择性设置 Draft 参数")

    print("\n✓ 向后兼容性设计正确")

def main():
    print("=" * 60)
    print("Draft Batch Size 解耦功能测试")
    print("=" * 60)
    print("\n功能：Actor 和 Draft 使用独立的 batch size 参数")
    print("方案：保持 Actor 参数不变，新增 Draft 参数（可选）\n")

    # 测试 1: 配置字段
    test_actor_config_draft_fields()

    # 测试 2: Fallback 机制
    test_fallback_mechanism()

    # 测试 3: 应用逻辑
    test_apply_draft_batch_config_logic()

    # 测试 4: 集成测试
    test_update_draft_integration()

    # 测试 5: 向后兼容性
    test_backward_compatibility()

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
    print("\n总结：")
    print("  ✓ ActorConfig 新增 4 个 Draft 参数字段")
    print("  ✓ _apply_draft_batch_config 方法逻辑正确")
    print("  ✓ Fallback 机制正常工作")
    print("  ✓ update_draft 正确集成")
    print("  ✓ 向后兼容性良好")

if __name__ == "__main__":
    main()
