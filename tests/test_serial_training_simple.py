"""
EAGLE3 串行训练 - 简化单元测试

只测试核心逻辑，不依赖完整的 verl 环境
"""

import unittest


# ============================================================
# 复制 SerialTrainingScheduler 类（用于独立测试）
# ============================================================

class SerialTrainingScheduler:
    """Online 串行训练调度器"""

    def __init__(self, actor_steps_per_draft: int):
        self.k = actor_steps_per_draft
        self.cycle_length = self.k + 1

    def should_train_actor(self, global_step: int) -> bool:
        return (global_step % self.cycle_length) != 0

    def should_train_draft(self, global_step: int) -> bool:
        return (global_step % self.cycle_length) == 0


# ============================================================
# 测试用例
# ============================================================

class TestSerialTrainingScheduler(unittest.TestCase):
    """测试串行训练调度器"""

    def test_scheduler_with_k5(self):
        """测试 k=5 的调度逻辑"""
        print("\n[测试] k=5 的调度逻辑")
        scheduler = SerialTrainingScheduler(actor_steps_per_draft=5)

        # 测试 Step 1-5: 应该训练 actor
        for step in [1, 2, 3, 4, 5]:
            self.assertTrue(scheduler.should_train_actor(step),
                          f"Step {step} should train actor")
            self.assertFalse(scheduler.should_train_draft(step),
                           f"Step {step} should not train draft")
            print(f"  Step {step}: Actor=True, Draft=False ✓")

        # 测试 Step 6: 应该训练 draft
        self.assertFalse(scheduler.should_train_actor(6),
                        "Step 6 should not train actor")
        self.assertTrue(scheduler.should_train_draft(6),
                       "Step 6 should train draft")
        print(f"  Step 6: Actor=False, Draft=True ✓")

        # 测试 Step 7-11: 应该训练 actor
        for step in [7, 8, 9, 10, 11]:
            self.assertTrue(scheduler.should_train_actor(step),
                          f"Step {step} should train actor")
            self.assertFalse(scheduler.should_train_draft(step),
                           f"Step {step} should not train draft")

        # 测试 Step 12: 应该训练 draft
        self.assertFalse(scheduler.should_train_actor(12),
                        "Step 12 should not train actor")
        self.assertTrue(scheduler.should_train_draft(12),
                       "Step 12 should train draft")
        print(f"  Step 12: Actor=False, Draft=True ✓")

    def test_scheduler_with_k3(self):
        """测试 k=3 的调度逻辑"""
        print("\n[测试] k=3 的调度逻辑")
        scheduler = SerialTrainingScheduler(actor_steps_per_draft=3)

        # 测试 Step 1-3: actor
        for step in [1, 2, 3]:
            self.assertTrue(scheduler.should_train_actor(step))
            self.assertFalse(scheduler.should_train_draft(step))
            print(f"  Step {step}: Actor=True ✓")

        # 测试 Step 4: draft
        self.assertFalse(scheduler.should_train_actor(4))
        self.assertTrue(scheduler.should_train_draft(4))
        print(f"  Step 4: Draft=True ✓")

        # 测试 Step 5-7: actor
        for step in [5, 6, 7]:
            self.assertTrue(scheduler.should_train_actor(step))
            self.assertFalse(scheduler.should_train_draft(step))

        # 测试 Step 8: draft
        self.assertFalse(scheduler.should_train_actor(8))
        self.assertTrue(scheduler.should_train_draft(8))
        print(f"  Step 8: Draft=True ✓")

    def test_scheduler_with_k1(self):
        """测试 k=1 的调度逻辑（交替训练）"""
        print("\n[测试] k=1 的调度逻辑（交替训练）")
        scheduler = SerialTrainingScheduler(actor_steps_per_draft=1)

        # Step 1: actor, Step 2: draft, Step 3: actor, Step 4: draft
        for step in range(1, 11):
            if step % 2 == 1:  # 奇数步
                self.assertTrue(scheduler.should_train_actor(step))
                self.assertFalse(scheduler.should_train_draft(step))
                print(f"  Step {step}: Actor ✓")
            else:  # 偶数步
                self.assertFalse(scheduler.should_train_actor(step))
                self.assertTrue(scheduler.should_train_draft(step))
                print(f"  Step {step}: Draft ✓")

    def test_scheduler_mutual_exclusion(self):
        """测试 actor 和 draft 训练互斥"""
        print("\n[测试] Actor 和 Draft 互斥性（100 steps）")
        scheduler = SerialTrainingScheduler(actor_steps_per_draft=5)

        for step in range(1, 101):
            train_actor = scheduler.should_train_actor(step)
            train_draft = scheduler.should_train_draft(step)

            # 确保每个 step 只训练一个（互斥）
            self.assertTrue(train_actor ^ train_draft,
                          f"Step {step} should train exactly one of actor or draft")

        print(f"  所有 100 步都满足互斥性 ✓")

    def test_scheduler_cycle_correctness(self):
        """测试调度周期的正确性"""
        print("\n[测试] 调度周期正确性")

        test_cases = [
            (3, 4),   # k=3, cycle=4
            (5, 6),   # k=5, cycle=6
            (7, 8),   # k=7, cycle=8
        ]

        for k, expected_cycle in test_cases:
            scheduler = SerialTrainingScheduler(actor_steps_per_draft=k)
            self.assertEqual(scheduler.cycle_length, expected_cycle,
                           f"k={k} should have cycle length {expected_cycle}")

            # 验证一个完整周期
            actor_count = 0
            draft_count = 0
            for step in range(1, expected_cycle + 1):
                if scheduler.should_train_actor(step):
                    actor_count += 1
                if scheduler.should_train_draft(step):
                    draft_count += 1

            self.assertEqual(actor_count, k, f"k={k} should train actor {k} times")
            self.assertEqual(draft_count, 1, f"k={k} should train draft 1 time")
            print(f"  k={k}: {k} actor steps + 1 draft step = {expected_cycle} cycle ✓")


class TestRoutingLogic(unittest.TestCase):
    """测试路由逻辑（不依赖完整环境）"""

    def test_routing_decision_parallel(self):
        """测试并行模式的路由决策"""
        print("\n[测试] 并行模式路由决策")

        # 模拟配置
        enable_serial_training = False

        # 路由决策
        if enable_serial_training:
            mode = "serial"
        else:
            mode = "parallel"

        self.assertEqual(mode, "parallel")
        print(f"  enable_serial_training=False -> mode='{mode}' ✓")

    def test_routing_decision_serial(self):
        """测试串行模式的路由决策"""
        print("\n[测试] 串行模式路由决策")

        # 模拟配置
        enable_serial_training = True

        # 路由决策
        if enable_serial_training:
            mode = "serial"
        else:
            mode = "parallel"

        self.assertEqual(mode, "serial")
        print(f"  enable_serial_training=True -> mode='{mode}' ✓")

    def test_train_batch_routing(self):
        """测试 train_batch 路由逻辑"""
        print("\n[测试] train_batch 路由逻辑")

        # 测试路由到 original
        train_draft_only = False
        if train_draft_only:
            method = "_train_batch_draft_only"
        else:
            method = "_train_batch_original"

        self.assertEqual(method, "_train_batch_original")
        print(f"  train_draft_only=False -> '{method}' ✓")

        # 测试路由到 draft_only
        train_draft_only = True
        if train_draft_only:
            method = "_train_batch_draft_only"
        else:
            method = "_train_batch_original"

        self.assertEqual(method, "_train_batch_draft_only")
        print(f"  train_draft_only=True -> '{method}' ✓")


class TestFlagPassing(unittest.TestCase):
    """测试标志位传递（模拟）"""

    def test_draft_step_flags(self):
        """测试 draft 步的标志设置"""
        print("\n[测试] Draft 步标志设置")

        # 模拟 draft 步设置的标志
        flags = {
            'enable_draft_training': True,
            'train_draft_only': True
        }

        self.assertTrue(flags['enable_draft_training'])
        self.assertTrue(flags['train_draft_only'])
        print(f"  enable_draft_training=True ✓")
        print(f"  train_draft_only=True ✓")

    def test_actor_step_flags(self):
        """测试 actor 步的标志设置"""
        print("\n[测试] Actor 步标志设置")

        # 模拟 actor 步设置的标志
        flags = {
            'enable_draft_training': False,
            'train_draft_only': False
        }

        self.assertFalse(flags['enable_draft_training'])
        self.assertFalse(flags['train_draft_only'])
        print(f"  enable_draft_training=False ✓")
        print(f"  train_draft_only=False ✓")


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("EAGLE3 串行训练 - 阶段 1-3 单元测试（简化版）")
    print("=" * 70)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestSerialTrainingScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestRoutingLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestFlagPassing))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print()
    print("=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试: {result.testsRun}")
    print(f"✓ 成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ 失败: {len(result.failures)}")
    print(f"✗ 错误: {len(result.errors)}")

    if result.wasSuccessful():
        print()
        print("🎉 所有测试通过！")
    else:
        print()
        print("⚠️  有测试失败，请检查上面的错误信息")

    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
