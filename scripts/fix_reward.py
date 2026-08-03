#!/usr/bin/env python3
"""GRPO/GEPO 奖励函数连续化工具 (治根因A)
将离散奖励改为按误差连续递减, 避免组内奖励打平导致空转。

用法:
  python fix_reward.py --test          # 单测: 8条不同答案验证 std > 0
  python fix_reward.py --example       # 打印改造前后对比示例
"""
import numpy as np, sys

# ============ 改造前: 离散奖励 (空转元凶) ============
def discrete_reward(pred, gold):
    if abs(pred - gold) < 1e-6:
        return 1.0
    elif abs(pred - gold) / max(abs(gold), 1.0) < 0.1:
        return 0.3
    else:
        return 0.0

# ============ 改造后: 连续奖励 ============
def continuous_reward(pred, gold):
    """误差连续递减: 100% 误差才归零, 避免错误答案被压成同一分值"""
    rel_err = abs(pred - gold) / max(abs(gold), 1.0)
    return 1.0 * max(0.0, 1.0 - min(rel_err, 1.0))

def _continuous_reward_factory(scale=1.0, threshold=1.0):
    """可配置版: 阈值放宽到 threshold 倍误差才归零"""
    def reward(pred, gold):
        rel_err = abs(pred - gold) / max(abs(gold), 1.0)
        return scale * max(0.0, 1.0 - min(rel_err / threshold, 1.0))
    return reward

# ============ 单测 ============
def unit_test():
    print("=== 单测: 8条不同completion的奖励方差 ===")
    gold = 100.0
    preds = [100.0, 99.0, 95.0, 90.0, 80.0, 50.0, 0.0, 200.0]

    disc = np.array([discrete_reward(p, gold) for p in preds])
    cont = np.array([continuous_reward(p, gold) for p in preds])

    print(f"{'答案':>8} {'离散奖励':>8} {'连续奖励':>8}")
    for p, d, c in zip(preds, disc, cont):
        print(f"{p:>8} {d:>8.2f} {c:>8.3f}")

    print(f"\n离散 std = {disc.std():.4f}  {'⚠️ 组内打平→空转!' if disc.std()==0 else '✅'}")
    print(f"连续 std = {cont.std():.4f}  {'✅ 奖励有方差, 可学习' if cont.std()>0 else '⚠️ 仍打平'}")
    return cont.std() > 0

def example():
    print("=== 改造示例 ===")
    print("错误答案从固定 0.3 改为: 0.3 * max(0.0, 1.0 - min(rel_err, 1.0))")
    print("其中 rel_err = |pred - gold| / max(|gold|, 1.0)")
    print("\n阈值放宽到 100% 误差才归零, 避免大部分错误答案又被压成 0")

if __name__ == '__main__':
    if '--test' in sys.argv:
        ok = unit_test()
        sys.exit(0 if ok else 1)
    elif '--example' in sys.argv:
        example()
    else:
        ok = unit_test()
        print()
        example()
        sys.exit(0 if ok else 1)
