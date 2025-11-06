"""
测试轮换逻辑 - 验证修复是否生效
"""

from test_utils import create_mock_helper
from fl_utils.factorized_attacker import FactorizedAttacker

print("="*70)
print("测试轮换逻辑")
print("="*70)

# 创建攻击器
helper = create_mock_helper({'num_adversaries': 3})
attacker = FactorizedAttacker(helper)

print(f"\n因子库大小: {len(attacker.factor_library)}")
print(f"k-of-m规则: {attacker.k}-of-{attacker.m}\n")

# 测试3个攻击者在10个epoch的组合
print("测试轮换效果:")
print("-" * 70)

for adversary_id in range(3):
    print(f"\nAdversary {adversary_id}:")
    combinations = []
    
    for epoch in range(10):
        combination = attacker.assign_factor_combination(adversary_id, epoch)
        factor_names = [attacker.factor_library[i].name for i in combination]
        combinations.append(tuple(combination))
        
        if epoch < 5:  # 只打印前5个
            print(f"  Epoch {epoch}: {combination} - {factor_names[:2]}...")
    
    # 统计唯一组合数
    unique = len(set(combinations))
    print(f"  → 10个epoch中有 {unique} 个不同组合")
    
    if unique >= 8:
        print(f"  ✓ 轮换效果良好（变化率{unique/10*100:.0f}%）")
    elif unique >= 5:
        print(f"  ⚠️  轮换效果一般（变化率{unique/10*100:.0f}%）")
    else:
        print(f"  ✗ 轮换效果不佳（变化率{unique/10*100:.0f}%）")

print("\n" + "="*70)
print("测试完成")
print("="*70)

# 检查不同攻击者的组合是否不同
print("\n检查攻击者间的差异:")
print("-" * 70)

epoch = 0
combos = []
for adv_id in range(3):
    combo = attacker.assign_factor_combination(adv_id, epoch)
    combos.append(set(combo))
    print(f"  Adversary {adv_id} (Epoch 0): {combo}")

# 检查是否有重叠
overlap_01 = len(combos[0] & combos[1])
overlap_02 = len(combos[0] & combos[2])
overlap_12 = len(combos[1] & combos[2])

print(f"\n  重叠情况:")
print(f"    Adv0 & Adv1: {overlap_01}/{attacker.m} 个因子重叠")
print(f"    Adv0 & Adv2: {overlap_02}/{attacker.m} 个因子重叠")
print(f"    Adv1 & Adv2: {overlap_12}/{attacker.m} 个因子重叠")

avg_overlap = (overlap_01 + overlap_02 + overlap_12) / 3
if avg_overlap < 1.5:
    print(f"  ✓ 攻击者间差异良好（平均重叠{avg_overlap:.1f}个）")
elif avg_overlap < 2.5:
    print(f"  ⚠️  攻击者间差异一般（平均重叠{avg_overlap:.1f}个）")
else:
    print(f"  ✗ 攻击者间差异不足（平均重叠{avg_overlap:.1f}个）")

print("\n" + "="*70)
