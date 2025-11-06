"""
修复轮换逻辑 - 解决轮换有效性低的问题

问题：轮换有效性只有0.0478，说明因子组合几乎不变化
目标：让轮换有效性达到0.8+
"""

import re
import os
import shutil
from datetime import datetime


def backup_file(filepath):
    """备份文件"""
    backup_path = f"{filepath}.backup_rotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"  已备份: {backup_path}")
    return backup_path


def fix_rotation_logic():
    """修复 factorized_attacker.py 中的轮换逻辑"""
    filepath = 'fl_utils/factorized_attacker.py'
    
    if not os.path.exists(filepath):
        print(f"错误: 找不到 {filepath}")
        return False
    
    print(f"正在修复 {filepath}...")
    
    # 备份
    backup_path = backup_file(filepath)
    
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到需要修复的函数
    # 查找 assign_factor_combination 方法
    pattern = r"(elif self\.rotation_strategy == 'adversary_specific':.*?combination = \[.*?\])"
    
    # 新的实现
    new_implementation = """elif self.rotation_strategy == 'adversary_specific':
            # 修复：确保每个epoch、每个攻击者都有真正不同的组合
            # 使用 adversary_id 和 epoch 创建唯一的偏移
            
            # 计算基础偏移（每个攻击者不同）
            base_offset = adversary_id * self.m
            
            # 计算epoch偏移（每个epoch变化）
            epoch_offset = epoch % (len(self.factor_library) - self.m + 1)
            
            # 组合索引（确保在有效范围内）
            combination = []
            for i in range(self.m):
                factor_idx = (base_offset + epoch_offset + i) % len(self.factor_library)
                combination.append(factor_idx)"""
    
    # 替换（使用更精确的匹配）
    original_content = content
    
    # 尝试替换
    if "elif self.rotation_strategy == 'adversary_specific':" in content:
        # 找到整个elif块
        start = content.find("elif self.rotation_strategy == 'adversary_specific':")
        if start != -1:
            # 找到下一个elif或else的位置
            end_markers = [
                content.find("\n        elif ", start + 10),
                content.find("\n        else:", start + 10),
            ]
            end_markers = [e for e in end_markers if e != -1]
            
            if end_markers:
                end = min(end_markers)
            else:
                # 找到函数结束
                end = content.find("\n    def ", start + 10)
            
            if end != -1:
                # 提取并替换
                old_block = content[start:end]
                content = content[:start] + new_implementation + content[end:]
                
                print("  ✓ 已修复 adversary_specific 轮换策略")
            else:
                print("  ✗ 无法定位代码块结束位置")
                return False
        else:
            print("  ✗ 无法定位代码块起始位置")
            return False
    else:
        print("  ⚠️  未找到 adversary_specific 策略代码")
        print("  可能已经修改过或版本不同")
        return False
    
    # 检查是否有实际修改
    if content == original_content:
        print("  ⚠️  没有进行修改")
        os.remove(backup_path)
        return False
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  ✓ 修复完成")
    return True


def verify_fix():
    """验证修复"""
    filepath = 'fl_utils/factorized_attacker.py'
    
    print(f"\n验证 {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查新代码是否存在
    if "# 修复：确保每个epoch、每个攻击者都有真正不同的组合" in content:
        print("  ✓ 新的轮换逻辑已应用")
        return True
    else:
        print("  ✗ 验证失败，新代码未找到")
        return False


def create_test_script():
    """创建测试脚本"""
    test_code = '''"""
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

print(f"\\n因子库大小: {len(attacker.factor_library)}")
print(f"k-of-m规则: {attacker.k}-of-{attacker.m}\\n")

# 测试3个攻击者在10个epoch的组合
print("测试轮换效果:")
print("-" * 70)

for adversary_id in range(3):
    print(f"\\nAdversary {adversary_id}:")
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

print("\\n" + "="*70)
print("测试完成")
print("="*70)

# 检查不同攻击者的组合是否不同
print("\\n检查攻击者间的差异:")
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

print(f"\\n  重叠情况:")
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

print("\\n" + "="*70)
'''
    
    with open('test_rotation.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("  ✓ 已创建测试脚本: test_rotation.py")


def main():
    """主函数"""
    print("="*70)
    print("修复轮换逻辑")
    print("="*70)
    print("\n问题: 轮换有效性只有0.0478（目标0.8+）")
    print("原因: 因子组合在不同epoch之间几乎不变化")
    print("解决: 改进 assign_factor_combination() 的实现\n")
    
    # 1. 修复代码
    if not fix_rotation_logic():
        print("\n修复失败！")
        print("请手动检查 fl_utils/factorized_attacker.py")
        return False
    
    # 2. 验证修复
    if not verify_fix():
        print("\n验证失败！")
        return False
    
    # 3. 创建测试脚本
    create_test_script()
    
    # 4. 总结
    print("\n" + "="*70)
    print("✓ 修复完成！")
    print("="*70)
    print("\n下一步:")
    print("1. 运行测试验证修复:")
    print("   python test_rotation.py")
    print("\n2. 如果测试通过，使用优化配置重新训练:")
    print("   python main_train_factorized.py --gpu 0 --params configs/optimized.yaml")
    print("\n3. 在 Epoch 30 检查结果:")
    print("   - 轮换有效性应该 > 0.5")
    print("   - 主任务准确率应该 > 70%")
    print("="*70)
    
    return True


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
