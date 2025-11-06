"""
快速防御验证脚本
快速测试所有防御机制是否正常工作

不需要完整训练，只测试聚合功能
"""

import torch
import numpy as np
import sys
import os

# 确保导入项目文件，而不是第三方库
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'defense_test_framework' in current_dir else current_dir
sys.path.insert(0, project_root)

from fl_defenses import create_defense

# 动态导入ResNet（避免路径问题）
try:
    from models.resnet import ResNet18
except ImportError:
    print("警告: 无法导入ResNet18，使用简化模型")
    import torch.nn as nn
    class ResNet18(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(64, num_classes)
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)


def create_mock_models(num_models=10, num_malicious=3):
    """创建模拟的本地模型"""
    # 创建全局模型
    global_model = ResNet18(num_classes=10).cuda()

    # 创建本地模型（带小扰动）
    local_models = {}

    for i in range(num_models):
        local_model = ResNet18(num_classes=10).cuda()
        local_model.load_state_dict(global_model.state_dict())

        # 添加随机扰动
        with torch.no_grad():
            for param in local_model.parameters():
                if i < num_malicious:
                    # 恶意模型：大扰动
                    noise = torch.randn_like(param) * 0.5
                else:
                    # 良性模型：小扰动
                    noise = torch.randn_like(param) * 0.01

                param.add_(noise)

        local_models[i] = local_model

    participant_ids = list(range(num_models))

    return global_model, local_models, participant_ids


def test_defense(defense_name, config):
    """测试单个防御"""
    print(f"\n{'='*60}")
    print(f"测试防御: {defense_name.upper()}")
    print(f"{'='*60}")

    try:
        # 创建防御
        defense = create_defense(defense_name, config)
        print(f"✓ {defense.name} 初始化成功")

        # 创建模拟模型
        global_model, local_models, participant_ids = create_mock_models()
        print(f"✓ 创建了 {len(local_models)} 个本地模型 (3个恶意)")

        # 计算聚合前的模型差异
        original_params = sum(p.sum().item() for p in global_model.parameters())

        # 执行聚合
        aggregated_state = defense.aggregate(global_model, local_models, participant_ids)
        print(f"✓ 聚合完成")

        # 加载聚合后的状态
        global_model.load_state_dict(aggregated_state)

        # 计算聚合后的模型差异
        new_params = sum(p.sum().item() for p in global_model.parameters())
        param_change = abs(new_params - original_params)

        print(f"  参数变化: {param_change:.6f}")
        print(f"✓ {defense.name} 测试通过\n")

        return True

    except Exception as e:
        print(f"✗ {defense.name} 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("="*60)
    print("快速防御验证")
    print("="*60)

    # 创建测试配置
    class TestConfig:
        def get(self, key, default=None):
            return {
                'num_adversaries': 3,
                'krum_num_selected': 5,
                'trimmed_mean_beta': 0.1,
                'clip_threshold': 10.0,
                'dp_noise_scale': 0.001,
            }.get(key, default)

    config = TestConfig()

    # 测试所有防御
    defenses = [
        'fedavg',
        'krum',
        'trimmed_mean',
        'median',
        'norm_clipping',
        'weak_dp',
        'foolsgold',
    ]

    results = {}
    for defense_name in defenses:
        results[defense_name] = test_defense(defense_name, config)

    # 打印摘要
    print("="*60)
    print("测试摘要")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for defense_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {defense_name:<20} {status}")

    print(f"\n通过: {passed}/{total}")
    print("="*60)

    if passed == total:
        print("\n✓ 所有防御机制验证通过！")
        print("\n接下来可以运行完整测试:")
        print("  python test_defenses.py --params configs/tmp.yaml --test_all --epochs 50")
    else:
        print("\n⚠️  部分防御验证失败，请检查错误信息")


if __name__ == '__main__':
    main()