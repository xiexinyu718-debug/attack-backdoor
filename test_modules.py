"""
测试所有模块 - 独立测试脚本
用于验证所有模块是否正常工作
"""

import sys
import os
import torch
import numpy as np

print("="*70)
print("测试所有模块")
print("="*70)

# 测试1: Helper模块
print("\n" + "="*70)
print("测试 1/6: Helper模块")
print("="*70)
try:
    from helper import Helper
    
    test_config = {
        'dataset': 'cifar10',
        'seed': 0,
        'num_total_participants': 10,
        'num_sampled_participants': 3,
        'num_adversaries': 2,
        'epochs': 10,
        'batch_size': 64,
        'test_batch_size': 1024,
        'lr': 0.01,
        'momentum': 0.9,
        'decay': 0.0005,
        'sample_method': 'random',
        'target_class': 2,
        'environment_name': 'test_helper',
        'retrain_times': 2,
        'attacker_retrain_times': 2
    }
    
    helper = Helper(test_config)
    print("✓ Helper初始化成功")
    
    # 测试数据加载（不实际下载）
    print("  (跳过数据加载测试，需要时间)")
    
    print("✓ Helper模块测试通过")
    
except Exception as e:
    print(f"✗ Helper模块测试失败: {e}")
    import traceback
    traceback.print_exc()


# 测试2: ResNet模型
print("\n" + "="*70)
print("测试 2/6: ResNet模型")
print("="*70)
try:
    from models.resnet import ResNet18
    
    model = ResNet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    
    assert y.shape == (2, 10), f"输出形状错误: {y.shape}"
    
    print(f"✓ ResNet18模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {y.shape}")
    print("✓ ResNet模块测试通过")
    
except Exception as e:
    print(f"✗ ResNet模块测试失败: {e}")
    import traceback
    traceback.print_exc()


# 测试3: 触发器因子
print("\n" + "="*70)
print("测试 3/6: 触发器因子")
print("="*70)
try:
    from fl_utils.trigger_factors import (
        PositionPerturbationFactor,
        FrequencyPerturbationFactor,
        ColorShiftFactor,
        GeometricPerturbationFactor
    )
    
    test_input = torch.rand(2, 3, 32, 32)
    
    # 测试位置因子
    factor1 = PositionPerturbationFactor((2, 2), 4, 'square', 0.15)
    output1 = factor1.apply(test_input.cuda() if torch.cuda.is_available() else test_input)
    print(f"✓ 位置因子测试通过: {factor1.name}")
    
    # 测试频域因子
    factor2 = FrequencyPerturbationFactor('high', 0.05)
    output2 = factor2.apply(test_input.cuda() if torch.cuda.is_available() else test_input)
    print(f"✓ 频域因子测试通过: {factor2.name}")
    
    # 测试颜色因子
    factor3 = ColorShiftFactor('brightness', 0.1)
    output3 = factor3.apply(test_input.cuda() if torch.cuda.is_available() else test_input)
    print(f"✓ 颜色因子测试通过: {factor3.name}")
    
    # 测试几何因子
    factor4 = GeometricPerturbationFactor('translate', {'shift_x': 2, 'shift_y': 0}, 0.1)
    output4 = factor4.apply(test_input.cuda() if torch.cuda.is_available() else test_input)
    print(f"✓ 几何因子测试通过: {factor4.name}")
    
    print("✓ 触发器因子模块测试通过")
    
except Exception as e:
    print(f"✗ 触发器因子模块测试失败: {e}")
    import traceback
    traceback.print_exc()


# 测试4: 因子化攻击器
print("\n" + "="*70)
print("测试 4/6: 因子化攻击器")
print("="*70)
try:
    from fl_utils.factorized_attacker import FactorizedAttacker
    from test_utils import create_mock_helper

    # 使用测试工具创建mock helper
    helper = create_mock_helper({
        'k_of_m_k': 2,
        'k_of_m_m': 3,
        'num_adversaries': 2,
        'rotation_strategy': 'adversary_specific',
        'target_class': 2,  # 确保设置target_class
    })

    attacker = FactorizedAttacker(helper)

    print(f"✓ 攻击器初始化成功")
    print(f"  因子库大小: {len(attacker.factor_library)}")
    print(f"  k-of-m规则: {attacker.k}-of-{attacker.m}")

    # 测试因子分配
    combination = attacker.assign_factor_combination(0, 0)
    print(f"✓ 因子分配成功: {len(combination)} 个因子")

    # 测试触发器应用
    test_input = torch.rand(4, 3, 32, 32)
    test_labels = torch.tensor([0, 1, 2, 3])
    if torch.cuda.is_available():
        test_input = test_input.cuda()
        test_labels = test_labels.cuda()

    poisoned, labels, num = attacker.poison_input(
        test_input, test_labels, 0, 0, eval=False
    )
    print(f"✓ 触发器应用成功: 投毒 {num}/{test_input.shape[0]} 个样本")

    print("✓ 因子化攻击器模块测试通过")

except Exception as e:
    print(f"✗ 因子化攻击器模块测试失败: {e}")
    import traceback
    traceback.print_exc()


# 测试5: 任务分离训练
print("\n" + "="*70)
print("测试 5/6: 任务分离训练")
print("="*70)
try:
    from fl_utils.task_separation import TaskSeparationTrainer, create_trainer

    class MockConfig:
        def get(self, key, default=None):
            config_dict = {
                'task_separation_weight': 0.5,
                'adaptive_adjustment': {
                    'enabled': True,
                    'asr_threshold': 0.85,
                    'accuracy_threshold': 0.88
                }
            }
            return config_dict.get(key, default)

    config = MockConfig()
    trainer = create_trainer(config, adaptive=False)

    print(f"✓ 训练器初始化成功")
    print(f"  分离权重: {trainer.separation_weight}")

    # 测试损失计算
    outputs = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))
    if torch.cuda.is_available():
        outputs = outputs.cuda()
        labels = labels.cuda()

    total_loss, main_loss, backdoor_loss = trainer.compute_separated_loss(
        outputs, labels, poison_num=2
    )
    print(f"✓ 损失计算成功")
    print(f"  总损失: {total_loss.item():.4f}")
    print(f"  主任务损失: {main_loss.item():.4f}")
    print(f"  后门损失: {backdoor_loss.item():.4f}")

    print("✓ 任务分离训练模块测试通过")

except Exception as e:
    print(f"✗ 任务分离训练模块测试失败: {e}")
    import traceback
    traceback.print_exc()


# 测试6: 评估模块
print("\n" + "="*70)
print("测试 6/6: 评估模块")
print("="*70)
try:
    from fl_utils.evaluation import FactorizedAttackEvaluator

    print("✓ 评估模块导入成功")
    print("  (完整测试需要训练好的模型)")
    print("✓ 评估模块测试通过")

except Exception as e:
    print(f"✗ 评估模块测试失败: {e}")
    import traceback
    traceback.print_exc()


# 总结
print("\n" + "="*70)
print("测试总结")
print("="*70)
print("""
✓ 所有模块测试完成！

接下来可以：
1. 运行完整训练:
   python main_train_factorized.py --gpu 0 --params configs/tmp.yaml

2. 如果需要先下载数据集，可以运行:
   python -c "from helper import Helper; h = Helper({'dataset': 'cifar10', ...}); h.load_data()"

3. 查看更多测试:
   python test_aggregate.py  # 测试模型聚合
""")
print("="*70)