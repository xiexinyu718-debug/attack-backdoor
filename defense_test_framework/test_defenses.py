"""
防御机制测试脚本
评估因子化触发器攻击在各种防御机制下的表现

使用方法:
    python test_defenses.py --params configs/tmp.yaml --defense krum
    python test_defenses.py --params configs/tmp.yaml --test_all
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
import copy
import json
from datetime import datetime

# 导入模块
# 注意：确保导入项目的helper，而不是第三方库的helper
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'defense_test_framework' in current_dir else current_dir
sys.path.insert(0, project_root)

# 现在可以安全导入
try:
    from helper import Helper
except ImportError:
    # 如果还是失败，尝试直接导入
    import importlib.util
    helper_path = os.path.join(project_root, 'helper.py')
    spec = importlib.util.spec_from_file_location("helper", helper_path)
    helper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper_module)
    Helper = helper_module.Helper

from fl_utils.factorized_attacker import FactorizedAttacker
from fl_utils.task_separation import create_trainer
from fl_utils.evaluation import FactorizedAttackEvaluator
from fl_defenses import create_defense


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_benign_client(helper, participant_id, model, epoch):
    """训练良性客户端"""
    import torch.nn as nn

    lr = helper.get_lr(epoch)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=helper.config.momentum,
        weight_decay=helper.config.decay
    )

    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for _ in range(helper.config.retrain_times):
        for inputs, labels in helper.train_data[participant_id]:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += inputs.shape[0]

    return model


def train_malicious_client(helper, participant_id, model, epoch, attacker, trainer):
    """训练恶意客户端"""
    adversary_id = helper.adversary_list.index(participant_id)

    # 分配因子组合
    if epoch % attacker.rotation_frequency == 0:
        attacker.assign_factor_combination(adversary_id, epoch)

    # 创建优化器
    lr = helper.get_lr(epoch)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=helper.config.momentum,
        weight_decay=helper.config.decay
    )

    # 使用任务分离训练
    for _ in range(helper.config.attacker_retrain_times):
        trainer.train_epoch(
            model, optimizer,
            helper.train_data[participant_id],
            attacker, adversary_id, epoch
        )

    return model


def test_with_defense(helper, attacker, trainer, defense, num_epochs=50):
    """
    使用指定防御测试攻击效果

    Args:
        helper: Helper对象
        attacker: FactorizedAttacker实例
        trainer: 任务分离训练器
        defense: 防御实例
        num_epochs: 测试轮数

    Returns:
        results: 测试结果
    """
    print(f"\n{'='*70}")
    print(f"测试防御: {defense.name}")
    print(f"{'='*70}")

    # 初始化评估器
    evaluator = FactorizedAttackEvaluator(helper, attacker)

    # 存储结果
    evaluation_history = []

    # 训练循环
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{num_epochs}")

        # 采样参与者
        sampled_participants = helper.sample_participants(epoch)
        malicious_in_round = [p for p in sampled_participants if p in helper.adversary_list]

        # 本地训练
        local_models = {}
        for participant_id in sampled_participants:
            local_model = copy.deepcopy(helper.global_model)

            if participant_id in helper.adversary_list:
                local_model = train_malicious_client(
                    helper, participant_id, local_model, epoch, attacker, trainer
                )
            else:
                local_model = train_benign_client(
                    helper, participant_id, local_model, epoch
                )

            local_models[participant_id] = local_model

        # 使用防御机制聚合
        aggregated_state = defense.aggregate(
            helper.global_model, local_models, sampled_participants
        )

        # 更新全局模型
        helper.global_model.load_state_dict(aggregated_state)

        # 定期评估
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            results = evaluator.comprehensive_evaluation(helper.global_model, epoch)
            results['epoch'] = epoch
            evaluation_history.append(results)

            print(f"  主任务准确率: {results['main_accuracy']:.2f}%")
            print(f"  平均ASR: {results['average_asr']:.2f}%")

    # 返回最终结果
    final_results = evaluation_history[-1]

    return {
        'defense_name': defense.name,
        'final_main_accuracy': final_results['main_accuracy'],
        'final_asr': final_results['average_asr'],
        'final_individual_asr': final_results['individual_asr'],
        'evaluation_history': evaluation_history,
    }


def test_all_defenses(helper, attacker, trainer, num_epochs=50):
    """测试所有防御机制"""
    print(f"\n{'='*70}")
    print(f"测试所有防御机制")
    print(f"{'='*70}")

    # 定义要测试的防御
    defense_configs = [
        {'name': 'fedavg', 'params': {}},
        {'name': 'krum', 'params': {'krum_num_selected': 5}},
        {'name': 'trimmed_mean', 'params': {'trimmed_mean_beta': 0.1}},
        {'name': 'median', 'params': {}},
        {'name': 'norm_clipping', 'params': {'clip_threshold': 10.0}},
        {'name': 'weak_dp', 'params': {'dp_noise_scale': 0.001}},
        {'name': 'foolsgold', 'params': {}},
    ]

    all_results = {}

    for defense_config in defense_configs:
        # 更新配置
        config_dict = vars(helper.config)
        config_dict.update(defense_config['params'])

        # 创建防御
        defense = create_defense(defense_config['name'], helper.config)

        # 重新初始化模型（确保每个防御从相同起点开始）
        helper.load_model()

        # 测试
        results = test_with_defense(helper, attacker, trainer, defense, num_epochs)
        all_results[defense_config['name']] = results

    return all_results


def print_comparison_table(all_results):
    """打印对比表格"""
    print(f"\n{'='*70}")
    print(f"防御机制对比结果")
    print(f"{'='*70}\n")

    # 表头
    print(f"{'防御机制':<20} {'主任务准确率':<15} {'平均ASR':<15} {'ASR下降':<15}")
    print(f"{'-'*70}")

    # 基准（无防御）
    baseline_asr = all_results.get('fedavg', {}).get('final_asr', 0)

    # 打印每个防御的结果
    for defense_name in ['fedavg', 'krum', 'trimmed_mean', 'median',
                         'norm_clipping', 'weak_dp', 'foolsgold']:
        if defense_name not in all_results:
            continue

        result = all_results[defense_name]
        main_acc = result['final_main_accuracy']
        asr = result['final_asr']
        asr_drop = baseline_asr - asr

        print(f"{result['defense_name']:<20} {main_acc:<15.2f} {asr:<15.2f} {asr_drop:<15.2f}")

    print(f"{'='*70}\n")


def generate_defense_report(all_results, save_path):
    """生成防御测试报告"""
    report = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results,
        'summary': {}
    }

    # 计算摘要统计
    baseline_asr = all_results.get('fedavg', {}).get('final_asr', 0)

    for defense_name, result in all_results.items():
        report['summary'][defense_name] = {
            'main_accuracy': result['final_main_accuracy'],
            'asr': result['final_asr'],
            'asr_drop': baseline_asr - result['final_asr'],
            'asr_drop_percentage': ((baseline_asr - result['final_asr']) / baseline_asr * 100) if baseline_asr > 0 else 0
        }

    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ 防御测试报告已保存: {save_path}")


def visualize_defense_results(all_results, save_dir='./defense_visualizations'):
    """可视化防御测试结果"""
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # 1. ASR对比柱状图
    fig, ax = plt.subplots(figsize=(12, 6))

    defense_names = []
    asrs = []
    main_accs = []

    for defense_name in ['fedavg', 'krum', 'trimmed_mean', 'median',
                         'norm_clipping', 'weak_dp', 'foolsgold']:
        if defense_name not in all_results:
            continue

        result = all_results[defense_name]
        defense_names.append(result['defense_name'])
        asrs.append(result['final_asr'])
        main_accs.append(result['final_main_accuracy'])

    x = np.arange(len(defense_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, asrs, width, label='ASR', color='indianred', alpha=0.7)
    bars2 = ax.bar(x + width/2, main_accs, width, label='Main Accuracy', color='steelblue', alpha=0.7)

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Defense Mechanism', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Defense Mechanisms Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(defense_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/defense_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存对比图: {save_dir}/defense_comparison.png")
    plt.close()

    # 2. 训练过程曲线
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, defense_name in enumerate(['fedavg', 'krum', 'trimmed_mean', 'median',
                                        'norm_clipping', 'weak_dp', 'foolsgold']):
        if defense_name not in all_results:
            continue

        result = all_results[defense_name]
        history = result['evaluation_history']

        epochs = [h['epoch'] for h in history]
        main_accs = [h['main_accuracy'] for h in history]
        asrs = [h['average_asr'] for h in history]

        ax = axes[idx]
        ax.plot(epochs, main_accs, 'b-o', label='Main Acc', linewidth=2, markersize=4)
        ax.plot(epochs, asrs, 'r-o', label='ASR', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(result['defense_name'], fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 隐藏未使用的子图
    for idx in range(len(all_results), 8):
        axes[idx].set_visible(False)

    plt.suptitle('Training Progress Under Different Defenses', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/defense_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存训练曲线: {save_dir}/defense_training_curves.png")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试因子化攻击在防御机制下的表现')
    parser.add_argument('--params', type=str, required=True, help='配置文件路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--defense', type=str, default=None,
                       help='测试单个防御 (fedavg/krum/trimmed_mean/median/norm_clipping/weak_dp/foolsgold)')
    parser.add_argument('--test_all', action='store_true', help='测试所有防御')
    parser.add_argument('--epochs', type=int, default=50, help='测试轮数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    args = parser.parse_args()

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU: {args.gpu}")

    # 加载配置
    with open(args.params, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    seed = args.seed if args.seed is not None else config.get('seed', 0)
    set_seed(seed)

    # 初始化系统
    print(f"\n初始化系统...")
    helper = Helper(config)
    helper.load_data()
    helper.load_model()
    helper.config_adversaries()

    # 初始化攻击器和训练器
    attacker = FactorizedAttacker(helper)
    trainer = create_trainer(helper.config, adaptive=True)

    # 测试防御
    if args.test_all:
        # 测试所有防御
        all_results = test_all_defenses(helper, attacker, trainer, args.epochs)

        # 打印对比表格
        print_comparison_table(all_results)

        # 生成报告
        report_path = './defense_test_report.json'
        generate_defense_report(all_results, report_path)

        # 可视化
        visualize_defense_results(all_results)

    elif args.defense:
        # 测试单个防御
        defense = create_defense(args.defense, helper.config)
        results = test_with_defense(helper, attacker, trainer, defense, args.epochs)

        print(f"\n{'='*70}")
        print(f"最终结果")
        print(f"{'='*70}")
        print(f"防御机制: {results['defense_name']}")
        print(f"主任务准确率: {results['final_main_accuracy']:.2f}%")
        print(f"平均ASR: {results['final_asr']:.2f}%")
        print(f"{'='*70}")

    else:
        print("错误: 请指定 --defense 或 --test_all")
        return

    print(f"\n✓ 测试完成！")


if __name__ == '__main__':
    main()