"""
因子化触发器攻击 - 持久性测试版本（集成MR模块）

新增功能：
1. 在指定轮次后停止投毒
2. 追踪停止后的ASR变化
3. 生成持久性分析报告
4. 集成Model Replacement (MR)模块

使用方法:
    python train_with_persistence_test.py --gpu 0 --params configs/persistence_test.yaml
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
import copy
from datetime import datetime
import json

# 导入模块
from helper import Helper
from fl_utils.factorized_attacker import FactorizedAttacker
from fl_utils.task_separation import create_trainer
from fl_utils.evaluation import FactorizedAttackEvaluator
from fl_utils.visualization import FactorizedAttackVisualizer
from model_replacement import ModelReplacementAttacker  # 新增: MR模块


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def should_poison(epoch, config):
    """
    判断当前轮次是否应该投毒

    Args:
        epoch: 当前轮次
        config: 配置对象

    Returns:
        bool: 是否应该投毒
    """
    # 【修复】兼容字典和对象两种形式
    if isinstance(config, dict):
        # 如果是字典，使用.get()
        poison_start = config.get('poison_start_epoch', 0)
        poison_stop = config.get('poison_stop_epoch', float('inf'))
    else:
        # 如果是对象，使用getattr()
        poison_start = getattr(config, 'poison_start_epoch', 0)
        poison_stop = getattr(config, 'poison_stop_epoch', float('inf'))

    # 【新增】调试输出（仅在第0轮显示）
    if epoch == 0:
        print(f"\n{'='*70}")
        print(f"should_poison函数调试信息 (Epoch 0):")
        print(f"  config类型: {type(config)}")
        print(f"  poison_start_epoch: {poison_start}")
        print(f"  poison_stop_epoch: {poison_stop}")
        print(f"{'='*70}\n")

    return poison_start <= epoch < poison_stop


def train_benign_client(helper, participant_id, model, epoch):
    """训练良性客户端"""
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

    accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / (len(helper.train_data[participant_id]) * helper.config.retrain_times)

    stats = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total_samples
    }

    return model, stats


def train_malicious_client(helper, participant_id, model, epoch, attacker, trainer, mr_attacker=None):
    """
    训练恶意客户端（使用因子化攻击、任务分离和MR）

    Args:
        helper: Helper对象
        participant_id: 参与者ID
        model: 本地模型
        epoch: 当前轮次
        attacker: FactorizedAttacker实例
        trainer: TaskSeparationTrainer实例
        mr_attacker: ModelReplacementAttacker实例(可选)

    Returns:
        model: 训练后的模型
        stats: 训练统计
    """
    print(f"\n{'='*60}")
    print(f"训练恶意客户端 {participant_id} (Epoch {epoch})")
    if mr_attacker is not None:
        print(f"  ✓ 启用Model Replacement (缩放因子: {mr_attacker._get_current_scale_factor(epoch):.2f})")
    print(f"{'='*60}")

    adversary_id = helper.adversary_list.index(participant_id)

    # 分配/更新因子组合
    if epoch % attacker.rotation_frequency == 0:
        combination = attacker.assign_factor_combination(adversary_id, epoch)
        factor_names = attacker.get_active_factors(adversary_id)
        print(f"  因子组合: {factor_names}")
        print(f"  k-of-m: {attacker.k}-of-{attacker.m}")

    current_intensity = attacker.intensity_schedule.get(epoch, 0.3)
    print(f"  当前强度: {current_intensity:.3f}")

    lr = helper.get_lr(epoch)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=helper.config.momentum,
        weight_decay=helper.config.decay
    )

    print(f"  开始训练 (重复 {helper.config.attacker_retrain_times} 次)...")

    all_stats = []
    for internal_epoch in range(helper.config.attacker_retrain_times):
        epoch_stats = trainer.train_epoch(
            model, optimizer,
            helper.train_data[participant_id],
            attacker, adversary_id, epoch
        )
        all_stats.append(epoch_stats)

    # 【新增】应用Model Replacement
    if mr_attacker is not None:
        print(f"  应用Model Replacement...")
        original_norm = mr_attacker._compute_update_norm(model, helper.global_model)
        model = mr_attacker.scale_malicious_update(model, helper.global_model, epoch)
        scaled_norm = mr_attacker._compute_update_norm(model, helper.global_model)
        print(f"    原始更新范数: {original_norm:.4f}")
        print(f"    放大后范数: {scaled_norm:.4f}")
        print(f"    放大倍数: {scaled_norm / original_norm:.2f}x")

    final_stats = {
        'avg_loss': np.mean([s['avg_loss'] for s in all_stats]),
        'accuracy': all_stats[-1]['accuracy'],
        'poisoned_samples': sum([s['poisoned_samples'] for s in all_stats]),
        'total_samples': all_stats[-1]['total_samples']
    }

    print(f"\n  训练完成:")
    print(f"    损失: {final_stats['avg_loss']:.4f}")
    print(f"    准确率: {final_stats['accuracy']:.2f}%")
    print(f"    投毒样本: {final_stats['poisoned_samples']}/{final_stats['total_samples']}")

    return model, final_stats


def federated_learning_round(helper, attacker, trainer, mr_attacker, epoch):
    """
    执行一轮联邦学习（支持持久性测试和MR）

    Args:
        helper: Helper对象
        attacker: FactorizedAttacker实例
        trainer: TaskSeparationTrainer实例
        mr_attacker: ModelReplacementAttacker实例
        epoch: 当前轮次

    Returns:
        local_models: {participant_id: model} 字典
        training_stats: 训练统计
    """
    print(f"\n{'='*70}")
    print(f"联邦学习轮次 {epoch}/{helper.config.epochs}")

    # 显示投毒状态
    is_poisoning = should_poison(epoch, helper.config)

    # 【修复】兼容对象和字典
    if isinstance(helper.config, dict):
        poison_start = helper.config.get('poison_start_epoch', 0)
        poison_stop = helper.config.get('poison_stop_epoch', float('inf'))
    else:
        poison_start = getattr(helper.config, 'poison_start_epoch', 0)
        poison_stop = getattr(helper.config, 'poison_stop_epoch', float('inf'))

    if epoch < poison_start:
        print(f"投毒状态: 未开始 (将在第 {poison_start} 轮开始)")
    elif is_poisoning:
        print(f"投毒状态: 进行中 (将在第 {poison_stop} 轮停止)")
    else:
        epochs_since_stop = epoch - poison_stop
        print(f"投毒状态: 已停止 (已停止 {epochs_since_stop} 轮)")

    print(f"{'='*70}")

    sampled_participants = helper.sample_participants(epoch)
    print(f"采样参与者: {sampled_participants}")

    malicious_in_round = [p for p in sampled_participants if p in helper.adversary_list]
    print(f"恶意客户端: {malicious_in_round if malicious_in_round else '无'}")

    local_models = {}
    training_stats = {}

    for participant_id in sampled_participants:
        local_model = copy.deepcopy(helper.global_model)

        if participant_id in helper.adversary_list and is_poisoning:
            # 恶意训练（投毒期间，带MR）
            local_model, stats = train_malicious_client(
                helper, participant_id, local_model, epoch, attacker, trainer, mr_attacker
            )
            stats['client_type'] = 'malicious_poisoning'
        elif participant_id in helper.adversary_list and not is_poisoning:
            # 曾经的攻击者，现在良性训练
            print(f"\n客户端 {participant_id}（曾为攻击者，现已停止投毒）")
            local_model, stats = train_benign_client(
                helper, participant_id, local_model, epoch
            )
            stats['client_type'] = 'malicious_stopped'
        else:
            # 良性训练
            local_model, stats = train_benign_client(
                helper, participant_id, local_model, epoch
            )
            stats['client_type'] = 'benign'

        local_models[participant_id] = local_model
        training_stats[participant_id] = stats

    return local_models, training_stats


def aggregate_models(helper, local_models):
    """聚合本地模型（FedAvg）"""
    global_state = helper.global_model.state_dict()

    weight_accumulator = {}
    for name, param in global_state.items():
        if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
            continue
        weight_accumulator[name] = torch.zeros_like(param, dtype=torch.float32)

    for participant_id, local_model in local_models.items():
        local_state = local_model.state_dict()
        for name, param in local_state.items():
            if name not in weight_accumulator:
                continue
            global_param = global_state[name]
            update = param.float() - global_param.float()
            weight_accumulator[name] += update

    num_participants = len(local_models)

    with torch.no_grad():
        for name, param in global_state.items():
            if name not in weight_accumulator:
                continue
            avg_update = weight_accumulator[name] / num_participants
            if param.dtype == torch.float32:
                param.add_(avg_update)
            elif param.dtype == torch.float16:
                param.add_(avg_update.half())
            elif param.dtype == torch.float64:
                param.add_(avg_update.double())
            else:
                param.add_(avg_update.to(param.dtype))


def analyze_persistence(evaluation_history, poison_stop_epoch):
    """
    分析后门持久性

    Args:
        evaluation_history: 评估历史记录
        poison_stop_epoch: 停止投毒的轮次

    Returns:
        persistence_analysis: 持久性分析结果
    """
    print(f"\n{'='*70}")
    print(f"分析后门持久性")
    print(f"{'='*70}")

    # 分离投毒前和投毒后的数据
    before_stop = [e for e in evaluation_history if e['epoch'] < poison_stop_epoch]
    after_stop = [e for e in evaluation_history if e['epoch'] >= poison_stop_epoch]

    if not before_stop or not after_stop:
        print("  ⚠️  数据不足，无法进行持久性分析")
        return {}

    # 计算停止前的平均ASR
    avg_asr_before = np.mean([e['average_asr'] for e in before_stop])

    # 分析停止后的ASR变化
    asr_changes = []
    for e in after_stop:
        epochs_after_stop = e['epoch'] - poison_stop_epoch
        asr = e['average_asr']
        retention = (asr / avg_asr_before * 100) if avg_asr_before > 0 else 0
        asr_changes.append({
            'epoch': e['epoch'],
            'epochs_after_stop': epochs_after_stop,
            'asr': asr,
            'retention': retention
        })

    # 计算半衰期（ASR降至停止前50%的轮次）
    half_life_epoch = None
    half_asr = avg_asr_before * 0.5
    for change in asr_changes:
        if change['asr'] <= half_asr:
            half_life_epoch = change['epochs_after_stop']
            break

    # 计算衰减率
    decay_rate = 0
    if len(after_stop) >= 2:
        recent_asrs = [e['average_asr'] for e in after_stop[:5]]
        decay_rate = (recent_asrs[0] - recent_asrs[-1]) / len(recent_asrs)
    else:
        decay_rate = 0

    # 评估持久性
    retention_rate = None
    persistence_level = None

    if len(after_stop) >= 5:
        final_asr = after_stop[-1]['average_asr']
        retention_rate = final_asr / avg_asr_before * 100 if avg_asr_before > 0 else 0

        print(f"\n持久性评估:")
        print(f"  停止前ASR: {avg_asr_before:.2f}%")
        print(f"  最终ASR: {final_asr:.2f}%")
        print(f"  保持率: {retention_rate:.2f}%")
        print(f"  平均衰减率: {decay_rate:.4f}% per epoch")

        if half_life_epoch:
            print(f"  半衰期: {half_life_epoch} 轮")
        else:
            print(f"  半衰期: >100 轮 (非常持久)")

        # 持久性等级
        if retention_rate > 80:
            persistence_level = "⭐⭐⭐⭐⭐ 极强"
        elif retention_rate > 60:
            persistence_level = "⭐⭐⭐⭐ 强"
        elif retention_rate > 40:
            persistence_level = "⭐⭐⭐ 中等"
        elif retention_rate > 20:
            persistence_level = "⭐⭐ 弱"
        else:
            persistence_level = "⭐ 很弱"

        print(f"  持久性等级: {persistence_level}")

    print(f"{'='*70}\n")

    return {
        'avg_asr_before_stop': avg_asr_before,
        'asr_changes': asr_changes,
        'decay_rate': decay_rate,
        'half_life_epoch': half_life_epoch,
        'retention_rate': retention_rate,
        'persistence_level': persistence_level
    }


def visualize_persistence(evaluation_history, poison_stop_epoch, save_dir):
    """可视化持久性分析"""
    import matplotlib.pyplot as plt

    epochs = [e['epoch'] for e in evaluation_history]
    asrs = [e['average_asr'] for e in evaluation_history]
    main_accs = [e['main_accuracy'] for e in evaluation_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 图1: ASR变化（突出停止投毒的时刻）
    ax1.plot(epochs, asrs, 'b-o', linewidth=2, markersize=4, label='ASR')
    ax1.axvline(x=poison_stop_epoch, color='r', linestyle='--',
                linewidth=2, label=f'停止投毒 (Epoch {poison_stop_epoch})')
    ax1.axvspan(0, poison_stop_epoch, alpha=0.2, color='red', label='投毒期')
    ax1.axvspan(poison_stop_epoch, epochs[-1], alpha=0.2, color='green', label='停止后')

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Backdoor Persistence: ASR Over Time (with MR)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 图2: 主任务准确率
    ax2.plot(epochs, main_accs, 'g-o', linewidth=2, markersize=4, label='Main Accuracy')
    ax2.axvline(x=poison_stop_epoch, color='r', linestyle='--', linewidth=2)
    ax2.axvspan(0, poison_stop_epoch, alpha=0.2, color='red')
    ax2.axvspan(poison_stop_epoch, epochs[-1], alpha=0.2, color='green')

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Main Task Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Main Task Performance', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'persistence_analysis_with_mr.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 持久性分析图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='后门持久性测试(带MR)')
    parser.add_argument('--params', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU设备ID')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（覆盖配置文件）')
    parser.add_argument('--disable-mr', action='store_true',
                       help='禁用Model Replacement')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU: {args.gpu}")

    with open(args.params, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"加载配置: {args.params}")

    seed = args.seed if args.seed is not None else config.get('seed', 0)
    set_seed(seed)
    print(f"随机种子: {seed}")

    # 显示持久性测试信息
    poison_stop = config.get('poison_stop_epoch', float('inf'))
    if poison_stop < float('inf'):
        print(f"\n{'='*70}")
        print(f"持久性测试模式")
        print(f"{'='*70}")
        print(f"将在第 {poison_stop} 轮停止投毒")
        print(f"之后继续训练 {config['epochs'] - poison_stop} 轮以观察后门持久性")
        print(f"{'='*70}\n")

    # 初始化系统
    print(f"\n初始化系统...")
    helper = Helper(config)
    helper.load_data()
    helper.load_model()
    helper.config_adversaries()

    # 初始化攻击器和训练器
    print(f"\n初始化因子化攻击器...")
    attacker = FactorizedAttacker(helper)

    # 【新增】初始化MR攻击器
    mr_attacker = None
    if not args.disable_mr:
        print(f"\n初始化Model Replacement攻击器...")
        mr_attacker = ModelReplacementAttacker(helper.config, helper)
    else:
        print(f"\n⚠️  Model Replacement已禁用")

    print(f"\n初始化任务分离训练器...")
    trainer = create_trainer(helper.config, adaptive=True)

    # 初始化评估器和可视化器
    evaluator = FactorizedAttackEvaluator(helper, attacker)
    visualizer = FactorizedAttackVisualizer(
        save_dir=config.get('visualization', {}).get('save_dir', './visualizations')
    )

    # 训练循环
    print(f"\n开始联邦学习训练...")
    evaluation_history = []

    for epoch in range(helper.config.epochs):
        # 执行一轮联邦学习
        local_models, training_stats = federated_learning_round(
            helper, attacker, trainer, mr_attacker, epoch
        )

        # 聚合模型
        print(f"\n聚合 {len(local_models)} 个本地模型...")
        aggregate_models(helper, local_models)

        # 定期评估
        eval_freq = config.get('eval_freq', 10)
        if epoch % eval_freq == 0 or epoch == helper.config.epochs - 1:
            print(f"\n{'='*70}")
            print(f"评估 (Epoch {epoch})")
            print(f"{'='*70}")

            results = evaluator.comprehensive_evaluation(helper.global_model, epoch)
            results['epoch'] = epoch
            results['is_poisoning'] = should_poison(epoch, helper.config)

            # 【新增】添加MR统计
            if mr_attacker is not None:
                results['mr_stats'] = mr_attacker.get_statistics()

            evaluation_history.append(results)

            # 如果刚刚停止投毒，打印提示
            if epoch == poison_stop:
                print(f"\n{'!'*70}")
                print(f"投毒已停止！开始观察后门持久性...")
                print(f"{'!'*70}\n")

        # 保存模型
        save_on_epochs = helper.config.get('save_on_epochs', [])
        if epoch in save_on_epochs or epoch == helper.config.epochs - 1:
            save_path = f"{helper.folder_path}/model_epoch_{epoch}.pt"
            save_data = {
                'epoch': epoch,
                'model_state_dict': helper.global_model.state_dict(),
            }

            # 【新增】保存MR统计
            if mr_attacker is not None:
                save_data['mr_stats'] = mr_attacker.get_statistics()
                save_data['mr_history'] = mr_attacker.history

            torch.save(save_data, save_path)
            print(f"模型已保存: {save_path}")

    # 最终评估和持久性分析
    print(f"\n{'='*70}")
    print(f"训练完成！生成持久性分析报告")
    print(f"{'='*70}")

    final_results = evaluator.comprehensive_evaluation(
        helper.global_model,
        helper.config.epochs
    )

    # 【新增】打印MR统计
    if mr_attacker is not None:
        mr_attacker.print_statistics()

    # 持久性分析
    persistence_analysis = analyze_persistence(
        evaluation_history,
        poison_stop
    )

    # 可视化持久性
    visualize_persistence(
        evaluation_history,
        poison_stop,
        visualizer.save_dir
    )

    # 生成完整报告
    print(f"\n生成最终报告...")
    report = {
        'configuration': {
            'dataset': helper.config.dataset,
            'num_adversaries': helper.config.num_adversaries,
            'k_of_m': f"{attacker.k}-of-{attacker.m}",
            'poison_stop_epoch': poison_stop,
            'total_epochs': helper.config.epochs,
            'mr_enabled': mr_attacker is not None,  # 新增
            'mr_scale_factor': mr_attacker.scale_factor if mr_attacker else None  # 新增
        },
        'final_results': final_results,
        'evaluation_history': evaluation_history,
        'persistence_analysis': persistence_analysis,
        'mr_statistics': mr_attacker.get_statistics() if mr_attacker else None  # 新增
    }

    report_path = f"{helper.folder_path}/persistence_report_with_mr.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n持久性测试报告已保存: {report_path}")

    # 打印摘要
    print(f"\n{'='*70}")
    print(f"持久性测试摘要")
    print(f"{'='*70}")
    print(f"停止投毒轮次: {poison_stop}")
    print(f"最终主任务准确率: {final_results['main_accuracy']:.2f}%")
    print(f"最终ASR: {final_results['average_asr']:.2f}%")
    if persistence_analysis.get('persistence_level'):
        print(f"持久性等级: {persistence_analysis['persistence_level']}")
        print(f"保持率: {persistence_analysis['retention_rate']:.2f}%")
    if mr_attacker is not None:
        print(f"MR平均缩放因子: {mr_attacker.get_statistics()['avg_scale_factor']:.2f}")
    print(f"{'='*70}")

    print(f"\n✓ 所有完成！")
    print(f"  - 模型保存在: {helper.folder_path}")
    print(f"  - 可视化保存在: {visualizer.save_dir}")
    print(f"  - 报告保存在: {report_path}")


if __name__ == '__main__':
    main()