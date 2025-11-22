"""
因子化触发器攻击 - 主训练脚本（集成MR模块）

新增功能:
    - 集成Model Replacement (MR)模块
    - 恶意客户端模型更新自动放大
    - MR统计信息追踪

使用方法:
    python main_train_factorized.py --gpu 0 --params configs/tmp.yaml
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

# 导入模块
from helper import Helper
from fl_utils.factorized_attacker import FactorizedAttacker
from fl_utils.task_separation import create_trainer
from fl_utils.evaluation import FactorizedAttackEvaluator
from fl_utils.visualization import FactorizedAttackVisualizer, visualize_complete_report
from model_replacement import ModelReplacementAttacker  # 新增: MR模块


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_benign_client(helper, participant_id, model, epoch):
    """
    训练良性客户端

    Args:
        helper: Helper对象
        participant_id: 参与者ID
        model: 本地模型
        epoch: 当前轮次

    Returns:
        model: 训练后的模型
        stats: 训练统计
    """
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

    # 获取攻击者索引
    adversary_id = helper.adversary_list.index(participant_id)

    # 分配/更新因子组合
    if epoch % attacker.rotation_frequency == 0:
        combination = attacker.assign_factor_combination(adversary_id, epoch)
        factor_names = attacker.get_active_factors(adversary_id)
        print(f"  因子组合: {factor_names}")
        print(f"  k-of-m: {attacker.k}-of-{attacker.m}")

    # 获取当前强度
    current_intensity = attacker.intensity_schedule.get(epoch, 0.3)
    print(f"  当前强度: {current_intensity:.3f}")

    # 创建优化器
    lr = helper.get_lr(epoch)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=helper.config.momentum,
        weight_decay=helper.config.decay
    )

    # 使用任务分离训练
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

    # 汇总统计
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
    执行一轮联邦学习

    Args:
        helper: Helper对象
        attacker: FactorizedAttacker实例
        trainer: TaskSeparationTrainer实例
        mr_attacker: ModelReplacementAttacker实例
        epoch: 当前轮次

    Returns:
        local_models: {participant_id: model} 字典
    """
    print(f"\n{'='*70}")
    print(f"联邦学习轮次 {epoch}/{helper.config.epochs}")
    print(f"{'='*70}")

    # 采样参与者
    sampled_participants = helper.sample_participants(epoch)
    print(f"采样参与者: {sampled_participants}")

    # 识别恶意客户端
    malicious_in_round = [p for p in sampled_participants if p in helper.adversary_list]
    print(f"恶意客户端: {malicious_in_round if malicious_in_round else '无'}")

    # 本地训练
    local_models = {}
    training_stats = {}

    for participant_id in sampled_participants:
        # 创建本地模型副本
        local_model = copy.deepcopy(helper.global_model)

        if participant_id in helper.adversary_list:
            # 恶意训练 (带MR)
            local_model, stats = train_malicious_client(
                helper, participant_id, local_model, epoch, attacker, trainer, mr_attacker
            )
        else:
            # 良性训练
            local_model, stats = train_benign_client(
                helper, participant_id, local_model, epoch
            )

        local_models[participant_id] = local_model
        training_stats[participant_id] = stats

    return local_models, training_stats


def aggregate_models(helper, local_models):
    """
    聚合本地模型（FedAvg）

    Args:
        helper: Helper对象
        local_models: {participant_id: model} 字典
    """
    global_state = helper.global_model.state_dict()

    # 初始化权重累加器（只对浮点参数）
    weight_accumulator = {}
    for name, param in global_state.items():
        # 跳过整数类型的参数（如BatchNorm的num_batches_tracked）
        if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
            continue
        weight_accumulator[name] = torch.zeros_like(param, dtype=torch.float32)

    # 累加所有本地模型的更新
    for participant_id, local_model in local_models.items():
        local_state = local_model.state_dict()
        for name, param in local_state.items():
            # 跳过整数类型的参数
            if name not in weight_accumulator:
                continue

            # 计算更新 (local - global)
            global_param = global_state[name]
            update = param.float() - global_param.float()
            weight_accumulator[name] += update

    # 平均并更新全局模型
    num_participants = len(local_models)

    with torch.no_grad():
        for name, param in global_state.items():
            # 跳过整数类型的参数
            if name not in weight_accumulator:
                continue

            # 计算平均更新
            avg_update = weight_accumulator[name] / num_participants

            # 应用更新，确保类型匹配
            if param.dtype == torch.float32:
                param.add_(avg_update)
            elif param.dtype == torch.float16:
                param.add_(avg_update.half())
            elif param.dtype == torch.float64:
                param.add_(avg_update.double())
            else:
                param.add_(avg_update.to(param.dtype))


def main():
    """主函数"""
    # 解析参数
    parser = argparse.ArgumentParser(description='因子化触发器攻击训练(带MR)')
    parser.add_argument('--params', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU设备ID')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（覆盖配置文件）')
    parser.add_argument('--disable-mr', action='store_true',
                       help='禁用Model Replacement')
    args = parser.parse_args()

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU: {args.gpu}")
    else:
        print("警告: CUDA不可用，使用CPU训练")

    # 加载配置
    with open(args.params, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"加载配置: {args.params}")

    # 设置随机种子
    seed = args.seed if args.seed is not None else config.get('seed', 0)
    set_seed(seed)
    print(f"随机种子: {seed}")

    # 初始化Helper
    print(f"\n初始化系统...")
    helper = Helper(config)
    helper.load_data()
    helper.load_model()
    helper.config_adversaries()

    # 初始化攻击器
    print(f"\n初始化因子化攻击器...")
    attacker = FactorizedAttacker(helper)

    # 【新增】初始化MR攻击器
    mr_attacker = None
    if not args.disable_mr:
        print(f"\n初始化Model Replacement攻击器...")
        mr_attacker = ModelReplacementAttacker(helper.config, helper)
    else:
        print(f"\n⚠️  Model Replacement已禁用")

    # 初始化任务分离训练器
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

            # 全面评估
            results = evaluator.comprehensive_evaluation(helper.global_model, epoch)
            results['epoch'] = epoch

            # 【新增】添加MR统计
            if mr_attacker is not None:
                results['mr_stats'] = mr_attacker.get_statistics()

            evaluation_history.append(results)

            # 可视化
            if config.get('visualization', {}).get('enabled', False):
                try:
                    # 获取测试样本
                    test_iter = iter(helper.test_data)
                    sample_batch, _ = next(test_iter)
                    sample_input = sample_batch[:1].cuda() if torch.cuda.is_available() else sample_batch[:1]

                    # 生成可视化
                    visualize_complete_report(
                        attacker, evaluator, evaluation_history,
                        sample_input, epoch,
                        save_dir=visualizer.save_dir
                    )
                except Exception as e:
                    print(f"可视化生成失败: {e}")

            # 自适应调整
            if hasattr(trainer, 'adjust_weight'):
                trainer.adjust_weight(
                    asr=results['average_asr'] / 100.0,
                    main_accuracy=results['main_accuracy'] / 100.0,
                    epoch=epoch
                )

        # 保存模型
        save_on_epochs = helper.config.get('save_on_epochs', [50, 100])
        if epoch in save_on_epochs or epoch == helper.config.epochs - 1:
            save_path = f"{helper.folder_path}/model_epoch_{epoch}.pt"
            save_data = {
                'epoch': epoch,
                'model_state_dict': helper.global_model.state_dict(),
                'attacker_state': {
                    'active_combinations': attacker.active_combinations,
                    'rotation_history': attacker.rotation_history
                }
            }

            # 【新增】保存MR统计
            if mr_attacker is not None:
                save_data['mr_stats'] = mr_attacker.get_statistics()
                save_data['mr_history'] = mr_attacker.history

            torch.save(save_data, save_path)
            print(f"模型已保存: {save_path}")

    # 最终评估和报告
    print(f"\n{'='*70}")
    print(f"训练完成！最终评估")
    print(f"{'='*70}")

    final_results = evaluator.comprehensive_evaluation(
        helper.global_model,
        helper.config.epochs
    )

    # 【新增】打印MR统计
    if mr_attacker is not None:
        mr_attacker.print_statistics()



    # 生成最终报告
    print(f"\n生成最终报告...")
    import json
    report = {
        'configuration': {
            'dataset': helper.config.dataset,
            'num_adversaries': helper.config.num_adversaries,
            'k_of_m': f"{attacker.k}-of-{attacker.m}",
            'rotation_strategy': attacker.rotation_strategy,
            'task_separation_weight': trainer.separation_weight,
            'total_epochs': helper.config.epochs,
            'mr_enabled': mr_attacker is not None,  # 新增
            'mr_scale_factor': mr_attacker.scale_factor if mr_attacker else None  # 新增
        },
        'final_results': final_results,
        'evaluation_history': evaluation_history,
        'baseline_comparison': comparison_results,
        'factor_info': attacker.get_factor_info(),
        'mr_statistics': mr_attacker.get_statistics() if mr_attacker else None  # 新增
    }

    report_path = f"{helper.folder_path}/final_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n最终报告已保存: {report_path}")

    # 打印摘要
    print(f"\n{'='*70}")
    print(f"训练摘要")
    print(f"{'='*70}")
    print(f"主任务准确率: {final_results['main_accuracy']:.2f}%")
    print(f"平均ASR: {final_results['average_asr']:.2f}%")
    print(f"因子多样性: {final_results['factor_diversity']:.4f}")
    print(f"轮换有效性: {final_results['rotation_effectiveness']:.4f}")
    if mr_attacker is not None:
        print(f"MR平均缩放因子: {mr_attacker.get_statistics()['avg_scale_factor']:.2f}")
    print(f"{'='*70}")

    print(f"\n✓ 所有完成！")
    print(f"  - 模型保存在: {helper.folder_path}")
    print(f"  - 可视化保存在: {visualizer.save_dir}")
    print(f"  - 报告保存在: {report_path}")


if __name__ == '__main__':
    main()
