"""
因子化触发器攻击 - 主训练脚本（集成自适应Model Replacement）

修改内容：
1. ✅ 导入AdaptiveModelReplacementAttacker
2. ✅ 替换MR攻击器初始化
3. ✅ 添加ASR快速评估函数
4. ✅ 在训练循环中更新gamma
5. ✅ 保存自适应MR统计信息
6. ✅ 支持自适应gamma的MR缩放

使用方法:
    python main_train_factorized_ADAPTIVE.py --gpu 0 --params configs/adaptive_mr_asr_based.yaml
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
from fl_utils.task_separation import create_trainer
from fl_utils.evaluation import FactorizedAttackEvaluator
from fl_utils.visualization import FactorizedAttackVisualizer, visualize_complete_report
from helper import Helper
from models.resnet import ResNet18
from fl_utils.factorized_attacker import FactorizedAttacker

# ⭐ 新增: 导入自适应Model Replacement
from adaptive_model_replacement import AdaptiveModelReplacementAttacker


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
    # 兼容字典和对象两种形式
    if isinstance(config, dict):
        poison_start = config.get('poison_start_epoch', 0)
        poison_stop = config.get('poison_stop_epoch', float('inf'))
    else:
        poison_start = getattr(config, 'poison_start_epoch', 0)
        poison_stop = getattr(config, 'poison_stop_epoch', float('inf'))

    return poison_start <= epoch < poison_stop


def get_poisoning_status_info(epoch, config):
    """
    获取当前投毒状态的详细信息

    Args:
        epoch: 当前轮次
        config: 配置对象

    Returns:
        dict: 投毒状态信息
    """
    if isinstance(config, dict):
        poison_start = config.get('poison_start_epoch', 0)
        poison_stop = config.get('poison_stop_epoch', float('inf'))
    else:
        poison_start = getattr(config, 'poison_start_epoch', 0)
        poison_stop = getattr(config, 'poison_stop_epoch', float('inf'))

    is_poisoning = poison_start <= epoch < poison_stop

    if epoch < poison_start:
        phase = "pre-attack"
        description = f"投毒前（将在第{poison_start}轮开始）"
    elif is_poisoning:
        progress = epoch - poison_start + 1
        total = poison_stop - poison_start
        description = f"投毒中（{progress}/{total}轮）"
        phase = "attacking"
    else:
        epochs_after = epoch - poison_stop + 1
        description = f"投毒后（已停止{epochs_after}轮）"
        phase = "post-attack"

    return {
        'phase': phase,
        'description': description,
        'is_poisoning': is_poisoning,
        'poison_start': poison_start,
        'poison_stop': poison_stop
    }


# ⭐ 新增: 快速ASR评估函数
def quick_evaluate_asr(helper, attacker, num_samples=200, adversary_id=0):
    """
    快速评估当前ASR（用于自适应MR调整）

    Args:
        helper: Helper对象
        attacker: FactorizedAttacker实例
        num_samples: 评估样本数

    Returns:
        asr: 攻击成功率 (0-1)
    """
    model = helper.global_model
    model.eval()
    device = next(model.parameters()).device

    if hasattr(helper.config, 'target_class'):
        target_class = helper.config.target_class
    elif isinstance(helper.config, dict) and 'target_class' in helper.config:
        target_class = helper.config['target_class']
    else:
        target_class = 0

    success = 0
    total = 0
    samples_processed = 0
    poison_start = getattr(helper.config, 'poison_start_epoch', 0)

    with torch.no_grad():
        for inputs, labels in helper.test_data:
            if samples_processed >= num_samples:
                break

            mask = labels != target_class
            inputs_clean = inputs[mask]
            labels_clean = labels[mask]

            if inputs_clean.shape[0] == 0:
                continue

            inputs_clean = inputs_clean.to(device)
            labels_clean = labels_clean.to(device)

            poisoned_inputs, poisoned_labels, _ = attacker.poison_input_with_task_separation(
                inputs_clean,
                labels_clean,
                adversary_id,
                epoch=max(poison_start, 0),
                eval_mode=True
            )

            outputs = model(poisoned_inputs)
            preds = outputs.argmax(dim=1)

            success += (preds == poisoned_labels).sum().item()
            total += poisoned_labels.size(0)
            samples_processed += poisoned_labels.size(0)

    asr = success / total if total > 0 else 0
    return asr


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

    # ⭐ 修复: 安全获取配置，提供默认值
    momentum = getattr(helper.config, 'momentum', 0.9)
    weight_decay = getattr(helper.config, 'decay', 5e-4)
    retrain_times = getattr(helper.config, 'retrain_times', 1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for _ in range(retrain_times):
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
    avg_loss = total_loss / (len(helper.train_data[participant_id]) * retrain_times) if len(helper.train_data[participant_id]) > 0 else 0

    stats = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total_samples,
        'client_type': 'benign'
    }

    return model, stats


def train_malicious_client(helper, participant_id, model, epoch, attacker, trainer,
                          mr_attacker=None, is_poisoning=True):
    """
    训练恶意客户端（支持自适应MR）

    Args:
        helper: Helper对象
        participant_id: 参与者ID
        model: 本地模型
        epoch: 当前轮次
        attacker: FactorizedAttacker实例
        trainer: TaskSeparationTrainer实例
        mr_attacker: AdaptiveModelReplacementAttacker实例(可选)
        is_poisoning: 是否在投毒期

    Returns:
        model: 训练后的模型
        stats: 训练统计
    """
    print(f"\n{'='*60}")
    print(f"训练恶意客户端 {participant_id} (Epoch {epoch})")
    print(f"  投毒状态: {'进行中' if is_poisoning else '已停止'}")

    # ⭐ 修改: 显示当前gamma
    if is_poisoning and mr_attacker is not None:
        current_gamma = mr_attacker.current_gamma
        print(f"  ✓ 启用自适应Model Replacement")
        print(f"  当前γ: {current_gamma:.2f}")
    print(f"{'='*60}")

    adversary_id = helper.adversary_list.index(participant_id)

    # 只在投毒期才分配因子组合
    if is_poisoning:
        # 分配/更新因子组合
        attacker.register_participation(adversary_id, epoch)
        if attacker._should_rotate(adversary_id):
            combination = attacker.assign_factor_combination(adversary_id, epoch)
            factor_names = attacker.get_active_factors(adversary_id)
            print(f"  因子组合: {factor_names}")
            print(f"  k-of-m: {attacker.k}-of-{attacker.m}")

        current_intensity = attacker.intensity_schedule.get(epoch, 0.3)
        print(f"  当前强度: {current_intensity:.3f}")

    # 创建优化器
    lr = helper.get_lr(epoch)

    # ⭐ 修复: 安全获取配置，提供默认值
    momentum = getattr(helper.config, 'momentum', 0.9)
    weight_decay = getattr(helper.config, 'decay', 5e-4)
    attacker_retrain_times = getattr(helper.config, 'attacker_retrain_times', 2)
    retrain_times = getattr(helper.config, 'retrain_times', 1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # 根据is_poisoning决定训练方式
    if is_poisoning:
        # 投毒训练
        print(f"  开始投毒训练 (重复 {attacker_retrain_times} 次)...")

        all_stats = []
        for internal_epoch in range(attacker_retrain_times):
            epoch_stats = trainer.train_epoch(
                model, optimizer,
                helper.train_data[participant_id],
                attacker, adversary_id, epoch
            )
            all_stats.append(epoch_stats)

        # ⭐ 修改: 应用自适应MR缩放
        if mr_attacker is not None:
            print(f"  应用自适应Model Replacement...")

            # 计算缩放前的L2范数
            original_norm = mr_attacker.compute_l2_norm(model, helper.global_model)

            # 应用缩放（使用当前的gamma）
            model = mr_attacker.scale_malicious_update(model, helper.global_model)

            # 计算缩放后的L2范数
            scaled_norm = mr_attacker.compute_l2_norm(model, helper.global_model)

            print(f"    原始更新范数: {original_norm:.4f}")
            print(f"    放大后范数: {scaled_norm:.4f}")
            print(f"    放大倍数: {scaled_norm / original_norm:.2f}x")
            print(f"    当前γ: {mr_attacker.current_gamma:.2f}")

        # 汇总统计
        final_stats = {
            'avg_loss': np.mean([s['avg_loss'] for s in all_stats]),
            'accuracy': all_stats[-1]['accuracy'],
            'poisoned_samples': sum([s['poisoned_samples'] for s in all_stats]),
            'total_samples': all_stats[-1]['total_samples'],
            'client_type': 'malicious_poisoning',
            'gamma': mr_attacker.current_gamma if mr_attacker else None,  # ⭐ 新增
            'l2_norm': scaled_norm if mr_attacker else None  # ⭐ 新增
        }
    else:
        # 投毒期外，按良性客户端训练
        print(f"  投毒已停止，按良性方式训练...")

        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for _ in range(retrain_times):  # 使用上面定义的retrain_times变量
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
        avg_loss = total_loss / (len(helper.train_data[participant_id]) * retrain_times) if len(helper.train_data[participant_id]) > 0 else 0

        final_stats = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_samples,
            'client_type': 'malicious_benign'
        }

    return model, final_stats


def federated_learning_round(helper, attacker, trainer, mr_attacker, epoch):
    """
    执行一轮联邦学习

    Args:
        helper: Helper对象
        attacker: FactorizedAttacker实例
        trainer: TaskSeparationTrainer实例
        mr_attacker: AdaptiveModelReplacementAttacker实例(可选)
        epoch: 当前轮次

    Returns:
        local_models: 本地模型列表
        training_stats: 训练统计
    """
    # 获取投毒状态
    poison_status = get_poisoning_status_info(epoch, helper.config)

    print(f"\n{'='*70}")
    print(f"Epoch {epoch} / {helper.config.epochs}")
    print(f"投毒状态: {poison_status['description']}")
    print(f"{'='*70}")

    # 采样参与者
    sampled_participants = helper.sample_participants(epoch)
    print(f"\n本轮参与者: {len(sampled_participants)} 个")

    local_models = []
    training_stats = {
        'benign_clients': [],
        'malicious_clients': [],
        'malicious_l2_norms': [],  # ⭐ 新增
        'benign_l2_norms': []  # ⭐ 新增
    }

    for participant_id in sampled_participants:
        # 复制全局模型
        local_model = copy.deepcopy(helper.global_model)

        is_malicious = participant_id in helper.adversary_list
        is_poisoning = poison_status['is_poisoning']

        if is_malicious:
            # 训练恶意客户端
            local_model, stats = train_malicious_client(
                helper, participant_id, local_model, epoch,
                attacker, trainer, mr_attacker, is_poisoning
            )
            training_stats['malicious_clients'].append(stats)

            # ⭐ 新增: 记录L2范数
            if 'l2_norm' in stats and stats['l2_norm'] is not None:
                training_stats['malicious_l2_norms'].append(stats['l2_norm'])
        else:
            # 训练良性客户端
            local_model, stats = train_benign_client(
                helper, participant_id, local_model, epoch
            )
            training_stats['benign_clients'].append(stats)

            # ⭐ 新增: 记录良性客户端的L2范数
            if mr_attacker is not None:
                benign_l2 = mr_attacker.compute_l2_norm(local_model, helper.global_model)
                training_stats['benign_l2_norms'].append(benign_l2)

        local_models.append(local_model)

    # ⭐ 新增: 打印L2比值统计
    if training_stats['malicious_l2_norms'] and training_stats['benign_l2_norms']:
        mal_l2 = np.mean(training_stats['malicious_l2_norms'])
        ben_l2 = np.mean(training_stats['benign_l2_norms'])
        l2_ratio = mal_l2 / ben_l2 if ben_l2 > 0 else 0

        print(f"\n📊 L2范数统计:")
        print(f"  恶意客户端平均: {mal_l2:.2f}")
        print(f"  良性客户端平均: {ben_l2:.2f}")
        print(f"  L2比值: {l2_ratio:.2f}")

        # 添加到统计
        training_stats['l2_ratio'] = l2_ratio

    return local_models, training_stats


def aggregate_models(helper, local_models):
    """
    聚合本地模型到全局模型（FedAvg）

    Args:
        helper: Helper对象
        local_models: 本地模型列表
    """
    global_state = helper.global_model.state_dict()
    weight_accumulator = {}

    # 初始化累加器
    for name, param in global_state.items():
        # 只累加浮点型参数，跳过整数型参数
        if param.dtype.is_floating_point:
            weight_accumulator[name] = torch.zeros_like(param, dtype=torch.float32)

    # 累加所有本地模型的更新
    for local_model in local_models:
        local_state = local_model.state_dict()

        for name, param in local_state.items():
            if name not in weight_accumulator:
                continue

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
    parser = argparse.ArgumentParser(description='因子化触发器攻击训练(集成自适应MR)')
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

    # 验证配置合理性
    poison_start = config.get('poison_start_epoch', 0)
    poison_stop = config.get('poison_stop_epoch', float('inf'))
    epochs = config.get('epochs', 100)

    print(f"\n{'='*70}")
    print(f"配置验证:")
    print(f"  总轮数: {epochs}")
    print(f"  投毒开始: 第{poison_start}轮")
    print(f"  投毒结束: 第{poison_stop}轮")

    if poison_start >= epochs:
        print(f"  ⚠️  警告: poison_start_epoch ({poison_start}) >= epochs ({epochs})")
        print(f"  ⚠️  将不会投毒！建议修改配置。")
    elif poison_stop > epochs:
        print(f"  ⚠️  提示: poison_stop_epoch ({poison_stop}) > epochs ({epochs})")
        print(f"  投毒将持续到训练结束。")
    else:
        print(f"  ✓ 配置合理")
    print(f"{'='*70}\n")

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

    # ⭐ 新增: 初始化自适应MR攻击器
    mr_attacker = None
    if not args.disable_mr and config.get('use_model_replacement', False):
        print(f"\n初始化自适应Model Replacement攻击器...")
        mr_attacker = AdaptiveModelReplacementAttacker(
            config=config,
            num_total_participants=helper.num_total_participants,
            num_sampled_participants=helper.num_sampled_participants
        )
        print(f"✅ 自适应MR已启用")
    else:
        if args.disable_mr:
            print(f"\n⚠️  Model Replacement已禁用（命令行参数）")
        else:
            print(f"\n⚠️  Model Replacement未配置（配置文件中无use_model_replacement）")

    # 初始化任务分离训练器
    print(f"\n初始化任务分离训练器...")
    trainer = create_trainer(helper.config, adaptive=True)

    # 初始化评估器和可视化器
    evaluator = FactorizedAttackEvaluator(helper, attacker)
    visualizer = FactorizedAttackVisualizer(
        save_dir=config.get('visualization', {}).get('save_dir', './visualizations')
    )

    # ⭐ 新增: ASR评估间隔
    evaluate_asr_interval = config.get('evaluate_asr_interval', 5)
    adaptive_start_epoch = config.get('adaptive_start_epoch', poison_start)
    print(f"ASR评估间隔: 每{evaluate_asr_interval}轮")

    # 训练循环
    print(f"\n开始联邦学习训练...")
    evaluation_history = []

    for epoch in range(helper.config.epochs):
        poison_active = should_poison(epoch, helper.config)

        # ⭐ 新增: 评估ASR并更新gamma（仅在达到自适应起点且处于投毒阶段）
        if (mr_attacker and mr_attacker.adaptive_enabled and poison_active
                and epoch >= adaptive_start_epoch and epoch % evaluate_asr_interval == 0):
            print(f"\n🔍 评估ASR用于自适应调整...")
            current_asr = quick_evaluate_asr(helper, attacker, num_samples=200)
            new_gamma = mr_attacker.update_gamma(current_asr=current_asr, current_epoch=epoch)

            print(f"  当前ASR: {current_asr:.2%}")
            print(f"  更新后γ: {new_gamma:.2f}")

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
            is_poisoning = should_poison(epoch, helper.config)
            results['is_poisoning'] = is_poisoning

            # 避免投毒前的ASR出现“伪100%”高值
            if not is_poisoning and epoch < poison_start:
                results['individual_asr'] = {
                    adv_id: 0.0 for adv_id in results.get('individual_asr', {})
                }
                results['average_asr'] = 0.0

            # ⭐ 新增: 添加自适应MR统计
            if mr_attacker is not None:
                mr_stats = mr_attacker.get_statistics()
                results['mr_stats'] = mr_stats
                results['current_gamma'] = mr_attacker.current_gamma

                # 如果有L2比值统计，也添加进去
                if 'l2_ratio' in training_stats:
                    results['l2_ratio'] = training_stats['l2_ratio']

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

            # ⭐ 新增: 保存自适应MR统计
            if mr_attacker is not None:
                save_data['mr_stats'] = mr_attacker.get_statistics()
                save_data['adaptive_mr_history'] = {
                    'gamma_history': mr_attacker.gamma_history,
                    'asr_history': mr_attacker.asr_history if hasattr(mr_attacker, 'asr_history') else []
                }

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

    # ⭐ 新增: 打印自适应MR统计
    if mr_attacker is not None and mr_attacker.adaptive_enabled:
        print(f"\n{'='*70}")
        print(f"自适应Model Replacement统计信息")
        print(f"{'='*70}")

        stats = mr_attacker.get_statistics()
        print(f"策略:           {stats['strategy']}")
        print(f"初始γ:          {stats['gamma_history'][0]:.2f}")
        print(f"最终γ:          {stats['gamma_history'][-1]:.2f}")
        print(f"平均γ:          {stats['avg_gamma']:.2f}")
        if 'final_asr' in stats:
            print(f"最终ASR:        {stats['final_asr']:.2%}")
        print(f"{'='*70}")

        # 保存统计到JSON
        import json
        mr_stats_path = f"{helper.folder_path}/adaptive_mr_stats.json"
        with open(mr_stats_path, 'w', encoding='utf-8') as f:
            # 转换为可序列化格式
            save_stats = {
                k: (v.tolist() if isinstance(v, np.ndarray) else
                    [float(x) for x in v] if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float, np.number)) else v)
                for k, v in stats.items()
            }
            json.dump(save_stats, f, indent=2, ensure_ascii=False)
        print(f"自适应MR统计已保存: {mr_stats_path}")

        # 绘制gamma演化曲线
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 子图1: Gamma vs Epoch
            ax1.plot(stats['gamma_history'], 'b-', linewidth=2)
            ax1.set_xlabel('Evaluation Step')
            ax1.set_ylabel('Gamma (γ)', color='b')
            ax1.set_title('Adaptive MR: Gamma Evolution')
            ax1.grid(alpha=0.3)

            # 子图2: ASR vs Epoch
            if 'asr_history' in stats and len(stats['asr_history']) > 0:
                ax2.plot([x*100 for x in stats['asr_history']], 'r-', linewidth=2)
                ax2.set_xlabel('Evaluation Step')
                ax2.set_ylabel('ASR (%)', color='r')
                ax2.set_title('Attack Success Rate Evolution')
                ax2.grid(alpha=0.3)

            plt.tight_layout()
            plot_path = f"{helper.folder_path}/adaptive_mr_evolution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"演化曲线已保存: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"⚠️  绘图失败: {e}")

    # 与基线对比
    from fl_utils.evaluation import compare_with_baselines
    comparison_results = compare_with_baselines(
        helper, attacker, helper.global_model, helper.config.epochs
    )

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
            'poison_start_epoch': poison_start,
            'poison_stop_epoch': poison_stop,
            'mr_enabled': mr_attacker is not None,
            'mr_adaptive': mr_attacker.adaptive_enabled if mr_attacker else False,  # ⭐ 新增
            'mr_strategy': mr_attacker.adaptive_strategy if (mr_attacker and mr_attacker.adaptive_enabled) else None,  # ⭐ 新增
            'mr_avg_gamma': stats['avg_gamma'] if (mr_attacker and mr_attacker.adaptive_enabled) else None  # ⭐ 新增
        },
        'final_results': final_results,
        'evaluation_history': evaluation_history,
        'baseline_comparison': comparison_results,
        'factor_info': attacker.get_factor_info(),
        'mr_statistics': mr_attacker.get_statistics() if mr_attacker else None
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

    # ⭐ 新增: 打印自适应MR摘要
    if mr_attacker is not None:
        if mr_attacker.adaptive_enabled:
            print(f"自适应MR平均γ: {stats['avg_gamma']:.2f}")
            print(f"最终γ: {stats['gamma_history'][-1]:.2f}")
        else:
            print(f"固定MR缩放因子: {mr_attacker.current_gamma:.2f}")

    print(f"{'='*70}")

    print(f"\n✓ 所有完成！")
    print(f"  - 模型保存在: {helper.folder_path}")
    print(f"  - 可视化保存在: {visualizer.save_dir}")
    print(f"  - 报告保存在: {report_path}")
    if mr_attacker and mr_attacker.adaptive_enabled:
        print(f"  - 自适应MR统计: {mr_stats_path}")
        print(f"  - Gamma演化图: {plot_path}")


if __name__ == '__main__':
    main()
