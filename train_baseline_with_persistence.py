"""
统一高级攻击训练脚本 - 整合所有优化 + ASR持久性测试
整合：
1. 因子化触发器
2. 5大核心创新（协同反推、聚合模拟、持久性评估、后门重放、协同优化）
3. 双阶段自适应更新
4. 主任务优化
5. ⭐ ASR持久性测试
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
import copy
import json
import logging
from torch.utils.data import DataLoader, Subset

from helper import Helper
# ⭐ 使用统一高级攻击器
from fl_utils.factorized_attacker import UnifiedAdvancedAttacker
from fl_utils.main_task_optimizer import MainTaskOptimizer
from fl_utils.task_separation import create_trainer
from fl_utils.evaluation import FactorizedAttackEvaluator
from fl_utils.visualization import FactorizedAttackVisualizer

# ⭐ 导入ADCA+DMBA模块
try:
    from fl_utils.adca_dmba_module import ADCADMBAIntegration
    ADCA_DMBA_AVAILABLE = True
    print("✓ ADCA+DMBA模块加载成功")
except ImportError:
    ADCA_DMBA_AVAILABLE = False
    print("⚠️ ADCA+DMBA模块未找到，将使用标准模式")
    print("  请确保adca_dmba_module.py在项目目录中")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DictObj(dict):
    """让 dict 同时支持 config['key'] 和 config.key"""
    def __init__(self, d=None, **kwargs):
        super().__init__(d or {}, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, DictObj):
                self[k] = DictObj(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config 中缺少键 '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"随机种子设置为: {seed}")


def get_benign_boost_params(config, epoch):
    schedule = config.get('benign_boost_schedule', None)
    if schedule is None:
        return config.get('benign_boost_enabled', False), config.get('benign_extra_epochs', 0)

    stage1_end = schedule.get('stage1_end_epoch', None)
    if stage1_end is None:
        return config.get('benign_boost_enabled', False), config.get('benign_extra_epochs', 0)

    stage = schedule.get('stage1') if epoch <= stage1_end else schedule.get('stage2')
    if stage is None:
        return config.get('benign_boost_enabled', False), config.get('benign_extra_epochs', 0)

    enabled = stage.get('enabled', config.get('benign_boost_enabled', False))
    extra_epochs = stage.get('extra_epochs', config.get('benign_extra_epochs', 0))
    return enabled, extra_epochs


def train_benign_client(helper, participant_id, model, epoch):
    """
    训练良性客户端（使用主任务优化器）

    Args:
        helper: Helper对象
        participant_id: 客户端ID
        model: 本地模型
        epoch: 当前轮次

    Returns:
        model: 训练后的模型
        stats: 训练统计
    """
    # ⭐ 初始化主任务优化器（只初始化一次）
    if not hasattr(helper, 'main_task_optimizer'):
        helper.main_task_optimizer = MainTaskOptimizer(
            base_lr=helper.config.get('lr', 0.06),
            scheduler_type=helper.config.get('lr_scheduler', 'cosine'),
            warmup_epochs=helper.config.get('warmup_epochs', 10),
            use_distillation=helper.config.get('knowledge_distillation', False),
            benign_boost=helper.config.get('benign_boost_enabled', False),
            benign_extra_epochs=helper.config.get('benign_extra_epochs', 2)
        )
        logger.info("主任务优化器初始化完成")

    # ⭐ 使用优化后的学习率
    benign_boost, benign_extra_epochs = get_benign_boost_params(helper.config, epoch)
    helper.main_task_optimizer.benign_boost = benign_boost
    helper.main_task_optimizer.benign_extra_epochs = benign_extra_epochs

    lr = helper.main_task_optimizer.get_lr(epoch, helper.config.epochs)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=helper.config.momentum,
        weight_decay=helper.config.decay
    )

    model.train()
    model = model.cuda()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    # ⭐ 计算训练轮数（支持良性增强）
    train_epochs = helper.config.retrain_times
    if helper.main_task_optimizer.benign_boost:
        train_epochs += helper.main_task_optimizer.benign_extra_epochs

    # 训练循环
    for internal_epoch in range(train_epochs):
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
    avg_loss = total_loss / (len(helper.train_data[participant_id]) * train_epochs)

    stats = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total_samples,
        'lr': lr,
        'train_epochs': train_epochs
    }

    return model, stats


def train_malicious_client(helper, participant_id, model, epoch, attacker, trainer):
    """
    训练恶意客户端（使用统一攻击器）

    Args:
        helper: Helper对象
        participant_id: 客户端ID
        model: 本地模型
        epoch: 当前轮次
        attacker: UnifiedAdvancedAttacker实例
        trainer: 训练器（保留用于兼容）

    Returns:
        model: 训练后的模型
        stats: 训练统计
    """
    print(f"\n{'=' * 60}")
    print(f"训练恶意客户端 {participant_id} (Epoch {epoch})")
    print(f"{'=' * 60}")

    adversary_id = helper.adversary_list.index(participant_id)

    # 注意：ADCA 协同聚合已移至训练后阶段（在 federated_learning_round 中）
    # 训练前加载是无效的，因为此时所有模型都是 global_model 的副本

    # ⭐ DMBA: 传递adca_dmba实例给attacker（如果存在）
    if hasattr(helper, 'adca_dmba') and helper.adca_dmba is not None:
        attacker.adca_dmba = helper.adca_dmba
        attacker.enable_dmba_replay = helper.config.get('enable_backdoor_replay', False)
    else:
        attacker.adca_dmba = None
        attacker.enable_dmba_replay = False

    # 显示因子信息
    if epoch % attacker.rotation_frequency == 0:
        combination = attacker.assign_factor_combination(adversary_id, epoch)
        factor_names = attacker.get_active_factors(adversary_id)
        print(f"  因子组合: {factor_names}")
        print(f"  k-of-m: {attacker.k}-of-{attacker.m}")

    current_intensity = attacker.intensity_schedule.get(epoch, 0.3)
    print(f"  当前强度: {current_intensity:.3f}")

    # ⭐ 使用统一攻击器的训练方法
    trained_model, stats = attacker.train_malicious_client(
        participant_id=participant_id,
        model=model,
        epoch=epoch,
        data_loader=helper.train_data[participant_id],
        adversary_id=adversary_id,
        trainer=trainer
    )

    print(f"\n  训练完成:")
    print(f"    损失: {stats['loss']:.4f}")
    print(f"    准确率: {stats['accuracy']:.2f}%")
    print(f"    投毒样本: {stats.get('poisoned_samples', 0)}/{stats['total_samples']}")

    if 'replay_samples' in stats or 'main_loss' in stats or 'backdoor_loss' in stats:
        print(f"    replay_samples: {stats.get('replay_samples', 0)}")
        if 'main_loss' in stats:
            print(f"    main_loss: {stats['main_loss']:.4f}")
        if 'backdoor_loss' in stats:
            print(f"    backdoor_loss: {stats['backdoor_loss']:.4f}")
        if 'kd_loss' in stats:
            print(f"    kd_loss: {stats['kd_loss']:.4f}")

    return trained_model, stats


def aggregate_models(helper, local_models):
    """
    聚合本地模型

    Args:
        helper: Helper对象
        local_models: 本地模型字典
    """
    # 使用FedAvg聚合
    global_dict = helper.global_model.state_dict()

    for key in global_dict.keys():
        # 聚合所有本地模型的参数
        global_dict[key] = torch.stack([
            local_models[pid].state_dict()[key].float()
            for pid in local_models.keys()
        ]).mean(dim=0)

    helper.global_model.load_state_dict(global_dict)


def federated_learning_round(helper, attacker, trainer, epoch):
    """
    执行一轮联邦学习（带协同优化）

    Args:
        helper: Helper对象
        attacker: UnifiedAdvancedAttacker实例
        trainer: 训练器
        epoch: 当前轮次

    Returns:
        local_models: 本地模型字典
        training_stats: 训练统计
    """
    print(f"\n{'='*70}")
    print(f"联邦学习 - Epoch {epoch}/{helper.config.epochs}")
    print(f"{'='*70}")

    # 采样客户端
    sampled_participants = helper.sample_participants(epoch)
    print(f"采样客户端: {sampled_participants}")
    print(f"  恶意: {[p for p in sampled_participants if p in helper.adversary_list]}")
    print(f"  良性: {[p for p in sampled_participants if p not in helper.adversary_list]}")

    local_models = {}
    training_stats = {}
    helper.client_models = []

    # ⭐ 收集恶意客户端的更新（用于协同优化）
    malicious_updates = []
    malicious_participant_ids = []

    # 保存全局模型初始状态
    global_model_initial = {k: v.cpu().clone() for k, v in helper.global_model.state_dict().items()}

    # 训练所有采样的客户端
    for participant_id in sampled_participants:
        local_model = copy.deepcopy(helper.global_model)

        if participant_id in helper.adversary_list:
            # 恶意客户端训练
            local_model, stats = train_malicious_client(
                helper, participant_id, local_model, epoch, attacker, trainer
            )

            # ⭐ 收集更新（用于协同优化）
            # 收集更新（用于协同优化）
            update = {}
            for name, param in local_model.state_dict().items():
                global_param = global_model_initial[name]

                # ⭐ 修复：跳过整数类型的参数
                if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                    continue

                # ⭐ 修复：确保是float类型
                update[name] = param.cpu().float() - global_param.float()

            if update:  # ⭐ 修复：只添加非空的update
                malicious_updates.append(update)
                malicious_participant_ids.append(participant_id)

        else:
            # 良性客户端训练
            local_model, stats = train_benign_client(
                helper, participant_id, local_model, epoch
            )

        local_models[participant_id] = local_model
        training_stats[participant_id] = stats
        helper.client_models.append(local_model)

    # ================================================================
    # ⭐ 训练后增强：ADCA 协同聚合 + 协同优化
    # ================================================================

    # ---- 阶段 A：ADCA 协同聚合（对真正训练过的恶意模型） ----
    adca_applied = False
    if (malicious_participant_ids
            and hasattr(helper, 'adca_dmba') and helper.adca_dmba is not None
            and helper.config.get('enable_malicious_coalition', False)):

        print(f"\n{'='*70}")
        print(f"ADCA协同聚合（训练后，{len(malicious_participant_ids)} 个恶意模型）")
        print(f"{'='*70}")

        try:
            # 收集真正训练过的恶意模型（不是global的副本！）
            trained_malicious_models = [
                local_models[pid] for pid in malicious_participant_ids
            ]

            # 用 ADCA 对训练后的恶意模型做注意力加权聚合
            enhanced_state = helper.adca_dmba.apply_coalition(
                trained_malicious_models,
                helper.global_model
            )

            if enhanced_state is not None:
                # 将增强后的 state 加载到所有恶意模型
                for pid in malicious_participant_ids:
                    device = next(local_models[pid].parameters()).device
                    enhanced_on_device = {
                        k: v.to(device) for k, v in enhanced_state.items()
                    }
                    local_models[pid].load_state_dict(enhanced_on_device)

                adca_applied = True
                print(f"✓ ADCA协同聚合完成，已增强 {len(malicious_participant_ids)} 个恶意模型")
            else:
                print("⚠️ ADCA返回None，跳过")

        except Exception as e:
            logger.warning(f"ADCA协同聚合失败: {e}")
            import traceback
            traceback.print_exc()

    # ---- 阶段 B：协同优化（如果启用，可与ADCA叠加） ----
    if malicious_updates and attacker.enable_collaboration and attacker.core_innovations_enabled:
        print(f"\n{'='*70}")
        print(f"应用协同优化（{len(malicious_updates)} 个恶意更新）")
        print(f"{'='*70}")

        try:
            # 计算全局模型变化（用于协同反推）
            global_change = {}
            for name in global_model_initial.keys():
                changes = []
                for pid in sampled_participants:
                    local_param = local_models[pid].state_dict()[name]
                    global_param = global_model_initial[name]

                    if local_param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                        continue

                    changes.append((local_param.cpu().float() - global_param.float()))

                if changes:
                    global_change[name] = torch.stack(changes).mean(dim=0)

            # 应用协同优化
            optimized_updates = attacker.aggregate_malicious_updates(
                malicious_updates,
                global_change,
                epoch
            )

            for i, pid in enumerate(malicious_participant_ids):
                target_model = local_models[pid]
                device = next(target_model.parameters()).device

                new_state = {}
                for name in target_model.state_dict().keys():
                    current_param = target_model.state_dict()[name]

                    if current_param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                        new_state[name] = current_param.clone()
                        continue

                    if name in optimized_updates[i]:
                        base_value = global_model_initial[name]
                        optimized_value = base_value.float() + optimized_updates[i][name].float()
                        new_state[name] = optimized_value.to(device)
                    else:
                        new_state[name] = current_param.clone()

                # ⭐ 修复：将优化后的参数实际写回模型
                target_model.load_state_dict(new_state)

            print("✓ 协同优化完成")

        except Exception as e:
            logger.warning(f"协同优化失败: {e}")
            logger.warning("使用原始更新继续")

    # 如果 ADCA 和协同优化都没跑，打印提示
    if not adca_applied and not (attacker.enable_collaboration and attacker.core_innovations_enabled):
        if malicious_participant_ids:
            print(f"\n⚠️ 警告：恶意更新未经任何放大就进入FedAvg，ASR可能很低")
            print(f"  建议：启用 enable_malicious_coalition 或 enable_collaboration")

    return local_models, training_stats


# ==================== 持久性测试模块 ====================
class PersistenceTester:
    """ASR持久性测试器 - 使用UnifiedAdvancedAttacker"""

    def __init__(self, helper, attacker, num_rounds=20, num_test_samples=1000):
        self.helper = helper
        self.attacker = attacker
        self.num_rounds = num_rounds
        self.num_test_samples = num_test_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.final_epoch = helper.config.epochs - 1

        logger.info(f"\n初始化持久性测试器...")
        logger.info(f"  测试轮数: {num_rounds}")
        logger.info(f"  测试样本: {num_test_samples}")
        self._prepare_test_data()

    def _prepare_test_data(self):
        """准备测试数据集"""
        logger.info(f"  准备测试数据...")

        # 干净测试集
        test_dataset = self.helper.test_dataset
        if len(test_dataset) > self.num_test_samples:
            indices = np.random.choice(
                len(test_dataset),
                self.num_test_samples,
                replace=False
            )
            self.clean_testset = Subset(test_dataset, indices)
        else:
            self.clean_testset = test_dataset

        self.clean_testloader = DataLoader(
            self.clean_testset,
            batch_size=64,
            shuffle=False,
            num_workers=2
        )

        # 后门测试集（为每个adversary创建）
        logger.info(f"  生成后门测试集...")
        self.backdoor_testloaders = []
        for adv_id in range(self.helper.config.num_adversaries):
            loader = self._create_backdoor_testloader(adv_id)
            self.backdoor_testloaders.append(loader)
            logger.info(f"    Adversary {adv_id}: {len(loader.dataset)} 样本")

        logger.info("  ✓ 测试数据准备完成")

    def _create_backdoor_testloader(self, adversary_id):
        """为指定adversary创建后门测试集 - 使用训练时相同的触发器"""
        backdoor_data = []
        backdoor_labels = []

        temp_loader = DataLoader(self.clean_testset, batch_size=64, shuffle=False, num_workers=2)

        for images, labels in temp_loader:
            # 过滤掉目标类
            mask = labels != self.helper.config.target_class
            if mask.sum() == 0:
                continue

            filtered_images = images[mask]

            # 【关键】使用和训练时完全相同的方法应用触发器
            poisoned_images, poisoned_labels, _ = self.attacker.poison_input_with_task_separation(
                filtered_images.cuda(),
                labels[mask].cuda(),
                adversary_id,
                self.final_epoch,  # 使用最后一个epoch的配置
                eval_mode=True
            )

            backdoor_data.append(poisoned_images.cpu())
            backdoor_labels.append(poisoned_labels.cpu())

        if len(backdoor_data) == 0:
            # 如果没有数据，创建空数据集
            backdoor_data = [torch.zeros(0, 3, 32, 32)]
            backdoor_labels = [torch.zeros(0, dtype=torch.long)]

        backdoor_data = torch.cat(backdoor_data, dim=0)
        backdoor_labels = torch.cat(backdoor_labels, dim=0)

        dataset = torch.utils.data.TensorDataset(backdoor_data, backdoor_labels)
        return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    def test_accuracy(self, model):
        """测试主任务准确率"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.clean_testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total if total > 0 else 0

    def test_asr(self, model, backdoor_loader):
        """测试ASR"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in backdoor_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total if total > 0 else 0

    def run_persistence_test(self, initial_model):
        """运行持久性测试"""
        print("\n" + "="*70)
        print(f"开始ASR持久性测试 ({self.num_rounds}轮)")
        print("="*70)

        # 测试初始状态
        print("\n[初始状态 - 训练完成时]")
        initial_acc = self.test_accuracy(initial_model)
        initial_asrs = []

        for adv_id, loader in enumerate(self.backdoor_testloaders):
            asr = self.test_asr(initial_model, loader)
            initial_asrs.append(asr)
            factors = self.attacker.get_active_factors(adv_id)
            print(f"  Adversary {adv_id} ({', '.join(factors[:2])}...): {asr:.2f}%")

        avg_initial_asr = np.mean(initial_asrs)
        print(f"\n主任务准确率: {initial_acc:.2f}%")
        print(f"平均ASR: {avg_initial_asr:.2f}%")

        # 创建良性客户端
        print(f"\n创建{10}个良性客户端用于持久性测试...")
        benign_clients = []
        train_dataset = self.helper.train_dataset
        num_clients = 10
        samples_per_client = len(train_dataset) // num_clients

        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            client_data = Subset(train_dataset, list(range(start_idx, end_idx)))
            benign_clients.append(BenignClient(i, client_data, self.device, self.helper.config))

        # 持久性测试循环
        print(f"\n开始{self.num_rounds}轮良性训练...")
        print("="*70)

        global_model = copy.deepcopy(initial_model)
        results = []

        for round_num in range(self.num_rounds):
            # 良性客户端训练
            client_models = [
                client.train(global_model)
                for client in benign_clients
            ]

            # 聚合
            global_state = self._aggregate_client_models(client_models)
            global_model.load_state_dict(global_state)

            # 测试
            main_acc = self.test_accuracy(global_model)
            round_asrs = [
                self.test_asr(global_model, loader)
                for loader in self.backdoor_testloaders
            ]
            avg_asr = np.mean(round_asrs)

            results.append({
                'round': round_num + 1,
                'main_acc': main_acc,
                'avg_asr': avg_asr,
                'individual_asrs': round_asrs
            })

            if (round_num + 1) % 5 == 0 or round_num == 0:
                decay = avg_initial_asr - avg_asr
                print(f"轮次 {round_num+1:2d}: 主任务={main_acc:.1f}%, "
                      f"ASR={avg_asr:.1f}%, 衰减={decay:.1f}%")

        # 生成报告
        self._generate_report(initial_acc, avg_initial_asr, initial_asrs, results)

        return results

    def _aggregate_client_models(self, client_models):
        """聚合客户端模型"""
        global_dict = {}
        for key in client_models[0].keys():
            global_dict[key] = torch.zeros_like(client_models[0][key])

        for key in global_dict.keys():
            for client_model in client_models:
                global_dict[key] += client_model[key]
            global_dict[key] = torch.div(global_dict[key], len(client_models))

        return global_dict

    def _generate_report(self, initial_acc, avg_initial_asr, initial_asrs, results):
        """生成持久性测试报告"""
        final_asr = results[-1]['avg_asr']
        asr_decay = avg_initial_asr - final_asr

        print("\n" + "="*70)
        print("持久性测试总结")
        print("="*70)
        print(f"测试轮次: {self.num_rounds}")
        print(f"初始ASR: {avg_initial_asr:.2f}%")
        print(f"最终ASR: {final_asr:.2f}%")
        print(f"总衰减: {asr_decay:.2f}% ({asr_decay/avg_initial_asr*100:.1f}%)")
        print(f"平均每轮衰减: {asr_decay/self.num_rounds:.2f}%")

        # 持久性评级
        print(f"\n持久性评估:")
        if final_asr > 80:
            rating = "⭐⭐⭐⭐⭐ 极强"
        elif final_asr > 60:
            rating = "⭐⭐⭐⭐ 很强"
        elif final_asr > 40:
            rating = "⭐⭐⭐ 较强"
        elif final_asr > 20:
            rating = "⭐⭐ 中等"
        else:
            rating = "⭐ 较弱"
        print(f"  {rating} - ASR保持在{final_asr:.1f}%")

        # 半衰期
        half_life = None
        for r in results:
            if r['avg_asr'] < avg_initial_asr / 2:
                half_life = r['round']
                break

        if half_life:
            print(f"  半衰期: {half_life} 轮")
        else:
            print(f"  半衰期: >{self.num_rounds} 轮")

        # 保存详细报告
        report = {
            'configuration': {
                'num_test_samples': self.num_test_samples,
                'num_rounds': self.num_rounds,
                'num_adversaries': self.helper.config.num_adversaries,
                'k_of_m': f"{self.attacker.k}-of-{self.attacker.m}",
                'rotation_frequency': self.attacker.rotation_frequency,
                'enable_collaboration': self.attacker.enable_collaboration
            },
            'initial_results': {
                'main_accuracy': initial_acc,
                'average_asr': avg_initial_asr,
                'individual_asrs': {str(i): asr for i, asr in enumerate(initial_asrs)}
            },
            'final_results': {
                'main_accuracy': results[-1]['main_acc'],
                'average_asr': final_asr,
                'individual_asrs': {str(i): asr for i, asr in enumerate(results[-1]['individual_asrs'])}
            },
            'persistence_metrics': {
                'total_decay': asr_decay,
                'decay_percentage': asr_decay/avg_initial_asr*100 if avg_initial_asr > 0 else 0,
                'avg_decay_per_round': asr_decay/self.num_rounds,
                'half_life': half_life if half_life else f">{self.num_rounds}",
                'rating': rating
            },
            'round_by_round': results
        }

        report_path = f"{self.helper.folder_path}/persistence_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 详细报告已保存: {report_path}")
        print("="*70)


class BenignClient:
    """良性客户端（用于持久性测试）"""
    def __init__(self, client_id, trainset, device, config):
        self.client_id = client_id
        self.device = device
        self.config = config
        self.trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    def train(self, global_model):
        """训练并返回state_dict"""
        # 获取模型类型
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,  # 固定学习率
            momentum=self.config.momentum,
            weight_decay=self.config.decay
        )
        criterion = nn.CrossEntropyLoss()

        # 训练5个本地epochs
        for epoch in range(5):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


# ==================== 主函数 ====================
def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='统一高级攻击训练 + 持久性测试')
    parser.add_argument('--params', type=str, required=True, help='配置文件路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    # 持久性测试参数
    parser.add_argument('--run_persistence', action='store_true',
                       help='训练完成后运行持久性测试')
    parser.add_argument('--persistence_rounds', type=int, default=20,
                       help='持久性测试轮数')
    parser.add_argument('--persistence_samples', type=int, default=1000,
                       help='持久性测试样本数')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"统一高级攻击训练系统")
    print(f"{'='*70}")

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU: {args.gpu}")
    else:
        print("使用CPU")

    # 加载配置
    with open(args.params, 'r', encoding='utf-8') as f:
        config = DictObj(yaml.safe_load(f))
    print(f"加载配置: {args.params}")

    # 设置随机种子
    seed = args.seed if args.seed is not None else config.get('seed', 0)
    set_seed(seed)

    # 初始化系统
    print(f"\n{'='*70}")
    print(f"初始化系统")
    print(f"{'='*70}")

    helper = Helper(config)
    helper.load_data()
    helper.load_model()
    helper.config_adversaries()

    print(f"✓ 数据集: {helper.config.dataset}")
    print(f"✓ 总客户端: {helper.config.num_total_participants}")
    print(f"✓ 恶意客户端: {helper.config.num_adversaries}")
    print(f"✓ 目标类别: {helper.config.target_class}")

    # ⭐ 初始化统一高级攻击器
    print(f"\n{'='*70}")
    print(f"初始化统一高级攻击器")
    print(f"{'='*70}")

    attacker = UnifiedAdvancedAttacker(
        helper,
        k=config.get('k_of_m_k', 3),
        m=config.get('k_of_m_m', 5),
        rotation_frequency=config.get('rotation_frequency', 8),
        enable_collaboration=config.get('enable_collaboration', True),
        weight_stability=config.get('weight_stability', 0.25),
        weight_importance=config.get('weight_importance', 0.30),
        weight_persistence=config.get('weight_persistence', 0.45),
        persistence_threshold=config.get('persistence_threshold', 0.6),
        importance_threshold=config.get('importance_threshold', 0.7),
        dynamic_lambda=config.get('dynamic_lambda', True),
        replay_buffer_size=config.get('replay_buffer_size', 150),
        replay_ratio=config.get('replay_ratio', 0.5),
        selection_ratio=config.get('selection_ratio', 0.3),
        boost_factor=config.get('boost_factor', 1.5)
    )

    # ⭐ 初始化ADCA+DMBA模块
    adca_dmba = None
    if ADCA_DMBA_AVAILABLE and (
        config.get('enable_malicious_coalition', False) or
        config.get('enable_backdoor_replay', False)
    ):
        print(f"\n{'='*70}")
        print(f"初始化ADCA+DMBA模块")
        print(f"{'='*70}")

        adca_dmba = ADCADMBAIntegration(config)
        helper.adca_dmba = adca_dmba

        print(f"✓ ADCA协同聚合: {config.get('enable_malicious_coalition', False)}")
        print(f"✓ DMBA后门重放: {config.get('enable_backdoor_replay', False)}")
    else:
        helper.adca_dmba = None
        if not ADCA_DMBA_AVAILABLE:
            print(f"\n⚠️ ADCA+DMBA模块不可用")
        else:
            print(f"\n⚠️ ADCA+DMBA未启用（检查配置文件）")

    # 保留trainer用于评估
    trainer = create_trainer(helper.config, adaptive=True)

    # 评估器和可视化器
    evaluator = FactorizedAttackEvaluator(helper, attacker)
    visualizer = FactorizedAttackVisualizer(
        save_dir=config.get('visualization', {}).get('save_dir', './visualizations')
    )

    # 训练循环
    print(f"\n{'='*70}")
    print(f"开始联邦学习训练")
    print(f"{'='*70}")

    evaluation_history = []

    for epoch in range(helper.config.epochs):
        # 执行一轮联邦学习（会保存client_models）
        local_models, training_stats = federated_learning_round(
            helper, attacker, trainer, epoch
        )

        # 聚合模型
        print(f"\n聚合 {len(local_models)} 个本地模型...")
        aggregate_models(helper, local_models)

        # 定期评估
        eval_freq = config.get('eval_freq', 10)
        if epoch % eval_freq == 0 or epoch == helper.config.epochs - 1:
            print(f"\n{'=' * 70}")
            print(f"评估 (Epoch {epoch})")
            print(f"{'=' * 70}")

            # 全面评估
            results = evaluator.comprehensive_evaluation(helper.global_model, epoch)
            results['epoch'] = epoch
            evaluation_history.append(results)

            # 打印关键指标
            print(f"主任务准确率: {results['main_accuracy']:.2f}%")
            print(f"平均ASR: {results['average_asr']:.2f}%")
            if results.get('stealthiness'):
                print(f"隐蔽性分数: {results['stealthiness']['normalized_stealthiness']:.4f}")

        # 保存模型
        save_on_epochs = helper.config.get('save_on_epochs', [50, 100, 150])
        if epoch in save_on_epochs or epoch == helper.config.epochs - 1:
            save_path = f"{helper.folder_path}/model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': helper.global_model.state_dict(),
                'attacker_stats': attacker.get_statistics() if hasattr(attacker, 'get_statistics') else {}
            }, save_path)
            print(f"模型已保存: {save_path}")

    # 最终评估和报告
    print(f"\n{'=' * 70}")
    print(f"训练完成！最终评估")
    print(f"{'=' * 70}")

    final_results = evaluator.comprehensive_evaluation(helper.global_model, helper.config.epochs)

    # ⭐ 生成完整报告
    print(f"\n生成最终报告...")

    attacker_stats = attacker.get_statistics() if hasattr(attacker, 'get_statistics') else {}

    report = {
        'configuration': {
            'dataset': helper.config.dataset,
            'num_adversaries': helper.config.num_adversaries,
            'k_of_m': f"{attacker.k}-of-{attacker.m}",
            'rotation_frequency': attacker.rotation_frequency,
            'enable_collaboration': attacker.enable_collaboration,
            'total_epochs': helper.config.epochs,
            'lr': config.get('lr', 0.06),
            'lr_scheduler': config.get('lr_scheduler', 'cosine'),
            'weight_persistence': config.get('weight_persistence', 0.45),
            'dynamic_lambda': config.get('dynamic_lambda', True),
        },
        'final_results': final_results,
        'evaluation_history': evaluation_history,
        'attacker_statistics': attacker_stats,
    }

    report_path = f"{helper.folder_path}/unified_attack_report_persistence.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"报告已保存: {report_path}")

    # 打印最终摘要
    print(f"\n{'=' * 70}")
    print(f"最终评估摘要")
    print(f"{'=' * 70}")
    print(f"主任务准确率: {final_results['main_accuracy']:.2f}%")
    print(f"平均ASR: {final_results['average_asr']:.2f}%")
    print(f"因子多样性: {final_results.get('factor_diversity', 0):.4f}")
    if final_results.get('stealthiness'):
        print(f"L2比值(恶意/良性): {final_results['stealthiness']['l2_ratio']:.4f}")
        print(f"归一化隐蔽分数: {final_results['stealthiness']['normalized_stealthiness']:.4f}")

    # 打印协同优化统计
    if attacker.core_innovations_enabled and attacker_stats:
    #         print(f"\n协同优化统计:")
        if 'collab_inference' in attacker_stats:
            collab_error = attacker_stats['collab_inference'].get('avg_error', 0)
            if collab_error is None:
                collab_error = 0
            print(f"  协同反推精度: {collab_error:.4f}")
        if 'dual_stage' in attacker_stats:
            dual_stats = attacker_stats['dual_stage']
            print(f"  双阶段策略分布:")
            print(f"    对齐: {dual_stats.get('align_ratio', 0):.2%}")
            print(f"    正交: {dual_stats.get('orthogonal_ratio', 0):.2%}")
            print(f"    跳过: {dual_stats.get('skip_ratio', 0):.2%}")

    print(f"{'=' * 70}")

    # ==================== 持久性测试 ====================
    if args.run_persistence:
        print(f"\n" + "="*70)
        print("开始ASR持久性测试")
        print("="*70)

        tester = PersistenceTester(
            helper,
            attacker,
            num_rounds=args.persistence_rounds,
            num_test_samples=args.persistence_samples
        )

        persistence_results = tester.run_persistence_test(helper.global_model)

        print("\n" + "="*70)
        print("全部完成！")
        print("="*70)
        print(f"\n生成的文件:")
        print(f"  - 训练报告: {report_path}")
        print(f"  - 持久性报告: {helper.folder_path}/persistence_report.json")
    else:
        print("\n提示: 使用 --run_persistence 参数可在训练后自动运行持久性测试")

    print(f"\n✅ 训练完成！")


if __name__ == '__main__':
    main()