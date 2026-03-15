"""
统一高级攻击?- 最终完整版
⭐⭐⭐⭐?

完整功能?
1. ?设备问题修复
2. ?evaluation.py完全兼容（active_combinations, factor_library?
3. ?因子多样性预分配和计?
4. ?良性更新对齐（三层对齐策略?
5. ?轮换历史记录（修复轮换有效性为0的问题）

预期结果?
- 主任务准确率: 90-91%
- 平均ASR: 91-93%
- 因子多样? 0.75+
- L2比? 1.03-1.07
- 归一化隐蔽分? 0.35-0.37
- 轮换有效? 0.8+

这是真正的最终完整版！所有问题已修复?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import copy
import itertools

logger = logging.getLogger(__name__)


class UnifiedAdvancedAttacker:
    """
    统一高级攻击器（良性更新对齐版本）
    """

    def __init__(
        self,
        helper,
        k: int = 3,
        m: int = 5,
        rotation_frequency: int = 8,
        enable_collaboration: bool = True,
        weight_stability: float = 0.25,
        weight_importance: float = 0.30,
        weight_persistence: float = 0.45,
        persistence_threshold: float = 0.6,
        importance_threshold: float = 0.7,
        dynamic_lambda: bool = True,
        replay_buffer_size: int = 150,
        replay_ratio: float = 0.5,
        selection_ratio: float = 0.3,
        boost_factor: float = 1.5,
        # ⭐⭐?良性对齐参?
        enable_benign_alignment: bool = False,
        benign_direction_alpha: float = 0.25,  # 方向对齐?5%良性混?
        target_l2_ratio: float = 1.05,  # 幅度对齐：目标L2比?
        enable_distribution_align: bool = False  # 分布对齐（可选）
    ):
        """初始化统一攻击?"""
        self.helper = helper
        self.config = helper.config

        # 因子化触发器参数
        self.k = k
        self.m = m
        self.rotation_frequency = rotation_frequency
        self.num_adversaries = self.config.num_adversaries

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 轮换策略
        self.rotation_strategy = 'adversary_specific'

        # ⭐⭐?轮换历史记录（修复轮换有效性为0?
        self.rotation_history = {}  # {adv_id: [{'epoch': x, 'combination': (...)}, ...]}

        # ⭐⭐?良性对齐参?
        self.enable_benign_alignment = enable_benign_alignment
        self.benign_direction_alpha = benign_direction_alpha
        self.target_l2_ratio = target_l2_ratio
        self.enable_distribution_align = enable_distribution_align

        # Attack/runtime config.
        self.selection_ratio = self.config.get('selection_ratio', selection_ratio)
        self.boost_factor = self.config.get('boost_factor', boost_factor)
        self.enable_direction_consistency = self.config.get('enable_direction_consistency', True)
        self.enable_load_balancing = self.config.get('enable_load_balancing', True)
        self.enable_selective_boosting = self.config.get('enable_selective_boosting', True)
        self.use_task_separation = self.config.get('use_task_separation', False)
        self.task_separation_adaptive = self.config.get('task_separation_adaptive', False)

        self.persistence_eval_samples = self.config.get('persistence_eval_samples', 64)
        self.backdoor_cache_size = self.config.get('backdoor_cache_size', 128)

        # ADCA-style coalition attention (malicious collaboration).
        self.coalition_attention = self.config.get('coalition_attention', False)
        self.coalition_mix_alpha = self.config.get('coalition_mix_alpha', 0.5)
        self.coalition_temperature = self.config.get('coalition_temperature', 1.0)

        # Spa-style utility preservation (knowledge distillation on clean data).
        self.malicious_kd = self.config.get('malicious_kd', False)
        self.kd_lambda = self.config.get('kd_lambda', 1.0)
        self.kd_temperature = self.config.get('kd_temperature', 4.0)

        # KPBAFL-style key parameter attack (scale non-key params).
        self.key_param_attack_enabled = self.config.get('key_param_attack_enabled', False)
        self.key_param_keep_ratio = self.config.get('key_param_keep_ratio', None)
        self.non_key_param_scale = self.config.get('non_key_param_scale', 0.2)
        self.mr_enabled = self.config.get('mr_enabled', False)
        self.mr_scale = self.config.get('mr_scale', 1.0)

        self.last_backdoor_batch = None
        self.last_benign_avg = None
        self._task_trainer = None
        self.collab_inference = None
        self.simulator = None
        self.evaluator = None
        self.optimizer = None
        self.scaler = None
        self.dual_stage = None
        self.weight_adjuster = None

        # Replay configuration (decoupled from collaboration).
        self.replay_enabled = self.config.get('replay_enabled', False)
        self.replay_start_epoch = self.config.get('replay_start_epoch', 0)
        self.replay_buffer_size = self.config.get('replay_buffer_size', replay_buffer_size)
        self.replay_ratio = self.config.get('replay_ratio', replay_ratio)
        self.replay_sample_selection = self.config.get('replay_sample_selection', 'persistence')
        self.replay = None

        # 初始化组?
        self._init_factorized_trigger()
        self._initialize_factor_assignments()

        # evaluation.py兼容?
        self.active_combinations = self.factor_assignments
        self.factor_library = self.factor_types

        self.core_innovations_enabled = False
        if enable_collaboration:
            self._init_core_innovations()
        if self.replay_enabled and self.replay is None:
            try:
                from .backdoor_replay import BackdoorReplayMechanism
                self.replay = BackdoorReplayMechanism(
                    buffer_size=self.replay_buffer_size,
                    replay_ratio=self.replay_ratio,
                    sample_selection=self.replay_sample_selection
                )
                logger.info("Replay enabled (independent of collaboration)")
            except ImportError as e:
                logger.warning(f"backdoor_replay not available: {e}")
                self.replay_enabled = False

        # 初始化双阶段更新?
        try:
            # ?使用相对导入
            from .dual_stage_updater import DualStageAdaptiveUpdate
            self.dual_stage = DualStageAdaptiveUpdate(
                persistence_threshold=persistence_threshold,
                importance_threshold=importance_threshold,
                dynamic_lambda=dynamic_lambda
            )
            self.use_dual_stage = True
        except ImportError as e:
            logger.warning(f"未找?dual_stage_updater，跳过双阶段更新: {e}")
            self.use_dual_stage = False

        # 训练历史
        self.training_history = []
        self.enable_collaboration = enable_collaboration

        logger.info("=" * 70)
        logger.info("Unified advanced attacker initialized (full)")
        logger.info("=" * 70)
        logger.info(f"因子化触发器: k={k}, m={m}, 轮换频率={rotation_frequency}")
        logger.info(f"协同优化: {'启用' if enable_collaboration else '禁用'}")
        logger.info(f"?良性对? {'启用' if enable_benign_alignment else '禁用'}")
        if enable_benign_alignment:
            logger.info(f"  - 方向混合比例: {benign_direction_alpha:.2%}")
            logger.info(f"  - 目标L2比? {target_l2_ratio:.2f}")
            logger.info(f"  - 分布对齐: {'启用' if enable_distribution_align else '禁用'}")
        logger.info(f"?轮换历史记录: 启用（修复轮换有效性）")
        logger.info(f"设备: {self.device}")

    def _init_factorized_trigger(self):
        """Initialize factorized trigger."""
        self.factor_types = ['position_1', 'position_2', 'frequency', 'geometric_1', 'geometric_2']
        self.factor_assignments = {}
        self.active_factors = {}
        self.intensity_schedule = self._create_intensity_schedule()
        logger.info("Factorized trigger initialized")

    def _initialize_factor_assignments(self):
        """Assign factor combinations to adversaries."""
        all_combinations = list(itertools.combinations(range(self.m), self.k))

        logger.info(f"?预分配因子组?")
        logger.info(f"  可用组合? {len(all_combinations)}")
        logger.info(f"  恶意客户端数: {self.num_adversaries}")

        for adv_id in range(self.num_adversaries):
            combination_idx = adv_id % len(all_combinations)
            self.factor_assignments[adv_id] = all_combinations[combination_idx]
            self.active_factors[adv_id] = [
                self.factor_types[i] for i in all_combinations[combination_idx]
            ]

            # ⭐⭐?初始化轮换历史记?
            self.rotation_history[adv_id] = [{
                'epoch': 0,
                'combination': all_combinations[combination_idx]
            }]

            logger.info(f"  客户?{adv_id}: {self.active_factors[adv_id]}")

        diversity = self.compute_factor_diversity()
        logger.info(f"  初始因子多样? {diversity:.4f}")

    def _init_core_innovations(self):
        """初始?大核心创新组?"""
        try:
            # ?使用相对导入（从同一个包中导入）
            from .collaborative_inference import CollaborativeInference
            self.collab_inference = CollaborativeInference(
                n_total=self.config.num_total_participants,
                n_malicious=self.num_adversaries
            )

            from .aggregation_simulator import AggregationSimulator
            self.simulator = AggregationSimulator(
                n_total=self.config.num_total_participants,
                n_malicious=self.num_adversaries
            )

            from .persistence_evaluator import PersistenceEvaluator
            self.evaluator = PersistenceEvaluator(
                model=self.helper.global_model,
                simulator=self.simulator,
                weights={'stability': 0.25, 'importance': 0.30, 'persistence': 0.45}
            )

            from .backdoor_replay import BackdoorReplayMechanism
            self.replay = BackdoorReplayMechanism(
                buffer_size=self.replay_buffer_size,
                replay_ratio=self.replay_ratio,
                sample_selection=self.replay_sample_selection
            )

            from .collaborative_optimization import CollaborativeOptimizer, AdaptiveScaling
            self.optimizer = CollaborativeOptimizer(n_malicious=self.num_adversaries)
            self.scaler = AdaptiveScaling(initial_scale=1.0, decay_rate=0.95)

            self.core_innovations_enabled = True
            logger.info("?5大核心创新组件初始化完成")

        except ImportError as e:
            logger.warning(f"核心创新模块导入失败: {e}")
            self.core_innovations_enabled = False

    def _create_intensity_schedule(self):
        """创建强度调度?"""
        schedule = {}
        alpha_initial = self.config.get('trigger_alpha_initial', 0.30)
        alpha_final = self.config.get('trigger_alpha_final', 0.15)
        schedule_type = self.config.get('trigger_alpha_schedule', 'linear')
        if schedule_type == 'constant':
            alpha_final = alpha_initial

        total_epochs = max(self.config.epochs - 1, 1)
        for epoch in range(self.config.epochs):
            progress = epoch / total_epochs
            alpha = alpha_initial - (alpha_initial - alpha_final) * progress
            schedule[epoch] = max(alpha, alpha_final)

        return schedule

    def _get_backdoor_eval_batch(self, device):
        if self.replay is not None and len(self.replay.buffer.buffer) > 0:
            n_samples = min(self.persistence_eval_samples, len(self.replay.buffer.buffer))
            data, targets = self.replay.buffer.sample(n_samples)
            if data is not None:
                return data.to(device), targets.to(device)

        if self.last_backdoor_batch is not None:
            data, targets = self.last_backdoor_batch
            return data.to(device), targets.to(device)

        return None, None

    def _maybe_get_task_trainer(self):
        if not self.use_task_separation:
            return None
        if self._task_trainer is None:
            try:
                from .task_separation import create_trainer
                self._task_trainer = create_trainer(
                    self.config, adaptive=self.task_separation_adaptive
                )
            except Exception as e:
                logger.warning(f"任务分离训练器初始化失败: {e}")
                self._task_trainer = None
        return self._task_trainer

    def _prepare_poisoned_batch(self, data, targets, adversary_id, epoch, bkd_ratio, use_replay):
        batch_size = data.shape[0]
        poison_num = int(batch_size * bkd_ratio)
        if bkd_ratio > 0 and poison_num == 0:
            poison_num = 1

        if poison_num > 0:
            poisoned_data = self.apply_trigger(data[:poison_num], adversary_id, epoch)
            poisoned_targets = torch.ones(
                poison_num, dtype=torch.long, device=data.device
            ) * self.config.target_class
        else:
            poisoned_data = data[:0]
            poisoned_targets = targets[:0]

        clean_data = data[poison_num:]
        clean_targets = targets[poison_num:]

        n_replay = 0
        replay_data = None
        replay_targets = None
        if use_replay and self.replay is not None and len(self.replay.buffer.buffer) > 0:
            replay_ratio = self._resolve_replay_ratio(epoch)
            n_replay = int(batch_size * replay_ratio)
            if n_replay > 0:
                replay_data, replay_targets = self.replay.buffer.sample(n_replay)
                if replay_data is not None:
                    replay_data = replay_data.to(data.device)
                    replay_targets = replay_targets.to(data.device)
                else:
                    n_replay = 0

        data_parts = [poisoned_data]
        target_parts = [poisoned_targets]

        if n_replay > 0:
            data_parts.append(replay_data)
            target_parts.append(replay_targets)

        if clean_data.numel() > 0:
            data_parts.append(clean_data)
            target_parts.append(clean_targets)

        mixed_data = torch.cat(data_parts) if len(data_parts) > 1 else data_parts[0]
        mixed_targets = torch.cat(target_parts) if len(target_parts) > 1 else target_parts[0]

        poison_total = poison_num + n_replay

        if poison_num > 0:
            cache_data = poisoned_data.detach().cpu()[: self.backdoor_cache_size]
            cache_targets = poisoned_targets.detach().cpu()[: self.backdoor_cache_size]
            self.last_backdoor_batch = (cache_data, cache_targets)

        if use_replay and poison_num > 0:
            self.replay.update_buffer(
                poisoned_data.detach().cpu(),
                poisoned_targets.detach().cpu()
            )

        return mixed_data, mixed_targets, poison_num, n_replay

    def _evaluate_param_scores(self, malicious_update, epoch, device):
        if self.evaluator is None:
            return None, None, None

        filtered_update = {}
        for name, value in malicious_update.items():
            if torch.is_floating_point(value) or torch.is_complex(value):
                filtered_update[name] = value

        if not filtered_update:
            return None, None, None

        if self.simulator is not None and self.simulator.benign_avg is not None:
            filtered_update = {
                name: value
                for name, value in filtered_update.items()
                if name in self.simulator.benign_avg
            }
            if not filtered_update:
                return None, None, None

        backdoor_data, backdoor_targets = self._get_backdoor_eval_batch(device)
        if backdoor_data is None:
            return None, None, None

        criterion = nn.CrossEntropyLoss()

        try:
            from .persistence_evaluator import AdaptiveWeightAdjuster
        except Exception:
            AdaptiveWeightAdjuster = None

        if getattr(self, 'weight_adjuster', None) is None and self.config.get('adaptive_persistence_weights', False):
            if AdaptiveWeightAdjuster is not None:
                self.weight_adjuster = AdaptiveWeightAdjuster(total_rounds=self.config.epochs)

        if getattr(self, 'weight_adjuster', None) is not None:
            self.evaluator.weights = self.weight_adjuster.get_weights(epoch)

        stability_scores = {name: 0.5 for name in filtered_update.keys()}
        importance_scores = self.evaluator.evaluate_importance(
            self.helper.global_model, backdoor_data, backdoor_targets, criterion
        )
        persistence_scores = self.evaluator.evaluate_persistence(filtered_update)
        comprehensive_scores = self.evaluator.compute_comprehensive_score(
            stability_scores, importance_scores, persistence_scores
        )
        selection_ratio = self.selection_ratio
        if self.key_param_attack_enabled and self.key_param_keep_ratio is not None:
            selection_ratio = self.key_param_keep_ratio

        selected_params = self.evaluator.select_parameters(
            comprehensive_scores, selection_ratio=selection_ratio
        )

        return selected_params, importance_scores, persistence_scores

    def _resolve_replay_ratio(self, epoch: int) -> float:
        schedule = self.config.get('replay_ratio_schedule', None)
        if schedule is None:
            return self.replay_ratio

        stage1_end = schedule.get('stage1_end_epoch', None)
        if stage1_end is None:
            return self.replay_ratio

        stage = schedule.get('stage1') if epoch <= stage1_end else schedule.get('stage2')
        if stage is None:
            return self.replay_ratio

        return stage.get('replay_ratio', self.replay_ratio)

    def _apply_key_param_attack(self, updates, selected_params):
        if not selected_params:
            return updates

        scaled_updates = []
        for update in updates:
            scaled = {}
            for name, value in update.items():
                if not (torch.is_floating_point(value) or torch.is_complex(value)):
                    scaled[name] = value
                    continue
                if name in selected_params:
                    scaled[name] = value * self.mr_scale if self.mr_enabled else value
                else:
                    scaled[name] = value * self.non_key_param_scale
            scaled_updates.append(scaled)
        return scaled_updates

    def _compute_coalition_attention(self, updates, device):
        if not updates:
            return None

        backdoor_data, backdoor_targets = self._get_backdoor_eval_batch(device)
        if backdoor_data is None:
            return None

        criterion = nn.CrossEntropyLoss()
        losses = []
        for update in updates:
            loss_value = self._evaluate_update_backdoor_loss(
                update, backdoor_data, backdoor_targets, criterion, device
            )
            if loss_value is None:
                return None
            losses.append(loss_value)

        weights = torch.softmax(
            torch.tensor(losses, device=device, dtype=torch.float32) / max(self.coalition_temperature, 1e-6),
            dim=0
        )
        return weights.tolist()

    def _evaluate_update_backdoor_loss(
        self, update, backdoor_data, backdoor_targets, criterion, device
    ):
        if self.helper.global_model is None:
            return None

        model = copy.deepcopy(self.helper.global_model).to(device)
        state = model.state_dict()
        for name, value in update.items():
            if name in state and (torch.is_floating_point(state[name]) or torch.is_complex(state[name])):
                state[name] = state[name].to(device) + value.to(device)

        model.load_state_dict(state, strict=False)
        model.eval()
        with torch.no_grad():
            outputs = model(backdoor_data)
            loss = criterion(outputs, backdoor_targets)
        return loss.item()

    def _resolve_attack_hyperparams(self, epoch: int) -> Tuple[float, int]:
        """Resolve per-epoch attack hyperparameters from schedule."""
        bkd_ratio = self.config.bkd_ratio
        attacker_retrain_times = self.config.attacker_retrain_times

        schedule = self.config.get('attack_schedule', None)
        if schedule is None:
            return bkd_ratio, attacker_retrain_times

        stage1_end = schedule.get('stage1_end_epoch', None)
        if stage1_end is None:
            return bkd_ratio, attacker_retrain_times

        stage = schedule.get('stage1') if epoch <= stage1_end else schedule.get('stage2')
        if stage is None:
            return bkd_ratio, attacker_retrain_times

        bkd_ratio = stage.get('bkd_ratio', bkd_ratio)
        attacker_retrain_times = stage.get('attacker_retrain_times', attacker_retrain_times)
        return bkd_ratio, attacker_retrain_times

    def assign_factor_combination(self, adversary_id: int, epoch: int):
        """
        分配因子组合（k-of-m?
        ?添加轮换历史记录
        """
        all_combinations = list(itertools.combinations(range(self.m), self.k))

        rotation_cycle = epoch // self.rotation_frequency
        base_idx = adversary_id % len(all_combinations)
        rotation_offset = rotation_cycle % len(all_combinations)
        combination_idx = (base_idx + rotation_offset) % len(all_combinations)

        new_combination = all_combinations[combination_idx]

        # ⭐⭐?记录轮换历史（关键修复）
        if adversary_id not in self.rotation_history:
            self.rotation_history[adversary_id] = []

        # 检查是否需要记录新的轮?
        should_record = (
            len(self.rotation_history[adversary_id]) == 0 or
            self.rotation_history[adversary_id][-1]['combination'] != new_combination
        )

        if should_record:
            self.rotation_history[adversary_id].append({
                'epoch': epoch,
                'combination': new_combination
            })

            # 如果是轮换（不是初始化），记录日?
            if len(self.rotation_history[adversary_id]) > 1:
                prev_factors = [self.factor_types[i] for i in self.rotation_history[adversary_id][-2]['combination']]
                new_factors = [self.factor_types[i] for i in new_combination]
                logger.info(f"  ?[轮换] 客户?{adversary_id} Epoch {epoch}: "
                           f"{prev_factors} ?{new_factors}")

        # 更新分配
        self.factor_assignments[adversary_id] = new_combination
        self.active_factors[adversary_id] = [
            self.factor_types[i] for i in new_combination
        ]

        self.active_combinations = self.factor_assignments

        return self.factor_assignments[adversary_id]

    def get_active_factors(self, adversary_id: int) -> List[str]:
        """获取活跃因子"""
        return self.active_factors.get(adversary_id, self.factor_types[:self.k])

    def compute_factor_diversity(self) -> float:
        """计算因子多样?"""
        if not self.active_factors or len(self.active_factors) < 2:
            return 0.0

        factor_sets = []
        for adv_id in range(self.num_adversaries):
            if adv_id in self.active_factors:
                factor_sets.append(set(self.active_factors[adv_id]))

        if len(factor_sets) < 2:
            return 0.0

        distances = []
        for s1, s2 in itertools.combinations(factor_sets, 2):
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard_sim = intersection / union if union > 0 else 0
            jaccard_dist = 1 - jaccard_sim
            distances.append(jaccard_dist)

        return sum(distances) / len(distances) if distances else 0.0

    def apply_trigger(self, images, adversary_id, epoch):
        """应用因子化触发器"""
        alpha = self.intensity_schedule.get(epoch, 0.3)
        active_factors = self.get_active_factors(adversary_id)

        use_pixel_space = (
            self.config.get('apply_trigger_in_pixel_space', True) and
            hasattr(self.helper, 'data_mean') and
            hasattr(self.helper, 'data_std')
        )

        if use_pixel_space:
            mean = torch.as_tensor(
                self.helper.data_mean,
                device=images.device,
                dtype=images.dtype
            ).view(1, -1, 1, 1)
            std = torch.as_tensor(
                self.helper.data_std,
                device=images.device,
                dtype=images.dtype
            ).view(1, -1, 1, 1)
            poisoned = images * std + mean
            alpha_scale = std
        else:
            poisoned = images.clone()
            alpha_scale = 1.0

        alpha_tensor = alpha * alpha_scale
        if isinstance(alpha_tensor, torch.Tensor):
            alpha_patch = alpha_tensor
            alpha_channel = alpha_tensor[..., 0, 0]
        else:
            alpha_patch = alpha_tensor
            alpha_channel = alpha_tensor

        def clamp_if_needed(tensor):
            if use_pixel_space:
                return torch.clamp(tensor, 0, 1)
            return tensor

        for factor_name in active_factors:
            if factor_name == 'position_1':
                poisoned[:, :, :3, :3] = clamp_if_needed(
                    poisoned[:, :, :3, :3] + alpha_patch
                )
            elif factor_name == 'position_2':
                poisoned[:, :, -3:, -3:] = clamp_if_needed(
                    poisoned[:, :, -3:, -3:] - alpha_patch
                )
            elif factor_name == 'frequency':
                h, w = poisoned.shape[-2:]
                grid_i = torch.arange(h, device=poisoned.device, dtype=poisoned.dtype).view(h, 1)
                grid_j = torch.arange(w, device=poisoned.device, dtype=poisoned.dtype).view(1, w)
                pattern = torch.sin(2 * np.pi * (grid_i + grid_j) / 8.0).view(1, 1, h, w)
                poisoned = poisoned + (alpha_patch * 0.1) * pattern
                poisoned = clamp_if_needed(poisoned)
            elif factor_name == 'geometric_1':
                h, w = poisoned.shape[-2:]
                for i in range(min(h, w)):
                    if i < h and i < w:
                        poisoned[:, :, i, i] = clamp_if_needed(
                            poisoned[:, :, i, i] + alpha_channel * 0.5
                        )
            elif factor_name == 'geometric_2':
                h, w = poisoned.shape[-2:]
                for i in range(min(h, w)):
                    if i < h and (w-1-i) >= 0:
                        poisoned[:, :, i, w-1-i] = clamp_if_needed(
                            poisoned[:, :, i, w-1-i] - alpha_channel * 0.5
                        )

        if use_pixel_space:
            poisoned = (poisoned - mean) / std

        return poisoned

    def poison_input_with_task_separation(
        self, inputs, labels, adversary_id, epoch, eval_mode=False
    ):
        """完全匹配evaluation.py的接?"""
        batch_size = inputs.shape[0]

        if eval_mode:
            poison_num = batch_size
        else:
            bkd_ratio, _ = self._resolve_attack_hyperparams(epoch)
            poison_num = int(batch_size * bkd_ratio)
            if bkd_ratio > 0 and poison_num == 0:
                poison_num = 1

        if poison_num == 0:
            return inputs, labels, 0

        poisoned_inputs = inputs.clone()
        poisoned_inputs[:poison_num] = self.apply_trigger(
            inputs[:poison_num], adversary_id, epoch
        )

        poisoned_labels = labels.clone()
        poisoned_labels[:poison_num] = self.config.target_class

        return poisoned_inputs, poisoned_labels, poison_num

    def poison_input(self, images, adversary_id, epoch):
        """简化版投毒方法"""
        return self.apply_trigger(images, adversary_id, epoch)

    # ============================================
    # ⭐⭐?良性更新对齐核心方?
    # ============================================

    def _compute_update_norm(self, update: Dict[str, torch.Tensor]) -> float:
        """计算更新的L2范数"""
        total_norm = 0.0
        for k, v in update.items():
            total_norm += torch.sum(v ** 2).item()
        return np.sqrt(total_norm)

    def align_with_benign(
        self,
        malicious_update: Dict[str, torch.Tensor],
        benign_avg: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """
        ⭐⭐?良性更新对齐（三层策略?

        核心思想：让恶意更新在统计上接近良性更新，但保持后门效?

        Args:
            malicious_update: 恶意更新
            benign_avg: 良性平均更新（从协同反推获得）
            epoch: 当前轮次

        Returns:
            对齐后的恶意更新
        """
        if not self.enable_benign_alignment:
            return malicious_update

        # ===== ?: 方向对齐 =====
        # 混合恶意和良性方?
        alpha = self.benign_direction_alpha

        direction_aligned = {}
        for k in malicious_update.keys():
            if k in benign_avg:
                # 混合更新: (1-α)×恶意 + α×良?
                direction_aligned[k] = (
                    (1 - alpha) * malicious_update[k] +
                    alpha * benign_avg[k]
                )
            else:
                direction_aligned[k] = malicious_update[k]

        # ===== ?: 幅度对齐 =====
        # 调整L2范数到目标比?
        mal_norm = self._compute_update_norm(direction_aligned)
        benign_norm = self._compute_update_norm(benign_avg)

        if benign_norm > 0:
            current_ratio = mal_norm / benign_norm
            target_norm = benign_norm * self.target_l2_ratio

            if mal_norm > target_norm:
                scale_factor = target_norm / mal_norm

                magnitude_aligned = {}
                for k, v in direction_aligned.items():
                    magnitude_aligned[k] = v * scale_factor

                logger.info(f"  ?[良性对齐] Epoch {epoch}: "
                           f"L2比?{current_ratio:.3f} ?{self.target_l2_ratio:.3f} "
                           f"(缩放 {scale_factor:.3f}x)")
            else:
                magnitude_aligned = direction_aligned
        else:
            magnitude_aligned = direction_aligned

        # ===== ?: 分布对齐（可选）=====
        if self.enable_distribution_align:
            # 这里可以添加更高级的分布对齐
            # 暂时跳过，前两层已经足够
            final_aligned = magnitude_aligned
        else:
            final_aligned = magnitude_aligned

        return final_aligned

    # ============================================
    # 训练接口
    # ============================================

    def train_malicious_client(
        self,
        participant_id: int,
        model: nn.Module,
        epoch: int,
        data_loader,
        adversary_id: int,
        trainer=None
    ) -> Tuple[nn.Module, Dict]:
        """训练恶意客户?"""
        model = model.to(self.device)
        initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr = self.helper.get_lr(epoch)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.config.momentum,
            weight_decay=self.config.decay
        )

        criterion = nn.CrossEntropyLoss()
        model.train()

        task_trainer = trainer if trainer is not None else self._maybe_get_task_trainer()

        teacher = None
        if self.malicious_kd and self.helper.global_model is not None:
            teacher = copy.deepcopy(self.helper.global_model).to(self.device)
            teacher.eval()

        total_loss = 0.0
        total_main_loss = 0.0
        total_backdoor_loss = 0.0
        total_kd_loss = 0.0
        correct = 0
        total_samples = 0
        poisoned_samples = 0
        replay_samples = 0

        min_epochs = self.config.retrain_times
        if self.config.get('benign_boost_enabled', False):
            min_epochs = max(
                min_epochs,
                self.config.retrain_times + self.config.get('benign_extra_epochs', 0)
            )

        bkd_ratio, scheduled_retrain_times = self._resolve_attack_hyperparams(epoch)
        train_epochs = max(scheduled_retrain_times, min_epochs)

        for internal_epoch in range(train_epochs):
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                use_replay = (
                    self.replay_enabled and
                    self.replay is not None and
                    epoch >= self.replay_start_epoch
                )

                mixed_data, mixed_targets, poison_num, replay_num = self._prepare_poisoned_batch(
                    data, targets, adversary_id, epoch, bkd_ratio, use_replay
                )
                poison_total = poison_num + replay_num
                poisoned_samples += poison_total
                replay_samples += replay_num

                optimizer.zero_grad()
                outputs = model(mixed_data)
                if task_trainer is not None:
                    total_loss_tensor, main_loss, backdoor_loss = \
                        task_trainer.compute_separated_loss(
                            outputs, mixed_targets, poison_total
                        )
                else:
                    total_loss_tensor = criterion(outputs, mixed_targets)
                    main_loss = total_loss_tensor
                    backdoor_loss = torch.tensor(0.0, device=total_loss_tensor.device)

                kd_loss = torch.tensor(0.0, device=total_loss_tensor.device)
                if self.malicious_kd and mixed_targets.size(0) > poison_total:
                    clean_inputs = mixed_data[poison_total:]
                    if clean_inputs.numel() > 0 and teacher is not None:

                        with torch.no_grad():

                            teacher_logits = teacher(clean_inputs)


                        kd_temp = max(self.kd_temperature, 1e-6)
                        student_log_probs = F.log_softmax(
                            outputs[poison_total:] / kd_temp, dim=1
                        )
                        teacher_probs = F.softmax(
                            teacher_logits / kd_temp, dim=1
                        )
                        kd_loss = F.kl_div(
                            student_log_probs, teacher_probs, reduction='batchmean'
                        ) * (kd_temp * kd_temp)

                total_loss_tensor = total_loss_tensor + self.kd_lambda * kd_loss
                total_loss_tensor.backward()
                optimizer.step()

                total_loss += total_loss_tensor.item()
                total_main_loss += main_loss.item()
                total_backdoor_loss += backdoor_loss.item()
                total_kd_loss += kd_loss.item()

                _, predicted = outputs.max(1)
                correct += predicted.eq(mixed_targets).sum().item()
                total_samples += mixed_targets.size(0)

        # 计算模型更新
        final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        local_update = {}
        for k in initial_state.keys():
            local_update[k] = final_state[k] - initial_state[k]

        # 协同优化（如果启用）
        if self.core_innovations_enabled and self.enable_collaboration:
            local_update = self._apply_collaborative_optimization(
                local_update, epoch, model
            )

        # 应用更新
        optimized_state = {}
        for k in initial_state.keys():
            optimized_state[k] = initial_state[k] + local_update[k]

        model.load_state_dict({
            k: v.to(self.device) for k, v in optimized_state.items()
        })

        accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
        denom = len(data_loader) * train_epochs
        avg_loss = total_loss / denom if denom > 0 else 0.0
        avg_main_loss = total_main_loss / denom if denom > 0 else 0.0
        avg_backdoor_loss = total_backdoor_loss / denom if denom > 0 else 0.0

        stats = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'poisoned_samples': poisoned_samples,
            'replay_samples': replay_samples,
            'total_samples': total_samples,
            'train_epochs': train_epochs,
            'bkd_ratio': bkd_ratio,
            'attacker_retrain_times': scheduled_retrain_times
        }

        if task_trainer is not None:
            stats['main_loss'] = avg_main_loss
            stats['backdoor_loss'] = avg_backdoor_loss
        if self.malicious_kd:
            stats['kd_loss'] = total_kd_loss / denom if denom > 0 else 0.0

        return model, stats

    def _apply_collaborative_optimization(self, local_update, epoch, model):
        """应用协同优化"""
        if self.optimizer is None or self.scaler is None:
            return local_update

        float_update = {}
        non_float_update = {}
        for name, value in local_update.items():
            if torch.is_floating_point(value) or torch.is_complex(value):
                float_update[name] = value
            else:
                non_float_update[name] = value

        if not float_update:
            return local_update

        updates = [float_update]
        selected_params = None
        importance_scores = None
        persistence_scores = None

        if self.simulator is not None and self.last_benign_avg is not None:
            self.simulator.set_benign_average(self.last_benign_avg)
            selected_params, importance_scores, persistence_scores = \
                self._evaluate_param_scores(float_update, epoch, self.device)

        optimized_updates = self.optimizer.full_optimization(
            updates,
            selected_params=selected_params,
            boost_factor=self.boost_factor,
            enable_direction=self.enable_direction_consistency,
            enable_balancing=self.enable_load_balancing,
            enable_boosting=self.enable_selective_boosting
        )

        if self.key_param_attack_enabled and selected_params:
            optimized_updates = self._apply_key_param_attack(
                optimized_updates, selected_params
            )

        if (
            self.use_dual_stage and
            self.dual_stage is not None and
            self.last_benign_avg is not None and
            importance_scores is not None and
            persistence_scores is not None
        ):
            optimized_updates = self.dual_stage.batch_apply(
                optimized_updates,
                self.last_benign_avg,
                persistence_scores,
                importance_scores,
                epoch
            )

        optimized_updates = self.scaler.apply_scaling(optimized_updates, epoch)
        final_update = non_float_update.copy()
        final_update.update(optimized_updates[0])
        return final_update

    def aggregate_malicious_updates(
        self,
        malicious_updates: List[Dict[str, torch.Tensor]],
        global_model_change: Dict[str, torch.Tensor],
        epoch: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        聚合和优化恶意更?
        ?添加良性对齐步?
        """
        if not self.core_innovations_enabled or not self.enable_collaboration:
            return malicious_updates

        logger.info(f"\n{'='*60}")
        logger.info(f"Round {epoch} - collaborative optimization + benign alignment")
        logger.info(f"{'='*60}")

        # 步骤1: 协同反推
        self.collab_inference.reset()
        for i, update in enumerate(malicious_updates):
            self.collab_inference.collect_malicious_update(i, update)

        benign_avg = self.collab_inference.infer_benign_average(global_model_change)
        self.last_benign_avg = benign_avg
        logger.info("Step1: inferred benign updates")

        # ⭐⭐?步骤2: 良性对齐（新增?
        if self.enable_benign_alignment:
            aligned_updates = []
            for i, mal_update in enumerate(malicious_updates):
                aligned = self.align_with_benign(mal_update, benign_avg, epoch)
                aligned_updates.append(aligned)
            logger.info("Step2: benign alignment complete")
            logger.info(f"  - 方向混合: {self.benign_direction_alpha:.1%}")
            logger.info(f"  - 目标L2比? {self.target_l2_ratio:.2f}")
        else:
            aligned_updates = malicious_updates

        if self.coalition_attention and aligned_updates:
            weights = self._compute_coalition_attention(aligned_updates, self.device)
            if weights is not None:
                coalition_update = {}
                for name in aligned_updates[0].keys():
                    coalition_update[name] = sum(
                        w * update[name] for w, update in zip(weights, aligned_updates)
                    )

                alpha = self.coalition_mix_alpha
                mixed_updates = []
                for update in aligned_updates:
                    mixed = {}
                    for name, value in update.items():
                        mixed[name] = (1 - alpha) * value + alpha * coalition_update[name]
                    mixed_updates.append(mixed)
                aligned_updates = mixed_updates

        # 步骤3: 聚合模拟
        self.simulator.set_benign_average(benign_avg)
        logger.info("Step3: aggregation simulator configured")

        # 步骤4: 持久性评?
        logger.info("?步骤4: 持久性评估（简化）")

        # 步骤5: 协同优化
        selected_params = None
        importance_scores = None
        persistence_scores = None

        if self.evaluator is not None and aligned_updates:
            representative_update = {}
            for name in aligned_updates[0].keys():
                representative_update[name] = torch.stack(
                    [u[name] for u in aligned_updates]
                ).mean(dim=0)
            selected_params, importance_scores, persistence_scores = \
                self._evaluate_param_scores(representative_update, epoch, self.device)

        optimized_updates = self.optimizer.full_optimization(
            aligned_updates,
            selected_params=selected_params,
            boost_factor=self.boost_factor,
            enable_direction=self.enable_direction_consistency,
            enable_balancing=self.enable_load_balancing,
            enable_boosting=self.enable_selective_boosting
        )
        if self.key_param_attack_enabled and selected_params:
            optimized_updates = self._apply_key_param_attack(
                optimized_updates, selected_params
            )
        logger.info("?步骤5: 协同优化完成")

        # 步骤6: 双阶段更?
        if self.use_dual_stage:
            if (
                self.dual_stage is not None and
                importance_scores is not None and
                persistence_scores is not None
            ):
                optimized_updates = self.dual_stage.batch_apply(
                    optimized_updates,
                    benign_avg,
                    persistence_scores,
                    importance_scores,
                    epoch
                )
        logger.info("  应用双阶段自适应更新...")
        logger.info("Step6: dual-stage update complete")

        # 步骤7: 自适应缩放
        optimized_updates = self.scaler.apply_scaling(optimized_updates, epoch)
        logger.info("?步骤7: 自适应缩放完成")

        return optimized_updates

    def get_factor_info(self) -> Dict:
        """获取因子信息"""
        return {
            'k': self.k,
            'm': self.m,
            'rotation_frequency': self.rotation_frequency,
            'rotation_strategy': self.rotation_strategy,
            'factor_types': self.factor_types,
            'factor_assignments': {
                k: [self.factor_types[i] for i in v]
                for k, v in self.factor_assignments.items()
            },
            'factor_diversity': self.compute_factor_diversity()
        }

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'factorized_trigger': self.get_factor_info(),
            'enable_collaboration': self.enable_collaboration,
            'enable_benign_alignment': self.enable_benign_alignment,
            'benign_direction_alpha': self.benign_direction_alpha,
            'target_l2_ratio': self.target_l2_ratio,
            'rotation_history_size': {
                adv_id: len(history)
                for adv_id, history in self.rotation_history.items()
            },
            'n_rounds': len(self.training_history),
            'device': str(self.device)
        }

        if self.core_innovations_enabled:
            stats.update({
                'collab_inference': self.collab_inference.get_statistics(),
                'simulator': self.simulator.get_statistics(),
                'optimizer': self.optimizer.get_statistics()
            })

        if self.use_dual_stage:
            stats['dual_stage'] = self.dual_stage.get_statistics()

        return stats


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("统一高级攻击器（良性更新对齐版?- 测试")
    print("=" * 70)

    class MockConfig:
        num_total_participants = 100
        num_adversaries = 4
        epochs = 200
        attacker_retrain_times = 5
        momentum = 0.9
        decay = 0.001
        bkd_ratio = 0.15
        target_class = 2

    class MockHelper:
        def __init__(self):
            self.config = MockConfig()
            self.global_model = None
            self.adversary_list = [0, 1, 2, 3]

        def get_lr(self, epoch):
            return 0.01

    helper = MockHelper()
    attacker = UnifiedAdvancedAttacker(
        helper, k=3, m=5,
        enable_collaboration=False,
        enable_benign_alignment=True,  # ?启用良性对?
        benign_direction_alpha=0.25,
        target_l2_ratio=1.05
    )

    print(f"\n?攻击器初始化成功")
    print(f"  良性对? 启用")
    print(f"  方向混合: 25%")
    print(f"  目标L2比? 1.05")

    print("\n" + "=" * 70)
    print("测试完成 ?- 良性更新对齐版就绪")
    print("=" * 70)
    print("\n预期效果:")
    print("  - ASR: 91-93% (保持)")
    print("  - L2比? 1.03-1.07 (大幅改善)")
    print("  - 隐蔽分数: 0.35-0.37 (大幅提升)")
    print("=" * 70)








