"""
自适应Model Replacement攻击实现
支持动态调整缩放因子，平衡训练速度和隐蔽性

核心思路:
- 早期(ASR低): 大gamma → 快速建立后门
- 中期(ASR中): 中gamma → 稳定提升
- 后期(ASR高): 小gamma → 提高隐蔽性
"""

import torch
import copy
import numpy as np


class AdaptiveModelReplacementAttacker:
    """自适应Model Replacement攻击器"""
    
    def __init__(self, config, num_total_participants=100, num_sampled_participants=20):
        """
        初始化自适应MR攻击器
        
        Args:
            config: 配置对象
            num_total_participants: 总参与者数 n
            num_sampled_participants: 每轮采样数 η
        """
        self.config = config
        self.n = num_total_participants
        self.eta = num_sampled_participants
        
        # 基础缩放因子 (n/η)
        self.base_scale_factor = self.n / self.eta
        
        # 自适应配置
        self.adaptive_enabled = config.get('mr_adaptive', False)
        self.adaptive_strategy = config.get('mr_adaptive_strategy', 'asr_based')
        
        # 初始化策略
        if self.adaptive_enabled:
            self._init_adaptive_strategy()
        else:
            # 固定gamma
            self.current_gamma = config.get('scale_factor_gamma', 
                                           config.get('mr_scale_factor', 
                                                     self.base_scale_factor))
        
        # 历史记录
        self.asr_history = []
        self.gamma_history = []
        self.l2_ratio_history = []

        self._print_init_info()

    def _init_adaptive_strategy(self):
        """初始化自适应策略"""

        if self.adaptive_strategy == 'asr_based':
            # 基于ASR的分段调整
            default_schedule = [
                # (ASR下界, ASR上界, gamma值)
                (0.0, 0.50, 50.0),   # 阶段1: 激进加速
                (0.50, 0.85, 20.0),  # 阶段2: 稳定提升
                (0.85, 1.00, 5.0),   # 阶段3: 隐蔽优化
            ]
            self.asr_schedule = self.config.get('mr_asr_schedule', default_schedule)

        elif self.adaptive_strategy == 'epoch_based':
            # 基于epoch的分段调整
            default_schedule = [
                # (起始epoch, 结束epoch, gamma值)
                (0, 20, 50.0),
                (20, 40, 20.0),
                (40, 1000, 5.0),
            ]
            self.epoch_schedule = self.config.get('mr_epoch_schedule', default_schedule)

        elif self.adaptive_strategy == 'smooth':
            # 平滑衰减: gamma = gamma_max * exp(-decay_rate * ASR)
            self.gamma_max = self.config.get('mr_gamma_max', 50.0)
            self.gamma_min = self.config.get('mr_gamma_min', 5.0)
            self.decay_rate = self.config.get('mr_decay_rate', 3.0)

        elif self.adaptive_strategy == 'hybrid':
            # 混合策略: ASR为主，epoch为辅
            self.asr_schedule = self.config.get('mr_asr_schedule', [
                (0.0, 0.50, 50.0),
                (0.50, 0.85, 20.0),
                (0.85, 1.00, 5.0),
            ])
            # epoch约束: 避免长期使用大gamma
            self.max_epoch_high_gamma = self.config.get('mr_max_epoch_high_gamma', 30)
            self.max_epoch_mid_gamma = self.config.get('mr_max_epoch_mid_gamma', 60)

        else:
            raise ValueError(f"Unknown adaptive strategy: {self.adaptive_strategy}")

        # 初始gamma
        self.current_gamma = self._get_initial_gamma()

    def _get_initial_gamma(self):
        """获取初始gamma值"""
        if self.adaptive_strategy == 'asr_based':
            return self.asr_schedule[0][2]
        elif self.adaptive_strategy == 'epoch_based':
            return self.epoch_schedule[0][2]
        elif self.adaptive_strategy == 'smooth':
            return self.gamma_max
        elif self.adaptive_strategy == 'hybrid':
            return self.asr_schedule[0][2]
        else:
            return self.base_scale_factor

    def update_gamma(self, current_asr=None, current_epoch=None):
        """
        更新当前的gamma值

        Args:
            current_asr: 当前ASR (0-1)
            current_epoch: 当前epoch

        Returns:
            new_gamma: 更新后的gamma值
        """
        if not self.adaptive_enabled:
            return self.current_gamma

        old_gamma = self.current_gamma

        if self.adaptive_strategy == 'asr_based':
            new_gamma = self._update_gamma_asr_based(current_asr)

        elif self.adaptive_strategy == 'epoch_based':
            new_gamma = self._update_gamma_epoch_based(current_epoch)

        elif self.adaptive_strategy == 'smooth':
            new_gamma = self._update_gamma_smooth(current_asr)

        elif self.adaptive_strategy == 'hybrid':
            new_gamma = self._update_gamma_hybrid(current_asr, current_epoch)

        else:
            new_gamma = self.current_gamma

        # 更新并记录
        self.current_gamma = new_gamma

        if current_asr is not None:
            self.asr_history.append(current_asr)
        self.gamma_history.append(new_gamma)

        # 如果gamma变化了，打印信息
        if abs(new_gamma - old_gamma) > 0.1:
            print(f"\n  🔄 自适应MR: γ {old_gamma:.1f} → {new_gamma:.1f}", end="")
            if current_asr is not None:
                print(f" (ASR={current_asr:.2%})", end="")
            if current_epoch is not None:
                print(f" (Epoch={current_epoch})", end="")
            print()

        return new_gamma

    def _update_gamma_asr_based(self, current_asr):
        """基于ASR更新gamma"""
        if current_asr is None:
            return self.current_gamma

        # 查找当前ASR所在的区间
        for asr_min, asr_max, gamma in self.asr_schedule:
            if asr_min <= current_asr < asr_max:
                return gamma

        # 默认返回最后一个阶段的gamma
        return self.asr_schedule[-1][2]

    def _update_gamma_epoch_based(self, current_epoch):
        """基于epoch更新gamma"""
        if current_epoch is None:
            return self.current_gamma

        # 查找当前epoch所在的区间
        for epoch_start, epoch_end, gamma in self.epoch_schedule:
            if epoch_start <= current_epoch < epoch_end:
                return gamma

        # 超出范围，返回最后一个阶段的gamma
        return self.epoch_schedule[-1][2]

    def _update_gamma_smooth(self, current_asr):
        """平滑衰减更新gamma"""
        if current_asr is None:
            return self.current_gamma

        # 指数衰减: gamma = gamma_min + (gamma_max - gamma_min) * exp(-decay_rate * ASR)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * \
                np.exp(-self.decay_rate * current_asr)

        return float(gamma)

    def _update_gamma_hybrid(self, current_asr, current_epoch):
        """混合策略更新gamma"""
        # 首先基于ASR确定gamma
        if current_asr is not None:
            gamma_from_asr = self._update_gamma_asr_based(current_asr)
        else:
            gamma_from_asr = self.current_gamma

        # epoch约束
        if current_epoch is not None:
            if current_epoch >= self.max_epoch_mid_gamma:
                # 强制降到最小gamma
                gamma_from_epoch = self.asr_schedule[-1][2]
            elif current_epoch >= self.max_epoch_high_gamma:
                # 至少降到中等gamma
                gamma_from_epoch = min(gamma_from_asr, self.asr_schedule[1][2])
            else:
                gamma_from_epoch = gamma_from_asr

            return min(gamma_from_asr, gamma_from_epoch)

        return gamma_from_asr

    def scale_malicious_update(self, local_model, global_model):
        """
        应用Model Replacement缩放

        公式: L_malicious = γ * (X - G) + G
        其中: X = 恶意本地模型, G = 全局模型, γ = 当前缩放因子

        Args:
            local_model: 恶意客户端的本地模型
            global_model: 当前全局模型

        Returns:
            scaled_model: 缩放后的模型
        """
        scaled_model = copy.deepcopy(local_model)
        global_params = dict(global_model.named_parameters())
        injection_ratio = self.config.get('mr_injection_ratio', 1.0)

        with torch.no_grad():
            for name, param in scaled_model.named_parameters():
                if not param.requires_grad:
                    continue

                global_param = global_params[name]
                delta = param.data - global_param.data
                scaled_param = global_param.data + injection_ratio * self.current_gamma * delta
                param.data.copy_(scaled_param)

        return scaled_model

    def compute_l2_norm(self, model1, model2):
        """计算两个模型之间的L2范数"""
        l2_norm = 0.0
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(), model2.named_parameters()
            ):
                if param1.requires_grad:
                    diff = param1.data - param2.data
                    l2_norm += torch.norm(diff).item() ** 2

        return np.sqrt(l2_norm)

    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'current_gamma': self.current_gamma,
            'base_scale_factor': self.base_scale_factor,
            'adaptive_enabled': self.adaptive_enabled,
            'strategy': self.adaptive_strategy if self.adaptive_enabled else 'fixed',
        }

        if len(self.asr_history) > 0:
            stats['asr_history'] = self.asr_history
            stats['gamma_history'] = self.gamma_history
            stats['avg_gamma'] = np.mean(self.gamma_history)
            stats['final_asr'] = self.asr_history[-1] if self.asr_history else 0

        return stats

    def _print_init_info(self):
        """打印初始化信息"""
        print(f"\n{'='*60}")
        print(f"🔧 自适应Model Replacement初始化")
        print(f"{'='*60}")
        print(f"总参与者数 (n):      {self.n}")
        print(f"每轮采样数 (η):      {self.eta}")
        print(f"基础缩放因子 (n/η):  {self.base_scale_factor:.2f}")
        print(f"自适应模式:          {self.adaptive_enabled}")

        if self.adaptive_enabled:
            print(f"自适应策略:          {self.adaptive_strategy}")
            print(f"初始γ:               {self.current_gamma:.2f}")

            if self.adaptive_strategy == 'asr_based':
                print(f"\nASR-based策略配置:")
                for asr_min, asr_max, gamma in self.asr_schedule:
                    print(f"  ASR {asr_min:.0%}-{asr_max:.0%}: γ={gamma:.1f}")

            elif self.adaptive_strategy == 'epoch_based':
                print(f"\nEpoch-based策略配置:")
                for epoch_start, epoch_end, gamma in self.epoch_schedule:
                    print(f"  Epoch {epoch_start}-{epoch_end}: γ={gamma:.1f}")

            elif self.adaptive_strategy == 'smooth':
                print(f"\nSmooth策略配置:")
                print(f"  γ_max = {self.gamma_max:.1f}")
                print(f"  γ_min = {self.gamma_min:.1f}")
                print(f"  衰减率 = {self.decay_rate:.2f}")

            elif self.adaptive_strategy == 'hybrid':
                print(f"\nHybrid策略配置:")
                for asr_min, asr_max, gamma in self.asr_schedule:
                    print(f"  ASR {asr_min:.0%}-{asr_max:.0%}: γ={gamma:.1f}")
                print(f"  Epoch约束:")
                print(f"    高γ最大epoch: {self.max_epoch_high_gamma}")
                print(f"    中γ最大epoch: {self.max_epoch_mid_gamma}")
        else:
            print(f"固定γ:               {self.current_gamma:.2f}")

        print(f"{'='*60}\n")


# 兼容性: 保留原始类名
class ModelReplacementAttacker(AdaptiveModelReplacementAttacker):
    """向后兼容的别名"""
    pass
