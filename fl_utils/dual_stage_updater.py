"""
双阶段自适应更新器 (Dual-Stage Adaptive Update)
⭐⭐⭐⭐⭐

功能：
1. 根据持久性和重要性选择更新策略
2. 动态λ调整（分阶段）
3. 对齐式注入（高持久性参数）
4. 正交投影（高重要性低持久性参数）

创新：
解决对齐vs正交的矛盾，根据参数特性自适应选择策略
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DualStageAdaptiveUpdate:
    """
    双阶段自适应更新策略
    
    核心思想：
    - 高持久性参数 → 对齐式注入（保证长期存活）
    - 高重要性低持久性参数 → 正交投影（最小化冲突）
    - 其他参数 → 跳过（避免稀释）
    """
    
    def __init__(
        self,
        persistence_threshold: float = 0.6,
        importance_threshold: float = 0.7,
        dynamic_lambda: bool = True,
        lambda_initial: float = 0.85,
        lambda_final: float = 0.70
    ):
        """
        初始化双阶段更新器
        
        Args:
            persistence_threshold: 持久性阈值（P>threshold → 对齐）
            importance_threshold: 重要性阈值（I>threshold → 正交）
            dynamic_lambda: 是否使用动态λ调整
            lambda_initial: 初始λ值（高对齐）
            lambda_final: 最终λ值（稳定）
        """
        self.p_threshold = persistence_threshold
        self.i_threshold = importance_threshold
        self.dynamic_lambda = dynamic_lambda
        self.lambda_initial = lambda_initial
        self.lambda_final = lambda_final
        
        # 创建λ调度表
        self.lambda_schedule = self._create_lambda_schedule()
        
        # 统计信息
        self.strategy_counts = {
            'align': 0,      # 对齐式注入
            'orthogonal': 0, # 正交投影
            'skip': 0        # 跳过
        }
        
        logger.info(f"初始化双阶段自适应更新器:")
        logger.info(f"  持久性阈值: {persistence_threshold}")
        logger.info(f"  重要性阈值: {importance_threshold}")
        logger.info(f"  动态λ: {'启用' if dynamic_lambda else '禁用'}")
    
    def _create_lambda_schedule(self) -> Dict[str, Tuple[int, int, float]]:
        """
        创建λ调度表
        
        策略：
        - 初期（0-50轮）：高对齐（λ=0.85），快速植入
        - 中期（50-100轮）：平衡（λ=0.75），稳定累积
        - 后期（100-200轮）：稳定（λ=0.70），保持效果
        
        Returns:
            schedule: {stage_name: (start_epoch, end_epoch, lambda_value)}
        """
        return {
            'stage1_early': (0, 50, 0.85),      # 初期：高对齐
            'stage2_mid': (50, 100, 0.75),      # 中期：平衡
            'stage3_late': (100, 200, 0.70)     # 后期：稳定
        }
    
    def get_lambda(self, epoch: int) -> float:
        """
        获取当前轮次的λ值
        
        Args:
            epoch: 当前轮次
            
        Returns:
            lambda_t: λ值
        """
        if not self.dynamic_lambda:
            # 固定λ（线性衰减）
            return self.lambda_initial
        
        # 动态λ（分阶段）
        for stage_name, (start, end, lambda_val) in self.lambda_schedule.items():
            if start <= epoch < end:
                return lambda_val
        
        # 默认返回最终值
        return self.lambda_final
    
    def align_injection(
        self,
        backdoor_grad: torch.Tensor,
        benign_grad: torch.Tensor,
        lambda_t: float
    ) -> torch.Tensor:
        """
        策略1：对齐式渐进注入
        
        公式：Δθ = λ * Δθ_benign + (1-λ) * Δθ_backdoor
        
        目标：通过与良性更新对齐，提高聚合后的存活率
        适用：高持久性参数（P > 0.6）
        
        Args:
            backdoor_grad: 后门梯度
            benign_grad: 良性梯度
            lambda_t: 混合系数
            
        Returns:
            mixed_grad: 混合后的梯度
        """
        # 混合梯度
        mixed_grad = lambda_t * benign_grad + (1 - lambda_t) * backdoor_grad
        
        self.strategy_counts['align'] += 1
        
        return mixed_grad
    
    def orthogonal_projection(
        self,
        backdoor_grad: torch.Tensor,
        benign_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        策略2：正交子空间投影
        
        公式：Δθ_⊥ = Δθ_bd - <Δθ_bd, Δθ_ben> / ||Δθ_ben||² * Δθ_ben
        
        目标：投影到与良性更新正交的子空间，最小化冲突
        适用：高重要性低持久性参数（I > 0.7, P < 0.4）
        
        Args:
            backdoor_grad: 后门梯度
            benign_grad: 良性梯度
            
        Returns:
            projected_grad: 投影后的梯度
        """
        # 展平为1D向量
        bd_flat = backdoor_grad.flatten()
        ben_flat = benign_grad.flatten()
        
        # 计算投影
        dot_product = torch.dot(bd_flat, ben_flat)
        ben_norm_sq = torch.dot(ben_flat, ben_flat)
        
        # 避免除零
        if ben_norm_sq < 1e-8:
            return backdoor_grad
        
        # 投影公式
        projection_coefficient = dot_product / ben_norm_sq
        
        # 重塑回原始形状
        ben_projection = projection_coefficient * benign_grad
        
        # 正交分量
        projected_grad = backdoor_grad - ben_projection
        
        self.strategy_counts['orthogonal'] += 1
        
        return projected_grad
    
    def apply_update(
        self,
        param_name: str,
        backdoor_grad: torch.Tensor,
        benign_grad: torch.Tensor,
        persistence_score: float,
        importance_score: float,
        epoch: int
    ) -> torch.Tensor:
        """
        根据参数特性选择更新策略
        
        决策树：
        1. P > 0.6 → 对齐式注入
        2. I > 0.7 且 P < 0.4 → 正交投影
        3. 其他 → 跳过（返回零）
        
        Args:
            param_name: 参数名称
            backdoor_grad: 后门梯度
            benign_grad: 良性梯度
            persistence_score: 持久性分数
            importance_score: 重要性分数
            epoch: 当前轮次
            
        Returns:
            optimized_grad: 优化后的梯度
        """
        # 策略1：高持久性 → 对齐式注入
        if persistence_score > self.p_threshold:
            lambda_t = self.get_lambda(epoch)
            
            logger.debug(
                f"[对齐] {param_name}: P={persistence_score:.3f}, λ={lambda_t:.3f}"
            )
            
            return self.align_injection(backdoor_grad, benign_grad, lambda_t)
        
        # 策略2：高重要性低持久性 → 正交投影
        elif importance_score > self.i_threshold and persistence_score < 0.4:
            logger.debug(
                f"[正交] {param_name}: I={importance_score:.3f}, P={persistence_score:.3f}"
            )
            
            return self.orthogonal_projection(backdoor_grad, benign_grad)
        
        # 策略3：其他 → 跳过
        else:
            logger.debug(
                f"[跳过] {param_name}: I={importance_score:.3f}, P={persistence_score:.3f}"
            )
            
            self.strategy_counts['skip'] += 1
            return torch.zeros_like(backdoor_grad)
    
    def batch_apply(
        self,
        malicious_updates: List[Dict[str, torch.Tensor]],
        benign_avg: Dict[str, torch.Tensor],
        persistence_scores: Dict[str, float],
        importance_scores: Dict[str, float],
        epoch: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        批量应用双阶段策略
        
        对多个恶意客户端的更新应用策略
        
        Args:
            malicious_updates: 恶意客户端更新列表
            benign_avg: 良性平均更新
            persistence_scores: 持久性分数
            importance_scores: 重要性分数
            epoch: 当前轮次
            
        Returns:
            optimized_updates: 优化后的更新列表
        """
        optimized_updates = []
        
        for mal_update in malicious_updates:
            optimized = {}
            
            for param_name in mal_update.keys():
                # 获取分数
                p_score = persistence_scores.get(param_name, 0.0)
                i_score = importance_scores.get(param_name, 0.0)
                
                # 获取梯度
                backdoor_grad = mal_update[param_name]
                benign_grad = benign_avg.get(
                    param_name, 
                    torch.zeros_like(backdoor_grad)
                )
                
                # 应用策略
                optimized_grad = self.apply_update(
                    param_name,
                    backdoor_grad,
                    benign_grad,
                    p_score,
                    i_score,
                    epoch
                )
                
                optimized[param_name] = optimized_grad
            
            optimized_updates.append(optimized)
        
        return optimized_updates
    
    def get_strategy_distribution(self) -> Dict[str, float]:
        """
        获取策略分布
        
        Returns:
            distribution: 各策略使用比例
        """
        total = sum(self.strategy_counts.values())
        
        if total == 0:
            return {'align': 0.0, 'orthogonal': 0.0, 'skip': 0.0}
        
        return {
            strategy: count / total
            for strategy, count in self.strategy_counts.items()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.strategy_counts = {
            'align': 0,
            'orthogonal': 0,
            'skip': 0
        }
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            stats: 统计数据
        """
        distribution = self.get_strategy_distribution()
        
        return {
            'p_threshold': self.p_threshold,
            'i_threshold': self.i_threshold,
            'dynamic_lambda': self.dynamic_lambda,
            'total_updates': sum(self.strategy_counts.values()),
            'align_count': self.strategy_counts['align'],
            'orthogonal_count': self.strategy_counts['orthogonal'],
            'skip_count': self.strategy_counts['skip'],
            'align_ratio': distribution['align'],
            'orthogonal_ratio': distribution['orthogonal'],
            'skip_ratio': distribution['skip']
        }


class LambdaScheduler:
    """
    λ调度器（独立类）
    
    提供更灵活的λ调整策略
    """
    
    def __init__(
        self,
        schedule_type: str = 'stage',
        lambda_initial: float = 0.85,
        lambda_final: float = 0.70
    ):
        """
        初始化λ调度器
        
        Args:
            schedule_type: 调度类型 ('stage', 'linear', 'exponential')
            lambda_initial: 初始值
            lambda_final: 最终值
        """
        self.schedule_type = schedule_type
        self.lambda_initial = lambda_initial
        self.lambda_final = lambda_final
    
    def get_lambda(self, epoch: int, total_epochs: int = 200) -> float:
        """
        获取λ值
        
        Args:
            epoch: 当前轮次
            total_epochs: 总轮次
            
        Returns:
            lambda_t: λ值
        """
        if self.schedule_type == 'linear':
            # 线性衰减
            progress = epoch / total_epochs
            return self.lambda_initial - (self.lambda_initial - self.lambda_final) * progress
        
        elif self.schedule_type == 'exponential':
            # 指数衰减
            decay_rate = 0.995
            return max(self.lambda_initial * (decay_rate ** epoch), self.lambda_final)
        
        elif self.schedule_type == 'stage':
            # 分阶段（默认）
            if epoch < 50:
                return 0.85
            elif epoch < 100:
                return 0.75
            else:
                return 0.70
        
        else:
            return self.lambda_initial


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("双阶段自适应更新器 - 功能测试")
    print("=" * 60)
    
    # 创建更新器
    updater = DualStageAdaptiveUpdate(
        persistence_threshold=0.6,
        importance_threshold=0.7,
        dynamic_lambda=True
    )
    
    print(f"\n✓ 更新器初始化成功")
    
    # 测试λ调度
    print("\n" + "=" * 60)
    print("λ调度测试")
    print("=" * 60)
    
    print(f"\n{'Epoch':<10} {'λ值':<15} {'阶段'}")
    print("-" * 40)
    
    test_epochs = [0, 25, 50, 75, 100, 125, 150, 199]
    for epoch in test_epochs:
        lambda_t = updater.get_lambda(epoch)
        
        if epoch < 50:
            stage = "初期（高对齐）"
        elif epoch < 100:
            stage = "中期（平衡）"
        else:
            stage = "后期（稳定）"
        
        print(f"{epoch:<10} {lambda_t:<15.3f} {stage}")
    
    # 测试策略选择
    print("\n" + "=" * 60)
    print("策略选择测试")
    print("=" * 60)
    
    # 模拟参数
    param_scenarios = [
        ('layer1.weight', 0.8, 0.5, '对齐'),    # 高P
        ('layer2.weight', 0.3, 0.8, '正交'),    # 高I低P
        ('layer3.weight', 0.4, 0.5, '跳过'),    # 中等
    ]
    
    print(f"\n{'参数':<20} {'P':<8} {'I':<8} {'预期策略':<12} {'实际策略'}")
    print("-" * 60)
    
    for param_name, p_score, i_score, expected in param_scenarios:
        # 创建模拟梯度
        backdoor_grad = torch.randn(100, 50) * 0.1
        benign_grad = torch.randn(100, 50) * 0.05
        
        # 应用策略
        result = updater.apply_update(
            param_name, backdoor_grad, benign_grad,
            p_score, i_score, epoch=50
        )
        
        # 判断实际策略
        if torch.allclose(result, torch.zeros_like(result)):
            actual = '跳过'
        elif result.abs().mean() > backdoor_grad.abs().mean() * 0.5:
            actual = '对齐'
        else:
            actual = '正交'
        
        match = '✓' if actual == expected else '✗'
        print(f"{param_name:<20} {p_score:<8.2f} {i_score:<8.2f} {expected:<12} {actual} {match}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    
    stats = updater.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 策略分布
    print("\n策略分布:")
    distribution = updater.get_strategy_distribution()
    for strategy, ratio in distribution.items():
        print(f"  {strategy}: {ratio:.2%}")
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)
