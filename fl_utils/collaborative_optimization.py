"""
协同优化策略 (Collaborative Optimization)
⭐⭐

功能：
1. 方向一致性优化
2. 负载均衡分配
3. 自适应缩放

目标：
最大化多个恶意客户端的协同效果
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def _is_float_tensor(t: torch.Tensor) -> bool:
    return torch.is_floating_point(t) or torch.is_complex(t)




class CollaborativeOptimizer:
    """
    协同优化器
    
    优化多个恶意客户端的更新，使其协同效果最大化
    """
    
    def __init__(self, n_malicious: int = 10):
        """
        初始化协同优化器
        
        Args:
            n_malicious: 恶意客户端数量
        """
        self.n_malicious = n_malicious
        self.optimization_history = []
        
        logger.info(f"初始化协同优化器: {n_malicious} 个恶意客户端")
    
    def optimize_direction_consistency(
        self,
        updates: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Direction-consistency optimization.

        Aligns each update direction to the average direction.
        Skips non-floating tensors (e.g., batch norm counters).
        """
        if len(updates) == 0:
            return []

        param_names = updates[0].keys()

        optimized_updates = []
        for update in updates:
            optimized = {}
            for param_name in param_names:
                current_update = update[param_name]
                if not _is_float_tensor(current_update):
                    optimized[param_name] = current_update
                    continue

                all_updates = [u[param_name] for u in updates]
                if not all(_is_float_tensor(u) for u in all_updates):
                    optimized[param_name] = current_update
                    continue

                avg_direction = torch.stack(all_updates).mean(dim=0)
                avg_direction = avg_direction / (torch.norm(avg_direction) + 1e-10)

                projection_length = torch.dot(
                    current_update.flatten(),
                    avg_direction.flatten()
                )

                optimized[param_name] = avg_direction * projection_length

            optimized_updates.append(optimized)

        logger.debug("Completed direction-consistency optimization")
        return optimized_updates
    def optimize_load_balancing(
        self,
        updates: List[Dict[str, torch.Tensor]],
        target_norm: Optional[float] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Load-balancing optimization.

        Scales each malicious update to a similar norm.
        """
        if len(updates) == 0:
            return []

        norms = []
        for update in updates:
            total_norm = 0.0
            for param in update.values():
                if not _is_float_tensor(param):
                    continue
                total_norm += torch.norm(param).item() ** 2
            norms.append(np.sqrt(total_norm))

        if target_norm is None:
            target_norm = np.mean(norms) if norms else 0.0

        balanced_updates = []
        for update, norm in zip(updates, norms):
            scale = target_norm / (norm + 1e-10) if norm > 0 else 1.0
            balanced = {}
            for param_name, param in update.items():
                if _is_float_tensor(param):
                    balanced[param_name] = param * scale
                else:
                    balanced[param_name] = param
            balanced_updates.append(balanced)

        logger.debug(f"Completed load balancing: target_norm={target_norm:.4f}")
        return balanced_updates
    def optimize_selective_boosting(
        self,
        updates: List[Dict[str, torch.Tensor]],
        selected_params: List[str],
        boost_factor: float = 1.5
    ) -> List[Dict[str, torch.Tensor]]:
        """Selectively boost high-persistence parameters."""
        boosted_updates = []

        for update in updates:
            boosted = {}
            for param_name, param in update.items():
                if param_name in selected_params and _is_float_tensor(param):
                    boosted[param_name] = param * boost_factor
                else:
                    boosted[param_name] = param
            boosted_updates.append(boosted)

        logger.debug(f"Boosted {len(selected_params)} parameters (factor={boost_factor})")
        return boosted_updates
    def full_optimization(
        self,
        updates: List[Dict[str, torch.Tensor]],
        selected_params: Optional[List[str]] = None,
        boost_factor: float = 1.5,
        enable_direction: bool = True,
        enable_balancing: bool = True,
        enable_boosting: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """Full collaborative optimization."""
        optimized = updates

        # 1. direction consistency
        if enable_direction:
            optimized = self.optimize_direction_consistency(optimized)
            logger.info("Direction-consistency optimization complete")

        # 2. load balancing
        if enable_balancing:
            optimized = self.optimize_load_balancing(optimized)
            logger.info("Load-balancing optimization complete")

        # 3. selective boosting
        if enable_boosting and selected_params:
            optimized = self.optimize_selective_boosting(
                optimized, selected_params, boost_factor
            )
            logger.info("Selective boosting complete")

        self.optimization_history.append({
            'n_updates': len(optimized),
            'enable_direction': enable_direction,
            'enable_balancing': enable_balancing,
            'enable_boosting': enable_boosting
        })

        return optimized
    def compute_consistency_score(
        self,
        updates: List[Dict[str, torch.Tensor]]
    ) -> float:
        """Compute direction consistency score (0-1)."""
        if len(updates) < 2:
            return 1.0

        param_names = updates[0].keys()
        consistencies = []

        for param_name in param_names:
            if not all(_is_float_tensor(u[param_name]) for u in updates):
                continue
            vectors = [u[param_name].flatten() for u in updates]

            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    v1 = vectors[i]
                    v2 = vectors[j]

                    cos_sim = torch.dot(v1, v2) / (
                        torch.norm(v1) * torch.norm(v2) + 1e-10
                    )
                    similarities.append(cos_sim.item())

            if similarities:
                consistencies.append(np.mean(similarities))

        return np.mean(consistencies) if consistencies else 0.0
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'n_optimizations': len(self.optimization_history),
            'n_malicious': self.n_malicious
        }


class AdaptiveScaling:
    """
    自适应缩放策略
    
    根据训练进度动态调整更新幅度
    """
    
    def __init__(self, initial_scale: float = 1.0, decay_rate: float = 0.95):
        self.initial_scale = initial_scale
        self.decay_rate = decay_rate
        self.current_round = 0
        
    def get_scale(self, round_idx: Optional[int] = None) -> float:
        """
        获取当前轮次的缩放因子
        
        Args:
            round_idx: 轮次索引（可选）
            
        Returns:
            scale: 缩放因子
        """
        if round_idx is not None:
            self.current_round = round_idx
        
        # 指数衰减
        scale = self.initial_scale * (self.decay_rate ** self.current_round)
        return max(scale, 0.1)  # 最小缩放0.1
    
    def apply_scaling(
        self,
        updates: List[Dict[str, torch.Tensor]],
        round_idx: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        应用自适应缩放
        
        Args:
            updates: 更新列表
            round_idx: 轮次索引
            
        Returns:
            scaled_updates: 缩放后的更新
        """
        scale = self.get_scale(round_idx)
        
        scaled_updates = []
        for update in updates:
            scaled = {}
            for param_name, param in update.items():
                scaled[param_name] = param * scale
            scaled_updates.append(scaled)
        
        logger.debug(f"自适应缩放: round={round_idx}, scale={scale:.4f}")
        return scaled_updates


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("协同优化策略 - 功能测试")
    print("=" * 60)
    
    # 创建优化器
    optimizer = CollaborativeOptimizer(n_malicious=10)
    
    # 模拟10个恶意客户端的更新
    updates = []
    for i in range(10):
        update = {
            'layer1.weight': torch.randn(100, 50) * (0.1 + i * 0.01),
            'layer1.bias': torch.randn(100) * (0.1 + i * 0.01),
        }
        updates.append(update)
    
    # 计算初始一致性
    initial_consistency = optimizer.compute_consistency_score(updates)
    print(f"\n✓ 初始一致性: {initial_consistency:.4f}")
    
    # 方向一致性优化
    direction_optimized = optimizer.optimize_direction_consistency(updates)
    direction_consistency = optimizer.compute_consistency_score(direction_optimized)
    print(f"✓ 方向优化后一致性: {direction_consistency:.4f}")
    print(f"  提升: {(direction_consistency - initial_consistency):.4f}")
    
    # 负载均衡优化
    balanced = optimizer.optimize_load_balancing(updates)
    print(f"\n✓ 负载均衡完成")
    
    # 计算范数
    norms = []
    for update in balanced:
        norm = sum(torch.norm(p).item() ** 2 for p in update.values())
        norms.append(np.sqrt(norm))
    print(f"  范数标准差: {np.std(norms):.6f}")
    
    # 完整优化
    print("\n" + "=" * 60)
    print("完整优化")
    print("=" * 60)
    
    selected_params = ['layer1.weight']
    optimized = optimizer.full_optimization(
        updates,
        selected_params=selected_params,
        boost_factor=1.5
    )
    
    final_consistency = optimizer.compute_consistency_score(optimized)
    print(f"\n✓ 最终一致性: {final_consistency:.4f}")
    
    # 测试自适应缩放
    print("\n" + "=" * 60)
    print("自适应缩放")
    print("=" * 60)
    
    scaler = AdaptiveScaling(initial_scale=1.0, decay_rate=0.95)
    for round_idx in [0, 10, 20, 30, 40]:
        scale = scaler.get_scale(round_idx)
        print(f"轮次 {round_idx}: scale = {scale:.4f}")
