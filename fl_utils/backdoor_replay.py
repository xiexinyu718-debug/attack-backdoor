"""
后门重放机制 (Backdoor Replay)
受 Act in Collusion 启发 ⭐⭐

功能：
1. 缓解灾难性遗忘
2. 混合历史样本训练
3. 基于持久性的智能重放

扩展：
- 持久性加权采样
- 自适应缓冲区管理
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class BackdoorReplayBuffer:
    """
    后门样本重放缓冲区
    
    存储历史后门样本，用于缓解遗忘
    """
    
    def __init__(
        self,
        buffer_size: int = 500,
        sample_selection: str = 'uniform'  # 'uniform', 'importance', 'persistence'
    ):
        """
        初始化重放缓冲区
        
        Args:
            buffer_size: 缓冲区大小
            sample_selection: 采样策略
        """
        self.buffer_size = buffer_size
        self.sample_selection = sample_selection
        
        # 缓冲区存储
        self.buffer = deque(maxlen=buffer_size)
        
        # 元数据
        self.importance_scores = deque(maxlen=buffer_size)
        self.persistence_scores = deque(maxlen=buffer_size)
        
        logger.info(f"初始化重放缓冲区: size={buffer_size}, strategy={sample_selection}")
    
    def add_samples(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        importance: Optional[float] = None,
        persistence: Optional[float] = None
    ):
        """
        添加样本到缓冲区
        
        Args:
            data: 数据
            targets: 标签
            importance: 重要性分数（可选）
            persistence: 持久性分数（可选）
        """
        for i in range(len(data)):
            sample = {
                'data': data[i].cpu(),
                'target': targets[i].cpu(),
                'importance': importance if importance is not None else 1.0,
                'persistence': persistence if persistence is not None else 1.0
            }
            self.buffer.append(sample)
        
        logger.debug(f"添加 {len(data)} 个样本到缓冲区 (总计: {len(self.buffer)})")
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从缓冲区采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            data, targets: 采样的数据和标签
        """
        if len(self.buffer) == 0:
            return None, None
        
        # 实际采样数量
        n_samples = min(batch_size, len(self.buffer))
        
        # 根据策略采样
        if self.sample_selection == 'uniform':
            indices = np.random.choice(len(self.buffer), n_samples, replace=False)
        
        elif self.sample_selection == 'importance':
            # 基于重要性的加权采样
            weights = [s['importance'] for s in self.buffer]
            weights = np.array(weights) / (sum(weights) + 1e-10)
            indices = np.random.choice(len(self.buffer), n_samples, replace=False, p=weights)
        
        elif self.sample_selection == 'persistence':
            # 基于持久性的加权采样 ⭐
            weights = [s['persistence'] for s in self.buffer]
            weights = np.array(weights) / (sum(weights) + 1e-10)
            indices = np.random.choice(len(self.buffer), n_samples, replace=False, p=weights)
        
        else:
            raise ValueError(f"未知的采样策略: {self.sample_selection}")
        
        # 收集样本
        data_list = []
        targets_list = []
        for idx in indices:
            sample = list(self.buffer)[idx]
            data_list.append(sample['data'])
            targets_list.append(sample['target'])
        
        data = torch.stack(data_list)
        targets = torch.stack(targets_list)
        
        return data, targets
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if len(self.buffer) == 0:
            return {'size': 0}
        
        importances = [s['importance'] for s in self.buffer]
        persistences = [s['persistence'] for s in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.buffer_size,
            'avg_importance': np.mean(importances),
            'avg_persistence': np.mean(persistences),
            'strategy': self.sample_selection
        }


class BackdoorReplayMechanism:
    """
    后门重放机制
    
    整合样本缓冲和训练策略
    """
    
    def __init__(
        self,
        buffer_size: int = 500,
        replay_ratio: float = 0.5,
        sample_selection: str = 'persistence'
    ):
        """
        初始化重放机制
        
        Args:
            buffer_size: 缓冲区大小
            replay_ratio: 重放样本比例（0-1）
            sample_selection: 采样策略
        """
        self.buffer = BackdoorReplayBuffer(buffer_size, sample_selection)
        self.replay_ratio = replay_ratio
        
        # 训练历史
        self.training_history = []
        
        logger.info(f"初始化重放机制: ratio={replay_ratio}")
    
    def prepare_training_batch(
        self,
        current_data: torch.Tensor,
        current_targets: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备训练批次（混合当前和历史样本）
        
        Args:
            current_data: 当前批次数据
            current_targets: 当前批次标签
            batch_size: 总批次大小
            
        Returns:
            mixed_data, mixed_targets: 混合后的数据和标签
        """
        # 计算重放样本数量
        n_replay = int(batch_size * self.replay_ratio)
        n_current = batch_size - n_replay
        
        # 从缓冲区采样
        if n_replay > 0 and len(self.buffer.buffer) > 0:
            replay_data, replay_targets = self.buffer.sample(n_replay)
            
            if replay_data is not None:
                # 从当前数据中随机选择
                current_indices = torch.randperm(len(current_data))[:n_current]
                current_subset = current_data[current_indices]
                target_subset = current_targets[current_indices]
                
                # 混合
                mixed_data = torch.cat([current_subset, replay_data.to(current_data.device)])
                mixed_targets = torch.cat([target_subset, replay_targets.to(current_targets.device)])
                
                # 打乱
                shuffle_idx = torch.randperm(len(mixed_data))
                mixed_data = mixed_data[shuffle_idx]
                mixed_targets = mixed_targets[shuffle_idx]
                
                logger.debug(f"混合批次: {n_current} 当前 + {n_replay} 重放")
                
                return mixed_data, mixed_targets
        
        # 如果缓冲区为空或不需要重放
        return current_data, current_targets
    
    def update_buffer(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        importance: Optional[float] = None,
        persistence: Optional[float] = None
    ):
        """
        更新缓冲区
        
        Args:
            data: 数据
            targets: 标签
            importance: 重要性分数
            persistence: 持久性分数
        """
        self.buffer.add_samples(data, targets, importance, persistence)
    
    def train_with_replay(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        current_data: torch.Tensor,
        current_targets: torch.Tensor,
        batch_size: int
    ) -> float:
        """
        使用重放机制训练
        
        Args:
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            current_data: 当前数据
            current_targets: 当前标签
            batch_size: 批次大小
            
        Returns:
            loss: 损失值
        """
        # 准备混合批次
        mixed_data, mixed_targets = self.prepare_training_batch(
            current_data, current_targets, batch_size
        )
        
        # 训练
        model.train()
        optimizer.zero_grad()
        
        outputs = model(mixed_data)
        loss = criterion(outputs, mixed_targets)
        
        loss.backward()
        optimizer.step()
        
        # 记录
        self.training_history.append({
            'loss': loss.item(),
            'n_replay': int(batch_size * self.replay_ratio),
            'n_current': batch_size - int(batch_size * self.replay_ratio)
        })
        
        return loss.item()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        buffer_stats = self.buffer.get_statistics()
        
        stats = {
            'buffer': buffer_stats,
            'replay_ratio': self.replay_ratio,
            'n_training_batches': len(self.training_history)
        }
        
        if self.training_history:
            stats['avg_loss'] = np.mean([h['loss'] for h in self.training_history])
        
        return stats


class PersistenceWeightedReplay(BackdoorReplayMechanism):
    """
    基于持久性的加权重放
    
    我们的扩展：优先重放高持久性参数对应的样本
    """
    
    def __init__(
        self,
        buffer_size: int = 500,
        replay_ratio: float = 0.5,
        persistence_threshold: float = 0.3
    ):
        super().__init__(buffer_size, replay_ratio, 'persistence')
        self.persistence_threshold = persistence_threshold
        
        logger.info(f"初始化持久性加权重放: threshold={persistence_threshold}")
    
    def add_samples_with_persistence(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        persistence_scores: Dict[str, float]
    ):
        """
        添加样本并计算持久性权重
        
        Args:
            data: 数据
            targets: 标签
            persistence_scores: 持久性分数字典
        """
        # 计算平均持久性
        avg_persistence = np.mean(list(persistence_scores.values()))
        
        # 只添加高持久性样本
        if avg_persistence >= self.persistence_threshold:
            self.buffer.add_samples(
                data, targets,
                importance=1.0,
                persistence=avg_persistence
            )
            logger.debug(f"添加高持久性样本 (P={avg_persistence:.4f})")
        else:
            logger.debug(f"跳过低持久性样本 (P={avg_persistence:.4f})")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("后门重放机制 - 功能测试")
    print("=" * 60)
    
    # 创建缓冲区
    buffer = BackdoorReplayBuffer(buffer_size=100, sample_selection='persistence')
    
    # 添加样本
    for i in range(5):
        data = torch.randn(10, 32)
        targets = torch.randint(0, 10, (10,))
        importance = np.random.rand()
        persistence = np.random.rand()
        
        buffer.add_samples(data, targets, importance, persistence)
        print(f"批次 {i+1}: 添加 10 个样本")
    
    # 采样
    sampled_data, sampled_targets = buffer.sample(32)
    print(f"\n✓ 采样: {len(sampled_data)} 个样本")
    
    # 统计
    stats = buffer.get_statistics()
    print("\n" + "=" * 60)
    print("缓冲区统计")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 测试重放机制
    print("\n" + "=" * 60)
    print("重放机制测试")
    print("=" * 60)
    
    replay = BackdoorReplayMechanism(
        buffer_size=100,
        replay_ratio=0.5,
        sample_selection='persistence'
    )
    
    # 更新缓冲区
    data = torch.randn(20, 32)
    targets = torch.randint(0, 10, (20,))
    replay.update_buffer(data, targets, importance=0.8, persistence=0.9)
    
    # 准备混合批次
    current_data = torch.randn(64, 32)
    current_targets = torch.randint(0, 10, (64,))
    
    mixed_data, mixed_targets = replay.prepare_training_batch(
        current_data, current_targets, batch_size=64
    )
    
    print(f"\n✓ 混合批次: {len(mixed_data)} 个样本")
    print(f"  预期: 32 当前 + 32 重放")
    
    # 统计
    replay_stats = replay.get_statistics()
    print("\n重放统计:")
    for key, value in replay_stats.items():
        print(f"{key}: {value}")
