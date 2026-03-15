"""
持久性评估器 (Persistence Evaluator)
核心创新 ⭐⭐⭐⭐⭐

功能：
1. 三维度评估：稳定性 + 重要性 + 持久性
2. 量化稀释效应
3. 参数选择优化

创新意义：
首次将"稀释效应"从定性描述转化为可计算、可优化的量化指标
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PersistenceEvaluator:
    """
    三维度持久性评估框架
    
    评估维度：
    1. Stability - 对主任务的影响小
    2. Importance - 对后门的贡献大
    3. Persistence - 抗稀释能力强
    """
    
    def __init__(
        self,
        model: nn.Module,
        simulator,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化评估器
        
        Args:
            model: 模型
            simulator: 聚合模拟器
            weights: 三个维度的权重
        """
        self.model = model
        self.simulator = simulator
        
        # 默认权重配置（持久性权重最高）
        self.weights = weights or {
            'stability': 0.2,
            'importance': 0.3,
            'persistence': 0.5  # 最高权重 ⭐
        }
        
        # 历史记录
        self.evaluation_history = []
        
        logger.info(f"初始化持久性评估器，权重: {self.weights}")
    
    def evaluate_stability(
        self,
        param_gradients: Dict[str, torch.Tensor],
        gradient_history: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, float]:
        """
        维度1：稳定性评估
        
        目标：选择对主任务影响小的参数
        方法：分析梯度方差
        
        Stability(θ) = sigmoid(-Var(∇θ L_main))
        
        Args:
            param_gradients: 当前梯度
            gradient_history: 历史梯度（可选）
            
        Returns:
            stability_scores: {param_name: score}
        """
        stability_scores = {}
        
        for param_name, grad in param_gradients.items():
            if gradient_history and len(gradient_history) > 1:
                # 计算历史梯度方差
                grads = [h[param_name] for h in gradient_history if param_name in h]
                if len(grads) > 1:
                    grad_stack = torch.stack(grads)
                    variance = torch.var(grad_stack, dim=0).mean().item()
                else:
                    variance = 0.0
            else:
                # 使用当前梯度的标准差作为代理
                variance = torch.std(grad).item()
            
            # 稳定性分数（方差越小越稳定）
            stability = self._sigmoid(-variance)
            stability_scores[param_name] = stability
        
        return stability_scores
    
    def evaluate_importance(
        self,
        model: nn.Module,
        backdoor_data: torch.Tensor,
        backdoor_targets: torch.Tensor,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        维度2：重要性评估
        
        目标：选择对后门贡献大的参数
        方法：计算后门损失的梯度
        
        Importance(θ) = |∇θ L_backdoor|_normalized
        
        Args:
            model: 模型
            backdoor_data: 后门数据
            backdoor_targets: 后门目标
            criterion: 损失函数
            
        Returns:
            importance_scores: {param_name: score}
        """
        model.zero_grad()
        
        # 前向传播
        outputs = model(backdoor_data)
        loss = criterion(outputs, backdoor_targets)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度
        importance_scores = {}
        all_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                importance_scores[name] = grad_norm
                all_grads.append(grad_norm)
        
        # 归一化
        if all_grads:
            max_grad = max(all_grads)
            for name in importance_scores:
                importance_scores[name] /= (max_grad + 1e-10)
        
        return importance_scores
    
    def evaluate_persistence(
        self,
        malicious_update: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        维度3：持久性评估 ⭐⭐⭐
        
        目标：选择抗稀释能力强的参数
        方法：通过聚合模拟器预测保留比例
        
        Persistence(θ) = |Δθ_sim| / |Δθ_mal|
        
        Args:
            malicious_update: 恶意更新
            
        Returns:
            persistence_scores: {param_name: score}
        """
        # 模拟聚合
        sim_result = self.simulator.simulate_aggregation(malicious_update)
        
        persistence_scores = {}
        for param_name in malicious_update.keys():
            mal_norm = torch.norm(malicious_update[param_name]).item()
            sim_norm = torch.norm(sim_result[param_name]).item()
            
            # 持久性分数 = 保留比例
            persistence = sim_norm / (mal_norm + 1e-10)
            persistence_scores[param_name] = persistence
        
        return persistence_scores
    
    def compute_comprehensive_score(
        self,
        stability_scores: Dict[str, float],
        importance_scores: Dict[str, float],
        persistence_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算综合评分
        
        Score = α×Stability + β×Importance + γ×Persistence
        
        Args:
            stability_scores: 稳定性分数
            importance_scores: 重要性分数
            persistence_scores: 持久性分数
            
        Returns:
            comprehensive_scores: 综合分数
        """
        comprehensive_scores = {}
        
        # 获取所有参数名
        param_names = set(stability_scores.keys()) & \
                     set(importance_scores.keys()) & \
                     set(persistence_scores.keys())
        
        for param_name in param_names:
            score = (
                self.weights['stability'] * stability_scores[param_name] +
                self.weights['importance'] * importance_scores[param_name] +
                self.weights['persistence'] * persistence_scores[param_name]
            )
            comprehensive_scores[param_name] = score
        
        return comprehensive_scores
    
    def select_parameters(
        self,
        comprehensive_scores: Dict[str, float],
        selection_ratio: float = 0.3,
        min_score: Optional[float] = None
    ) -> List[str]:
        """
        选择参数
        
        Args:
            comprehensive_scores: 综合分数
            selection_ratio: 选择比例
            min_score: 最低分数阈值（可选）
            
        Returns:
            selected_params: 选中的参数名列表
        """
        # 按分数排序
        sorted_params = sorted(
            comprehensive_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择top-k
        n_select = int(len(sorted_params) * selection_ratio)
        selected = [name for name, score in sorted_params[:n_select]]
        
        # 应用最低分数阈值
        if min_score is not None:
            selected = [
                name for name in selected 
                if comprehensive_scores[name] >= min_score
            ]
        
        logger.info(f"选择了 {len(selected)}/{len(sorted_params)} 个参数")
        
        return selected
    
    def full_evaluation(
        self,
        model: nn.Module,
        backdoor_data: torch.Tensor,
        backdoor_targets: torch.Tensor,
        malicious_update: Dict[str, torch.Tensor],
        criterion: nn.Module,
        param_gradients: Optional[Dict[str, torch.Tensor]] = None,
        gradient_history: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        完整的三维度评估
        
        Args:
            model: 模型
            backdoor_data: 后门数据
            backdoor_targets: 后门目标
            malicious_update: 恶意更新
            criterion: 损失函数
            param_gradients: 参数梯度（可选）
            gradient_history: 梯度历史（可选）
            
        Returns:
            comprehensive_scores: 综合分数
            details: 详细评估结果
        """
        # 1. 稳定性评估
        if param_gradients is not None:
            stability_scores = self.evaluate_stability(
                param_gradients, gradient_history
            )
        else:
            # 如果没有提供梯度，使用默认值
            stability_scores = {
                name: 0.5 for name in malicious_update.keys()
            }
        
        # 2. 重要性评估
        importance_scores = self.evaluate_importance(
            model, backdoor_data, backdoor_targets, criterion
        )
        
        # 3. 持久性评估
        persistence_scores = self.evaluate_persistence(malicious_update)
        
        # 4. 综合评分
        comprehensive_scores = self.compute_comprehensive_score(
            stability_scores, importance_scores, persistence_scores
        )
        
        # 记录评估结果
        evaluation_result = {
            'stability': stability_scores,
            'importance': importance_scores,
            'persistence': persistence_scores,
            'comprehensive': comprehensive_scores
        }
        self.evaluation_history.append(evaluation_result)
        
        # 详细信息
        details = {
            'n_params': len(comprehensive_scores),
            'avg_stability': np.mean(list(stability_scores.values())),
            'avg_importance': np.mean(list(importance_scores.values())),
            'avg_persistence': np.mean(list(persistence_scores.values())),
            'avg_comprehensive': np.mean(list(comprehensive_scores.values())),
            'max_score': max(comprehensive_scores.values()),
            'min_score': min(comprehensive_scores.values())
        }
        
        return comprehensive_scores, details
    
    @staticmethod
    def _sigmoid(x: float, k: float = 1.0) -> float:
        """Sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-k * x))
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.evaluation_history:
            return {}
        
        # 计算平均分数
        avg_scores = {
            'stability': [],
            'importance': [],
            'persistence': [],
            'comprehensive': []
        }
        
        for eval_result in self.evaluation_history:
            for key in avg_scores.keys():
                if key in eval_result:
                    avg_scores[key].append(
                        np.mean(list(eval_result[key].values()))
                    )
        
        return {
            'n_evaluations': len(self.evaluation_history),
            'avg_stability': np.mean(avg_scores['stability']),
            'avg_importance': np.mean(avg_scores['importance']),
            'avg_persistence': np.mean(avg_scores['persistence']),
            'avg_comprehensive': np.mean(avg_scores['comprehensive']),
            'weights': self.weights
        }


class AdaptiveWeightAdjuster:
    """
    自适应权重调整器
    
    根据训练阶段动态调整三个维度的权重
    """
    
    def __init__(self, total_rounds: int = 50):
        self.total_rounds = total_rounds
        
    def get_weights(self, current_round: int) -> Dict[str, float]:
        """
        获取当前轮次的权重
        
        策略：
        - 早期（<20轮）：更注重稳定性
        - 中期（20-35轮）：平衡
        - 后期（>35轮）：更注重持久性
        
        Args:
            current_round: 当前轮次
            
        Returns:
            weights: 权重配置
        """
        progress = current_round / self.total_rounds
        
        if progress < 0.4:  # 早期
            return {
                'stability': 0.4,
                'importance': 0.3,
                'persistence': 0.3
            }
        elif progress < 0.7:  # 中期
            return {
                'stability': 0.2,
                'importance': 0.3,
                'persistence': 0.5
            }
        else:  # 后期
            return {
                'stability': 0.1,
                'importance': 0.3,
                'persistence': 0.6
            }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("持久性评估器 - 功能测试")
    print("=" * 60)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    
    # 创建模拟器
    from aggregation_simulator import AggregationSimulator
    simulator = AggregationSimulator(n_total=100, n_malicious=10)
    benign_avg = {name: torch.randn_like(param) * 0.01 
                  for name, param in model.named_parameters()}
    simulator.set_benign_average(benign_avg)
    
    # 创建评估器
    evaluator = PersistenceEvaluator(model, simulator)
    
    # 准备测试数据
    backdoor_data = torch.randn(32, 10)
    backdoor_targets = torch.randint(0, 2, (32,))
    malicious_update = {name: torch.randn_like(param) * 0.05 
                       for name, param in model.named_parameters()}
    criterion = nn.CrossEntropyLoss()
    
    # 完整评估
    scores, details = evaluator.full_evaluation(
        model, backdoor_data, backdoor_targets,
        malicious_update, criterion
    )
    
    print("\n✓ 评估完成")
    print(f"  参数数量: {details['n_params']}")
    print(f"  平均稳定性: {details['avg_stability']:.4f}")
    print(f"  平均重要性: {details['avg_importance']:.4f}")
    print(f"  平均持久性: {details['avg_persistence']:.4f}")
    print(f"  平均综合分: {details['avg_comprehensive']:.4f}")
    
    # 参数选择
    selected = evaluator.select_parameters(scores, selection_ratio=0.3)
    print(f"\n✓ 选择参数数: {len(selected)}")
    
    # 测试自适应权重
    print("\n" + "=" * 60)
    print("自适应权重调整")
    print("=" * 60)
    
    adjuster = AdaptiveWeightAdjuster(total_rounds=50)
    for round_idx in [0, 10, 20, 30, 40, 49]:
        weights = adjuster.get_weights(round_idx)
        print(f"轮次 {round_idx}: {weights}")
