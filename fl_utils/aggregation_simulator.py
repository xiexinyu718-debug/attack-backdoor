"""
聚合模拟器 (Aggregation Simulator)
核心创新 ⭐⭐⭐

功能：
1. 预测参数在联邦聚合后的状态
2. 支持多轮模拟
3. 用于持久性评估

数学原理：
Δθ_sim = (1/n) × [(n-m)×Δθ_benign + m×Δθ_mal]
"""

import numpy as np
import torch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def _is_float_tensor(t: torch.Tensor) -> bool:
    return torch.is_floating_point(t) or torch.is_complex(t)




class AggregationSimulator:
    """
    聚合模拟器
    
    预测参数在经过联邦平均聚合后的变化
    """
    
    def __init__(self, n_total: int = 100, n_malicious: int = 10):
        """
        初始化聚合模拟器
        
        Args:
            n_total: 总客户端数量
            n_malicious: 恶意客户端数量
        """
        self.n = n_total
        self.m = n_malicious
        self.benign_avg = None
        self.simulation_history = []
        
        logger.info(f"初始化聚合模拟器: n={n_total}, m={n_malicious}")
    
    def set_benign_average(self, benign_avg: Dict[str, torch.Tensor]):
        """
        设置估计的良性平均更新
        
        Args:
            benign_avg: 良性客户端的平均更新
        """
        self.benign_avg = benign_avg
        logger.info("设置良性平均更新")
    
    def simulate_aggregation(
        self,
        malicious_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Simulate one round of federated aggregation."""
        if self.benign_avg is None:
            raise ValueError("benign_avg not set")

        simulated_change = {}
        for param_name, mal_param in malicious_update.items():
            benign_param = self.benign_avg.get(param_name)
            if benign_param is None:
                simulated_change[param_name] = mal_param.clone()
                continue
            if not _is_float_tensor(mal_param) or not _is_float_tensor(benign_param):
                simulated_change[param_name] = mal_param.clone()
                continue

            benign_part = (self.n - self.m) * benign_param
            mal_part = self.m * mal_param
            simulated_change[param_name] = (benign_part + mal_part) / self.n

        self.simulation_history.append({
            'malicious_update': malicious_update,
            'simulated_result': simulated_change
        })

        return simulated_change
    def simulate_multi_round(
        self,
        malicious_updates: List[Dict[str, torch.Tensor]],
        n_rounds: int = 10
    ) -> List[Dict[str, torch.Tensor]]:
        """
        模拟多轮聚合
        
        Args:
            malicious_updates: 多轮恶意更新列表
            n_rounds: 模拟轮数
            
        Returns:
            results: 每轮的模拟结果
        """
        results = []
        for i in range(min(n_rounds, len(malicious_updates))):
            sim_result = self.simulate_aggregation(malicious_updates[i])
            results.append(sim_result)
            logger.debug(f"模拟第 {i+1} 轮聚合")
        
        return results
    
    def compute_retention_ratio(
        self,
        malicious_update: Dict[str, torch.Tensor],
        simulated_result: Dict[str, torch.Tensor],
        param_name: Optional[str] = None
    ) -> float:
        """Compute retention ratio of malicious updates."""
        if param_name:
            mal_param = malicious_update.get(param_name)
            sim_param = simulated_result.get(param_name)
            if mal_param is None or sim_param is None:
                return 0.0
            if not _is_float_tensor(mal_param) or not _is_float_tensor(sim_param):
                return 0.0
            mal_norm = torch.norm(mal_param).item()
            sim_norm = torch.norm(sim_param).item()
            return sim_norm / (mal_norm + 1e-10)

        retentions = []
        for name, mal_param in malicious_update.items():
            sim_param = simulated_result.get(name)
            if sim_param is None:
                continue
            if not _is_float_tensor(mal_param) or not _is_float_tensor(sim_param):
                continue
            mal_norm = torch.norm(mal_param).item()
            sim_norm = torch.norm(sim_param).item()
            retentions.append(sim_norm / (mal_norm + 1e-10))

        return float(np.mean(retentions)) if retentions else 0.0
    def analyze_dilution_effect(
        self,
        malicious_update: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Analyze dilution effect for each parameter."""
        sim_result = self.simulate_aggregation(malicious_update)

        dilution_analysis = {}
        for param_name, mal_param in malicious_update.items():
            if not _is_float_tensor(mal_param):
                continue
            retention = self.compute_retention_ratio(
                malicious_update, sim_result, param_name
            )
            dilution_analysis[param_name] = 1.0 - retention

        return dilution_analysis
    def predict_parameter_evolution(
        self,
        initial_param: torch.Tensor,
        malicious_updates: List[torch.Tensor],
        n_rounds: int = 10
    ) -> List[torch.Tensor]:
        """
        预测参数在多轮训练中的演化
        
        Args:
            initial_param: 初始参数值
            malicious_updates: 每轮的恶意更新
            n_rounds: 预测轮数
            
        Returns:
            evolution: 参数演化轨迹
        """
        evolution = [initial_param.clone()]
        current_param = initial_param.clone()
        
        for i in range(min(n_rounds, len(malicious_updates))):
            # 模拟聚合
            mal_update_dict = {'param': malicious_updates[i]}
            benign_avg_dict = {'param': self.benign_avg.get('param', torch.zeros_like(initial_param))}
            
            self.benign_avg = benign_avg_dict
            sim_result = self.simulate_aggregation(mal_update_dict)
            
            # 更新参数
            current_param = current_param + sim_result['param']
            evolution.append(current_param.clone())
        
        return evolution
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            stats: 统计数据
        """
        if not self.simulation_history:
            return {'n_simulations': 0}
        
        # 计算平均保留比例
        avg_retention = np.mean([
            self.compute_retention_ratio(h['malicious_update'], h['simulated_result'])
            for h in self.simulation_history
        ])
        
        return {
            'n_simulations': len(self.simulation_history),
            'avg_retention': avg_retention,
            'avg_dilution': 1.0 - avg_retention,
            'malicious_ratio': self.m / self.n,
            'theoretical_retention': self.m / self.n  # 理论保留比例
        }
    
    def reset(self):
        """重置模拟器"""
        self.simulation_history = []
        logger.debug("重置聚合模拟器")


class MultiRoundSimulator:
    """
    多轮聚合模拟器
    
    支持更复杂的多轮模拟场景
    """
    
    def __init__(self, base_simulator: AggregationSimulator):
        self.base_simulator = base_simulator
        self.round_history = []
    
    def simulate_training_process(
        self,
        initial_model: Dict[str, torch.Tensor],
        malicious_strategy: callable,
        n_rounds: int = 50
    ) -> List[Dict[str, torch.Tensor]]:
        """
        模拟完整训练过程
        
        Args:
            initial_model: 初始模型参数
            malicious_strategy: 恶意策略函数(round) -> malicious_update
            n_rounds: 训练轮数
            
        Returns:
            model_history: 每轮的模型状态
        """
        model_history = [initial_model]
        current_model = {k: v.clone() for k, v in initial_model.items()}
        
        for round_idx in range(n_rounds):
            # 生成恶意更新
            mal_update = malicious_strategy(round_idx)
            
            # 模拟聚合
            sim_result = self.base_simulator.simulate_aggregation(mal_update)
            
            # 更新模型
            for param_name in current_model.keys():
                current_model[param_name] = current_model[param_name] + sim_result[param_name]
            
            model_history.append({k: v.clone() for k, v in current_model.items()})
            
            self.round_history.append({
                'round': round_idx,
                'malicious_update': mal_update,
                'simulated_result': sim_result,
                'model_state': {k: v.clone() for k, v in current_model.items()}
            })
        
        return model_history
    
    def analyze_convergence(self) -> Dict:
        """
        分析收敛性
        
        Returns:
            analysis: 收敛分析结果
        """
        if len(self.round_history) < 2:
            return {}
        
        # 计算参数变化趋势
        param_changes = []
        for i in range(1, len(self.round_history)):
            prev_model = self.round_history[i-1]['model_state']
            curr_model = self.round_history[i]['model_state']
            
            change = 0.0
            for param_name in prev_model.keys():
                diff = torch.norm(curr_model[param_name] - prev_model[param_name]).item()
                change += diff
            
            param_changes.append(change)
        
        return {
            'avg_change': np.mean(param_changes),
            'final_change': param_changes[-1],
            'is_converging': param_changes[-1] < param_changes[0],
            'change_history': param_changes
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("聚合模拟器 - 功能测试")
    print("=" * 60)
    
    # 创建模拟器
    simulator = AggregationSimulator(n_total=100, n_malicious=10)
    
    # 设置良性更新
    benign_avg = {
        'layer1.weight': torch.randn(1000) * 0.01,
        'layer1.bias': torch.randn(100) * 0.01,
    }
    simulator.set_benign_average(benign_avg)
    
    # 创建恶意更新
    malicious_update = {
        'layer1.weight': torch.randn(1000) * 0.05,  # 更大的更新
        'layer1.bias': torch.randn(100) * 0.05,
    }
    
    # 模拟聚合
    sim_result = simulator.simulate_aggregation(malicious_update)
    
    # 计算保留比例
    retention = simulator.compute_retention_ratio(malicious_update, sim_result)
    print(f"\n✓ 保留比例: {retention:.4f}")
    print(f"✓ 理论保留比例: {10/100:.4f}")
    print(f"✓ 稀释效应: {1-retention:.4f}")
    
    # 分析稀释效应
    print("\n" + "=" * 60)
    print("稀释效应分析")
    print("=" * 60)
    
    dilution = simulator.analyze_dilution_effect(malicious_update)
    for param_name, dilution_ratio in dilution.items():
        print(f"{param_name}: 稀释 {dilution_ratio:.2%}")
    
    # 统计信息
    stats = simulator.get_statistics()
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")
