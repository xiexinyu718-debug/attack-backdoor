"""
联邦学习防御机制模块
实现常见的后门防御方法

支持的防御方法:
1. Krum/Multi-Krum - 选择距离最小的模型
2. Trimmed Mean - 修剪极端值后平均
3. Median - 中位数聚合
4. Norm Clipping - 梯度范数裁剪
5. Weak DP - 弱差分隐私
6. FoolsGold - 基于梯度相似度检测
"""

import torch
import numpy as np
from collections import defaultdict
import copy


class DefenseBase:
    """防御基类"""
    def __init__(self, config):
        self.config = config
        self.name = "Base Defense"
        
    def aggregate(self, global_model, local_models, participant_ids):
        """
        聚合本地模型
        
        Args:
            global_model: 全局模型
            local_models: {participant_id: model} 字典
            participant_ids: 参与者ID列表
            
        Returns:
            aggregated_state: 聚合后的模型状态
        """
        raise NotImplementedError


class FedAvgDefense(DefenseBase):
    """FedAvg - 基准方法（无防御）"""
    def __init__(self, config):
        super().__init__(config)
        self.name = "FedAvg (No Defense)"
        
    def aggregate(self, global_model, local_models, participant_ids):
        """标准FedAvg聚合"""
        global_state = global_model.state_dict()
        
        # 初始化权重累加器
        weight_accumulator = {}
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                continue
            weight_accumulator[name] = torch.zeros_like(param, dtype=torch.float32)
        
        # 累加所有本地模型的更新
        for participant_id in participant_ids:
            local_model = local_models[participant_id]
            local_state = local_model.state_dict()
            
            for name, param in local_state.items():
                if name not in weight_accumulator:
                    continue
                
                global_param = global_state[name]
                update = param.float() - global_param.float()
                weight_accumulator[name] += update
        
        # 平均并更新
        num_participants = len(participant_ids)
        aggregated_state = {}
        
        for name, param in global_state.items():
            if name not in weight_accumulator:
                aggregated_state[name] = param.clone()
                continue
            
            avg_update = weight_accumulator[name] / num_participants
            aggregated_state[name] = param + avg_update.to(param.dtype)
        
        return aggregated_state


class KrumDefense(DefenseBase):
    """
    Krum防御
    选择与其他模型距离之和最小的k个模型
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "Krum"
        self.num_selected = config.get('krum_num_selected', 1)  # Multi-Krum的k值
        
    def aggregate(self, global_model, local_models, participant_ids):
        """Krum聚合"""
        # 计算所有模型对之间的距离
        distances = self._compute_pairwise_distances(local_models, participant_ids)
        
        # 计算每个模型的得分（距离之和）
        num_participants = len(participant_ids)
        num_to_consider = num_participants - self.config.get('num_adversaries', 0) - 2
        
        scores = {}
        for i, pid_i in enumerate(participant_ids):
            # 找到最近的num_to_consider个模型
            dists_to_others = []
            for j, pid_j in enumerate(participant_ids):
                if i != j:
                    dists_to_others.append(distances[i][j])
            
            dists_to_others.sort()
            score = sum(dists_to_others[:num_to_consider])
            scores[pid_i] = score
        
        # 选择得分最低的k个模型
        selected_participants = sorted(scores.keys(), key=lambda x: scores[x])[:self.num_selected]
        
        print(f"  [Krum] 选择了 {len(selected_participants)} 个模型: {selected_participants}")
        
        # 对选中的模型进行FedAvg
        selected_models = {pid: local_models[pid] for pid in selected_participants}
        fedavg = FedAvgDefense(self.config)
        return fedavg.aggregate(global_model, selected_models, selected_participants)
    
    def _compute_pairwise_distances(self, local_models, participant_ids):
        """计算模型之间的欧氏距离"""
        n = len(participant_ids)
        distances = [[0.0] * n for _ in range(n)]
        
        # 将模型参数展平
        flattened_params = {}
        for pid in participant_ids:
            params = []
            for param in local_models[pid].parameters():
                params.append(param.data.view(-1))
            flattened_params[pid] = torch.cat(params)
        
        # 计算距离矩阵
        for i, pid_i in enumerate(participant_ids):
            for j, pid_j in enumerate(participant_ids):
                if i < j:
                    dist = torch.norm(flattened_params[pid_i] - flattened_params[pid_j]).item()
                    distances[i][j] = dist
                    distances[j][i] = dist
        
        return distances


class TrimmedMeanDefense(DefenseBase):
    """
    Trimmed Mean防御
    移除最大和最小的beta比例的更新，然后平均
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "Trimmed Mean"
        self.beta = config.get('trimmed_mean_beta', 0.1)  # 修剪比例
        
    def aggregate(self, global_model, local_models, participant_ids):
        """Trimmed Mean聚合"""
        global_state = global_model.state_dict()
        num_participants = len(participant_ids)
        num_to_trim = int(num_participants * self.beta)
        
        print(f"  [Trimmed Mean] 修剪每端 {num_to_trim} 个极端值")
        
        aggregated_state = {}
        
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                aggregated_state[name] = param.clone()
                continue
            
            # 收集所有本地模型在这个参数上的值
            updates = []
            for pid in participant_ids:
                local_param = local_models[pid].state_dict()[name]
                update = local_param.float() - param.float()
                updates.append(update)
            
            # 对每个元素进行修剪和平均
            stacked_updates = torch.stack(updates)  # [num_participants, ...]
            
            # 沿着参与者维度排序并修剪
            if num_to_trim > 0:
                sorted_updates, _ = torch.sort(stacked_updates, dim=0)
                trimmed_updates = sorted_updates[num_to_trim:-num_to_trim]
                mean_update = trimmed_updates.mean(dim=0)
            else:
                mean_update = stacked_updates.mean(dim=0)
            
            aggregated_state[name] = param + mean_update.to(param.dtype)
        
        return aggregated_state


class MedianDefense(DefenseBase):
    """
    Median防御
    对每个参数取中位数
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "Median"
        
    def aggregate(self, global_model, local_models, participant_ids):
        """Median聚合"""
        global_state = global_model.state_dict()
        aggregated_state = {}
        
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                aggregated_state[name] = param.clone()
                continue
            
            # 收集所有本地模型在这个参数上的值
            updates = []
            for pid in participant_ids:
                local_param = local_models[pid].state_dict()[name]
                update = local_param.float() - param.float()
                updates.append(update)
            
            # 计算中位数
            stacked_updates = torch.stack(updates)
            median_update = torch.median(stacked_updates, dim=0)[0]
            
            aggregated_state[name] = param + median_update.to(param.dtype)
        
        return aggregated_state


class NormClippingDefense(DefenseBase):
    """
    Norm Clipping防御
    限制每个客户端更新的L2范数
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "Norm Clipping"
        self.clip_threshold = config.get('clip_threshold', 10.0)
        
    def aggregate(self, global_model, local_models, participant_ids):
        """带范数裁剪的聚合"""
        global_state = global_model.state_dict()
        
        # 计算并裁剪每个客户端的更新
        clipped_updates = {}
        for pid in participant_ids:
            local_state = local_models[pid].state_dict()
            
            # 计算更新向量
            update_dict = {}
            update_norm = 0.0
            
            for name, param in global_state.items():
                if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                    continue
                
                local_param = local_state[name]
                update = local_param.float() - param.float()
                update_dict[name] = update
                update_norm += torch.sum(update ** 2).item()
            
            update_norm = np.sqrt(update_norm)
            
            # 如果范数超过阈值，进行裁剪
            if update_norm > self.clip_threshold:
                clip_factor = self.clip_threshold / update_norm
                print(f"  [Clipping] 客户端 {pid}: 范数 {update_norm:.2f} -> {self.clip_threshold:.2f}")
                for name in update_dict:
                    update_dict[name] *= clip_factor
            
            clipped_updates[pid] = update_dict
        
        # 平均裁剪后的更新
        aggregated_state = {}
        weight_accumulator = {}
        
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                aggregated_state[name] = param.clone()
                continue
            
            weight_accumulator[name] = torch.zeros_like(param, dtype=torch.float32)
            
            for pid in participant_ids:
                weight_accumulator[name] += clipped_updates[pid][name]
            
            avg_update = weight_accumulator[name] / len(participant_ids)
            aggregated_state[name] = param + avg_update.to(param.dtype)
        
        return aggregated_state


class WeakDPDefense(DefenseBase):
    """
    Weak DP防御
    在聚合时添加高斯噪声
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "Weak DP"
        self.noise_scale = config.get('dp_noise_scale', 0.001)
        
    def aggregate(self, global_model, local_models, participant_ids):
        """带差分隐私的聚合"""
        # 先用FedAvg聚合
        fedavg = FedAvgDefense(self.config)
        aggregated_state = fedavg.aggregate(global_model, local_models, participant_ids)
        
        # 添加高斯噪声
        for name, param in aggregated_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                continue
            
            noise = torch.randn_like(param) * self.noise_scale
            aggregated_state[name] = param + noise
        
        print(f"  [Weak DP] 添加噪声 (scale={self.noise_scale})")
        
        return aggregated_state


class FoolsGoldDefense(DefenseBase):
    """
    FoolsGold防御
    基于历史梯度相似度检测Sybil攻击
    """
    def __init__(self, config):
        super().__init__(config)
        self.name = "FoolsGold"
        self.history = defaultdict(list)  # 存储历史更新
        self.learning_rate = 1.0
        
    def aggregate(self, global_model, local_models, participant_ids):
        """FoolsGold聚合"""
        global_state = global_model.state_dict()
        
        # 计算每个客户端的更新向量
        updates = {}
        for pid in participant_ids:
            local_state = local_models[pid].state_dict()
            update_vec = []
            
            for name, param in global_state.items():
                if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                    continue
                
                local_param = local_state[name]
                update = local_param.float() - param.float()
                update_vec.append(update.view(-1))
            
            updates[pid] = torch.cat(update_vec)
            self.history[pid].append(updates[pid])
        
        # 计算相似度矩阵
        similarity_matrix = self._compute_similarity_matrix(participant_ids)
        
        # 计算每个客户端的权重
        weights = self._compute_weights(similarity_matrix, participant_ids)
        
        print(f"  [FoolsGold] 客户端权重: {[f'{w:.3f}' for w in weights.values()]}")
        
        # 加权聚合
        aggregated_state = {}
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                aggregated_state[name] = param.clone()
                continue
            
            weighted_update = torch.zeros_like(param, dtype=torch.float32)
            
            for pid in participant_ids:
                local_param = local_models[pid].state_dict()[name]
                update = local_param.float() - param.float()
                weighted_update += weights[pid] * update
            
            aggregated_state[name] = param + weighted_update.to(param.dtype)
        
        return aggregated_state
    
    def _compute_similarity_matrix(self, participant_ids):
        """计算历史梯度相似度矩阵"""
        n = len(participant_ids)
        similarity = np.zeros((n, n))
        
        for i, pid_i in enumerate(participant_ids):
            if len(self.history[pid_i]) == 0:
                continue
            
            for j, pid_j in enumerate(participant_ids):
                if i >= j or len(self.history[pid_j]) == 0:
                    continue
                
                # 计算余弦相似度
                vec_i = torch.stack(self.history[pid_i]).mean(dim=0)
                vec_j = torch.stack(self.history[pid_j]).mean(dim=0)
                
                cos_sim = torch.dot(vec_i, vec_j) / (torch.norm(vec_i) * torch.norm(vec_j) + 1e-9)
                similarity[i][j] = cos_sim.item()
                similarity[j][i] = cos_sim.item()
        
        return similarity
    
    def _compute_weights(self, similarity_matrix, participant_ids):
        """基于相似度计算权重"""
        n = len(participant_ids)
        weights = {}
        
        for i, pid in enumerate(participant_ids):
            # 计算与其他客户端的最大相似度
            max_similarity = np.max(similarity_matrix[i])
            
            # 权重与相似度成反比
            weight = 1.0 / (1.0 + max_similarity)
            weights[pid] = weight
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for pid in weights:
                weights[pid] /= total_weight
        else:
            # 如果都是0，使用均等权重
            for pid in weights:
                weights[pid] = 1.0 / n
        
        return weights


def create_defense(defense_name, config):
    """
    工厂函数：创建防御实例
    
    Args:
        defense_name: 防御名称
        config: 配置对象
        
    Returns:
        防御实例
    """
    defense_map = {
        'fedavg': FedAvgDefense,
        'none': FedAvgDefense,
        'krum': KrumDefense,
        'trimmed_mean': TrimmedMeanDefense,
        'median': MedianDefense,
        'norm_clipping': NormClippingDefense,
        'weak_dp': WeakDPDefense,
        'foolsgold': FoolsGoldDefense,
    }
    
    defense_name = defense_name.lower()
    if defense_name not in defense_map:
        raise ValueError(f"未知防御: {defense_name}，可用: {list(defense_map.keys())}")
    
    return defense_map[defense_name](config)


if __name__ == '__main__':
    print("测试防御机制模块\n")
    
    # 创建测试配置
    class TestConfig:
        def get(self, key, default=None):
            return {
                'num_adversaries': 4,
                'krum_num_selected': 5,
                'trimmed_mean_beta': 0.1,
                'clip_threshold': 10.0,
                'dp_noise_scale': 0.001,
            }.get(key, default)
    
    config = TestConfig()
    
    # 测试每种防御
    defenses = ['fedavg', 'krum', 'trimmed_mean', 'median', 
                'norm_clipping', 'weak_dp', 'foolsgold']
    
    for defense_name in defenses:
        defense = create_defense(defense_name, config)
        print(f"✓ {defense.name} 初始化成功")
    
    print("\n✓ 防御机制模块测试完成")