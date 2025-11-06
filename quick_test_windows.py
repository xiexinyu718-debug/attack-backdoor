"""
Windows专用快速防御验证脚本
不依赖项目其他文件，可以独立运行
"""

import torch
import numpy as np


# ============= 简化的防御实现 =============

class DefenseBase:
    """防御基类"""
    def __init__(self):
        self.name = "Base Defense"
        
    def aggregate(self, global_model, local_models, participant_ids):
        raise NotImplementedError


class FedAvgDefense(DefenseBase):
    """FedAvg - 基准方法（无防御）"""
    def __init__(self):
        super().__init__()
        self.name = "FedAvg (No Defense)"
        
    def aggregate(self, global_model, local_models, participant_ids):
        global_state = global_model.state_dict()
        weight_accumulator = {}
        
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                continue
            weight_accumulator[name] = torch.zeros_like(param, dtype=torch.float32)
        
        for participant_id in participant_ids:
            local_model = local_models[participant_id]
            local_state = local_model.state_dict()
            
            for name, param in local_state.items():
                if name not in weight_accumulator:
                    continue
                global_param = global_state[name]
                update = param.float() - global_param.float()
                weight_accumulator[name] += update
        
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
    """Krum防御"""
    def __init__(self):
        super().__init__()
        self.name = "Krum"
        self.num_selected = 5
        
    def aggregate(self, global_model, local_models, participant_ids):
        distances = self._compute_pairwise_distances(local_models, participant_ids)
        num_participants = len(participant_ids)
        num_to_consider = num_participants - 3 - 2
        
        scores = {}
        for i, pid_i in enumerate(participant_ids):
            dists_to_others = []
            for j, pid_j in enumerate(participant_ids):
                if i != j:
                    dists_to_others.append(distances[i][j])
            dists_to_others.sort()
            score = sum(dists_to_others[:num_to_consider])
            scores[pid_i] = score
        
        selected_participants = sorted(scores.keys(), key=lambda x: scores[x])[:self.num_selected]
        selected_models = {pid: local_models[pid] for pid in selected_participants}
        
        fedavg = FedAvgDefense()
        return fedavg.aggregate(global_model, selected_models, selected_participants)
    
    def _compute_pairwise_distances(self, local_models, participant_ids):
        n = len(participant_ids)
        distances = [[0.0] * n for _ in range(n)]
        
        flattened_params = {}
        for pid in participant_ids:
            params = []
            for param in local_models[pid].parameters():
                params.append(param.data.view(-1))
            flattened_params[pid] = torch.cat(params)
        
        for i, pid_i in enumerate(participant_ids):
            for j, pid_j in enumerate(participant_ids):
                if i < j:
                    dist = torch.norm(flattened_params[pid_i] - flattened_params[pid_j]).item()
                    distances[i][j] = dist
                    distances[j][i] = dist
        
        return distances


class TrimmedMeanDefense(DefenseBase):
    """Trimmed Mean防御"""
    def __init__(self):
        super().__init__()
        self.name = "Trimmed Mean"
        self.beta = 0.1
        
    def aggregate(self, global_model, local_models, participant_ids):
        global_state = global_model.state_dict()
        num_participants = len(participant_ids)
        num_to_trim = int(num_participants * self.beta)
        aggregated_state = {}
        
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                aggregated_state[name] = param.clone()
                continue
            
            updates = []
            for pid in participant_ids:
                local_param = local_models[pid].state_dict()[name]
                update = local_param.float() - param.float()
                updates.append(update)
            
            stacked_updates = torch.stack(updates)
            
            if num_to_trim > 0:
                sorted_updates, _ = torch.sort(stacked_updates, dim=0)
                trimmed_updates = sorted_updates[num_to_trim:-num_to_trim]
                mean_update = trimmed_updates.mean(dim=0)
            else:
                mean_update = stacked_updates.mean(dim=0)
            
            aggregated_state[name] = param + mean_update.to(param.dtype)
        
        return aggregated_state


class MedianDefense(DefenseBase):
    """Median防御"""
    def __init__(self):
        super().__init__()
        self.name = "Median"
        
    def aggregate(self, global_model, local_models, participant_ids):
        global_state = global_model.state_dict()
        aggregated_state = {}
        
        for name, param in global_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                aggregated_state[name] = param.clone()
                continue
            
            updates = []
            for pid in participant_ids:
                local_param = local_models[pid].state_dict()[name]
                update = local_param.float() - param.float()
                updates.append(update)
            
            stacked_updates = torch.stack(updates)
            median_update = torch.median(stacked_updates, dim=0)[0]
            aggregated_state[name] = param + median_update.to(param.dtype)
        
        return aggregated_state


class NormClippingDefense(DefenseBase):
    """Norm Clipping防御"""
    def __init__(self):
        super().__init__()
        self.name = "Norm Clipping"
        self.clip_threshold = 10.0
        
    def aggregate(self, global_model, local_models, participant_ids):
        global_state = global_model.state_dict()
        clipped_updates = {}
        
        for pid in participant_ids:
            local_state = local_models[pid].state_dict()
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
            
            if update_norm > self.clip_threshold:
                clip_factor = self.clip_threshold / update_norm
                for name in update_dict:
                    update_dict[name] *= clip_factor
            
            clipped_updates[pid] = update_dict
        
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
    """Weak DP防御"""
    def __init__(self):
        super().__init__()
        self.name = "Weak DP"
        self.noise_scale = 0.001
        
    def aggregate(self, global_model, local_models, participant_ids):
        fedavg = FedAvgDefense()
        aggregated_state = fedavg.aggregate(global_model, local_models, participant_ids)
        
        for name, param in aggregated_state.items():
            if param.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                continue
            noise = torch.randn_like(param) * self.noise_scale
            aggregated_state[name] = param + noise
        
        return aggregated_state


class FoolsGoldDefense(DefenseBase):
    """FoolsGold防御（简化版）"""
    def __init__(self):
        super().__init__()
        self.name = "FoolsGold"
        
    def aggregate(self, global_model, local_models, participant_ids):
        # 简化实现：使用均等权重
        fedavg = FedAvgDefense()
        return fedavg.aggregate(global_model, local_models, participant_ids)


# ============= 简化的ResNet模型 =============

class SimpleResNet(torch.nn.Module):
    """简化的ResNet用于测试"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============= 测试函数 =============

def create_mock_models(num_models=10, num_malicious=3):
    """创建模拟的本地模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = SimpleResNet(num_classes=10).to(device)
    
    local_models = {}
    for i in range(num_models):
        local_model = SimpleResNet(num_classes=10).to(device)
        local_model.load_state_dict(global_model.state_dict())
        
        with torch.no_grad():
            for param in local_model.parameters():
                if i < num_malicious:
                    noise = torch.randn_like(param) * 0.5
                else:
                    noise = torch.randn_like(param) * 0.01
                param.add_(noise)
        
        local_models[i] = local_model
    
    participant_ids = list(range(num_models))
    return global_model, local_models, participant_ids


def test_defense(defense_name):
    """测试单个防御"""
    print(f"\n{'='*60}")
    print(f"测试防御: {defense_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # 创建防御
        defense_map = {
            'fedavg': FedAvgDefense,
            'krum': KrumDefense,
            'trimmed_mean': TrimmedMeanDefense,
            'median': MedianDefense,
            'norm_clipping': NormClippingDefense,
            'weak_dp': WeakDPDefense,
            'foolsgold': FoolsGoldDefense,
        }
        
        defense = defense_map[defense_name]()
        print(f"✓ {defense.name} 初始化成功")
        
        # 创建模拟模型
        global_model, local_models, participant_ids = create_mock_models()
        print(f"✓ 创建了 {len(local_models)} 个本地模型 (3个恶意)")
        
        # 执行聚合
        aggregated_state = defense.aggregate(global_model, local_models, participant_ids)
        print(f"✓ 聚合完成")
        
        # 加载聚合后的状态
        global_model.load_state_dict(aggregated_state)
        print(f"✓ {defense.name} 测试通过\n")
        
        return True
        
    except Exception as e:
        print(f"✗ {defense.name} 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("="*60)
    print("Windows专用快速防御验证")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 测试所有防御
    defenses = [
        'fedavg',
        'krum',
        'trimmed_mean',
        'median',
        'norm_clipping',
        'weak_dp',
        'foolsgold',
    ]
    
    results = {}
    for defense_name in defenses:
        results[defense_name] = test_defense(defense_name)
    
    # 打印摘要
    print("="*60)
    print("测试摘要")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for defense_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {defense_name:<20} {status}")
    
    print(f"\n通过: {passed}/{total}")
    print("="*60)
    
    if passed == total:
        print("\n✓ 所有防御机制验证通过！")
        print("\n接下来您需要:")
        print("1. 将防御测试文件放到您的项目目录")
        print("2. 确保helper.py、fl_utils等文件在正确位置")
        print("3. 运行完整测试（需要项目的其他文件）")
    else:
        print("\n⚠️  部分防御验证失败，请检查错误信息")


if __name__ == '__main__':
    main()
