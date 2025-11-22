"""
Model Replacement (MR) 模块
实现模型替换攻击策略,通过放大恶意更新来增强后门植入效果

参考论文: "How To Backdoor Federated Learning" (Bagdasaryan et al., 2020)

核心思想:
    恶意客户端通过缩放因子γ放大其模型更新,使得后门在聚合后依然有效
    γ = n/η, 其中:
    - n: 总参与者数量
    - η: 每轮采样的参与者数量
    
公式:
    L^{t+1}_m = γ(X - G^t) + G^t
    其中:
    - L^{t+1}_m: 放大后的恶意模型
    - X: 原始恶意模型
    - G^t: 当前全局模型
"""

import torch
import copy


class ModelReplacementAttacker:
    """Model Replacement攻击器"""
    
    def __init__(self, config, helper=None):
        """
        初始化MR攻击器
        
        Args:
            config: 配置对象,需要包含:
                - num_total_participants: 总参与者数
                - num_sampled_participants: 每轮采样参与者数
                - mr_scale_factor: 手动设置的缩放因子(可选)
                - mr_adaptive: 是否使用自适应缩放(可选)
            helper: Helper对象(可选,用于获取配置)
        """
        if helper is not None:
            self.config = helper.config
        else:
            self.config = config
        
        # 基础缩放因子: γ = n/η
        self.base_scale_factor = self._compute_base_scale_factor()
        
        # 实际使用的缩放因子
        self.scale_factor = self._get_effective_scale_factor()
        
        # 自适应参数
        self.adaptive = getattr(self.config, 'mr_adaptive', False)
        self.min_scale = getattr(self.config, 'mr_min_scale', 1.0)
        self.max_scale = getattr(self.config, 'mr_max_scale', self.base_scale_factor * 2)
        
        # 统计信息
        self.history = {
            'scale_factors': [],
            'update_norms': [],
            'epochs': []
        }
        
        print(f"\n{'='*60}")
        print(f"Model Replacement (MR) 攻击器初始化")
        print(f"{'='*60}")
        print(f"  总参与者数: {self.config.num_total_participants}")
        print(f"  采样参与者数: {self.config.num_sampled_participants}")
        print(f"  基础缩放因子 γ: {self.base_scale_factor:.2f}")
        print(f"  实际缩放因子: {self.scale_factor:.2f}")
        print(f"  自适应模式: {'是' if self.adaptive else '否'}")
        if self.adaptive:
            print(f"  缩放范围: [{self.min_scale:.2f}, {self.max_scale:.2f}]")
        print(f"{'='*60}\n")
    
    def _compute_base_scale_factor(self):
        """
        计算基础缩放因子
        
        Returns:
            γ = n/η
        """
        n = self.config.num_total_participants
        eta = self.config.num_sampled_participants
        return n / eta
    
    def _get_effective_scale_factor(self):
        """
        获取实际使用的缩放因子
        
        Returns:
            scale_factor: 缩放因子
        """
        # 如果配置中指定了缩放因子,使用指定值
        if hasattr(self.config, 'mr_scale_factor') and self.config.mr_scale_factor is not None:
            return self.config.mr_scale_factor
        
        # 否则使用基础缩放因子
        return self.base_scale_factor
    
    def scale_malicious_update(self, malicious_model, global_model, epoch=None):
        """
        放大恶意模型更新
        
        Args:
            malicious_model: 恶意客户端训练后的模型
            global_model: 当前全局模型
            epoch: 当前轮次(用于自适应缩放)
            
        Returns:
            scaled_model: 放大后的恶意模型
        """
        # 深拷贝,避免修改原模型
        scaled_model = copy.deepcopy(malicious_model)
        
        # 获取当前缩放因子
        current_scale = self._get_current_scale_factor(epoch)
        
        # 计算并记录更新范数
        update_norm = self._compute_update_norm(malicious_model, global_model)
        
        # 应用模型替换公式
        # L^{t+1}_m = γ(X - G^t) + G^t
        scaled_state = scaled_model.state_dict()
        global_state = global_model.state_dict()
        malicious_state = malicious_model.state_dict()
        
        with torch.no_grad():
            for name in scaled_state.keys():
                # 跳过非浮点参数
                if scaled_state[name].dtype not in [torch.float32, torch.float16, torch.float64]:
                    continue
                
                # 计算更新: X - G^t
                update = malicious_state[name].float() - global_state[name].float()
                
                # 放大更新: γ(X - G^t)
                scaled_update = current_scale * update
                
                # 应用放大后的更新: γ(X - G^t) + G^t
                scaled_state[name].copy_((global_state[name].float() + scaled_update).to(scaled_state[name].dtype))
        
        # 更新统计
        if epoch is not None:
            self.history['scale_factors'].append(current_scale)
            self.history['update_norms'].append(update_norm)
            self.history['epochs'].append(epoch)
        
        return scaled_model
    
    def _get_current_scale_factor(self, epoch):
        """
        获取当前轮次的缩放因子(支持自适应)
        
        Args:
            epoch: 当前轮次
            
        Returns:
            current_scale: 当前缩放因子
        """
        if not self.adaptive or epoch is None:
            return self.scale_factor
        
        # 自适应策略:根据训练进度调整缩放因子
        # 早期使用较小的缩放因子,后期逐渐增大
        total_epochs = getattr(self.config, 'epochs', 100)
        progress = min(epoch / total_epochs, 1.0)
        
        # 线性增长
        current_scale = self.min_scale + (self.scale_factor - self.min_scale) * progress
        current_scale = min(current_scale, self.max_scale)
        
        return current_scale
    
    def _compute_update_norm(self, model1, model2):
        """
        计算两个模型之间的L2范数
        
        Args:
            model1: 模型1
            model2: 模型2
            
        Returns:
            norm: L2范数
        """
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        
        total_norm = 0.0
        for name in state1.keys():
            if state1[name].dtype in [torch.float32, torch.float16, torch.float64]:
                diff = state1[name].float() - state2[name].float()
                total_norm += torch.norm(diff, p=2).item() ** 2
        
        return (total_norm ** 0.5)
    
    def get_statistics(self):
        """
        获取MR攻击器的统计信息
        
        Returns:
            stats: 统计信息字典
        """
        if not self.history['scale_factors']:
            return {
                'avg_scale_factor': self.scale_factor,
                'avg_update_norm': 0.0,
                'num_attacks': 0
            }
        
        import numpy as np
        return {
            'avg_scale_factor': np.mean(self.history['scale_factors']),
            'min_scale_factor': np.min(self.history['scale_factors']),
            'max_scale_factor': np.max(self.history['scale_factors']),
            'avg_update_norm': np.mean(self.history['update_norms']),
            'num_attacks': len(self.history['epochs'])
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"Model Replacement 统计信息")
        print(f"{'='*60}")
        print(f"  攻击次数: {stats['num_attacks']}")
        print(f"  平均缩放因子: {stats['avg_scale_factor']:.2f}")
        if self.adaptive:
            print(f"  最小缩放因子: {stats['min_scale_factor']:.2f}")
            print(f"  最大缩放因子: {stats['max_scale_factor']:.2f}")
        print(f"  平均更新范数: {stats['avg_update_norm']:.4f}")
        print(f"{'='*60}\n")


def apply_model_replacement(malicious_model, global_model, config, epoch=None):
    """
    便捷函数:对恶意模型应用模型替换
    
    Args:
        malicious_model: 恶意模型
        global_model: 全局模型
        config: 配置对象
        epoch: 当前轮次
        
    Returns:
        scaled_model: 放大后的模型
    """
    attacker = ModelReplacementAttacker(config)
    return attacker.scale_malicious_update(malicious_model, global_model, epoch)


if __name__ == '__main__':
    print("测试Model Replacement模块\n")
    
    # 创建测试配置
    class TestConfig:
        def __init__(self):
            self.num_total_participants = 100
            self.num_sampled_participants = 10
            self.mr_adaptive = True
            self.epochs = 100
    
    config = TestConfig()
    
    # 创建MR攻击器
    mr_attacker = ModelReplacementAttacker(config)
    
    # 测试缩放因子计算
    print("\n测试缩放因子:")
    for epoch in [0, 25, 50, 75, 99]:
        scale = mr_attacker._get_current_scale_factor(epoch)
        print(f"  Epoch {epoch}: 缩放因子 = {scale:.2f}")
    
    # 创建测试模型
    from models.resnet import ResNet18
    
    global_model = ResNet18(num_classes=10)
    malicious_model = copy.deepcopy(global_model)
    
    # 修改恶意模型(模拟训练)
    with torch.no_grad():
        for param in malicious_model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    # 应用MR
    print("\n测试模型替换:")
    scaled_model = mr_attacker.scale_malicious_update(
        malicious_model, global_model, epoch=50
    )
    
    # 计算范数
    original_norm = mr_attacker._compute_update_norm(malicious_model, global_model)
    scaled_norm = mr_attacker._compute_update_norm(scaled_model, global_model)
    
    print(f"  原始更新范数: {original_norm:.4f}")
    print(f"  放大后范数: {scaled_norm:.4f}")
    print(f"  放大倍数: {scaled_norm / original_norm:.2f}")
    
    # 打印统计
    mr_attacker.print_statistics()
    
    print("\n✓ Model Replacement模块测试完成")
