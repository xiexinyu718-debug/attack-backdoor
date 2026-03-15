"""
主任务优化器 (Main Task Optimizer)
⭐⭐⭐

功能：
1. 余弦退火学习率调度
2. 知识蒸馏（可选）
3. 良性增强训练
4. 预热策略

目标：
将主任务准确率从81%提升到82-83%
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MainTaskOptimizer:
    """
    主任务优化器
    
    提升良性客户端的训练效果
    """
    
    def __init__(
        self,
        base_lr: float = 0.06,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 10,
        use_distillation: bool = False,
        distill_temperature: float = 4.0,
        distill_alpha: float = 0.7,
        benign_boost: bool = False,
        benign_extra_epochs: int = 2
    ):
        """
        初始化主任务优化器
        
        Args:
            base_lr: 基础学习率（提高到0.06）
            scheduler_type: 学习率调度类型 ('cosine', 'step', 'exponential')
            warmup_epochs: 预热轮数
            use_distillation: 是否使用知识蒸馏
            distill_temperature: 蒸馏温度
            distill_alpha: 蒸馏权重
            benign_boost: 是否启用良性增强
            benign_extra_epochs: 良性额外训练轮数
        """
        self.base_lr = base_lr
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.use_distillation = use_distillation
        self.distill_temperature = distill_temperature
        self.distill_alpha = distill_alpha
        self.benign_boost = benign_boost
        self.benign_extra_epochs = benign_extra_epochs
        
        logger.info(f"初始化主任务优化器:")
        logger.info(f"  基础学习率: {base_lr}")
        logger.info(f"  调度器: {scheduler_type}")
        logger.info(f"  知识蒸馏: {'启用' if use_distillation else '禁用'}")
        logger.info(f"  良性增强: {'启用' if benign_boost else '禁用'}")
    
    def get_lr(self, epoch: int, total_epochs: int) -> float:
        """
        获取当前轮次的学习率
        
        Args:
            epoch: 当前轮次
            total_epochs: 总轮次
            
        Returns:
            lr: 学习率
        """
        # 预热阶段
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        
        # 主训练阶段
        if self.scheduler_type == 'cosine':
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
            return max(lr, 0.001)
        
        elif self.scheduler_type == 'step':
            # 阶梯衰减
            if epoch < total_epochs * 0.5:
                return self.base_lr
            elif epoch < total_epochs * 0.75:
                return self.base_lr * 0.1
            else:
                return self.base_lr * 0.01
        
        elif self.scheduler_type == 'exponential':
            # 指数衰减
            decay_rate = 0.96
            lr = self.base_lr * (decay_rate ** epoch)
            return max(lr, 0.001)
        
        else:
            return self.base_lr
    
    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        知识蒸馏损失
        
        Loss = α * L_CE + (1-α) * L_KD
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            targets: 真实标签
            criterion: 损失函数
            
        Returns:
            loss: 蒸馏损失
        """
        # 交叉熵损失
        ce_loss = criterion(student_outputs, targets)
        
        # KL散度损失
        T = self.distill_temperature
        
        # 软标签
        student_soft = nn.functional.log_softmax(student_outputs / T, dim=1)
        teacher_soft = nn.functional.softmax(teacher_outputs / T, dim=1)
        
        # KL散度
        kd_loss = nn.functional.kl_div(
            student_soft, teacher_soft, reduction='batchmean'
        ) * (T ** 2)
        
        # 组合损失
        total_loss = self.distill_alpha * ce_loss + (1 - self.distill_alpha) * kd_loss
        
        return total_loss
    
    def train_benign_client(
        self,
        model: nn.Module,
        data_loader,
        epoch: int,
        total_epochs: int,
        retrain_times: int,
        momentum: float = 0.9,
        weight_decay: float = 0.001,
        teacher_model: Optional[nn.Module] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        训练良性客户端（优化版）
        
        Args:
            model: 模型
            data_loader: 数据加载器
            epoch: 当前轮次
            total_epochs: 总轮次
            retrain_times: 重复训练次数
            momentum: 动量
            weight_decay: 权重衰减
            teacher_model: 教师模型（用于蒸馏）
            device: 设备
            
        Returns:
            stats: 训练统计
        """
        # 获取优化学习率
        lr = self.get_lr(epoch, total_epochs)
        
        # 创建优化器
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练模式
        model.train()
        if teacher_model is not None and self.use_distillation:
            teacher_model.eval()
        
        # 统计信息
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        # 基础训练轮数
        train_epochs = retrain_times
        
        # 良性增强：额外训练
        if self.benign_boost:
            train_epochs += self.benign_extra_epochs
        
        # 训练循环
        for internal_epoch in range(train_epochs):
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(data)
                
                # 计算损失
                if self.use_distillation and teacher_model is not None:
                    # 知识蒸馏
                    with torch.no_grad():
                        teacher_outputs = teacher_model(data)
                    
                    loss = self.distillation_loss(
                        outputs, teacher_outputs, targets, criterion
                    )
                else:
                    # 标准交叉熵
                    loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total_samples += data.shape[0]
        
        # 计算指标
        accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / (len(data_loader) * train_epochs)
        
        stats = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_samples,
            'lr': lr,
            'train_epochs': train_epochs
        }
        
        return stats
    
    def create_lr_schedule(self, total_epochs: int) -> Dict[int, float]:
        """
        创建完整的学习率调度表
        
        Args:
            total_epochs: 总轮次
            
        Returns:
            schedule: {epoch: lr}
        """
        schedule = {}
        for epoch in range(total_epochs):
            schedule[epoch] = self.get_lr(epoch, total_epochs)
        return schedule


class CosineAnnealingWithWarmup:
    """
    带预热的余弦退火调度器（PyTorch风格）
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        warmup_epochs: int = 10,
        eta_min: float = 0.001
    ):
        """
        初始化调度器
        
        Args:
            optimizer: 优化器
            T_max: 最大轮次
            warmup_epochs: 预热轮数
            eta_min: 最小学习率
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火
            progress = (self.current_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # 更新优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        
        return lr


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("主任务优化器 - 功能测试")
    print("=" * 60)
    
    # 创建优化器
    optimizer = MainTaskOptimizer(
        base_lr=0.06,
        scheduler_type='cosine',
        warmup_epochs=10,
        use_distillation=True,
        benign_boost=True
    )
    
    print(f"\n✓ 优化器初始化成功")
    
    # 测试学习率调度
    print("\n" + "=" * 60)
    print("学习率调度测试")
    print("=" * 60)
    
    total_epochs = 200
    
    print(f"\n{'Epoch':<10} {'学习率':<15} {'阶段'}")
    print("-" * 40)
    
    test_epochs = [0, 5, 10, 20, 50, 100, 150, 199]
    for epoch in test_epochs:
        lr = optimizer.get_lr(epoch, total_epochs)
        
        if epoch < optimizer.warmup_epochs:
            stage = "预热"
        elif epoch < total_epochs * 0.5:
            stage = "初期"
        elif epoch < total_epochs * 0.75:
            stage = "中期"
        else:
            stage = "后期"
        
        print(f"{epoch:<10} {lr:<15.6f} {stage}")
    
    # 测试完整调度表
    schedule = optimizer.create_lr_schedule(total_epochs)
    print(f"\n✓ 创建完整调度表: {len(schedule)} 个轮次")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("学习率统计")
    print("=" * 60)
    lrs = list(schedule.values())
    print(f"最大学习率: {max(lrs):.6f}")
    print(f"最小学习率: {min(lrs):.6f}")
    print(f"平均学习率: {np.mean(lrs):.6f}")
    
    # 测试知识蒸馏
    if optimizer.use_distillation:
        print("\n" + "=" * 60)
        print("知识蒸馏测试")
        print("=" * 60)
        
        # 创建模拟输出
        batch_size = 32
        num_classes = 10
        
        student_outputs = torch.randn(batch_size, num_classes)
        teacher_outputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        criterion = nn.CrossEntropyLoss()
        
        # 计算蒸馏损失
        distill_loss = optimizer.distillation_loss(
            student_outputs, teacher_outputs, targets, criterion
        )
        
        print(f"✓ 蒸馏损失: {distill_loss.item():.4f}")
        print(f"  温度: {optimizer.distill_temperature}")
        print(f"  α权重: {optimizer.distill_alpha}")
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)
