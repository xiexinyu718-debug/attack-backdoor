"""
任务分离训练模块
实现后门任务和主任务的完全分离
确保投毒样本和正常样本分别优化
"""

import torch
import torch.nn.functional as F


class TaskSeparationTrainer:
    """
    任务分离训练器
    
    核心思想：
    - 主任务损失：针对正常样本
    - 后门任务损失：针对投毒样本
    - 总损失：加权组合，确保两个任务分离
    """
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 任务分离权重
        self.separation_weight = config.get('task_separation_weight', 0.5)
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        
        print(f"任务分离训练器初始化:")
        print(f"  - 分离权重: {self.separation_weight}")
        print(f"  - 主任务权重: {1 - self.separation_weight}")
        print(f"  - 后门任务权重: {self.separation_weight}")
    
    def compute_separated_loss(self, outputs, labels, poison_num):
        """
        计算任务分离的损失
        
        Args:
            outputs: 模型输出 [batch_size, num_classes]
            labels: 标签（前poison_num个是投毒标签）
            poison_num: 投毒样本数量
            
        Returns:
            total_loss: 总损失
            main_loss: 主任务损失
            backdoor_loss: 后门任务损失
        """
        batch_size = outputs.shape[0]
        
        # 分离主任务和后门任务
        if poison_num > 0 and poison_num < batch_size:
            # 情况1：既有投毒样本，又有正常样本
            # 后门任务损失（前poison_num个样本）
            backdoor_outputs = outputs[:poison_num]
            backdoor_labels = labels[:poison_num]
            backdoor_loss = self.criterion(backdoor_outputs, backdoor_labels)
            
            # 主任务损失（剩余样本）
            main_outputs = outputs[poison_num:]
            main_labels = labels[poison_num:]
            main_loss = self.criterion(main_outputs, main_labels)
            
        elif poison_num == batch_size:
            # 情况2：全是投毒样本（通常在评估时）
            backdoor_loss = self.criterion(outputs, labels)
            main_loss = torch.tensor(0.0).cuda()
            
        else:
            # 情况3：全是正常样本
            main_loss = self.criterion(outputs, labels)
            backdoor_loss = torch.tensor(0.0).cuda()
        
        # 加权组合
        total_loss = (1 - self.separation_weight) * main_loss + \
                     self.separation_weight * backdoor_loss
        
        return total_loss, main_loss, backdoor_loss
    
    def train_step(self, model, optimizer, inputs, labels, poison_num):
        """
        执行一个训练步骤（任务分离）
        
        Args:
            model: 模型
            optimizer: 优化器
            inputs: 输入数据
            labels: 标签
            poison_num: 投毒样本数量
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 前向传播
        outputs = model(inputs)
        
        # 计算任务分离损失
        total_loss, main_loss, backdoor_loss = self.compute_separated_loss(
            outputs, labels, poison_num
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 计算准确率
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / inputs.shape[0]
        
        # 返回详细信息
        loss_dict = {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'backdoor_loss': backdoor_loss.item(),
            'accuracy': accuracy,
            'poison_num': poison_num,
            'total_samples': inputs.shape[0]
        }
        
        return loss_dict
    
    def train_epoch(self, model, optimizer, dataloader, attacker, adversary_id, epoch):
        """
        训练一个完整的epoch（使用任务分离）
        
        Args:
            model: 模型
            optimizer: 优化器
            dataloader: 数据加载器
            attacker: 攻击器（用于生成投毒数据）
            adversary_id: 攻击者ID
            epoch: 当前轮次
            
        Returns:
            epoch_stats: epoch级别的统计信息
        """
        model.train()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_backdoor_loss = 0.0
        correct = 0
        total_samples = 0
        total_poisoned = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # 使用攻击器生成投毒数据
            poisoned_inputs, poisoned_labels, poison_num = \
                attacker.poison_input_with_task_separation(
                    inputs, labels, adversary_id, epoch, eval_mode=False
                )
            
            # 执行训练步骤
            loss_dict = self.train_step(
                model, optimizer, poisoned_inputs, poisoned_labels, poison_num
            )
            
            # 累积统计
            total_loss += loss_dict['total_loss']
            total_main_loss += loss_dict['main_loss']
            total_backdoor_loss += loss_dict['backdoor_loss']
            correct += loss_dict['accuracy'] * loss_dict['total_samples'] / 100.0
            total_samples += loss_dict['total_samples']
            total_poisoned += poison_num
        
        # 计算平均值
        num_batches = len(dataloader)
        epoch_stats = {
            'avg_loss': total_loss / num_batches,
            'avg_main_loss': total_main_loss / num_batches,
            'avg_backdoor_loss': total_backdoor_loss / num_batches,
            'accuracy': 100.0 * correct / total_samples if total_samples > 0 else 0,
            'total_samples': total_samples,
            'poisoned_samples': total_poisoned,
            'poison_ratio': total_poisoned / total_samples if total_samples > 0 else 0
        }
        
        return epoch_stats


class AdaptiveTaskSeparation(TaskSeparationTrainer):
    """
    自适应任务分离训练器
    
    根据训练进度和性能指标动态调整分离权重
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 自适应参数
        self.adaptive_enabled = config.get('adaptive_adjustment', {}).get('enabled', False)
        self.asr_threshold = config.get('adaptive_adjustment', {}).get('asr_threshold', 0.85)
        self.acc_threshold = config.get('adaptive_adjustment', {}).get('accuracy_threshold', 0.88)
        
        # 权重历史
        self.weight_history = []
        
        if self.adaptive_enabled:
            print(f"  - 启用自适应调整")
            print(f"    ASR阈值: {self.asr_threshold}")
            print(f"    准确率阈值: {self.acc_threshold}")
    
    def adjust_weight(self, asr, main_accuracy, epoch):
        """
        根据性能指标自适应调整分离权重
        
        Args:
            asr: 当前攻击成功率（0-1）
            main_accuracy: 主任务准确率（0-1）
            epoch: 当前轮次
        """
        if not self.adaptive_enabled:
            return
        
        original_weight = self.separation_weight
        
        # 规则1：如果ASR太低，增加后门任务权重
        if asr < self.asr_threshold:
            self.separation_weight = min(0.9, self.separation_weight + 0.05)
            print(f"  [自适应] ASR={asr:.2f} < {self.asr_threshold:.2f}, "
                  f"增加后门权重: {original_weight:.2f} -> {self.separation_weight:.2f}")
        
        # 规则2：如果主任务准确率太低，减少后门任务权重
        elif main_accuracy < self.acc_threshold:
            self.separation_weight = max(0.1, self.separation_weight - 0.05)
            print(f"  [自适应] Acc={main_accuracy:.2f} < {self.acc_threshold:.2f}, "
                  f"减少后门权重: {original_weight:.2f} -> {self.separation_weight:.2f}")
        
        # 记录历史
        self.weight_history.append({
            'epoch': epoch,
            'weight': self.separation_weight,
            'asr': asr,
            'accuracy': main_accuracy
        })


def create_trainer(config, adaptive=False):
    """
    工厂函数：创建训练器
    
    Args:
        config: 配置对象
        adaptive: 是否使用自适应训练器
        
    Returns:
        trainer: 训练器实例
    """
    if adaptive:
        return AdaptiveTaskSeparation(config)
    else:
        return TaskSeparationTrainer(config)


if __name__ == '__main__':
    # 测试代码
    print("测试任务分离训练模块\n")
    
    # 创建模拟配置
    class MockConfig:
        def get(self, key, default=None):
            config_dict = {
                'task_separation_weight': 0.5,
                'adaptive_adjustment': {
                    'enabled': True,
                    'asr_threshold': 0.85,
                    'accuracy_threshold': 0.88
                }
            }
            return config_dict.get(key, default)
    
    config = MockConfig()
    
    # 测试基础训练器
    print("1. 基础训练器:")
    trainer = TaskSeparationTrainer(config)
    
    # 模拟输出和标签
    outputs = torch.randn(8, 10).cuda()
    labels = torch.randint(0, 10, (8,)).cuda()
    poison_num = 2
    
    total_loss, main_loss, backdoor_loss = trainer.compute_separated_loss(
        outputs, labels, poison_num
    )
    print(f"  总损失: {total_loss.item():.4f}")
    print(f"  主任务损失: {main_loss.item():.4f}")
    print(f"  后门损失: {backdoor_loss.item():.4f}")
    
    # 测试自适应训练器
    print("\n2. 自适应训练器:")
    adaptive_trainer = AdaptiveTaskSeparation(config)
    
    print(f"  初始权重: {adaptive_trainer.separation_weight:.2f}")
    
    # 模拟调整
    adaptive_trainer.adjust_weight(asr=0.75, main_accuracy=0.90, epoch=10)
    print(f"  调整后权重: {adaptive_trainer.separation_weight:.2f}")
    
    adaptive_trainer.adjust_weight(asr=0.90, main_accuracy=0.85, epoch=11)
    print(f"  再次调整权重: {adaptive_trainer.separation_weight:.2f}")
