"""
Helper模块 - 联邦学习辅助类
负责数据加载、模型管理、参与者采样、攻击者配置等核心功能

新增功能:
- Model Replacement (MR): 放大恶意更新以确保后门持久性
  参考论文: "Beyond Traditional Threats: A Persistent Backdoor Attack on Federated Learning"

MR使用方法:
    1. 在配置中启用: use_model_replacement=True
    2. 设置缩放因子: scale_factor_gamma=1000.0 (或自动计算: n/η)
    3. 设置攻击期: poison_start_epoch, poison_stop_epoch
    4. 在训练循环中调用: helper.apply_model_replacement(local_model, global_model, pid, epoch)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
from collections import defaultdict


class Config:
    """配置类 - 将字典转换为对象属性访问"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """字典式访问"""
        return getattr(self, key, default)

    def __repr__(self):
        return f"Config({self.__dict__})"


class Helper:
    """
    联邦学习Helper类

    核心功能:
    1. 数据加载和分割
    2. 模型初始化和管理
    3. 参与者采样
    4. 攻击者配置
    5. 学习率调度
    """

    def __init__(self, config):
        """
        初始化Helper

        Args:
            config: 配置字典或Config对象
        """
        if isinstance(config, dict):
            self.config = Config(config)
        else:
            self.config = config

        # 基础配置
        self.dataset = self.config.dataset
        self.seed = self.config.get('seed', 0)

        # 联邦学习配置
        self.num_total_participants = self.config.num_total_participants
        self.num_sampled_participants = self.config.num_sampled_participants
        self.num_adversaries = self.config.num_adversaries

        # 训练配置
        self.epochs = self.config.epochs
        self.batch_size = self.config.batch_size
        self.test_batch_size = self.config.get('test_batch_size', 1024)

        # ============= Model Replacement配置 =============
        self.use_model_replacement = self.config.get('use_model_replacement', False)
        self.scale_factor_gamma = self.config.get('scale_factor_gamma', 1000.0)

        # 计算缩放因子（如果没有直接指定）
        if self.use_model_replacement and not self.config.get('scale_factor_gamma'):
            # γ = n / η
            n = self.num_total_participants
            eta = self.config.get('lr', 0.1)
            self.scale_factor_gamma = n / eta
            print(f"  自动计算缩放因子: γ = {n}/{eta} = {self.scale_factor_gamma}")
        # ===============================================

        # 创建保存目录
        self.folder_path = self.create_folder()

        # 初始化容器
        self.train_data = {}  # {participant_id: DataLoader}
        self.test_data = None
        self.global_model = None
        self.client_models = []
        self.adversary_list = []

        print(f"Helper初始化完成:")
        print(f"  数据集: {self.dataset}")
        print(f"  参与者总数: {self.num_total_participants}")
        print(f"  每轮采样: {self.num_sampled_participants}")
        print(f"  攻击者数量: {self.num_adversaries}")
        if self.use_model_replacement:
            print(f"  Model Replacement: 启用")
            print(f"  缩放因子γ: {self.scale_factor_gamma}")
        print(f"  保存路径: {self.folder_path}\n")

    def create_folder(self):
        """创建实验保存目录"""
        environment_name = self.config.get('environment_name', 'fl_experiment')
        folder_path = f'./saved_models/{environment_name}'
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def load_data(self):
        """
        加载和分割数据集
        支持: CIFAR-10, CIFAR-100, MNIST
        """
        print(f"加载数据集: {self.dataset}...")

        if self.dataset.lower() == 'cifar10':
            self._load_cifar10()
        elif self.dataset.lower() == 'cifar100':
            self._load_cifar100()
        elif self.dataset.lower() == 'mnist':
            self._load_mnist()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset}")

        print(f"✓ 数据加载完成")
        print(f"  训练集参与者: {len(self.train_data)}")
        print(f"  测试集大小: {len(self.test_data.dataset)}\n")

    def _load_cifar10(self):
        """加载CIFAR-10数据集"""
        # 数据预处理
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010)),
        ])

        # 加载数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transform_test
        )

        # 分割训练数据给各个参与者
        self._split_train_data(train_dataset)

        # 创建测试数据加载器
        self.test_data = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=0
        )

        # 设置类别数
        self.num_classes = 10

    def _load_cifar100(self):
        """加载CIFAR-100数据集"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                               (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                               (0.2675, 0.2565, 0.2761)),
        ])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True,
            transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True,
            transform=transform_test
        )

        self._split_train_data(train_dataset)

        self.test_data = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=0
        )

        self.num_classes = 100

    def _load_mnist(self):
        """加载MNIST数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True,
            transform=transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True,
            transform=transform
        )

        self._split_train_data(train_dataset)

        self.test_data = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=0
        )

        self.num_classes = 10

    def _split_train_data(self, train_dataset):
        """
        分割训练数据给各个参与者
        支持IID和Non-IID（Dirichlet分布）
        """
        sample_method = self.config.get('sample_method', 'random')

        if sample_method == 'random':
            # IID分割
            self._split_iid(train_dataset)
        elif sample_method == 'dirichlet':
            # Non-IID分割（Dirichlet分布）
            self._split_dirichlet(train_dataset)
        else:
            raise ValueError(f"不支持的分割方法: {sample_method}")

    def _split_iid(self, dataset):
        """IID数据分割"""
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)

        # 计算每个参与者的数据量
        samples_per_participant = total_size // self.num_total_participants

        for i in range(self.num_total_participants):
            start_idx = i * samples_per_participant
            end_idx = (i + 1) * samples_per_participant if i < self.num_total_participants - 1 else total_size

            participant_indices = indices[start_idx:end_idx]
            participant_dataset = Subset(dataset, participant_indices)

            self.train_data[i] = DataLoader(
                participant_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )

    def _split_dirichlet(self, dataset):
        """Non-IID数据分割（Dirichlet分布）"""
        alpha = self.config.get('dirichlet_alpha', 0.5)

        # 获取所有标签
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            # 手动提取标签
            labels = np.array([dataset[i][1] for i in range(len(dataset))])

        num_classes = len(np.unique(labels))

        # 为每个类别的样本生成索引
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        # 为每个参与者分配数据
        participant_indices = [[] for _ in range(self.num_total_participants)]

        for c in range(num_classes):
            # 使用Dirichlet分布生成分配比例
            proportions = np.random.dirichlet(
                np.repeat(alpha, self.num_total_participants)
            )

            # 根据比例分配样本
            proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)
            proportions = [0] + proportions.tolist()

            for i in range(self.num_total_participants):
                start = proportions[i]
                end = proportions[i + 1]
                participant_indices[i].extend(class_indices[c][start:end])

        # 创建DataLoader
        for i in range(self.num_total_participants):
            if len(participant_indices[i]) > 0:
                participant_dataset = Subset(dataset, participant_indices[i])
                self.train_data[i] = DataLoader(
                    participant_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0
                )
            else:
                # 如果某个参与者没有分配到数据，给它一些随机样本
                participant_indices[i] = random.sample(
                    range(len(dataset)),
                    max(1, len(dataset) // self.num_total_participants // 10)
                )
                participant_dataset = Subset(dataset, participant_indices[i])
                self.train_data[i] = DataLoader(
                    participant_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0
                )

    def load_model(self):
        """加载模型"""
        print(f"初始化模型...")

        # 根据数据集选择模型
        if self.dataset.lower() in ['cifar10', 'cifar100']:
            from models.resnet import ResNet18
            self.global_model = ResNet18(num_classes=self.num_classes)
        elif self.dataset.lower() == 'mnist':
            # 简单的CNN模型用于MNIST
            self.global_model = self._create_mnist_model()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset}")

        # 移动到GPU
        self.global_model = self.global_model.cuda()

        print(f"✓ 模型初始化完成: {self.global_model.__class__.__name__}\n")

    def _create_mnist_model(self):
        """创建MNIST简单CNN模型"""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.relu(x)
                x = nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return x

        return SimpleCNN()

    def config_adversaries(self):
        """配置攻击者"""
        print(f"配置攻击者...")

        # 随机选择攻击者
        self.adversary_list = random.sample(
            range(self.num_total_participants),
            self.num_adversaries
        )

        print(f"✓ 攻击者配置完成: {self.adversary_list}\n")

    def sample_participants(self, epoch):
        """
        采样参与者

        Args:
            epoch: 当前轮次

        Returns:
            sampled_participants: 采样的参与者ID列表
        """
        # 确保至少有一个攻击者被采样
        available_adversaries = [adv for adv in self.adversary_list]

        if len(available_adversaries) > 0:
            # 随机选择一个攻击者
            selected_adversary = random.choice(available_adversaries)

            # 从非攻击者中采样剩余参与者
            benign_participants = [
                i for i in range(self.num_total_participants)
                if i not in self.adversary_list
            ]

            num_benign = self.num_sampled_participants - 1
            if num_benign > 0:
                selected_benign = random.sample(benign_participants,
                                               min(num_benign, len(benign_participants)))
            else:
                selected_benign = []

            sampled_participants = [selected_adversary] + selected_benign
        else:
            # 如果没有攻击者，随机采样
            sampled_participants = random.sample(
                range(self.num_total_participants),
                self.num_sampled_participants
            )

        return sampled_participants

    def get_lr(self, epoch):
        """
        获取学习率（支持学习率调度）

        Args:
            epoch: 当前轮次

        Returns:
            lr: 学习率
        """
        lr_method = self.config.get('lr_method', 'constant')
        base_lr = self.config.lr

        if lr_method == 'constant':
            return base_lr

        elif lr_method == 'linear':
            # 线性衰减
            decay_rate = self.config.get('lr_decay_rate', 0.99)
            return base_lr * (decay_rate ** epoch)

        elif lr_method == 'step':
            # 阶梯衰减
            decay_epochs = self.config.get('lr_decay_epochs', [50, 75])
            decay_factor = self.config.get('lr_decay_factor', 0.1)

            lr = base_lr
            for decay_epoch in decay_epochs:
                if epoch >= decay_epoch:
                    lr *= decay_factor
            return lr

        elif lr_method == 'cosine':
            # 余弦退火
            import math
            return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))

        else:
            return base_lr

    def apply_model_replacement(self, local_model, global_model, participant_id, epoch):
        """
        应用Model Replacement放大恶意更新

        这是PDF论文中的关键技术！通过放大恶意更新，确保后门在平均聚合中存活。

        Args:
            local_model: 本地训练后的模型
            global_model: 全局模型
            participant_id: 参与者ID
            epoch: 当前轮次

        Returns:
            scaled_model: 如果是恶意客户端且MR启用，返回放大后的模型；否则返回原模型
        """
        import copy

        # 检查是否启用MR
        if not self.use_model_replacement:
            return local_model

        # 检查是否为恶意客户端
        if participant_id not in self.adversary_list:
            return local_model

        # 检查是否在攻击期内
        poison_start = self.config.get('poison_start_epoch', 0)
        poison_stop = self.config.get('poison_stop_epoch', float('inf'))

        if not (poison_start <= epoch < poison_stop):
            return local_model

        # ========== 应用Model Replacement ==========
        print(f"\n  [MR] 应用Model Replacement (客户端 {participant_id}, γ={self.scale_factor_gamma})")

        # 计算模型更新
        update_dict = {}
        update_norms = []

        for (name, local_param), (_, global_param) in zip(
            local_model.named_parameters(),
            global_model.named_parameters()
        ):
            # 计算更新
            update = local_param.data - global_param.data
            update_dict[name] = update

            # 记录范数（用于调试）
            if 'weight' in name and len(update.shape) >= 2:
                before_norm = torch.norm(update).item()
                after_norm = before_norm * self.scale_factor_gamma
                update_norms.append((name, before_norm, after_norm))

        # 显示前3个层的放大效果
        if update_norms:
            print(f"  [MR] 更新放大效果（前3层）:")
            for name, before, after in update_norms[:3]:
                print(f"    {name}: ||Δ|| = {before:.4f} → {after:.2f} (×{self.scale_factor_gamma})")

        # 创建放大后的模型
        scaled_model = copy.deepcopy(global_model)
        for name, param in scaled_model.named_parameters():
            global_param_data = dict(global_model.named_parameters())[name].data
            param.data = global_param_data + self.scale_factor_gamma * update_dict[name]

        print(f"  [MR] ✓ 模型更新已放大{self.scale_factor_gamma}倍\n")

        return scaled_model

    def test_model(self, model=None):
        """
        测试模型准确率

        Args:
            model: 要测试的模型（默认使用全局模型）

        Returns:
            accuracy: 准确率（百分比）
        """
        if model is None:
            model = self.global_model

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_data:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        model.train()
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy

    @property
    def target_class(self):
        """获取目标类别"""
        return self.config.get('target_class', 0)

    def should_apply_mr(self, participant_id, epoch):
        """
        检查是否应该对该参与者应用Model Replacement

        Args:
            participant_id: 参与者ID
            epoch: 当前轮次

        Returns:
            bool: 是否应该应用MR
        """
        # 检查全局开关
        if not self.use_model_replacement:
            return False

        # 检查是否为恶意客户端
        if participant_id not in self.adversary_list:
            return False

        # 检查是否在攻击期内
        poison_start = self.config.get('poison_start_epoch', 0)
        poison_stop = self.config.get('poison_stop_epoch', float('inf'))

        return poison_start <= epoch < poison_stop

    @property
    def save_every(self):
        """获取保存频率"""
        return self.config.get('save_on_epochs', [50, 100])

    def get_attack_phase_info(self, epoch):
        """
        获取当前轮次的攻击阶段信息

        Args:
            epoch: 当前轮次

        Returns:
            dict: 包含阶段信息的字典
        """
        poison_start = self.config.get('poison_start_epoch', 0)
        poison_stop = self.config.get('poison_stop_epoch', float('inf'))

        is_attacking = poison_start <= epoch < poison_stop

        if epoch < poison_start:
            phase = "pre-attack"
            description = "攻击前期（良性训练）"
        elif is_attacking:
            phase = "attacking"
            description = f"攻击期（{epoch - poison_start + 1}/{poison_stop - poison_start}轮）"
        else:
            phase = "post-attack"
            description = f"攻击后期（停止{epoch - poison_stop + 1}轮）"

        return {
            'phase': phase,
            'description': description,
            'is_attacking': is_attacking,
            'poison_start': poison_start,
            'poison_stop': poison_stop,
            'epochs_since_start': max(0, epoch - poison_start + 1) if is_attacking else 0,
            'epochs_since_stop': max(0, epoch - poison_stop + 1) if not is_attacking and epoch >= poison_stop else 0
        }


if __name__ == '__main__':
    # 测试代码
    print("测试Helper模块\n")

    # 创建测试配置
    test_config = {
        'dataset': 'cifar10',
        'seed': 0,
        'num_total_participants': 100,
        'num_sampled_participants': 10,
        'num_adversaries': 14,  # 改为14（与PDF一致）
        'epochs': 200,  # 改为200
        'batch_size': 64,
        'test_batch_size': 1024,
        'lr': 0.1,  # 改为0.1（与PDF一致）
        'momentum': 0.9,
        'decay': 0.0001,
        'sample_method': 'dirichlet',
        'dirichlet_alpha': 0.5,
        'target_class': 2,
        'environment_name': 'test_helper_mr',
        'retrain_times': 2,
        'attacker_retrain_times': 6,  # 改为6（与PDF一致）
        # ========== Model Replacement配置 ==========
        'use_model_replacement': True,
        'scale_factor_gamma': 1000.0,  # γ = n/η = 100/0.1
        'poison_start_epoch': 100,  # 先收敛100轮
        'poison_stop_epoch': 114,   # 投毒14轮
        # =========================================
    }

    # 创建Helper
    helper = Helper(test_config)

    # 加载数据
    helper.load_data()

    # 加载模型
    helper.load_model()

    # 配置攻击者
    helper.config_adversaries()

    # 测试采样
    print("\n测试参与者采样:")
    for i in range(3):
        sampled = helper.sample_participants(i)
        print(f"  Epoch {i}: {sampled}")

    # 测试学习率
    print("\n测试学习率调度:")
    for epoch in [0, 25, 50, 75, 99]:
        lr = helper.get_lr(epoch)
        print(f"  Epoch {epoch}: LR = {lr:.6f}")

    # 测试模型
    print("\n测试模型准确率:")
    acc = helper.test_model()
    print(f"  初始准确率: {acc:.2f}%")

    # ========== 测试Model Replacement功能 ==========
    if helper.use_model_replacement:
        print("\n测试Model Replacement功能:")

        # 测试不同epoch的攻击阶段
        test_epochs = [0, 50, 100, 105, 110, 114, 150, 200]
        print("\n  攻击阶段信息:")
        for epoch in test_epochs:
            phase_info = helper.get_attack_phase_info(epoch)
            print(f"    Epoch {epoch:3d}: {phase_info['description']}")

        # 测试should_apply_mr
        print("\n  是否应用MR:")
        adversary_id = helper.adversary_list[0]
        benign_id = 0 if 0 not in helper.adversary_list else 1

        for epoch in [50, 100, 105, 114, 150]:
            apply_adv = helper.should_apply_mr(adversary_id, epoch)
            apply_ben = helper.should_apply_mr(benign_id, epoch)
            print(f"    Epoch {epoch}: 攻击者={apply_adv}, 良性={apply_ben}")

        # 测试apply_model_replacement（使用虚拟模型）
        print("\n  测试MR放大效果:")
        print(f"    缩放因子 γ = {helper.scale_factor_gamma}")
        print(f"    攻击期: Epoch {helper.config.poison_start_epoch}-{helper.config.poison_stop_epoch}")
    # ==============================================

    print("\n✓ Helper模块测试完成")