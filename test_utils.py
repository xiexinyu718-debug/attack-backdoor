"""
测试工具 - 提供测试用的Mock对象和辅助函数
"""


class MockConfig:
    """
    模拟配置对象
    支持属性访问和字典式访问
    """
    def __init__(self, **kwargs):
        # 默认配置
        defaults = {
            # 基础配置
            'dataset': 'cifar10',
            'seed': 0,
            'epochs': 100,
            'poison_epochs': 100,
            'poison_start_epoch': 0,
            
            # 联邦学习配置
            'num_total_participants': 100,
            'num_sampled_participants': 10,
            'num_adversaries': 4,
            'sample_method': 'random',
            'dirichlet_alpha': 0.9,
            
            # 训练配置
            'lr': 0.01,
            'target_lr': 0.02,
            'lr_method': 'linear',
            'momentum': 0.9,
            'decay': 0.0005,
            'batch_size': 64,
            'test_batch_size': 1024,
            'retrain_times': 2,
            'attacker_retrain_times': 2,
            
            # 聚合方法
            'agg_method': 'avg',
            'clip_factor': 1,
            
            # 因子化触发器配置
            'attacker_method': 'factorized',
            'k_of_m_k': 2,
            'k_of_m_m': 3,
            'rotation_strategy': 'adversary_specific',
            'rotation_frequency': 1,
            
            # 动态优化
            'initial_intensity': 0.1,
            'final_intensity': 0.5,
            
            # 任务分离
            'task_separation_weight': 0.5,
            
            # 攻击目标
            'target_class': 2,
            'bkd_ratio': 0.25,
            
            # 保存配置
            'environment_name': 'test_env',
            'save_model': True,
            'save_on_epochs': [50, 100],
            'results_dir': './results/test',
        }
        
        # 合并用户提供的配置
        defaults.update(kwargs)
        
        # 设置所有配置为属性
        for key, value in defaults.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        """字典式访问"""
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        """支持 config['key'] 访问"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """支持 config['key'] = value 赋值"""
        setattr(self, key, value)
    
    def update(self, other):
        """更新配置"""
        if isinstance(other, dict):
            for key, value in other.items():
                setattr(self, key, value)
        else:
            for key in dir(other):
                if not key.startswith('_'):
                    setattr(self, key, getattr(other, key))


class MockHelper:
    """
    模拟Helper对象
    用于测试，不需要实际加载数据和模型
    """
    def __init__(self, config=None):
        if config is None:
            self.config = MockConfig()
        elif isinstance(config, dict):
            self.config = MockConfig(**config)
        else:
            self.config = config
        
        # 基础属性
        self.dataset = self.config.dataset
        self.num_total_participants = self.config.num_total_participants
        self.num_sampled_participants = self.config.num_sampled_participants
        self.num_adversaries = self.config.num_adversaries
        
        # 攻击者列表
        import random
        random.seed(self.config.seed)
        self.adversary_list = random.sample(
            range(self.num_total_participants),
            self.num_adversaries
        )
        
        # 占位符
        self.train_data = {}
        self.test_data = None
        self.global_model = None
        self.client_models = []
        self.folder_path = './test_models'
    
    def get_lr(self, epoch):
        """获取学习率"""
        return self.config.lr
    
    def sample_participants(self, epoch):
        """采样参与者"""
        import random
        return random.sample(
            range(self.num_total_participants),
            self.num_sampled_participants
        )


def create_mock_helper(custom_config=None):
    """
    创建模拟Helper对象的便捷函数
    
    Args:
        custom_config: 自定义配置字典
        
    Returns:
        MockHelper实例
    """
    if custom_config is None:
        return MockHelper()
    else:
        return MockHelper(custom_config)


def print_test_header(test_name):
    """打印测试标题"""
    print("\n" + "="*70)
    print(f"测试: {test_name}")
    print("="*70)


def print_test_result(test_name, success, message=""):
    """打印测试结果"""
    if success:
        print(f"✓ {test_name} 测试通过")
        if message:
            print(f"  {message}")
    else:
        print(f"✗ {test_name} 测试失败")
        if message:
            print(f"  {message}")


# 预定义的测试配置
TEST_CONFIGS = {
    'minimal': {
        'num_total_participants': 10,
        'num_sampled_participants': 3,
        'num_adversaries': 2,
        'epochs': 5,
        'batch_size': 32,
    },
    
    'small': {
        'num_total_participants': 20,
        'num_sampled_participants': 5,
        'num_adversaries': 2,
        'epochs': 10,
        'batch_size': 64,
    },
    
    'standard': {
        'num_total_participants': 100,
        'num_sampled_participants': 10,
        'num_adversaries': 4,
        'epochs': 100,
        'batch_size': 64,
    },
    
    'large': {
        'num_total_participants': 200,
        'num_sampled_participants': 20,
        'num_adversaries': 10,
        'epochs': 200,
        'batch_size': 128,
    }
}


def get_test_config(name='standard'):
    """
    获取预定义的测试配置
    
    Args:
        name: 配置名称 ('minimal', 'small', 'standard', 'large')
        
    Returns:
        MockConfig实例
    """
    if name not in TEST_CONFIGS:
        raise ValueError(f"未知配置: {name}，可用配置: {list(TEST_CONFIGS.keys())}")
    
    return MockConfig(**TEST_CONFIGS[name])


if __name__ == '__main__':
    print("测试工具模块\n")
    
    # 测试MockConfig
    print_test_header("MockConfig")
    
    config = MockConfig(epochs=50, num_adversaries=5)
    
    # 测试属性访问
    assert config.epochs == 50
    assert config.num_adversaries == 5
    assert config.target_class == 2  # 默认值
    print(f"✓ 属性访问正常")
    
    # 测试字典式访问
    assert config.get('epochs') == 50
    assert config.get('unknown_key', 'default') == 'default'
    print(f"✓ 字典式访问正常")
    
    # 测试[]访问
    assert config['epochs'] == 50
    config['new_param'] = 999
    assert config.new_param == 999
    print(f"✓ []访问正常")
    
    print_test_result("MockConfig", True, "所有功能正常")
    
    # 测试MockHelper
    print_test_header("MockHelper")
    
    helper = create_mock_helper({'num_adversaries': 3})
    
    assert len(helper.adversary_list) == 3
    print(f"✓ 攻击者列表: {helper.adversary_list}")
    
    lr = helper.get_lr(10)
    print(f"✓ 学习率: {lr}")
    
    participants = helper.sample_participants(0)
    print(f"✓ 采样参与者: {participants[:5]}...")
    
    print_test_result("MockHelper", True, "所有功能正常")
    
    # 测试预定义配置
    print_test_header("预定义配置")
    
    for name in TEST_CONFIGS.keys():
        config = get_test_config(name)
        print(f"  {name}: {config.num_total_participants} 参与者, "
              f"{config.num_adversaries} 攻击者, "
              f"{config.epochs} 轮次")
    
    print_test_result("预定义配置", True, "所有配置加载正常")
    
    print("\n" + "="*70)
    print("✓ 测试工具模块验证完成")
    print("="*70)
