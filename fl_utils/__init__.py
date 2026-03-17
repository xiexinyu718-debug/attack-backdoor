"""
FL Utils模块 - 因子化触发器攻击工具包

提供以下功能:
1. 触发器因子定义 (trigger_factors.py)
2. 因子化攻击器 (factorized_attacker.py)
3. 任务分离训练 (task_separation.py)
4. 评估指标 (evaluation.py)
5. 可视化工具 (visualization.py)
"""

from .trigger_factors import (
    TriggerFactor,
    PositionPerturbationFactor,
    FrequencyPerturbationFactor,
    ColorShiftFactor,
    GeometricPerturbationFactor,
    create_factor_from_config,
    PREDEFINED_FACTORS
)

from .factorized_attacker import FactorizedAttacker

from .task_separation import (
    TaskSeparationTrainer,
    AdaptiveTaskSeparation,
    create_trainer
)

from .evaluation import (
    FactorizedAttackEvaluator,
    compare_with_baselines
)

from .visualization import (
    FactorizedAttackVisualizer,
    visualize_complete_report
)


__all__ = [
    # 触发器因子
    'TriggerFactor',
    'PositionPerturbationFactor',
    'FrequencyPerturbationFactor',
    'ColorShiftFactor',
    'GeometricPerturbationFactor',
    'create_factor_from_config',
    'PREDEFINED_FACTORS',
    
    # 攻击器
    'FactorizedAttacker',
    
    # 训练器
    'TaskSeparationTrainer',
    'AdaptiveTaskSeparation',
    'create_trainer',
    
    # 评估
    'FactorizedAttackEvaluator',
    'compare_with_baselines',
    
    # 可视化
    'FactorizedAttackVisualizer',
    'visualize_complete_report',
]

__version__ = '1.0.0'
__author__ = 'Research Team'
__description__ = 'Factorized Trigger Backdoor Attack Framework for Federated Learning'