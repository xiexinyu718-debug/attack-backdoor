# 因子化触发器攻击 - 模块化代码包

## 📁 目录结构

```
factorized_backdoor_attack/
│
├── fl_utils/                          # 核心功能模块
│   ├── __init__.py                    # 包初始化
│   ├── trigger_factors.py             # 触发器因子定义
│   ├── factorized_attacker.py         # 因子化攻击器
│   ├── task_separation.py             # 任务分离训练
│   ├── evaluation.py                  # 评估指标
│   ├── visualization.py               # 可视化工具
│   ├── helper.py                      # Helper类（数据、模型管理）
│   └── aggregator.py                  # 聚合器
│
├── main/                              # 主程序
│   ├── train_factorized.py           # 主训练脚本
│   └── configs/                       # 配置文件
│       └── factorized_config.yaml     # 因子化攻击配置
│
├── models/                            # 模型定义
│   └── resnet.py                      # ResNet模型
│
└── README_MODULAR.md                  # 本文档
```

## 🎯 模块说明

### 1. `trigger_factors.py` - 触发器因子模块

**功能**: 定义各种类型的触发器子因子

**核心类**:
- `TriggerFactor`: 基类
- `PositionPerturbationFactor`: 位置扰动因子
- `FrequencyPerturbationFactor`: 频域扰动因子
- `ColorShiftFactor`: 颜色偏移因子
- `GeometricPerturbationFactor`: 几何扰动因子

**使用示例**:
```python
from fl_utils.trigger_factors import PositionPerturbationFactor

# 创建位置扰动因子
factor = PositionPerturbationFactor(
    position=(2, 2),      # 左上角
    size=4,               # 4x4区域
    pattern='square',     # 正方形
    intensity=0.15        # 强度15%
)

# 应用到输入
perturbed = factor.apply(inputs)
mask = factor.get_mask(inputs)
```

**关键特性**:
- ✓ 每个因子单独影响小（~10-15%）
- ✓ 支持4种因子类型
- ✓ 可组合使用
- ✓ 独立的mask生成

---

### 2. `factorized_attacker.py` - 因子化攻击器

**功能**: 实现k-of-m组合规则、动态优化、轮换策略

**核心类**:
- `FactorizedAttacker`: 主攻击器类

**使用示例**:
```python
from fl_utils.factorized_attacker import FactorizedAttacker

# 初始化攻击器
attacker = FactorizedAttacker(helper)

# 为攻击者分配因子组合
combination = attacker.assign_factor_combination(
    adversary_id=0,
    epoch=10
)

# 应用k-of-m触发器
poisoned_inputs, poisoned_labels, poison_num = \
    attacker.poison_input(
        inputs, labels,
        adversary_id=0,
        epoch=10,
        eval=False  # 训练模式
    )
```

**关键方法**:
- `_init_factor_library()`: 初始化因子库
- `assign_factor_combination()`: 分配因子组合（轮换）
- `apply_k_of_m_trigger()`: 应用k-of-m规则
- `poison_input()`: 投毒接口

**关键特性**:
- ✓ k-of-m组合规则
- ✓ 4种轮换策略
- ✓ 动态强度调度
- ✓ 轮换历史追踪

---

### 3. `task_separation.py` - 任务分离训练

**功能**: 实现后门任务和主任务的分离训练

**核心类**:
- `TaskSeparationTrainer`: 基础训练器
- `AdaptiveTaskSeparation`: 自适应训练器

**使用示例**:
```python
from fl_utils.task_separation import create_trainer

# 创建训练器
trainer = create_trainer(config, adaptive=True)

# 训练一个epoch
epoch_stats = trainer.train_epoch(
    model, optimizer, dataloader,
    attacker, adversary_id, epoch
)

# 自适应调整（如果启用）
trainer.adjust_weight(
    asr=0.85,
    main_accuracy=0.90,
    epoch=10
)
```

**损失函数**:
```python
# 主任务损失（正常样本）
L_main = CE(outputs[poison_num:], labels[poison_num:])

# 后门任务损失（投毒样本）
L_backdoor = CE(outputs[:poison_num], labels[:poison_num])

# 总损失（任务分离）
L_total = (1 - w) * L_main + w * L_backdoor
```

**关键特性**:
- ✓ 样本级任务分离
- ✓ 自适应权重调整
- ✓ 独立损失计算
- ✓ 详细统计信息

---

### 4. `evaluation.py` - 评估指标模块

**功能**: 提供全面的评估指标

**核心类**:
- `FactorizedAttackEvaluator`: 评估器

**使用示例**:
```python
from fl_utils.evaluation import FactorizedAttackEvaluator

# 创建评估器
evaluator = FactorizedAttackEvaluator(helper, attacker)

# 评估主任务
main_acc = evaluator.evaluate_main_task(model)

# 评估攻击成功率
asr = evaluator.evaluate_attack_success_rate(model, adv_id, epoch)

# 全面评估
results = evaluator.comprehensive_evaluation(model, epoch)
```

**评估指标**:
1. **主任务准确率**: 良性样本的分类准确率
2. **攻击成功率（ASR）**: 投毒样本被分类为目标类的比例
3. **因子多样性**: Shannon熵衡量因子使用分布
4. **隐蔽性**: 模型更新的L2范数
5. **k-of-m有效性**: 不同k值的ASR对比
6. **轮换有效性**: 因子组合变化率

---

### 5. `visualization.py` - 可视化工具

**功能**: 生成各种可视化图表

**核心类**:
- `FactorizedAttackVisualizer`: 可视化器

**使用示例**:
```python
from fl_utils.visualization import FactorizedAttackVisualizer

# 创建可视化器
visualizer = FactorizedAttackVisualizer(save_dir='./vis')

# 1. 可视化因子组合
visualizer.visualize_factor_combination(
    attacker, sample_input, adv_id, epoch
)

# 2. 可视化轮换模式
visualizer.visualize_rotation_pattern(attacker, epoch)

# 3. 可视化强度调度
visualizer.visualize_intensity_schedule(attacker, current_epoch=50)

# 4. 可视化评估指标
visualizer.visualize_evaluation_metrics(evaluation_history)
```

**生成的图表**:
- 因子组合效果图（原始→逐步应用→最终）
- 轮换模式时间线
- 动态强度曲线
- 评估指标趋势图
- k-of-m对比柱状图

---

### 6. `train_factorized.py` - 主训练脚本

**功能**: 整合所有模块，执行完整训练流程

**使用方法**:
```bash
# 基础使用
python train_factorized.py --gpu 0 --params configs/factorized_config.yaml

# 指定随机种子
python train_factorized.py --gpu 0 --params configs/factorized_config.yaml --seed 42
```

**训练流程**:
```
1. 初始化系统
   ├─ Helper（数据、模型）
   ├─ FactorizedAttacker（攻击器）
   ├─ TaskSeparationTrainer（训练器）
   ├─ Evaluator（评估器）
   └─ Visualizer（可视化器）

2. 联邦学习循环 (每个epoch)
   ├─ 采样参与者
   ├─ 本地训练
   │  ├─ 良性客户端: 正常训练
   │  └─ 恶意客户端: 因子化攻击训练
   ├─ 模型聚合
   └─ 定期评估和可视化

3. 最终报告
   ├─ 全面评估
   ├─ 基线对比
   └─ 生成JSON报告
```

---

## 🚀 快速开始

### Step 1: 准备环境

```bash
# 安装依赖
pip install torch torchvision numpy matplotlib pyyaml scipy

# 克隆代码
git clone <repository>
cd factorized_backdoor_attack
```

### Step 2: 配置实验

编辑 `main/configs/factorized_config.yaml`:

```yaml
# 基础配置
dataset: 'cifar10'
num_adversaries: 4
epochs: 100

# k-of-m规则
k_of_m_k: 2
k_of_m_m: 3

# 轮换策略
rotation_strategy: 'adversary_specific'

# 任务分离
task_separation_weight: 0.5
```

### Step 3: 运行训练

```bash
cd main
python train_factorized.py --gpu 0 --params configs/factorized_config.yaml
```

### Step 4: 查看结果

训练完成后，检查以下目录：
- `saved_models/`: 保存的模型
- `visualizations/`: 可视化图表
- `final_report.json`: 完整评估报告

---

## 📊 配置说明

### 场景配置

**高隐蔽性场景**:
```yaml
k_of_m_k: 3               # 需要3个因子
k_of_m_m: 5               # 从5个中选
num_adversaries: 2        # 少数攻击者
bkd_ratio: 0.15          # 低投毒率
task_separation_weight: 0.3
```

**平衡场景（推荐）**:
```yaml
k_of_m_k: 2               # 需要2个因子
k_of_m_m: 3               # 从3个中选
num_adversaries: 4        # 中等攻击者
bkd_ratio: 0.25          # 标准投毒率
task_separation_weight: 0.5
```

**高成功率场景**:
```yaml
k_of_m_k: 2               # 需要2个因子
k_of_m_m: 2               # 从2个中选
num_adversaries: 6        # 较多攻击者
bkd_ratio: 0.35          # 高投毒率
task_separation_weight: 0.7
```

---

## 🔧 自定义扩展

### 添加新的触发器因子

1. 在 `trigger_factors.py` 中定义新类:

```python
class CustomFactor(TriggerFactor):
    def __init__(self, param1, param2, intensity=0.1):
        super().__init__(name="Custom", intensity=intensity)
        self.param1 = param1
        self.param2 = param2
    
    def apply(self, inputs):
        # 实现你的扰动逻辑
        perturbed = inputs.clone()
        # ... 自定义处理 ...
        return perturbed
    
    def get_mask(self, inputs):
        # 返回影响区域的mask
        return mask
```

2. 在 `factorized_attacker.py` 的 `_init_factor_library()` 中添加:

```python
def _init_factor_library(self):
    factors = []
    # ... 现有因子 ...
    
    # 添加自定义因子
    factors.append(CustomFactor(
        param1=value1,
        param2=value2,
        intensity=0.1
    ))
    
    return factors
```

### 自定义轮换策略

在 `factorized_attacker.py` 的 `assign_factor_combination()` 中添加:

```python
elif self.rotation_strategy == 'custom':
    # 实现你的轮换逻辑
    combination = your_custom_logic(adversary_id, epoch)
```

### 自定义评估指标

在 `evaluation.py` 中添加新方法:

```python
class FactorizedAttackEvaluator:
    # ... 现有方法 ...
    
    def evaluate_custom_metric(self, model, epoch):
        """你的自定义评估指标"""
        # 实现评估逻辑
        return metric_value
```

---

## 📈 性能监控

### 训练过程监控

查看实时输出:
```
=====================================================
联邦学习轮次 10/100
=====================================================
采样参与者: [3, 15, 27, 42, 58, 71, 83, 95, 12, 64]
恶意客户端: [3]

============================================================
训练恶意客户端 3 (Epoch 10)
============================================================
  因子组合: ['Position_(2, 2)_square', 'Frequency_high', 'Color_brightness']
  k-of-m: 2-of-3
  当前强度: 0.234
  开始训练 (重复 2 次)...

  训练完成:
    损失: 0.8543
    准确率: 78.34%
    投毒样本: 128/512
```

### 评估报告

查看 `final_report.json`:
```json
{
  "configuration": {
    "k_of_m": "2-of-3",
    "num_adversaries": 4,
    "rotation_strategy": "adversary_specific"
  },
  "final_results": {
    "main_accuracy": 91.23,
    "average_asr": 93.45,
    "factor_diversity": 0.7234,
    "rotation_effectiveness": 0.8567
  }
}
```

---

## 🐛 调试技巧

### 1. 检查因子效果

```python
# 单独测试每个因子
for i, factor in enumerate(attacker.factor_library):
    print(f"测试因子 {i}: {factor.name}")
    perturbed = factor.apply(test_input)
    diff = torch.norm(perturbed - test_input)
    print(f"  扰动范数: {diff:.6f}")
```

### 2. 验证k-of-m规则

```python
# 测试不同k值的效果
for k in range(1, attacker.m + 1):
    attacker.k = k
    asr = evaluator.evaluate_attack_success_rate(model, 0, epoch)
    print(f"k={k}: ASR={asr:.2f}%")
```

### 3. 可视化单个样本

```python
# 查看具体样本的扰动效果
test_sample = test_batch[0:1]
visualizer.visualize_factor_combination(
    attacker, test_sample, adv_id=0, epoch=10
)
```

### 4. 追踪轮换历史

```python
# 打印轮换历史
for adv_id, history in attacker.rotation_history.items():
    print(f"\nAdversary {adv_id} 轮换历史:")
    for h in history:
        print(f"  Epoch {h['epoch']}: {h['combination']}")
```

---

## ❓ 常见问题

### Q1: 如何调整攻击强度？

**A**: 修改以下参数:
```yaml
# 1. 因子强度（在配置中或代码中）
trigger_factorization:
  factors:
    position_perturbation:
      intensity: 0.20  # 增加到20%

# 2. 动态强度范围
dynamic_optimization:
  initial_intensity: 0.15  # 提高初始强度
  final_intensity: 0.60    # 提高最终强度

# 3. 投毒比例
bkd_ratio: 0.30  # 增加到30%
```

### Q2: 为什么ASR不高？

**可能原因和解决方案**:
1. **k值过大**: 降低k值，如从3-of-5改为2-of-3
2. **强度太低**: 增加因子强度或动态强度范围
3. **轮换太频繁**: 降低轮换频率
4. **任务分离权重低**: 增加 `task_separation_weight`

### Q3: 如何提高隐蔽性？

**建议**:
1. **增加k和m**: 使用3-of-5或4-of-7
2. **降低因子强度**: 每个因子强度<0.10
3. **使用diverse轮换**: 确保因子分布均匀
4. **降低投毒比例**: `bkd_ratio` < 0.20

### Q4: 内存不足怎么办？

**解决方案**:
```yaml
# 1. 减小batch size
batch_size: 32  # 从64降到32

# 2. 减少因子数量
# 在 _init_factor_library() 中注释掉一些因子

# 3. 降低可视化频率
visualization:
  enabled: true
  save_interval: 20  # 每20个epoch保存一次
```

---

## 📚 代码示例

### 完整示例：自定义实验

```python
# custom_experiment.py
import sys
sys.path.append("../")
import yaml
from fl_utils.helper import Helper
from fl_utils.factorized_attacker import FactorizedAttacker
from fl_utils.evaluation import FactorizedAttackEvaluator

# 1. 加载配置
with open('configs/custom_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 初始化
helper = Helper(config)
helper.load_data()
helper.load_model()

# 3. 创建攻击器
attacker = FactorizedAttacker(helper)

# 4. 自定义修改
attacker.k = 2  # 修改k值
attacker.m = 4  # 修改m值

# 5. 评估
evaluator = FactorizedAttackEvaluator(helper, attacker)
results = evaluator.comprehensive_evaluation(helper.global_model, 0)

print(f"Main Accuracy: {results['main_accuracy']:.2f}%")
print(f"Average ASR: {results['average_asr']:.2f}%")
```

---

## 🎓 学习路径

### 初学者
1. 先运行默认配置，观察结果
2. 阅读 `trigger_factors.py`，理解因子概念
3. 修改k-of-m参数，观察ASR变化
4. 查看可视化，理解因子组合效果

### 进阶用户
1. 实现自定义触发器因子
2. 设计新的轮换策略
3. 添加新的评估指标
4. 尝试不同的任务分离策略

### 研究者
1. 修改动态优化算法
2. 实现多目标优化
3. 设计自适应k值选择
4. 研究因子依赖关系

---

## 📖 参考文献

如果使用本代码，请引用：
```bibtex
@article{factorized_backdoor2024,
  title={Factorized Trigger Backdoor Attacks in Federated Learning},
  author={Your Name},
  year={2024}
}
```

---

## 📧 支持

遇到问题？
- 查看代码注释
- 运行测试代码（每个模块的 `if __name__ == '__main__'` 部分）
- 使用调试模式运行

---

**祝学习愉快！** 🎉
