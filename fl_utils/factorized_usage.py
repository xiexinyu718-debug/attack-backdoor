"""
因子化触发器攻击使用示例
展示如何在联邦学习中使用因子化触发器、k-of-m规则、动态优化等功能
"""

import torch
import copy
import numpy as np
from fl_utils.factorized_attacker import FactorizedTriggerAttacker


def train_malicious_with_factorized_attack(helper, participant_id, model, 
                                           epoch, attacker):
    """
    使用因子化触发器训练恶意客户端
    
    核心特性:
    1. 触发器因子化
    2. k-of-m组合规则
    3. 动态优化
    4. 任务分离
    5. 跨轮次轮换
    """
    print(f'\n{"="*70}')
    print(f'Training Malicious Client {participant_id} at Epoch {epoch}')
    print(f'{"="*70}')
    
    # 获取攻击者在全局列表中的索引
    adversary_index = helper.adversary_list.index(participant_id)
    
    # 步骤1: 分配或更新因子组合（跨轮次轮换）
    if epoch % attacker.config.get('rotation_frequency', 1) == 0:
        print(f'\n[Rotation] Assigning new factor combination for adversary {adversary_index}')
        factor_combination = attacker.assign_factor_combination(adversary_index, epoch)
        
        # 打印当前使用的因子
        factor_names = [attacker.factor_library[i].name for i in factor_combination]
        print(f'  Active factors: {factor_names}')
        print(f'  k-of-m rule: {attacker.k}-of-{attacker.m}')
    
    # 步骤2: 获取动态强度
    current_intensity = attacker.intensity_schedule.get(epoch, 0.3)
    print(f'\n[Dynamic Optimization] Current intensity: {current_intensity:.3f}')
    
    # 步骤3: 使用任务分离策略训练
    lr = helper.get_lr(epoch)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=helper.config.momentum,
        weight_decay=helper.config.decay
    )
    
    print(f'\n[Training] Starting local training with task separation')
    results = attacker.train_with_dynamic_optimization(
        model=model,
        dataloader=helper.train_data[participant_id],
        adversary_id=adversary_index,
        epoch=epoch,
        optimizer=optimizer
    )
    
    # 打印训练结果
    print(f'\n[Results]')
    print(f'  Loss: {results["loss"]:.4f}')
    print(f'  Accuracy: {results["accuracy"]:.2f}%')
    print(f'  Poisoned samples: {results["poisoned_samples"]}/{results["total_samples"]}')
    
    # 步骤4: 评估本地模型的攻击成功率
    if epoch % helper.config.get('eval_freq', 10) == 0:
        print(f'\n[Evaluation] Testing backdoor effectiveness')
        local_asr = attacker.evaluate_attack_success(
            model=model,
            test_loader=helper.test_data,
            adversary_id=adversary_index,
            epoch=epoch
        )
        print(f'  Local ASR: {local_asr:.2f}%')
    
    return model, results


def evaluate_factorized_attack(helper, model, attacker, epoch):
    """
    全面评估因子化攻击的效果
    """
    print(f'\n{"="*70}')
    print(f'Comprehensive Evaluation at Epoch {epoch}')
    print(f'{"="*70}')
    
    results = {}
    
    # 1. 主任务准确率
    print(f'\n[1/6] Evaluating main task accuracy...')
    main_acc = test_main_task(helper, model)
    results['main_accuracy'] = main_acc
    print(f'  Main Task Accuracy: {main_acc:.2f}%')
    
    # 2. 全局攻击成功率（所有攻击者的平均）
    print(f'\n[2/6] Evaluating global attack success rate...')
    global_asrs = []
    for adv_id in range(helper.config.num_adversaries):
        asr = attacker.evaluate_attack_success(
            model, helper.test_data, adv_id, epoch
        )
        global_asrs.append(asr)
        print(f'  Adversary {adv_id}: ASR = {asr:.2f}%')
    
    results['global_asr'] = np.mean(global_asrs)
    results['individual_asrs'] = global_asrs
    print(f'  Average ASR: {results["global_asr"]:.2f}%')
    
    # 3. 因子多样性分析
    print(f'\n[3/6] Analyzing factor diversity...')
    diversity_score = analyze_factor_diversity(attacker)
    results['factor_diversity'] = diversity_score
    print(f'  Diversity Score: {diversity_score:.4f}')
    
    # 4. 隐蔽性评估
    print(f'\n[4/6] Evaluating stealthiness...')
    if hasattr(helper, 'global_model'):
        stealth_scores = []
        for adv_id in range(helper.config.num_adversaries):
            if adv_id in attacker.active_combinations:
                local_model = helper.client_models[helper.adversary_list[adv_id]]
                stealth = calculate_stealthiness(local_model, helper.global_model)
                stealth_scores.append(stealth)
        results['stealthiness'] = np.mean(stealth_scores) if stealth_scores else 0
        print(f'  Average Stealth Score: {results["stealthiness"]:.4f}')
    
    # 5. k-of-m规则有效性测试
    print(f'\n[5/6] Testing k-of-m rule effectiveness...')
    k_of_m_results = test_k_of_m_variations(helper, model, attacker, epoch)
    results['k_of_m_analysis'] = k_of_m_results
    for config, asr in k_of_m_results.items():
        print(f'  {config}: ASR = {asr:.2f}%')
    
    # 6. 轮换策略效果
    print(f'\n[6/6] Analyzing rotation strategy effectiveness...')
    rotation_effectiveness = analyze_rotation_effectiveness(attacker)
    results['rotation_effectiveness'] = rotation_effectiveness
    print(f'  Rotation Effectiveness: {rotation_effectiveness:.4f}')
    
    # 汇总报告
    print(f'\n{"="*70}')
    print(f'Evaluation Summary:')
    print(f'  Main Accuracy: {results["main_accuracy"]:.2f}%')
    print(f'  Global ASR: {results["global_asr"]:.2f}%')
    print(f'  Factor Diversity: {results["factor_diversity"]:.4f}')
    print(f'  Stealthiness: {results.get("stealthiness", "N/A")}')
    print(f'{"="*70}')
    
    return results


def test_main_task(helper, model):
    """测试主任务准确率（良性样本）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in helper.test_data:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    model.train()
    return 100.0 * correct / total if total > 0 else 0


def analyze_factor_diversity(attacker):
    """
    分析因子使用的多样性
    多样性越高，攻击越难被检测
    """
    if len(attacker.active_combinations) == 0:
        return 0.0
    
    # 统计每个因子的使用频率
    factor_counts = {}
    total_uses = 0
    
    for combination in attacker.active_combinations.values():
        for factor_idx in combination:
            factor_counts[factor_idx] = factor_counts.get(factor_idx, 0) + 1
            total_uses += 1
    
    # 计算Shannon熵作为多样性指标
    if total_uses == 0:
        return 0.0
    
    entropy = 0.0
    for count in factor_counts.values():
        p = count / total_uses
        if p > 0:
            entropy -= p * np.log2(p)
    
    # 归一化到[0, 1]
    max_entropy = np.log2(len(attacker.factor_library))
    diversity = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return diversity


def calculate_stealthiness(local_model, global_model):
    """
    计算模型更新的隐蔽性
    使用L2范数衡量，范数越小越隐蔽
    """
    total_norm = 0.0
    
    for (name1, param1), (name2, param2) in zip(
        local_model.named_parameters(),
        global_model.named_parameters()
    ):
        if name1 == name2:
            diff = param1.data - param2.data
            total_norm += torch.sum(diff ** 2).item()
    
    return np.sqrt(total_norm)


def test_k_of_m_variations(helper, model, attacker, epoch):
    """
    测试不同k-of-m配置的效果
    """
    results = {}
    original_k = attacker.k
    original_m = attacker.m
    
    # 测试不同的k-of-m组合
    test_configs = [
        (1, 3, 'Single Factor'),
        (2, 3, 'Dual Factors (Recommended)'),
        (3, 3, 'All Factors'),
    ]
    
    for k, m, desc in test_configs:
        attacker.k = k
        attacker.m = m
        
        # 评估ASR
        asrs = []
        for adv_id in range(min(3, helper.config.num_adversaries)):  # 测试前3个攻击者
            asr = attacker.evaluate_attack_success(
                model, helper.test_data, adv_id, epoch
            )
            asrs.append(asr)
        
        avg_asr = np.mean(asrs) if asrs else 0
        results[f'k={k},m={m} ({desc})'] = avg_asr
    
    # 恢复原始配置
    attacker.k = original_k
    attacker.m = original_m
    
    return results


def analyze_rotation_effectiveness(attacker):
    """
    分析轮换策略的有效性
    通过检查历史组合的多样性来评估
    """
    if len(attacker.rotation_history) == 0:
        return 0.0
    
    # 计算每个攻击者的组合变化率
    change_rates = []
    
    for adv_id, history in attacker.rotation_history.items():
        if len(history) < 2:
            continue
        
        changes = 0
        for i in range(1, len(history)):
            # 计算相邻两次组合的差异
            prev = set(history[i-1])
            curr = set(history[i])
            if prev != curr:
                changes += 1
        
        change_rate = changes / (len(history) - 1) if len(history) > 1 else 0
        change_rates.append(change_rate)
    
    # 返回平均变化率
    return np.mean(change_rates) if change_rates else 0.0


def visualize_attack_process(helper, attacker, epoch, save_dir='./visualizations'):
    """
    可视化攻击过程
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f'\n[Visualization] Generating visualizations...')
    
    # 1. 可视化每个攻击者的因子组合效果
    for adv_id in range(min(3, helper.config.num_adversaries)):
        # 获取一个测试样本
        test_iter = iter(helper.test_data)
        sample_batch, _ = next(test_iter)
        sample = sample_batch[:1].cuda()
        
        # 可视化因子组合
        save_path = os.path.join(save_dir, f'adversary_{adv_id}_epoch_{epoch}.png')
        attacker.visualize_factor_combination(
            sample_input=sample,
            adversary_id=adv_id,
            epoch=epoch,
            save_path=save_path
        )
    
    # 2. 可视化轮换模式
    visualize_rotation_pattern(attacker, epoch, save_dir)
    
    # 3. 可视化动态强度变化
    visualize_intensity_schedule(attacker, epoch, save_dir)
    
    print(f'[Visualization] Saved to {save_dir}')


def visualize_rotation_pattern(attacker, epoch, save_dir):
    """可视化轮换模式"""
    import matplotlib.pyplot as plt
    
    if len(attacker.rotation_history) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for adv_id, history in attacker.rotation_history.items():
        if len(history) == 0:
            continue
        
        # 将每个组合转换为字符串标识
        labels = ['-'.join(map(str, combo)) for combo in history]
        epochs = list(range(len(labels)))
        
        ax.plot(epochs, [adv_id] * len(epochs), 'o-', label=f'Adversary {adv_id}')
        
        # 标注组合
        for i, label in enumerate(labels):
            ax.annotate(label, (i, adv_id), fontsize=8, ha='center')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Adversary ID')
    ax.set_title('Factor Combination Rotation Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'rotation_pattern_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_intensity_schedule(attacker, epoch, save_dir):
    """可视化动态强度调度"""
    import matplotlib.pyplot as plt
    
    epochs = sorted(attacker.intensity_schedule.keys())
    intensities = [attacker.intensity_schedule[e] for e in epochs]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, intensities, 'b-', linewidth=2)
    plt.axvline(x=epoch, color='r', linestyle='--', label=f'Current Epoch: {epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Trigger Intensity')
    plt.title('Dynamic Trigger Intensity Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'intensity_schedule_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_with_traditional_attacks(helper):
    """
    与传统攻击方法对比
    """
    print(f'\n{"="*70}')
    print('Comparing Factorized Attack with Traditional Methods')
    print(f'{"="*70}')
    
    # 定义对比方法
    methods = {
        'BadNets (Fixed Trigger)': {
            'k_of_m_k': 1,
            'k_of_m_m': 1,
            'rotation_strategy': None,
            'dynamic_optimization': False
        },
        'Distributed Fixed Patterns': {
            'k_of_m_k': 1,
            'k_of_m_m': 1,
            'rotation_strategy': 'sequential',
            'dynamic_optimization': False
        },
        'Adaptive Trigger (A3FL)': {
            'k_of_m_k': 1,
            'k_of_m_m': 1,
            'rotation_strategy': None,
            'dynamic_optimization': True
        },
        'Factorized (Ours)': {
            'k_of_m_k': 2,
            'k_of_m_m': 3,
            'rotation_strategy': 'adversary_specific',
            'dynamic_optimization': True
        }
    }
    
    results = {}
    
    for method_name, config in methods.items():
        print(f'\nTesting {method_name}...')
        # 这里应该运行完整的训练和评估
        # 简化示例
        results[method_name] = {
            'ASR': 0.0,
            'Main Acc': 0.0,
            'Stealth': 0.0
        }
    
    # 打印对比表格
    print(f'\n{"="*70}')
    print('Comparison Results:')
    print(f'{"Method":<30} {"ASR":<10} {"Main Acc":<12} {"Stealth":<10}')
    print(f'{"-"*70}')
    for method, res in results.items():
        print(f'{method:<30} {res["ASR"]:<10.2f} {res["Main Acc"]:<12.2f} {res["Stealth"]:<10.4f}')
    print(f'{"="*70}')
    
    return results


# ============= 主训练循环 =============

def main_training_loop(helper):
    """
    主训练循环，集成因子化触发器攻击
    """
    print(f'\n{"="*70}')
    print('Federated Learning with Factorized Trigger Attack')
    print(f'{"="*70}')
    
    # 初始化攻击器
    attacker = FactorizedTriggerAttacker(helper)
    
    # 打印攻击配置
    stats = attacker.get_factor_statistics()
    print(f'\nAttack Configuration:')
    print(f'  Factor Library Size: {stats["factor_library_size"]}')
    print(f'  k-of-m Rule: {stats["k_of_m"]}')
    print(f'  Active Adversaries: {stats["active_adversaries"]}')
    print(f'  Rotation Strategy: {stats["rotation_strategy"]}')
    
    # 训练循环
    evaluation_results = []
    
    for epoch in range(helper.config.epochs):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch+1}/{helper.config.epochs}')
        print(f'{"="*70}')
        
        # 1. 采样参与者
        sampled_participants = helper.sample_participants(epoch)
        print(f'Sampled participants: {sampled_participants}')
        
        # 2. 本地训练
        local_models = {}
        local_results = {}
        
        for participant_id in sampled_participants:
            local_model = copy.deepcopy(helper.global_model)
            
            if participant_id in helper.adversary_list:
                # 恶意训练（使用因子化攻击）
                local_model, results = train_malicious_with_factorized_attack(
                    helper, participant_id, local_model, epoch, attacker
                )
                local_results[participant_id] = results
            else:
                # 良性训练
                # local_model = train_benign(helper, participant_id, local_model, epoch)
                pass
            
            local_models[participant_id] = local_model
        
        # 3. 聚合
        # helper.aggregate_models(local_models)
        
        # 4. 评估
        if epoch % helper.config.get('eval_freq', 10) == 0:
            results = evaluate_factorized_attack(helper, helper.global_model, attacker, epoch)
            evaluation_results.append(results)
            
            # 可视化
            if helper.config.get('visualization', {}).get('enabled', False):
                visualize_attack_process(helper, attacker, epoch)
        
        # 5. 自适应调整（如果启用）
        if helper.config.get('dynamic_optimization', {}).get('adaptive_adjustment', {}).get('enabled', False):
            adaptive_adjustment(helper, attacker, evaluation_results, epoch)
    
    # 最终评估
    print(f'\n{"="*70}')
    print('Final Evaluation')
    print(f'{"="*70}')
    final_results = evaluate_factorized_attack(
        helper, helper.global_model, attacker, helper.config.epochs
    )
    
    # 生成报告
    generate_final_report(helper, attacker, evaluation_results, final_results)
    
    return final_results


def adaptive_adjustment(helper, attacker, evaluation_results, epoch):
    """
    根据评估结果自适应调整攻击参数
    """
    if len(evaluation_results) == 0:
        return
    
    latest_results = evaluation_results[-1]
    
    # 根据ASR调整
    asr_threshold = helper.config.get('dynamic_optimization', {}).get('adaptive_adjustment', {}).get('asr_threshold', 0.85)
    if latest_results['global_asr'] < asr_threshold * 100:
        print(f'\n[Adaptive] ASR too low, increasing trigger intensity')
        # 增加触发器强度
        for e in attacker.intensity_schedule:
            if e >= epoch:
                attacker.intensity_schedule[e] = min(1.0, attacker.intensity_schedule[e] * 1.1)
    
    # 根据主任务准确率调整
    acc_threshold = helper.config.get('dynamic_optimization', {}).get('adaptive_adjustment', {}).get('accuracy_threshold', 0.88)
    if latest_results['main_accuracy'] < acc_threshold * 100:
        print(f'\n[Adaptive] Main accuracy too low, decreasing trigger intensity')
        # 降低触发器强度
        for e in attacker.intensity_schedule:
            if e >= epoch:
                attacker.intensity_schedule[e] = max(0.1, attacker.intensity_schedule[e] * 0.9)


def generate_final_report(helper, attacker, evaluation_results, final_results):
    """
    生成最终报告
    """
    import json
    import os
    
    report = {
        'configuration': {
            'k_of_m': f'{attacker.k}-of-{attacker.m}',
            'num_adversaries': helper.config.num_adversaries,
            'rotation_strategy': attacker.rotation_strategy,
            'num_factors': len(attacker.factor_library)
        },
        'final_results': final_results,
        'training_history': evaluation_results,
        'factor_statistics': attacker.get_factor_statistics()
    }
    
    # 保存JSON报告
    report_dir = helper.config.get('results_dir', './results')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'factorized_attack_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'\n[Report] Final report saved to {report_path}')
    
    # 打印摘要
    print(f'\n{"="*70}')
    print('FINAL RESULTS SUMMARY')
    print(f'{"="*70}')
    print(f'Main Task Accuracy: {final_results["main_accuracy"]:.2f}%')
    print(f'Global Attack Success Rate: {final_results["global_asr"]:.2f}%')
    print(f'Factor Diversity: {final_results["factor_diversity"]:.4f}')
    print(f'k-of-m Configuration: {attacker.k}-of-{attacker.m}')
    print(f'{"="*70}')


if __name__ == '__main__':
    print(__doc__)
    print("\nThis is a usage example for the Factorized Trigger Attacker.")
    print("\nKey Features:")
    print("1. Trigger Factorization - Multiple sub-factors with minimal individual impact")
    print("2. k-of-m Combination Rule - Attack triggers only when k factors present")
    print("3. Dynamic Optimization - Intensity increases over training")
    print("4. Task Separation - Backdoor and main task completely separated")
    print("5. Cross-round Rotation - Different factor combinations each round")
