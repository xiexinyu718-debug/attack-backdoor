"""
可视化工具模块
提供各种可视化功能：因子效果、轮换模式、强度调度等
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch


class FactorizedAttackVisualizer:
    """
    因子化攻击可视化器
    """
    
    def __init__(self, save_dir='./visualizations'):
        """
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    def visualize_factor_combination(self, attacker, sample_input, adversary_id, 
                                    epoch, save_name=None):
        """
        可视化因子组合效果
        展示：原始图像 -> 逐个应用因子 -> 最终结果
        
        Args:
            attacker: FactorizedAttacker实例
            sample_input: 样本输入 [1, 3, 32, 32]
            adversary_id: 攻击者ID
            epoch: 当前轮次
            save_name: 保存文件名（可选）
        """
        # 获取因子组合
        if adversary_id not in attacker.active_combinations:
            attacker.assign_factor_combination(adversary_id, epoch)
        
        combination = attacker.active_combinations[adversary_id]
        
        # 获取强度倍数
        intensity_multiplier = attacker.intensity_schedule.get(epoch, 0.3)
        
        # 准备显示
        num_factors = len(combination)
        fig, axes = plt.subplots(1, num_factors + 2, figsize=(4*(num_factors + 2), 4))
        
        # 1. 显示原始图像
        original = sample_input[0].cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(np.clip(original, 0, 1))
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. 逐个应用和显示因子
        current = sample_input.clone()
        for i, factor_idx in enumerate(combination):
            factor = attacker.factor_library[factor_idx]
            
            # 临时调整强度
            original_intensity = factor.intensity
            factor.intensity = original_intensity * intensity_multiplier
            
            # 应用因子
            current = factor.apply(current)
            
            # 恢复强度
            factor.intensity = original_intensity
            
            # 显示
            img = current[0].cpu().numpy().transpose(1, 2, 0)
            axes[i+1].imshow(np.clip(img, 0, 1))
            axes[i+1].set_title(f'+ {factor.name}\n(I={factor.intensity*intensity_multiplier:.3f})', 
                               fontsize=10)
            axes[i+1].axis('off')
        
        # 3. 显示最终结果
        final = current[0].cpu().numpy().transpose(1, 2, 0)
        axes[-1].imshow(np.clip(final, 0, 1))
        axes[-1].set_title('Final Result', fontsize=12, fontweight='bold')
        axes[-1].axis('off')
        
        # 设置总标题
        factor_names = [attacker.factor_library[i].name for i in combination]
        plt.suptitle(f'Adversary {adversary_id} - Epoch {epoch}\n'
                    f'Factors: {", ".join(factor_names)}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        if save_name is None:
            save_name = f'factors_adv{adversary_id}_epoch{epoch}.png'
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存因子组合可视化: {save_path}")
        plt.close()
    
    def visualize_rotation_pattern(self, attacker, epoch, save_name=None):
        """
        可视化轮换模式
        展示每个攻击者的因子组合随时间的变化
        
        Args:
            attacker: FactorizedAttacker实例
            epoch: 当前轮次
            save_name: 保存文件名（可选）
        """
        if len(attacker.rotation_history) == 0:
            print("  没有轮换历史可以可视化")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 为每个攻击者绘制轮换时间线
        colors = plt.cm.tab10(np.linspace(0, 1, len(attacker.rotation_history)))
        
        for idx, (adv_id, history) in enumerate(attacker.rotation_history.items()):
            if len(history) == 0:
                continue
            
            epochs = [h['epoch'] for h in history]
            
            # 绘制时间线
            ax.plot(epochs, [adv_id] * len(epochs), 
                   'o-', color=colors[idx], 
                   label=f'Adversary {adv_id}', 
                   markersize=8, linewidth=2)
            
            # 标注因子组合
            for h in history:
                e = h['epoch']
                combo = h['combination']
                combo_str = ','.join([str(c) for c in combo])
                
                # 只在关键点显示标注（避免过于拥挤）
                if e % 5 == 0 or e == history[-1]['epoch']:
                    ax.annotate(combo_str, (e, adv_id), 
                              fontsize=8, ha='center', va='bottom',
                              bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor=colors[idx], alpha=0.3))
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Adversary ID', fontsize=12, fontweight='bold')
        ax.set_title('Factor Combination Rotation Pattern Over Time', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_name is None:
            save_name = f'rotation_pattern_epoch{epoch}.png'
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存轮换模式可视化: {save_path}")
        plt.close()
    
    def visualize_intensity_schedule(self, attacker, current_epoch=None, save_name=None):
        """
        可视化动态强度调度
        
        Args:
            attacker: FactorizedAttacker实例
            current_epoch: 当前轮次（可选，用于标记）
            save_name: 保存文件名（可选）
        """
        epochs = sorted(attacker.intensity_schedule.keys())
        intensities = [attacker.intensity_schedule[e] for e in epochs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制强度曲线
        ax.plot(epochs, intensities, 'b-', linewidth=3, label='Intensity Schedule')
        
        # 标记当前轮次
        if current_epoch is not None and current_epoch in attacker.intensity_schedule:
            current_intensity = attacker.intensity_schedule[current_epoch]
            ax.axvline(x=current_epoch, color='r', linestyle='--', 
                      linewidth=2, label=f'Current Epoch: {current_epoch}')
            ax.plot(current_epoch, current_intensity, 'ro', markersize=12)
            ax.annotate(f'I={current_intensity:.3f}', 
                       (current_epoch, current_intensity),
                       fontsize=12, fontweight='bold',
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # 标注关键阶段
        total_epochs = len(epochs)
        ax.axvspan(0, total_epochs*0.2, alpha=0.2, color='green', label='Early (Low Intensity)')
        ax.axvspan(total_epochs*0.2, total_epochs*0.6, alpha=0.2, color='yellow', label='Mid (Increasing)')
        ax.axvspan(total_epochs*0.6, total_epochs, alpha=0.2, color='red', label='Late (High Intensity)')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trigger Intensity', fontsize=12, fontweight='bold')
        ax.set_title('Dynamic Trigger Intensity Schedule', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_name is None:
            save_name = 'intensity_schedule.png'
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存强度调度可视化: {save_path}")
        plt.close()
    
    def visualize_evaluation_metrics(self, evaluation_history, save_name=None):
        """
        可视化评估指标的变化
        
        Args:
            evaluation_history: 评估历史列表
            save_name: 保存文件名（可选）
        """
        if len(evaluation_history) == 0:
            print("  没有评估历史可以可视化")
            return
        
        # 提取数据
        epochs = [e.get('epoch', i) for i, e in enumerate(evaluation_history)]
        main_accs = [e.get('main_accuracy', 0) for e in evaluation_history]
        asrs = [e.get('average_asr', 0) for e in evaluation_history]
        diversities = [e.get('factor_diversity', 0) for e in evaluation_history]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 主任务准确率
        axes[0, 0].plot(epochs, main_accs, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Main Task Accuracy', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=88, color='r', linestyle='--', label='Threshold (88%)')
        axes[0, 0].legend()
        
        # 2. 攻击成功率
        axes[0, 1].plot(epochs, asrs, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ASR (%)')
        axes[0, 1].set_title('Attack Success Rate', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=85, color='g', linestyle='--', label='Target (85%)')
        axes[0, 1].legend()
        
        # 3. 因子多样性
        axes[1, 0].plot(epochs, diversities, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Diversity Score')
        axes[1, 0].set_title('Factor Diversity', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # 4. 综合对比
        ax2 = axes[1, 1].twinx()
        l1 = axes[1, 1].plot(epochs, main_accs, 'b-o', label='Main Acc', linewidth=2)
        l2 = ax2.plot(epochs, asrs, 'r-o', label='ASR', linewidth=2)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Main Accuracy (%)', color='b')
        ax2.set_ylabel('ASR (%)', color='r')
        axes[1, 1].set_title('Main Task vs Backdoor Task', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 合并图例
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        axes[1, 1].legend(lns, labs, loc='best')
        
        plt.suptitle('Factorized Attack Performance Metrics', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        if save_name is None:
            save_name = 'evaluation_metrics.png'
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存评估指标可视化: {save_path}")
        plt.close()
    
    def visualize_k_of_m_comparison(self, k_of_m_results, save_name=None):
        """
        可视化不同k-of-m配置的对比
        
        Args:
            k_of_m_results: {config_name: asr} 字典
            save_name: 保存文件名（可选）
        """
        configs = list(k_of_m_results.keys())
        asrs = list(k_of_m_results.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 创建条形图
        bars = ax.bar(range(len(configs)), asrs, color='steelblue', alpha=0.7)
        
        # 标注数值
        for i, (bar, asr) in enumerate(zip(bars, asrs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{asr:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('k-of-m Configuration Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存
        if save_name is None:
            save_name = 'k_of_m_comparison.png'
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存k-of-m对比可视化: {save_path}")
        plt.close()


def visualize_complete_report(attacker, evaluator, evaluation_history, 
                              sample_input, epoch, save_dir='./visualizations'):
    """
    生成完整的可视化报告
    
    Args:
        attacker: FactorizedAttacker实例
        evaluator: FactorizedAttackEvaluator实例
        evaluation_history: 评估历史
        sample_input: 样本输入
        epoch: 当前轮次
        save_dir: 保存目录
    """
    print(f"\n生成完整可视化报告...")
    
    visualizer = FactorizedAttackVisualizer(save_dir)
    
    # 1. 因子组合效果（前3个攻击者）
    print(f"\n1. 生成因子组合可视化...")
    num_adversaries = min(3, len(attacker.active_combinations))
    for adv_id in range(num_adversaries):
        visualizer.visualize_factor_combination(
            attacker, sample_input, adv_id, epoch
        )
    
    # 2. 轮换模式
    print(f"\n2. 生成轮换模式可视化...")
    visualizer.visualize_rotation_pattern(attacker, epoch)
    
    # 3. 强度调度
    print(f"\n3. 生成强度调度可视化...")
    visualizer.visualize_intensity_schedule(attacker, epoch)
    
    # 4. 评估指标
    if len(evaluation_history) > 0:
        print(f"\n4. 生成评估指标可视化...")
        visualizer.visualize_evaluation_metrics(evaluation_history)
    
    print(f"\n✓ 可视化报告已保存到: {save_dir}\n")


if __name__ == '__main__':
    print("测试可视化模块\n")
    
    # 创建测试数据
    print("生成测试可视化...")
    visualizer = FactorizedAttackVisualizer('./test_visualizations')
    
    # 测试强度调度可视化
    class MockAttacker:
        intensity_schedule = {i: 0.1 + 0.4 * (1 - np.exp(-3 * i/100)) 
                             for i in range(100)}
    
    mock_attacker = MockAttacker()
    visualizer.visualize_intensity_schedule(mock_attacker, current_epoch=50)
    
    print("\n测试完成！检查 ./test_visualizations 目录")
