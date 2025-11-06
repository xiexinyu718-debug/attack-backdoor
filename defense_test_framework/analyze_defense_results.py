"""
防御测试结果分析脚本
自动分析和可视化防御测试结果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_report(report_path='defense_test_report.json'):
    """加载测试报告"""
    if not os.path.exists(report_path):
        print(f"错误: 找不到报告文件 {report_path}")
        print("请先运行: python test_defenses.py --params configs/defense_test.yaml --test_all")
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary(report):
    """打印测试摘要"""
    print("\n" + "="*70)
    print("防御测试结果摘要")
    print("="*70)
    print(f"测试时间: {report['test_date']}")
    print()
    
    # 基准ASR（无防御）
    baseline_asr = report['summary'].get('fedavg', {}).get('asr', 96)
    
    # 打印表格
    print(f"{'防御机制':<20} {'主任务准确率':<12} {'ASR':<12} {'ASR下降':<12} {'效果':<12}")
    print("-" * 70)
    
    for defense in ['fedavg', 'krum', 'trimmed_mean', 'median', 
                   'norm_clipping', 'weak_dp', 'foolsgold']:
        if defense not in report['summary']:
            continue
        
        summary = report['summary'][defense]
        main_acc = summary['main_accuracy']
        asr = summary['asr']
        asr_drop = summary['asr_drop']
        asr_drop_pct = summary['asr_drop_percentage']
        
        # 评估效果
        if asr_drop_pct > 50:
            effect = "🛡️🛡️🛡️ 强"
        elif asr_drop_pct > 30:
            effect = "🛡️🛡️ 中等"
        elif asr_drop_pct > 15:
            effect = "🛡️ 弱"
        else:
            effect = "❌ 无效"
        
        print(f"{defense:<20} {main_acc:<12.2f} {asr:<12.2f} "
              f"{asr_drop:.2f} ({asr_drop_pct:.1f}%) {effect}")
    
    print("="*70)


def analyze_defense_ranking(report):
    """分析防御效果排名"""
    print("\n" + "="*70)
    print("防御效果排名（按ASR下降排序）")
    print("="*70)
    
    ranked = sorted(report['summary'].items(), 
                   key=lambda x: x[1]['asr_drop'], 
                   reverse=True)
    
    for i, (defense, summary) in enumerate(ranked, 1):
        if defense == 'fedavg':
            continue  # 跳过基准
        
        asr = summary['asr']
        asr_drop = summary['asr_drop']
        asr_drop_pct = summary['asr_drop_percentage']
        
        print(f"{i}. {defense:20} | ASR: {asr:5.1f}% | "
              f"下降: {asr_drop:5.1f}% ({asr_drop_pct:5.1f}%)")
    
    print("="*70)


def analyze_attack_robustness(report):
    """分析攻击鲁棒性"""
    print("\n" + "="*70)
    print("攻击鲁棒性分析")
    print("="*70)
    
    # 找出最强防御
    best_defense = min(report['summary'].items(), 
                      key=lambda x: x[1]['asr'] if x[0] != 'fedavg' else float('inf'))
    
    defense_name = best_defense[0]
    min_asr = best_defense[1]['asr']
    
    print(f"\n最强防御: {defense_name}")
    print(f"在最强防御下的ASR: {min_asr:.2f}%")
    
    # 评级
    if min_asr > 80:
        rating = "⭐⭐⭐⭐⭐ 极强"
        desc = "几乎不受防御影响"
    elif min_asr > 60:
        rating = "⭐⭐⭐⭐ 强"
        desc = "在强防御下仍有效"
    elif min_asr > 40:
        rating = "⭐⭐⭐ 中等"
        desc = "部分防御有效"
    elif min_asr > 20:
        rating = "⭐⭐ 弱"
        desc = "多数防御能显著降低ASR"
    else:
        rating = "⭐ 很弱"
        desc = "容易被防御"
    
    print(f"攻击鲁棒性评级: {rating}")
    print(f"评价: {desc}")
    
    # 对抗各类防御的表现
    print("\n对抗不同类型防御的表现:")
    
    strong_defenses = ['krum', 'foolsgold']
    medium_defenses = ['trimmed_mean', 'median']
    weak_defenses = ['norm_clipping', 'weak_dp']
    
    for category, defenses in [('强防御', strong_defenses), 
                               ('中等防御', medium_defenses),
                               ('弱防御', weak_defenses)]:
        asrs = [report['summary'][d]['asr'] for d in defenses if d in report['summary']]
        if asrs:
            avg_asr = np.mean(asrs)
            print(f"  {category:8} 平均ASR: {avg_asr:5.1f}%")
    
    print("="*70)


def analyze_accuracy_cost(report):
    """分析防御的准确率代价"""
    print("\n" + "="*70)
    print("防御的准确率代价分析")
    print("="*70)
    
    baseline_acc = report['summary'].get('fedavg', {}).get('main_accuracy', 86)
    
    print(f"基准准确率（无防御）: {baseline_acc:.2f}%\n")
    
    for defense in ['krum', 'trimmed_mean', 'median', 
                   'norm_clipping', 'weak_dp', 'foolsgold']:
        if defense not in report['summary']:
            continue
        
        acc = report['summary'][defense]['main_accuracy']
        acc_drop = baseline_acc - acc
        asr_drop = report['summary'][defense]['asr_drop']
        
        # 计算效率：ASR下降 / 准确率损失
        if abs(acc_drop) < 0.1:
            efficiency = "∞"
        else:
            efficiency = f"{asr_drop / max(abs(acc_drop), 0.1):.2f}"
        
        status = "✓" if acc_drop < 2 else "⚠️" if acc_drop < 5 else "✗"
        
        print(f"{defense:20} | 准确率: {acc:.2f}% | "
              f"下降: {acc_drop:+5.2f}% | 效率: {efficiency:>6} | {status}")
    
    print("\n说明:")
    print("  ✓  准确率下降 < 2% : 代价很小")
    print("  ⚠️  准确率下降 2-5% : 代价适中")
    print("  ✗  准确率下降 > 5% : 代价较大")
    print("  效率 = ASR下降 / 准确率损失（越高越好）")
    
    print("="*70)


def generate_improvement_suggestions(report):
    """生成改进建议"""
    print("\n" + "="*70)
    print("攻击改进建议")
    print("="*70)
    
    # 找出效果最好的防御
    best_defenses = sorted(report['summary'].items(), 
                          key=lambda x: x[1]['asr_drop'], 
                          reverse=True)[:3]
    
    print("\n最有效的3个防御:")
    for i, (defense, summary) in enumerate(best_defenses, 1):
        if defense == 'fedavg':
            continue
        print(f"  {i}. {defense} (ASR下降 {summary['asr_drop']:.1f}%)")
    
    print("\n针对性改进建议:")
    
    # 针对Krum
    if 'krum' in dict(best_defenses).keys():
        krum_asr = report['summary']['krum']['asr']
        if krum_asr < 60:
            print("\n🎯 针对Krum防御:")
            print("   - Krum选择距离最近的模型，您的攻击被部分检测")
            print("   - 建议: 降低初始触发器强度（initial_intensity: 0.08）")
            print("   - 建议: 增加良性训练轮数（retrain_times: 3）")
            print("   - 建议: 使用更渐进的强度调度")
    
    # 针对FoolsGold
    if 'foolsgold' in dict(best_defenses).keys():
        fg_asr = report['summary']['foolsgold']['asr']
        if fg_asr < 60:
            print("\n🎯 针对FoolsGold防御:")
            print("   - FoolsGold检测梯度相似度，攻击者可能被识别")
            print("   - 建议: 使用更多样化的轮换策略（rotation_strategy: 'diverse'）")
            print("   - 建议: 增加轮换频率（rotation_frequency: 3）")
            print("   - 建议: 为不同攻击者分配完全不同的因子集")
    
    # 针对Trimmed Mean
    if 'trimmed_mean' in dict(best_defenses).keys():
        tm_asr = report['summary']['trimmed_mean']['asr']
        if tm_asr < 70:
            print("\n🎯 针对Trimmed Mean防御:")
            print("   - Trimmed Mean修剪极端更新")
            print("   - 建议: 降低任务分离权重（task_separation_weight: 0.25）")
            print("   - 建议: 使用更温和的后门强度")
            print("   - 建议: 增加投毒比例但降低单样本强度")
    
    # 通用建议
    print("\n💡 通用改进方向:")
    avg_strong_defense_asr = np.mean([
        report['summary'][d]['asr'] 
        for d in ['krum', 'foolsgold', 'trimmed_mean', 'median']
        if d in report['summary']
    ])
    
    if avg_strong_defense_asr < 60:
        print("   - 您的攻击在强防御下效果下降明显")
        print("   - 优先改进: 增加隐蔽性和多样性")
        print("   - 可尝试: 对抗训练（在防御存在下训练攻击）")
    elif avg_strong_defense_asr < 80:
        print("   - 您的攻击已具有一定鲁棒性")
        print("   - 可进一步优化: 微调参数提升效果")
        print("   - 可探索: 自适应攻击策略")
    else:
        print("   - 🎉 恭喜！您的攻击具有很强的鲁棒性！")
        print("   - 建议: 测试更强的防御组合")
        print("   - 建议: 在更严格的设置下评估（更多良性客户端）")
    
    print("="*70)


def plot_defense_radar_chart(report, save_path='defense_radar.png'):
    """绘制防御效果雷达图"""
    print("\n生成防御效果雷达图...")
    
    # 选择要展示的防御
    defenses = ['krum', 'trimmed_mean', 'median', 'norm_clipping', 'weak_dp', 'foolsgold']
    
    # 提取数据
    asr_drops = []
    acc_drops = []
    
    baseline_acc = report['summary']['fedavg']['main_accuracy']
    
    for defense in defenses:
        if defense in report['summary']:
            asr_drops.append(report['summary'][defense]['asr_drop_percentage'])
            acc_drop = baseline_acc - report['summary'][defense]['main_accuracy']
            acc_drops.append(max(0, acc_drop))  # 负值设为0
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(defenses), endpoint=False).tolist()
    asr_drops += asr_drops[:1]  # 闭合
    angles += angles[:1]
    
    # 绘制
    ax.plot(angles, asr_drops, 'o-', linewidth=2, label='ASR下降%', color='red')
    ax.fill(angles, asr_drops, alpha=0.25, color='red')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace('_', ' ').title() for d in defenses], size=10)
    
    # 设置y轴
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.title('Defense Effectiveness Radar Chart\n(ASR Reduction Percentage)', 
             size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 雷达图已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("防御测试结果分析")
    print("="*70)
    
    # 加载报告
    report = load_report()
    if report is None:
        return
    
    # 各种分析
    print_summary(report)
    analyze_defense_ranking(report)
    analyze_attack_robustness(report)
    analyze_accuracy_cost(report)
    generate_improvement_suggestions(report)
    
    # 生成额外可视化
    plot_defense_radar_chart(report)
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)
    print("\n下一步:")
    print("1. 根据改进建议调整攻击参数")
    print("2. 在configs/defense_test.yaml中修改配置")
    print("3. 重新测试: python test_defenses.py --params configs/defense_test.yaml --test_all")
    print("="*70)


if __name__ == '__main__':
    main()
