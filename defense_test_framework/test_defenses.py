"""
test_defenses.py  (v4-compatible)
=================================
Tests the enhanced factorized attack (FSA + LWMP) against all defense mechanisms.
Uses the EXACT same attack pipeline as train_enhanced_attack.py v4.

Usage:
    # From project root:
    python defense_test_framework/test_defenses.py --params configs/defense_test.yaml --test_all --gpu 0

    # Or from inside defense_test_framework/:
    cd defense_test_framework
    python test_defenses.py --params ../configs/defense_test.yaml --test_all --gpu 0

    # Single defense:
    python defense_test_framework/test_defenses.py --params configs/defense_test.yaml --defense krum --gpu 0

    # Quick test (fewer epochs):
    python defense_test_framework/test_defenses.py --params configs/defense_test.yaml --test_all --epochs 30
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
import matplotlib
matplotlib.use('Agg')

# Handle import paths: works whether run from project root or from defense_test_framework/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir) if os.path.basename(_this_dir) == 'defense_test_framework' else _this_dir
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import argparse, yaml, torch, torch.nn as nn
import numpy as np, random, copy, json
from datetime import datetime

from helper import Helper
from fl_utils.factorized_attacker import FactorizedAttacker
from fl_utils.evaluation import FactorizedAttackEvaluator
from fl_utils.enhanced_poisoning import create_enhanced_trainer
from fl_defenses import create_defense


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cfg(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# -----------------------------------------------------------------------
# Training functions — identical to train_enhanced_attack.py v4
# -----------------------------------------------------------------------

def train_benign_client(helper, pid, model, epoch):
    lr = helper.get_lr(epoch)
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=getattr(helper.config, "momentum", 0.9),
                          weight_decay=getattr(helper.config, "decay", 5e-4))
    model.train()
    tl = cor = ts = 0
    for _ in range(getattr(helper.config, "retrain_times", 2)):
        for x, y in helper.train_data[pid]:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad(); o = model(x)
            loss = nn.functional.cross_entropy(o, y)
            loss.backward(); opt.step()
            tl += loss.item(); cor += o.max(1)[1].eq(y).sum().item(); ts += x.size(0)
    nb = len(helper.train_data[pid]) * getattr(helper.config, "retrain_times", 2)
    return model, {"loss": tl/max(nb,1), "accuracy": 100.*cor/max(ts,1), "total_samples": ts}


def _compute_l2(model, ref_model):
    s = 0.0
    with torch.no_grad():
        for (_, p1), (_, p2) in zip(model.named_parameters(), ref_model.named_parameters()):
            s += torch.sum((p1.data - p2.data)**2).item()
    return np.sqrt(s)


def _get_poisoned_batch(helper, attacker, adv_id, epoch):
    for x, y in helper.test_data:
        x, y = x.cuda(), y.cuda()
        px, py, _ = attacker.poison_input_with_task_separation(
            x, y, adv_id, epoch, eval_mode=True)
        return (px, py)
    return None


def sample_participants_fixed(helper, epoch):
    n_samp = helper.num_sampled_participants
    n_mal = min(_cfg(helper.config, "num_malicious_per_round",
                     len(helper.adversary_list)),
                len(helper.adversary_list), n_samp)
    sel_mal = random.sample(helper.adversary_list, n_mal)
    benign_pool = [i for i in range(helper.num_total_participants)
                   if i not in helper.adversary_list]
    sel_ben = random.sample(benign_pool, min(n_samp - n_mal, len(benign_pool)))
    out = sel_mal + sel_ben
    random.shuffle(out)
    return out


def train_malicious_client(helper, pid, model, epoch, attacker, enh):
    adv_id = helper.adversary_list.index(pid)

    # Rotation — compatible with both old and new attacker
    if hasattr(attacker, 'register_participation'):
        attacker.register_participation(adv_id, epoch)
    else:
        if epoch % attacker.rotation_frequency == 0:
            attacker.assign_factor_combination(adv_id, epoch)

    pb = _get_poisoned_batch(helper, attacker, adv_id, epoch)
    enh.prepare_round(model, helper.train_data[pid], poisoned_batch=pb, epoch=epoch)

    lr = helper.get_lr(epoch)
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=getattr(helper.config, "momentum", 0.9),
                          weight_decay=getattr(helper.config, "decay", 5e-4))

    retrain = getattr(helper.config, "attacker_retrain_times", 3)
    all_s = []
    for _ in range(retrain):
        all_s.append(enh.train_epoch(model, opt, helper.train_data[pid],
                                     attacker, adv_id, epoch))

    enh.finalise_round(model)
    return model, {
        "avg_loss": np.mean([s["avg_loss"] for s in all_s]),
        "accuracy": all_s[-1]["accuracy"],
        "poisoned_samples": sum(s["poisoned_samples"] for s in all_s),
        "total_samples": all_s[-1]["total_samples"],
    }


# -----------------------------------------------------------------------
# Backward-compatible client_models dict
# -----------------------------------------------------------------------

class ClientModelsCompat(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ordered = []

    def set_with_order(self, pid, model):
        self[pid] = model
        self._ordered.append(pid)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, int) and 0 <= key < len(self._ordered):
                return super().__getitem__(self._ordered[key])
            raise


# -----------------------------------------------------------------------
# Core: test one defense
# -----------------------------------------------------------------------

def test_with_defense(helper, attacker, enh_trainer, defense, num_epochs):
    """
    Run FL with a specific defense aggregation for num_epochs.
    Returns evaluation history and final metrics.
    """
    print(f"\n{'='*70}")
    print(f"Testing defense: {defense.name}  ({num_epochs} epochs)")
    print(f"{'='*70}")

    evaluator = FactorizedAttackEvaluator(helper, attacker)
    eval_history = []

    for epoch in range(num_epochs):
        sampled = sample_participants_fixed(helper, epoch)
        mal_ids = [p for p in sampled if p in helper.adversary_list]
        ben_ids = [p for p in sampled if p not in helper.adversary_list]

        if epoch % 10 == 0:
            print(f"\n  Epoch {epoch}/{num_epochs}  (mal={len(mal_ids)}, ben={len(ben_ids)})")

        helper.client_models = ClientModelsCompat()
        local_models = {}

        # Phase 1: benign
        benign_l2s = []
        for pid in ben_ids:
            lm = copy.deepcopy(helper.global_model)
            lm, _ = train_benign_client(helper, pid, lm, epoch)
            local_models[pid] = lm
            helper.client_models.set_with_order(pid, lm)
            benign_l2s.append(_compute_l2(lm, helper.global_model))

        avg_ben_l2 = np.mean(benign_l2s) if benign_l2s else 0.0
        enh_trainer.set_benign_reference_norm(avg_ben_l2)

        # Phase 2: malicious
        for pid in mal_ids:
            lm = copy.deepcopy(helper.global_model)
            lm, _ = train_malicious_client(helper, pid, lm, epoch, attacker, enh_trainer)
            local_models[pid] = lm
            helper.client_models.set_with_order(pid, lm)

        # ★ Use defense aggregation instead of plain FedAvg
        aggregated_state = defense.aggregate(
            helper.global_model, local_models, list(local_models.keys())
        )
        helper.global_model.load_state_dict(aggregated_state)

        # Evaluate periodically
        eval_freq = max(num_epochs // 10, 5)
        if epoch % eval_freq == 0 or epoch == num_epochs - 1:
            results = evaluator.comprehensive_evaluation(helper.global_model, epoch)
            results['epoch'] = epoch
            eval_history.append(results)

            if epoch % 10 == 0:
                print(f"    Acc={results['main_accuracy']:.2f}%  ASR={results['average_asr']:.2f}%")

    final = eval_history[-1] if eval_history else {}
    return {
        'defense_name': defense.name,
        'final_main_accuracy': final.get('main_accuracy', 0),
        'final_asr': final.get('average_asr', 0),
        'final_individual_asr': final.get('individual_asr', {}),
        'final_stealthiness': final.get('stealthiness', {}),
        'evaluation_history': eval_history,
    }


# -----------------------------------------------------------------------
# Test all defenses
# -----------------------------------------------------------------------

def test_all_defenses(helper, attacker, enh_trainer, num_epochs, config):
    """Test all defense mechanisms, resetting model for each."""

    defense_configs = [
        {'name': 'fedavg',        'params': {}},
        {'name': 'krum',          'params': {'krum_num_selected': 5}},
        {'name': 'trimmed_mean',  'params': {'trimmed_mean_beta': 0.1}},
        {'name': 'median',        'params': {}},
        {'name': 'norm_clipping', 'params': {'clip_threshold': 10.0}},
        {'name': 'weak_dp',       'params': {'dp_noise_scale': 0.001}},
        {'name': 'foolsgold',     'params': {}},
    ]

    all_results = {}
    # Save initial model state to reset between defenses
    initial_state = copy.deepcopy(helper.global_model.state_dict())

    for dc in defense_configs:
        print(f"\n{'#'*70}")
        print(f"# Defense: {dc['name'].upper()}")
        print(f"{'#'*70}")

        # Reset model to same starting point
        helper.global_model.load_state_dict(copy.deepcopy(initial_state))

        # Merge defense-specific params into config
        for k, v in dc['params'].items():
            if isinstance(helper.config, dict):
                helper.config[k] = v
            else:
                setattr(helper.config, k, v)

        defense = create_defense(dc['name'], helper.config)
        results = test_with_defense(helper, attacker, enh_trainer, defense, num_epochs)
        all_results[dc['name']] = results

    return all_results


# -----------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------

def print_comparison_table(all_results):
    baseline_asr = all_results.get('fedavg', {}).get('final_asr', 0)
    baseline_acc = all_results.get('fedavg', {}).get('final_main_accuracy', 0)

    print(f"\n{'='*80}")
    print(f"DEFENSE COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Defense':<20} {'Main Acc':>10} {'ASR':>10} {'ASR Drop':>10} {'Drop %':>10} {'Effect':>10}")
    print(f"{'-'*80}")

    for name in ['fedavg', 'krum', 'trimmed_mean', 'median',
                  'norm_clipping', 'weak_dp', 'foolsgold']:
        if name not in all_results:
            continue
        r = all_results[name]
        acc = r['final_main_accuracy']
        asr = r['final_asr']
        drop = baseline_asr - asr
        drop_pct = (drop / baseline_asr * 100) if baseline_asr > 0 else 0

        if drop_pct > 50:     eff = "STRONG"
        elif drop_pct > 30:   eff = "MODERATE"
        elif drop_pct > 15:   eff = "WEAK"
        else:                 eff = "NONE"

        print(f"{r['defense_name']:<20} {acc:>10.2f} {asr:>10.2f} {drop:>10.2f} {drop_pct:>9.1f}% {eff:>10}")

    print(f"{'='*80}")


def generate_report(all_results, config, save_path):
    baseline_asr = all_results.get('fedavg', {}).get('final_asr', 0)

    report = {
        'test_date': datetime.now().isoformat(),
        'version': 'v4-enhanced',
        'attack_config': {
            'k_of_m': f"{config.get('k_of_m_k', 2)}-of-{config.get('k_of_m_m', 3)}",
            'num_adversaries': config.get('num_adversaries', 4),
            'num_malicious_per_round': config.get('num_malicious_per_round', 2),
            'bkd_ratio': config.get('bkd_ratio', 0.20),
            'lambda_align': config.get('lambda_align', 0.3),
            'lwmp_strategy': config.get('lwmp_strategy', 'auto'),
            'norm_projection': config.get('norm_projection', True),
        },
        'results': {},
        'summary': {},
    }

    for name, r in all_results.items():
        asr = r['final_asr']
        report['results'][name] = {
            'defense_name': r['defense_name'],
            'main_accuracy': r['final_main_accuracy'],
            'asr': asr,
            'individual_asr': r.get('final_individual_asr', {}),
            'stealthiness': r.get('final_stealthiness', {}),
        }
        report['summary'][name] = {
            'main_accuracy': r['final_main_accuracy'],
            'asr': asr,
            'asr_drop': baseline_asr - asr,
            'asr_drop_percentage': ((baseline_asr - asr) / baseline_asr * 100) if baseline_asr > 0 else 0,
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved: {save_path}")
    return report


def visualize_results(all_results, save_dir='./defense_visualizations'):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    names, asrs, accs = [], [], []
    for name in ['fedavg','krum','trimmed_mean','median','norm_clipping','weak_dp','foolsgold']:
        if name not in all_results:
            continue
        r = all_results[name]
        names.append(r['defense_name'])
        asrs.append(r['final_asr'])
        accs.append(r['final_main_accuracy'])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, asrs, w, label='ASR', color='indianred', alpha=0.7)
    b2 = ax.bar(x + w/2, accs, w, label='Main Accuracy', color='steelblue', alpha=0.7)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Defense', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Enhanced Attack (v4) vs Defense Mechanisms', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/defense_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/defense_comparison.png")
    plt.close()

    # Training curves per defense
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    idx = 0
    for name in ['fedavg','krum','trimmed_mean','median','norm_clipping','weak_dp','foolsgold']:
        if name not in all_results or idx >= 7:
            continue
        r = all_results[name]
        hist = r.get('evaluation_history', [])
        if not hist:
            idx += 1
            continue

        eps = [h['epoch'] for h in hist]
        ma = [h['main_accuracy'] for h in hist]
        asr = [h['average_asr'] for h in hist]

        ax = axes[idx]
        ax.plot(eps, ma, 'b-o', label='Acc', markersize=3, linewidth=1.5)
        ax.plot(eps, asr, 'r-o', label='ASR', markersize=3, linewidth=1.5)
        ax.set_title(r['defense_name'], fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('%')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        idx += 1

    for i in range(idx, 8):
        axes[i].set_visible(False)

    plt.suptitle('Training under each defense', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/defense_curves.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/defense_curves.png")
    plt.close()


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Test enhanced v4 attack against defenses')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--defense', type=str, default=None,
                        help='Single defense: fedavg/krum/trimmed_mean/median/norm_clipping/weak_dp/foolsgold')
    parser.add_argument('--test_all', action='store_true')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count (default: from config)')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"GPU: {args.gpu}")

    with open(args.params, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    num_epochs = args.epochs if args.epochs else config.get('epochs', 80)
    seed = args.seed if args.seed else config.get('seed', 42)
    set_seed(seed)

    # Init
    print(f"\nInitialising system...")
    helper = Helper(config)
    helper.load_data()
    helper.load_model()
    helper.config_adversaries()

    attacker = FactorizedAttacker(helper)
    enh_trainer = create_enhanced_trainer(config, helper.global_model)

    n_mal = config.get('num_malicious_per_round',
                       min(config.get('num_adversaries', 4),
                           config.get('num_sampled_participants', 10)))

    print(f"\n  Attack: Enhanced v4 (FSA + LWMP)")
    print(f"  Mal/round: {n_mal}/{config.get('num_sampled_participants', 10)}")
    print(f"  Test epochs: {num_epochs}")

    if args.test_all:
        all_results = test_all_defenses(helper, attacker, enh_trainer, num_epochs, config)
        print_comparison_table(all_results)
        report_path = config.get('results_dir', '.') + '/defense_test_report.json'
        os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
        generate_report(all_results, config, report_path)
        visualize_results(all_results,
                          save_dir=config.get('results_dir', './defense_visualizations'))

    elif args.defense:
        helper.load_model()  # fresh model
        defense = create_defense(args.defense, helper.config)
        results = test_with_defense(helper, attacker, enh_trainer, defense, num_epochs)

        print(f"\n{'='*70}")
        print(f"Defense: {results['defense_name']}")
        print(f"Main accuracy: {results['final_main_accuracy']:.2f}%")
        print(f"ASR: {results['final_asr']:.2f}%")
        print(f"{'='*70}")
    else:
        print("Error: specify --defense <name> or --test_all")

    print(f"\nDone!")


if __name__ == '__main__':
    main()