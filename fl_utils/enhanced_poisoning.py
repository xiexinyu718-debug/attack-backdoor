"""
Enhanced Poisoning Module  (v4 — full integration)
===================================================
v4 addresses all integration audit findings:

1. [CRITICAL] trigger_factors clamp(0,1) — fixed via separate patch script
2. [HIGH] Rotation: register_participation() + remove double-assign
3. [MEDIUM] Adaptive task separation weight (ASR-based adjustment)
4. [MEDIUM] MR compatibility: auto-disable norm_projection when MR active
5. [MEDIUM] Poison timing: respect poison_start_epoch / poison_stop_epoch
6. [LOW] FSA: reuse first forward pass features (avoid double forward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Feature-Space Alignment
# ---------------------------------------------------------------------------

class FeatureExtractorHook:
    def __init__(self):
        self.features: Optional[torch.Tensor] = None
        self._hook = None

    def hook_fn(self, module, input, output):
        self.features = output

    def register(self, layer: nn.Module):
        self._hook = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def _find_feature_layer(model, layer_name=None):
    if layer_name:
        m = model
        for p in layer_name.split("."):
            m = getattr(m, p)
        return m
    if hasattr(model, "layer4"): return model.layer4
    if hasattr(model, "features"): return model.features
    ch = list(model.children())
    return ch[-2] if len(ch) >= 2 else ch[-1]


class FeatureSpaceAlignment:
    def __init__(self, model, target_class, feature_layer=None,
                 distance="cosine", momentum=0.9, device=torch.device("cuda")):
        self.target_class = target_class
        self.distance = distance
        self.momentum = momentum
        self.device = device
        self._hook = FeatureExtractorHook()
        self._hook.register(_find_feature_layer(model, feature_layer))
        self._centroid: Optional[torch.Tensor] = None

    def update_centroid(self, model, clean_loader, max_batches=10):
        model.eval()
        feat_sum, count = None, 0
        with torch.no_grad():
            for i, (imgs, labs) in enumerate(clean_loader):
                if i >= max_batches: break
                mask = labs == self.target_class
                if mask.sum() == 0: continue
                _ = model(imgs[mask].to(self.device))
                f = self._pool(self._hook.features)
                feat_sum = f.sum(0) if feat_sum is None else feat_sum + f.sum(0)
                count += f.size(0)
        model.train()
        if count == 0: return
        bc = feat_sum / count
        self._centroid = bc if self._centroid is None else (
            self.momentum * self._centroid + (1 - self.momentum) * bc)

    def compute_loss_from_hook(self):
        """Compute alignment loss using ALREADY-captured hook features (no extra forward)."""
        if self._centroid is None or self._hook.features is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        feats = self._pool(self._hook.features)
        c = self._centroid.detach()
        if self.distance == "cosine":
            return (1.0 - F.cosine_similarity(feats, c.unsqueeze(0), dim=1)).mean()
        return torch.norm(feats - c.unsqueeze(0), dim=1).mean()

    def cleanup(self):
        self._hook.remove()

    @staticmethod
    def _pool(f):
        if f.dim() == 4: return f.mean(dim=[2, 3])
        if f.dim() == 3: return f.mean(dim=2)
        return f


# ---------------------------------------------------------------------------
# 2. Layer-Wise Model Poisoning
# ---------------------------------------------------------------------------

class LayerWisePoisoning:
    def __init__(self, model, strategy="auto", top_k_ratio=0.4,
                 critical_scale=1.2, noncritical_scale=0.5,
                 max_l2_ratio=1.5, manual_layers=None):
        self.strategy = strategy
        self.top_k_ratio = top_k_ratio
        self.critical_scale = critical_scale
        self.noncritical_scale = noncritical_scale
        self.max_l2_ratio = max_l2_ratio

        if strategy == "manual" and manual_layers:
            self.critical_names = self._match_manual(model, manual_layers)
        elif strategy == "deep":
            self.critical_names = self._select_deep(model)
        else:
            self.critical_names: set = set()
        self._snapshot: Dict[str, torch.Tensor] = {}

    def analyse_sensitivity(self, model, poisoned_batch, criterion=nn.CrossEntropyLoss()):
        model.train()
        imgs, labs = poisoned_batch
        imgs, labs = imgs.cuda(), labs.cuda()
        model.zero_grad()
        loss = criterion(model(imgs), labs)
        loss.backward()
        sens: Dict[str, list] = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                k = ".".join(n.split(".")[:2])
                sens.setdefault(k, []).append(p.grad.abs().mean().item())
        avg = {k: np.mean(v) for k, v in sens.items()}
        ranked = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        n = max(1, int(len(ranked) * self.top_k_ratio))
        self.critical_names = {name for name, _ in ranked[:n]}
        model.zero_grad()
        print(f"  [LWMP] {len(ranked)} groups, {n} critical")

    def snapshot_params(self, model):
        self._snapshot = {n: p.data.clone() for n, p in model.named_parameters()}

    def get_snapshot(self):
        return self._snapshot

    def mask_and_scale_gradients(self, model):
        if not self.critical_names: return
        for n, p in model.named_parameters():
            if p.grad is None: continue
            k = ".".join(n.split(".")[:2])
            p.grad.data.mul_(self.critical_scale if k in self.critical_names
                             else self.noncritical_scale)

    def clamp_updates(self, model):
        if not self._snapshot: return
        groups: Dict[str, list] = {}
        for n, p in model.named_parameters():
            if n not in self._snapshot: continue
            k = ".".join(n.split(".")[:2])
            groups.setdefault(k, []).append((n, p, p.data - self._snapshot[n]))
        with torch.no_grad():
            for k, entries in groups.items():
                dn = np.sqrt(sum(torch.sum(d**2).item() for _, _, d in entries))
                sn = np.sqrt(sum(torch.sum(self._snapshot[n]**2).item()
                                 for n, _, _ in entries)) + 1e-10
                thr = self.max_l2_ratio * sn * 0.01
                if dn > thr > 0:
                    s = thr / dn
                    for n, p, d in entries:
                        p.data.copy_(self._snapshot[n] + d * s)

    def compute_poison_loss(self, model):
        if not self._snapshot:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            k = ".".join(n.split(".")[:2])
            if k in self.critical_names and n in self._snapshot:
                loss = loss + torch.norm(p - self._snapshot[n].detach()) ** 2
        return loss

    def _match_manual(self, model, patterns):
        m = set()
        for n, _ in model.named_parameters():
            k = ".".join(n.split(".")[:2])
            if any(p in n for p in patterns): m.add(k)
        return m

    def _select_deep(self, model):
        keys = []
        for n, _ in model.named_parameters():
            k = ".".join(n.split(".")[:2])
            if k not in keys: keys.append(k)
        n = max(1, int(len(keys) * self.top_k_ratio))
        return set(keys[-n:])


# ---------------------------------------------------------------------------
# 3. Enhanced Malicious Trainer  (v4)
# ---------------------------------------------------------------------------

class EnhancedMaliciousTrainer:
    """
    v4 integrations:
    - Adaptive task_separation_weight (ASR-driven)
    - MR compatibility flag
    - Norm projection with benign reference
    - FSA reuses forward-pass hook (no double forward)
    - Poison timing awareness
    """

    def __init__(self, config, model,
                 # FSA
                 lambda_align=0.3, lambda_align_start=0.0, fsa_warmup_epochs=80,
                 fsa_distance="cosine", fsa_momentum=0.9, fsa_feature_layer=None,
                 # LWMP
                 lambda_poison=0.0, lwmp_strategy="auto", lwmp_top_k_ratio=0.4,
                 lwmp_critical_scale=1.2, lwmp_noncritical_scale=0.5,
                 lwmp_max_l2_ratio=1.5, lwmp_manual_layers=None,
                 # Norm projection
                 norm_projection=True, target_l2_ratio=1.0,
                 # Adaptive separation
                 adaptive_separation=True, sep_asr_low=0.70, sep_asr_high=0.90,
                 ):
        self.config = config
        self.lambda_align_target = lambda_align
        self.lambda_align_start = lambda_align_start
        self.fsa_warmup_epochs = fsa_warmup_epochs
        self.lambda_align = lambda_align_start
        self.lambda_poison = lambda_poison
        self.norm_projection = norm_projection
        self.target_l2_ratio = target_l2_ratio

        # Adaptive task separation
        self.adaptive_separation = adaptive_separation
        self.sep_asr_low = sep_asr_low
        self.sep_asr_high = sep_asr_high

        # Read initial weight from config
        tc = (config.target_class if hasattr(config, "target_class")
              else config.get("target_class", 0) if isinstance(config, dict)
              else getattr(config, "target_class", 0))
        self.target_class = tc

        self.separation_weight = (
            config.get("task_separation_weight", 0.45)
            if hasattr(config, "get")
            else getattr(config, "task_separation_weight", 0.45))

        self.criterion = nn.CrossEntropyLoss()
        self.fsa = None
        self.lwmp = None

        self._fsa_kw = dict(target_class=tc, feature_layer=fsa_feature_layer,
                            distance=fsa_distance, momentum=fsa_momentum)
        self._lwmp_kw = dict(strategy=lwmp_strategy, top_k_ratio=lwmp_top_k_ratio,
                             critical_scale=lwmp_critical_scale,
                             noncritical_scale=lwmp_noncritical_scale,
                             max_l2_ratio=lwmp_max_l2_ratio,
                             manual_layers=lwmp_manual_layers)

        self._step_count = 0
        self._cum_align = self._cum_poison = self._cum_clean = 0.0
        self._benign_ref_norm: Optional[float] = None
        self._mr_active = False  # set True externally when MR is used

        print(f"\n  [EnhancedTrainer v4] Initialised")
        print(f"    lambda_align       = {lambda_align_start} -> {lambda_align} (warmup {fsa_warmup_epochs}ep)")
        print(f"    lambda_poison      = {lambda_poison}")
        print(f"    LWMP crit/noncrit  = {lwmp_critical_scale}/{lwmp_noncritical_scale}")
        print(f"    LWMP max_l2_ratio  = {lwmp_max_l2_ratio}")
        print(f"    Norm projection    = {norm_projection}, target_l2 = {target_l2_ratio}")
        print(f"    Adaptive sep       = {adaptive_separation}")

    # --- scheduling ---

    def _update_lambda_align(self, epoch):
        if epoch >= self.fsa_warmup_epochs:
            self.lambda_align = self.lambda_align_target
        else:
            p = epoch / max(self.fsa_warmup_epochs, 1)
            self.lambda_align = self.lambda_align_start + \
                (self.lambda_align_target - self.lambda_align_start) * p

    def adjust_separation_weight(self, asr: float, main_acc: float, epoch: int):
        """Adaptive task separation: shift weight based on current ASR."""
        if not self.adaptive_separation:
            return
        old = self.separation_weight
        if asr < self.sep_asr_low:
            self.separation_weight = min(0.7, self.separation_weight + 0.03)
        elif asr > self.sep_asr_high and main_acc < 0.88:
            self.separation_weight = max(0.2, self.separation_weight - 0.03)
        if abs(self.separation_weight - old) > 0.001:
            print(f"  [AdaptSep] w: {old:.3f} -> {self.separation_weight:.3f} "
                  f"(ASR={asr:.2%}, Acc={main_acc:.2%})")

    # --- MR compatibility ---

    def set_mr_active(self, active: bool):
        """Call with True when Model Replacement is enabled for this round."""
        self._mr_active = active
        if active and self.norm_projection:
            print(f"  [NormProj] Auto-disabled (MR active)")

    # --- benign reference ---

    def set_benign_reference_norm(self, norm: float):
        self._benign_ref_norm = norm

    # --- round hooks ---

    def prepare_round(self, model, clean_loader, poisoned_batch=None, epoch=0):
        device = next(model.parameters()).device
        self._update_lambda_align(epoch)

        # FSA
        if self.fsa is not None:
            self.fsa.cleanup()
        self.fsa = FeatureSpaceAlignment(model, device=device, **self._fsa_kw)
        self.fsa.update_centroid(model, clean_loader, max_batches=10)

        # LWMP
        self.lwmp = LayerWisePoisoning(model, **self._lwmp_kw)
        if self._lwmp_kw["strategy"] == "auto" and poisoned_batch is not None:
            self.lwmp.analyse_sensitivity(model, poisoned_batch)
        self.lwmp.snapshot_params(model)

        self._step_count = 0
        self._cum_align = self._cum_poison = self._cum_clean = 0.0

    def finalise_round(self, model):
        """LWMP clamp -> norm projection (unless MR active) -> cleanup."""
        if self.lwmp is not None:
            self.lwmp.clamp_updates(model)

        # Norm projection: skip if MR is active (MR does its own scaling)
        do_proj = (self.norm_projection and not self._mr_active
                   and self.lwmp is not None
                   and self._benign_ref_norm is not None
                   and self._benign_ref_norm > 0)
        if do_proj:
            self._apply_norm_projection(model, self.lwmp.get_snapshot())

        if self.fsa is not None:
            self.fsa.cleanup()
            self.fsa = None

        n = max(self._step_count, 1)
        print(f"  [Trainer] summary: L_clean={self._cum_clean/n:.4f} "
              f"L_align={self._cum_align/n:.4f}(lam={self.lambda_align:.3f}) "
              f"L_poison={self._cum_poison/n:.4f} "
              f"sep_w={self.separation_weight:.3f}")

    def _apply_norm_projection(self, model, snapshot):
        dsq = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in snapshot:
                    dsq += torch.sum((p.data - snapshot[n]) ** 2).item()
        dn = np.sqrt(dsq)
        tn = self.target_l2_ratio * self._benign_ref_norm
        if dn > tn > 0:
            s = tn / dn
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in snapshot:
                        p.data.copy_(snapshot[n] + (p.data - snapshot[n]) * s)
            print(f"  [NormProj] {dn:.4f} -> {tn:.4f} (x{s:.3f})")

    # --- step-level (v4: reuse hook features for FSA) ---

    def train_step(self, model, optimizer, inputs, labels, poison_num):
        model.train()

        # Single forward pass — FSA hook captures features automatically
        outputs = model(inputs)
        bs = outputs.size(0)

        # L_clean (task-separated)
        if 0 < poison_num < bs:
            bl = self.criterion(outputs[:poison_num], labels[:poison_num])
            ml = self.criterion(outputs[poison_num:], labels[poison_num:])
            L_clean = (1 - self.separation_weight) * ml + self.separation_weight * bl
        else:
            L_clean = self.criterion(outputs, labels)

        # L_align — uses hook features from the forward pass above (NO double forward)
        if self.fsa and poison_num > 0 and self.lambda_align > 0:
            # The hook captured features for the FULL batch.
            # We need features for only the poisoned portion.
            # Since the hook stores the full output, we slice it.
            full_feat = self.fsa._hook.features
            if full_feat is not None:
                # Pool and slice to poisoned portion
                pooled = self.fsa._pool(full_feat)
                poison_feats = pooled[:poison_num]
                if self.fsa._centroid is not None:
                    c = self.fsa._centroid.detach()
                    if self.fsa.distance == "cosine":
                        sim = F.cosine_similarity(poison_feats, c.unsqueeze(0), dim=1)
                        L_align = (1.0 - sim).mean()
                    else:
                        L_align = torch.norm(poison_feats - c.unsqueeze(0), dim=1).mean()
                else:
                    L_align = torch.tensor(0.0, device=inputs.device)
            else:
                L_align = torch.tensor(0.0, device=inputs.device)
        else:
            L_align = torch.tensor(0.0, device=inputs.device)

        # L_poison
        L_poison = (self.lwmp.compute_poison_loss(model)
                    if self.lwmp and self.lambda_poison > 0
                    else torch.tensor(0.0, device=inputs.device))

        L_total = L_clean + self.lambda_align * L_align + self.lambda_poison * L_poison

        optimizer.zero_grad()
        L_total.backward()
        if self.lwmp:
            self.lwmp.mask_and_scale_gradients(model)
        optimizer.step()

        _, pred = outputs.max(1)
        acc = 100.0 * pred.eq(labels).sum().item() / bs

        self._step_count += 1
        self._cum_clean += L_clean.item()
        self._cum_align += L_align.item()
        self._cum_poison += L_poison.item()

        return {"total_loss": L_total.item(), "clean_loss": L_clean.item(),
                "align_loss": L_align.item(), "poison_loss": L_poison.item(),
                "accuracy": acc, "poison_num": poison_num, "total_samples": bs}

    # --- epoch-level ---

    def train_epoch(self, model, optimizer, dataloader, attacker, adversary_id, epoch):
        model.train()
        tl = tm = ta = tp = 0.0
        cor = ts = tpois = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            pi, pl, pn = attacker.poison_input_with_task_separation(
                inputs, labels, adversary_id, epoch, eval_mode=False)
            s = self.train_step(model, optimizer, pi, pl, pn)
            tl += s["total_loss"]; tm += s["clean_loss"]
            ta += s["align_loss"]; tp += s["poison_loss"]
            cor += s["accuracy"] * s["total_samples"] / 100.0
            ts += s["total_samples"]; tpois += pn
        nb = max(len(dataloader), 1)
        return {"avg_loss": tl/nb, "avg_main_loss": tm/nb,
                "avg_align_loss": ta/nb, "avg_poison_loss": tp/nb,
                "accuracy": 100.0*cor/ts if ts else 0,
                "total_samples": ts, "poisoned_samples": tpois,
                "poison_ratio": tpois/ts if ts else 0}


# ---------------------------------------------------------------------------
# 4. Factory
# ---------------------------------------------------------------------------

def create_enhanced_trainer(config, model, **ov):
    def _c(k, d):
        if k in ov: return ov[k]
        return config.get(k, d) if isinstance(config, dict) else getattr(config, k, d)

    return EnhancedMaliciousTrainer(
        config, model,
        lambda_align=_c("lambda_align", 0.3),
        lambda_align_start=_c("lambda_align_start", 0.0),
        fsa_warmup_epochs=_c("fsa_warmup_epochs", 80),
        fsa_distance=_c("fsa_distance", "cosine"),
        fsa_momentum=_c("fsa_momentum", 0.9),
        fsa_feature_layer=_c("fsa_feature_layer", None),
        lambda_poison=_c("lambda_poison", 0.0),
        lwmp_strategy=_c("lwmp_strategy", "auto"),
        lwmp_top_k_ratio=_c("lwmp_top_k_ratio", 0.4),
        lwmp_critical_scale=_c("lwmp_critical_scale", 1.2),
        lwmp_noncritical_scale=_c("lwmp_noncritical_scale", 0.5),
        lwmp_max_l2_ratio=_c("lwmp_max_l2_ratio", 1.5),
        lwmp_manual_layers=_c("lwmp_manual_layers", None),
        norm_projection=_c("norm_projection", True),
        target_l2_ratio=_c("target_l2_ratio", 1.0),
        adaptive_separation=_c("adaptive_separation", True),
        sep_asr_low=_c("sep_asr_low", 0.70),
        sep_asr_high=_c("sep_asr_high", 0.90),
    )