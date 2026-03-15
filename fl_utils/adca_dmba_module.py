"""
ADCA + DMBA 重写修复版
=======================
修复原版导致 ASR=0 / loss=NaN 的三个致命 bug：

致命Bug1: softmax 在 ~11M 维参数向量上运行 → 每个值≈1e-7 → 所有参数被归零
致命Bug2: 范数裁剪阈值=100，但 ResNet18 参数向量范数≈3000+ → 模型缩小30倍
致命Bug3: 上述两个 bug 导致 enhanced ≈ 0.05*global → 模型坍塌 → NaN

修复方案:
  - 改为逐层(state_dict)聚合，不再 flatten 整个模型
  - 逐层计算注意力权重(基于与全局模型的偏差)，维度低、数值稳定
  - 增强公式只放大 delta(偏差)，不破坏参数原始量级
  - 正确处理 BatchNorm 的 running_mean/running_var/num_batches_tracked

使用方法：直接替换原来的 adca_dmba_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np


class ADCAModule:
    """
    ADCA: 注意力驱动的恶意客户端协同聚合（重写修复版）

    核心思想:
      1. 在 state_dict 层面逐层操作（不 flatten 整个模型）
      2. 每层计算各恶意客户端的"注意力权重" = 该层偏离全局模型的幅度
         偏离大 → 后门信号强 → 权重高
      3. 加权平均后，对 delta(聚合-全局) 做小幅放大(coalition_alpha)
      4. 完整保留 BatchNorm buffers
    """

    def __init__(self, config):
        self.coalition_alpha = config.get('coalition_alpha', 0.1)
        self.enable_coalition = config.get('enable_malicious_coalition', True)
        # 每层 delta 的最大放大倍数上限（安全阀）
        self.max_amplification = config.get('coalition_max_amplification', 3.0)
        print(f"[ADCA] 初始化完成 (alpha={self.coalition_alpha})")

    def malicious_coalition_aggregation(self, adversary_models, global_model):
        """
        恶意客户端协同聚合

        参数:
            adversary_models: 恶意客户端模型列表
            global_model: 全局模型

        返回:
            enhanced_state_dict | None
        """
        if not self.enable_coalition or len(adversary_models) < 2:
            return None

        try:
            num_adv = len(adversary_models)
            global_sd = global_model.state_dict()
            adv_sds = [m.state_dict() for m in adversary_models]

            enhanced_sd = {}

            for key in global_sd:
                global_val = global_sd[key].float()

                # ---- 非浮点参数(如 num_batches_tracked)：直接取第一个 ----
                if global_val.dtype not in (torch.float32, torch.float16, torch.bfloat16,
                                            torch.float64):
                    enhanced_sd[key] = adv_sds[0][key].clone()
                    continue

                # ---- BatchNorm 的 running_mean / running_var：简单平均即可 ----
                if 'running_mean' in key or 'running_var' in key:
                    avg = torch.stack([sd[key].float() for sd in adv_sds]).mean(dim=0)
                    enhanced_sd[key] = avg.to(global_sd[key].dtype)
                    continue

                # ---- 可训练参数：注意力加权聚合 ----
                adv_vals = torch.stack([sd[key].float() for sd in adv_sds])  # [N, ...]

                # 1) 计算每个恶意客户端在该层的 delta = adv - global
                deltas = adv_vals - global_val.unsqueeze(0)  # [N, ...]

                # 2) 注意力权重 = 各客户端 delta 的 L2 范数
                #    偏离全局越多 → 后门信号越强 → 权重越高
                delta_norms = torch.tensor([
                    d.norm().item() for d in deltas
                ], device=global_val.device)

                if delta_norms.sum() < 1e-12:
                    # 所有客户端和全局一样，无需增强
                    enhanced_sd[key] = global_val.to(global_sd[key].dtype)
                    continue

                # softmax 只在 num_adversaries(通常4个) 维度上 → 数值稳定
                weights = F.softmax(delta_norms, dim=0)  # [N]

                # 3) 加权平均 delta
                #    reshape weights 以广播: [N, 1, 1, ...] 匹配参数维度
                w_shape = [-1] + [1] * (deltas.dim() - 1)
                weighted_delta = (deltas * weights.view(w_shape)).sum(dim=0)

                # 4) 增强: 在加权平均的基础上，放大 delta
                #    enhanced = global + (1 + alpha) * weighted_delta
                amplification = 1.0 + self.coalition_alpha

                # 安全阀: 限制 delta 的范数不超过原始平均范数的 max_amplification 倍
                avg_delta_norm = delta_norms.mean().item()
                enhanced_delta = amplification * weighted_delta
                enhanced_delta_norm = enhanced_delta.norm().item()

                if avg_delta_norm > 1e-12 and enhanced_delta_norm > self.max_amplification * avg_delta_norm:
                    scale = self.max_amplification * avg_delta_norm / (enhanced_delta_norm + 1e-12)
                    enhanced_delta = enhanced_delta * scale

                enhanced_val = global_val + enhanced_delta

                # 最终 NaN/Inf 检查
                if torch.isnan(enhanced_val).any() or torch.isinf(enhanced_val).any():
                    print(f"[ADCA] 警告: {key} 出现 NaN/Inf，回退到简单平均")
                    enhanced_val = adv_vals.mean(dim=0)

                enhanced_sd[key] = enhanced_val.to(global_sd[key].dtype)

            print(f"[ADCA] 协同聚合完成 (alpha={self.coalition_alpha}, "
                  f"adversaries={num_adv})")
            return enhanced_sd

        except Exception as e:
            print(f"[ADCA] 协同聚合失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class DMBAModule:
    """
    DMBA: 分布式多目标后门攻击与重放机制
    """

    def __init__(self, config):
        self.replay_buffer_size = config.get('replay_buffer_size', 150)
        self.replay_ratio = config.get('replay_ratio', 0.5)
        self.importance_threshold = config.get('importance_threshold', 0.4)

        self.replay_buffers = {}
        self.importance_scores = {}

        print(f"[DMBA] 初始化完成 (buffer_size={self.replay_buffer_size}, "
              f"ratio={self.replay_ratio})")

    def initialize_buffer(self, adversary_id):
        """初始化重放缓冲区"""
        if adversary_id not in self.replay_buffers:
            self.replay_buffers[adversary_id] = deque(maxlen=self.replay_buffer_size)
            self.importance_scores[adversary_id] = []

    def add_to_buffer(self, adversary_id, backdoor_images, backdoor_labels, model=None):
        """添加后门样本到缓冲区"""
        self.initialize_buffer(adversary_id)

        if model is not None:
            importance = self._compute_importance(model, backdoor_images, backdoor_labels)
        else:
            importance = torch.ones(len(backdoor_images))

        for i in range(len(backdoor_images)):
            sample = {
                'image': backdoor_images[i].detach().cpu(),
                'label': backdoor_labels[i].detach().cpu(),
                'importance': importance[i].item()
            }
            self.replay_buffers[adversary_id].append(sample)

    def sample_replay(self, adversary_id, batch_size):
        """从缓冲区采样"""
        self.initialize_buffer(adversary_id)
        buffer = self.replay_buffers[adversary_id]

        if len(buffer) == 0:
            return None, None

        sample_size = min(batch_size, len(buffer))

        # 重要性采样
        if len(buffer) > sample_size:
            importance_scores = [s['importance'] for s in buffer]
            total_importance = sum(importance_scores)

            if total_importance > 0:
                probabilities = [s / total_importance for s in importance_scores]
                # 防止浮点误差导致概率和不为1
                prob_arr = np.array(probabilities, dtype=np.float64)
                prob_arr = prob_arr / prob_arr.sum()
                indices = np.random.choice(
                    len(buffer),
                    size=sample_size,
                    replace=False,
                    p=prob_arr
                )
            else:
                indices = random.sample(range(len(buffer)), sample_size)

            samples = [buffer[i] for i in indices]
        else:
            samples = list(buffer)

        images = torch.stack([s['image'] for s in samples])
        labels = torch.stack([s['label'] for s in samples])

        return images, labels

    def mix_with_replay(self, adversary_id, current_images, current_labels, device='cuda'):
        """混合当前批次与重放样本"""
        batch_size = len(current_images)
        replay_size = int(batch_size * self.replay_ratio)

        replay_images, replay_labels = self.sample_replay(adversary_id, replay_size)

        if replay_images is None:
            return current_images, current_labels

        replay_images = replay_images.to(device)
        replay_labels = replay_labels.to(device)

        mixed_images = torch.cat([current_images, replay_images], dim=0)
        mixed_labels = torch.cat([current_labels, replay_labels], dim=0)

        # 随机打乱
        indices = torch.randperm(len(mixed_images), device=device)
        mixed_images = mixed_images[indices]
        mixed_labels = mixed_labels[indices]

        return mixed_images, mixed_labels

    def _compute_importance(self, model, images, labels):
        """计算样本重要性（损失越大越需要重放）"""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                dev = next(model.parameters()).device
                imgs = images.to(dev) if images.device != dev else images
                labs = labels.to(dev) if labels.device != dev else labels
                outputs = model(imgs)
                loss = F.cross_entropy(outputs, labs, reduction='none')
            return loss.detach().cpu()
        except Exception:
            return torch.ones(len(images))
        finally:
            if was_training:
                model.train()

    def get_buffer_size(self, adversary_id):
        """获取缓冲区大小"""
        if adversary_id in self.replay_buffers:
            return len(self.replay_buffers[adversary_id])
        return 0


class ADCADMBAIntegration:
    """
    ADCA + DMBA 集成类（重写修复版）
    """

    def __init__(self, config):
        self.adca = ADCAModule(config)
        self.dmba = DMBAModule(config)
        self.config = config

        print(f"\n{'='*70}")
        print(f"ADCA + DMBA 模块初始化完成（重写修复版）")
        print(f"  ADCA coalition_alpha = {self.adca.coalition_alpha}")
        print(f"  DMBA buffer_size = {self.dmba.replay_buffer_size}, "
              f"ratio = {self.dmba.replay_ratio}")
        print(f"{'='*70}\n")

    def apply_coalition(self, adversary_models, global_model):
        """应用ADCA协同聚合"""
        return self.adca.malicious_coalition_aggregation(
            adversary_models, global_model
        )

    def add_backdoor_to_buffer(self, adversary_id, images, labels, model=None):
        """添加后门样本到DMBA缓冲区"""
        self.dmba.add_to_buffer(adversary_id, images, labels, model)

    def get_mixed_batch(self, adversary_id, images, labels, device='cuda'):
        """获取混合了重放样本的批次"""
        return self.dmba.mix_with_replay(adversary_id, images, labels, device)

    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'adca_enabled': self.adca.enable_coalition,
            'coalition_alpha': self.adca.coalition_alpha,
            'replay_buffer_size': self.dmba.replay_buffer_size,
            'replay_ratio': self.dmba.replay_ratio,
            'buffer_sizes': {}
        }
        for adv_id in self.dmba.replay_buffers:
            stats['buffer_sizes'][adv_id] = self.dmba.get_buffer_size(adv_id)
        return stats


# ======================== 自测 ========================
if __name__ == "__main__":
    print("=" * 70)
    print("ADCA + DMBA 重写修复版 — 自测")
    print("=" * 70)

    config = {
        'coalition_alpha': 0.1,
        'enable_malicious_coalition': True,
        'replay_buffer_size': 150,
        'replay_ratio': 0.5,
        'importance_threshold': 0.4,
    }

    integration = ADCADMBAIntegration(config)

    # ---- 测试 ADCA ----
    print("\n[测试 ADCA]")
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = F.relu(self.bn(self.conv(x)))
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)

    global_model = TinyModel()
    adversaries = [TinyModel() for _ in range(4)]

    # 模拟恶意训练：给每个adversary加随机扰动
    for m in adversaries:
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.1)

    result = integration.apply_coalition(adversaries, global_model)
    if result is not None:
        # 验证 state_dict 完整性
        missing = set(global_model.state_dict().keys()) - set(result.keys())
        extra = set(result.keys()) - set(global_model.state_dict().keys())
        assert len(missing) == 0, f"缺少 keys: {missing}"
        assert len(extra) == 0, f"多余 keys: {extra}"

        # 验证参数量级没有被破坏
        for key in result:
            orig_norm = global_model.state_dict()[key].float().norm().item()
            new_norm = result[key].float().norm().item()
            if orig_norm > 1e-6:
                ratio = new_norm / orig_norm
                assert 0.1 < ratio < 20.0, \
                    f"{key}: 范数比={ratio:.2f}，参数量级被破坏!"

        # 验证没有 NaN
        for key in result:
            assert not torch.isnan(result[key]).any(), f"{key} 包含 NaN!"

        print("  ✅ ADCA 测试通过: state_dict 完整、无NaN、参数量级正常")
    else:
        print("  ⚠️ ADCA 返回 None（不应该在此测试中发生）")

    # ---- 测试 DMBA ----
    print("\n[测试 DMBA]")
    fake_images = torch.randn(8, 3, 32, 32)
    fake_labels = torch.randint(0, 10, (8,))

    integration.add_backdoor_to_buffer(0, fake_images, fake_labels)
    assert integration.dmba.get_buffer_size(0) == 8

    mixed_imgs, mixed_labs = integration.get_mixed_batch(
        0, fake_images, fake_labels, device='cpu'
    )
    assert len(mixed_imgs) >= len(fake_images)
    assert not torch.isnan(mixed_imgs).any()
    print(f"  ✅ DMBA 测试通过: buffer={integration.dmba.get_buffer_size(0)}, "
          f"mixed_batch={len(mixed_imgs)}")

    print("\n" + "=" * 70)
    print("全部测试通过!")
    print("=" * 70)