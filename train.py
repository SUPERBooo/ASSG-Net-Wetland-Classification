# -------------------- train.py --------------------
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix, cohen_kappa_score, accuracy_score,
    recall_score, precision_score, f1_score
)

from model import FusionModel
from dataloader import get_dataloaders, RemoteSensingDataset
from configs import config


# =========================
# 损失函数：Focal CE（支持label smoothing、类权重、ignore_index）
# =========================
def focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor = None,
    ignore_index: int = 0,
    gamma: float = 1.5,
    label_smoothing: float = 0.05
) -> torch.Tensor:
    """
    logits: [B,C,H,W], targets: [B,H,W]
    使用 F.cross_entropy(reduction='none', label_smoothing=...) 得到每像素CE，
    然后乘以 focal 权重 ((1-pt)^gamma)，再对有效像素平均。
    """
    B, C, H, W = logits.shape

    # 每像素 CE（未聚合）
    ce_map = F.cross_entropy(
        logits, targets,
        weight=weight,
        ignore_index=ignore_index,
        reduction='none',
        label_smoothing=label_smoothing
    )  # [B,H,W]

    # pt：目标类的概率
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)                        # [B,C,H,W]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # [B,H,W]
        pt = torch.clamp(pt, min=1e-6, max=1.0)

    focal_w = (1.0 - pt) ** gamma                               # [B,H,W]

    # 只统计有效像素
    mask = (targets != ignore_index).float()                    # [B,H,W]
    loss = (focal_w * ce_map * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return loss


# =========================
# Tversky Loss（更稳的Dice变体）
# =========================
def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.6,
    beta: float = 0.4,
    eps: float = 1e-6,
    ignore_index: int = 0,
    num_classes: int = 6
) -> torch.Tensor:
    """
    logits: [B,C,H,W], targets: [B,H,W]
    对 ignore 像素不参与计算。
    """
    probs = F.softmax(logits, dim=1)
    B, C, H, W = probs.shape

    mask = (targets != ignore_index).float()                    # [B,H,W]
    targets_clamped = targets.clone()
    targets_clamped[targets == ignore_index] = 0

    onehot = F.one_hot(targets_clamped, num_classes=C).permute(0, 3, 1, 2).float()
    onehot = onehot * mask.unsqueeze(1)
    probs = probs * mask.unsqueeze(1)

    TP = (probs * onehot).sum(dim=(0, 2, 3))
    FP = (probs * (1.0 - onehot)).sum(dim=(0, 2, 3))
    FN = ((1.0 - probs) * onehot).sum(dim=(0, 2, 3))

    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return 1.0 - tversky.mean()


# =========================
# EMA（Exponential Moving Average）辅助
# =========================
def init_ema_state(model: nn.Module) -> dict:
    # 直接用当前权重初始化 shadow
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def ema_update(model: nn.Module, ema_state: dict, decay: float = 0.999):
    msd = model.state_dict()
    for k, v in msd.items():
        if k in ema_state:
            ema_state[k].mul_(decay).add_(v, alpha=1.0 - decay)
        else:
            ema_state[k] = v.detach().clone()

@torch.no_grad()
def swap_to_ema_weights(model: nn.Module, ema_state: dict):
    """
    用 EMA 权重替换当前模型权重，同时返回原始权重备份，便于验证后恢复。
    """
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state, strict=False)
    return backup

@torch.no_grad()
def load_state_dict_safe(model: nn.Module, state: dict):
    model.load_state_dict(state, strict=False)


def train():
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    os.makedirs(config.metrics_dir, exist_ok=True)
    os.makedirs(config.confusion_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 首次运行会计算统计量 & 生成超像素
    stats_path = os.path.join(config.output_dir, "dataset_stats.npz")
    if not os.path.exists(stats_path):
        print("首次运行，正在计算数据统计量并生成超像素...")
        _ = RemoteSensingDataset(config.output_dir)

    # 构建数据集/加载器
    full_dataset = RemoteSensingDataset(config.output_dir)
    train_loader, val_loader = get_dataloaders(config.output_dir, config.batch_size)

    # ======== 类权重：log(1 + 1/freq)，不归一 ========
    class_counts = full_dataset.class_counts
    print("\n原始类别分布:", class_counts)
    valid_classes = list(range(1, config.num_classes))
    if len(class_counts) < config.num_classes:
        raise ValueError(f"数据只包含 {len(class_counts)} 个类别，但配置要求 {config.num_classes} 个")

    valid_counts = class_counts[valid_classes]
    print("有效类别样本数:", valid_counts)

    eps = 1e-8
    inv_freq = 1.0 / (valid_counts.astype(np.float64) + eps)    # 反频率
    log_inv = np.log(1.0 + inv_freq)                            # 压制极端值
    print("log反频率（未归一）:", np.round(log_inv, 6))

    full_weights = np.zeros(config.num_classes, dtype=np.float32)
    full_weights[valid_classes] = log_inv.astype(np.float32)
    print("最终权重分配:", np.round(full_weights, 6))

    weights_tensor = torch.tensor(full_weights, dtype=torch.float32, device=device)

    # ======== 模型 / 优化器 / 调度器 ========
    model = FusionModel().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"使用 {torch.cuda.device_count()} 块GPU")

    # 初始 lr=2e-4 + warmup=10 epoch
    base_lr = 2e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    # 使用 ReduceLROnPlateau 基于验证 OA 调整学习率（warmup 之后照常工作）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )

    # ======== EMA ========
    ema_decay = 0.999
    ema_state = init_ema_state(model)

    # ======== 训练监控 ========
    metrics = []
    best_acc = 0.0
    warmup_epochs = 10
    focal_gamma = 1.5
    tv_alpha, tv_beta = 0.6, 0.4
    tv_weight = 0.2  # Tversky损失的权重
    lbl_smooth = 0.05

    for epoch in range(config.epochs):
        # ---- Warmup：线性提升到 base_lr ----
        if epoch < warmup_epochs:
            lr_now = base_lr * float(epoch + 1) / float(warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now

        # ========== Train ==========
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs} [Train]')

        for radar, optical, labels, spx in train_bar:
            radar = radar.to(device, non_blocking=True)
            optical = optical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spx = spx.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, reg_loss = model(radar, optical, spx)

            # 分离出两部分损失
            ce = focal_ce_loss(
                logits, labels,
                weight=weights_tensor,
                ignore_index=config.ignore_index,
                gamma=focal_gamma,
                label_smoothing=lbl_smooth
            )
            tv = tversky_loss(
                logits, labels,
                alpha=tv_alpha, beta=tv_beta,
                ignore_index=config.ignore_index,
                num_classes=config.num_classes
            )
            seg_loss = 0.8 * ce + tv_weight * tv

            if torch.isnan(seg_loss):
                raise RuntimeError("训练过程中出现NaN损失，请检查数据！")

            total_loss = seg_loss + config.agsm_reg_lambda * reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA 更新
            if isinstance(model, nn.DataParallel):
                ema_update(model.module, ema_state, decay=ema_decay)
            else:
                ema_update(model, ema_state, decay=ema_decay)

            train_loss += float(total_loss.item())
            train_bar.set_postfix(
                loss=float(total_loss.item()),
                seg=float(seg_loss.item()),
                ce=float(ce.item()),
                tv=float(tv.item()),
                reg=float(reg_loss.item())
            )

        # ========== Validation（使用 EMA 权重评估，更稳） ==========
        model.eval()
        # 切换到 EMA 权重
        if isinstance(model, nn.DataParallel):
            backup_state = swap_to_ema_weights(model.module, ema_state)
        else:
            backup_state = swap_to_ema_weights(model, ema_state)

        val_loss = 0.0
        all_preds, all_labels = [], []
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config.epochs} [Val]')
        with torch.no_grad():
            for radar, optical, labels, spx in val_bar:
                radar = radar.to(device, non_blocking=True)
                optical = optical.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                spx = spx.to(device, non_blocking=True)

                logits, _ = model(radar, optical, spx)

                # 与训练相同的损失口径
                ce = focal_ce_loss(
                    logits, labels,
                    weight=weights_tensor,
                    ignore_index=config.ignore_index,
                    gamma=focal_gamma,
                    label_smoothing=lbl_smooth
                )
                tv = tversky_loss(
                    logits, labels,
                    alpha=tv_alpha, beta=tv_beta,
                    ignore_index=config.ignore_index,
                    num_classes=config.num_classes
                )
                loss = 0.8 * ce + tv_weight * tv
                val_loss += float(loss.item())

                # 评估
                _, preds = torch.max(logits.permute(0, 2, 3, 1).reshape(-1, config.num_classes), dim=1)
                mask = labels.reshape(-1) != config.ignore_index
                valid_preds = preds[mask].cpu().numpy()
                valid_l = labels.reshape(-1)[mask].cpu().numpy()
                all_preds.extend(valid_preds)
                all_labels.extend(valid_l)

                val_bar.set_postfix(loss=float(loss.item()))

        # 恢复原训练权重
        if isinstance(model, nn.DataParallel):
            load_state_dict_safe(model.module, backup_state)
        else:
            load_state_dict_safe(model, backup_state)

        # 计算指标
        oa = accuracy_score(all_labels, all_preds)
        valid_classes = list(range(1, config.num_classes))
        aa = recall_score(all_labels, all_preds, average='macro', labels=valid_classes, zero_division=0)
        kappa = cohen_kappa_score(all_labels, all_preds, labels=valid_classes)
        precision_per_class = precision_score(all_labels, all_preds, average=None, labels=np.arange(config.num_classes), zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, labels=np.arange(config.num_classes), zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, labels=np.arange(config.num_classes), zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=np.arange(config.num_classes))
        np.save(os.path.join(config.confusion_dir, f'confusion_matrix_epoch_{epoch + 1}.npy'), conf_matrix)

        epoch_metrics = {
            'epoch': epoch + 1,
            'phase': 'val',
            'loss': val_loss / max(len(val_loader), 1),
            'oa': oa, 'aa': aa, 'kappa': kappa
        }
        for i in range(config.num_classes):
            epoch_metrics[f'precision_{i}'] = precision_per_class[i]
            epoch_metrics[f'recall_{i}'] = recall_per_class[i]
            epoch_metrics[f'f1_{i}'] = f1_per_class[i]
        metrics.append(epoch_metrics)
        pd.DataFrame(metrics).to_csv(os.path.join(config.metrics_dir, 'metrics.csv'), index=False)

        # 基于 OA 的调度器
        scheduler.step(oa)

        # 保存最佳（保存 EMA 权重更实用）
        if oa > best_acc:
            best_acc = oa
            # 用 EMA 权重保存
            if isinstance(model, nn.DataParallel):
                torch.save(ema_state, config.model_save_path)
            else:
                torch.save(ema_state, config.model_save_path)
            print(f"保存最佳模型(EMA)，准确率：{best_acc:.2%}")

        # 打印 epoch 小结
        print(f"[Epoch {epoch+1}] Val Loss={epoch_metrics['loss']:.4f} | OA={oa:.4f} | AA={aa:.4f} | Kappa={kappa:.4f}")


if __name__ == "__main__":
    train()
