from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from configs import config
from dataloader import get_dataloaders
from model import FusionModel, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    labels = np.asarray(config.valid_classes, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    union = support + pred_count - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)

    oa = accuracy_score(y_true, y_pred)
    aa = float(recall.mean())
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    macro_f1 = float(f1.mean())
    miou = float(iou.mean())

    return {
        "oa": float(oa),
        "aa": aa,
        "kappa": float(kappa),
        "macro_f1": macro_f1,
        "miou": miou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "confusion_matrix": cm,
    }


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> Tuple[float, float, Dict[str, object]]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_cls = 0.0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        bar = tqdm(loader, desc=desc)
        for radar, optical, labels, superpixels, _ in bar:
            radar = radar.to(device, non_blocking=True)
            optical = optical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            superpixels = superpixels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits, sparse_loss = model(radar, optical, superpixels)
            cls_loss = criterion(logits, labels)
            loss = cls_loss + config.sparsity_lambda * sparse_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item())
            total_cls += float(cls_loss.item())

            pred = logits.argmax(dim=1)
            mask = labels != config.ignore_index
            all_true.append(labels[mask].detach().cpu().numpy())
            all_pred.append(pred[mask].detach().cpu().numpy())

            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                sparse=f"{sparse_loss.item():.4f}",
            )

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    metrics = calculate_metrics(y_true, y_pred)
    return total_loss / max(len(loader), 1), total_cls / max(len(loader), 1), metrics


def flatten_metrics(prefix: str, metrics: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {
        f"{prefix}_oa": float(metrics["oa"]),
        f"{prefix}_aa": float(metrics["aa"]),
        f"{prefix}_kappa": float(metrics["kappa"]),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_miou": float(metrics["miou"]),
    }
    for j, cls in enumerate(config.valid_classes):
        out[f"{prefix}_precision_{cls}"] = float(metrics["precision"][j])
        out[f"{prefix}_recall_{cls}"] = float(metrics["recall"][j])
        out[f"{prefix}_f1_{cls}"] = float(metrics["f1"][j])
        out[f"{prefix}_iou_{cls}"] = float(metrics["iou"][j])
    return out


def train_one_run(run_idx: int, seed: int, train_loader, val_loader, test_loader, weights: np.ndarray):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionModel().to(device)
    print(f"Run {run_idx + 1}: trainable parameters = {count_parameters(model):,} ({count_parameters(model)/1e6:.3f} M)")

    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_t, ignore_index=config.ignore_index)

    # Manuscript says Adam; use Adam rather than AdamW for implementation-text consistency.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    run_dir = config.metrics_dir / f"run_{run_idx + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoints_path.mkdir(parents=True, exist_ok=True)
    config.confusion_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.checkpoints_path / f"best_model_run_{run_idx + 1}.pth"

    best_val_oa = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_cls, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer,
            desc=f"Run {run_idx + 1} Epoch {epoch}/{config.epochs} [Train]",
        )
        val_loss, val_cls, val_metrics = run_epoch(
            model, val_loader, criterion, device, None,
            desc=f"Run {run_idx + 1} Epoch {epoch}/{config.epochs} [Val]",
        )

        row: Dict[str, float] = {
            "run": run_idx + 1,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_cls,
            "val_loss": val_loss,
            "val_cls_loss": val_cls,
        }
        row.update(flatten_metrics("train", train_metrics))
        row.update(flatten_metrics("val", val_metrics))
        history.append(row)

        if config.save_every_epoch_metrics:
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch}: val OA={val_metrics['oa']:.4f}, AA={val_metrics['aa']:.4f}, "
            f"Kappa={val_metrics['kappa']:.4f}, mIoU={val_metrics['miou']:.4f}, "
            f"Macro-F1={val_metrics['macro_f1']:.4f}"
        )

        if float(val_metrics["oa"]) > best_val_oa:
            best_val_oa = float(val_metrics["oa"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "run": run_idx + 1,
                    "epoch": epoch,
                    "val_oa": best_val_oa,
                    "class_weights": weights,
                },
                ckpt_path,
            )
            np.save(
                config.confusion_dir / f"val_best_run_{run_idx + 1}.npy",
                val_metrics["confusion_matrix"],
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_cls, test_metrics = run_epoch(
        model, test_loader, criterion, device, None,
        desc=f"Run {run_idx + 1} [Test]",
    )

    np.save(
        config.confusion_dir / f"test_run_{run_idx + 1}.npy",
        test_metrics["confusion_matrix"],
    )

    summary = {
        "run": run_idx + 1,
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_oa": float(checkpoint["val_oa"]),
        "test_loss": test_loss,
        "test_cls_loss": test_cls,
    }
    summary.update(flatten_metrics("test", test_metrics))
    pd.DataFrame([summary]).to_csv(run_dir / "test_summary.csv", index=False)
    return summary


def main() -> None:
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, class_counts, class_weights, split, stats = get_dataloaders()

    print("Training pixel counts:", class_counts.tolist())
    print("ICF class weights (mean=1 over classes 1..5):", np.round(class_weights, 4).tolist())
    print(
        f"Block split: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}"
    )

    with open(config.metrics_dir / "training_setup.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_counts": class_counts.tolist(),
                "class_weights": class_weights.tolist(),
                "split_sizes": {k: len(v) for k, v in split.items()},
                "run_seeds": list(config.run_seeds),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summaries = []
    for run_idx, seed in enumerate(config.run_seeds):
        summaries.append(
            train_one_run(
                run_idx,
                seed,
                train_loader,
                val_loader,
                test_loader,
                class_weights,
            )
        )

    df = pd.DataFrame(summaries)
    df.to_csv(config.metrics_dir / "five_run_test_results.csv", index=False)

    numeric_cols = [c for c in df.columns if c.startswith("test_") and c not in {"test_loss", "test_cls_loss"}]
    agg_rows = []
    for col in numeric_cols:
        agg_rows.append(
            {
                "metric": col,
                "mean": float(df[col].mean()),
                "std": float(df[col].std(ddof=1)),
            }
        )
    pd.DataFrame(agg_rows).to_csv(config.metrics_dir / "five_run_mean_std.csv", index=False)
    print("Five-run results saved to:", config.metrics_dir)


if __name__ == "__main__":
    main()
