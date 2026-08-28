from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import config


def _available_block_ids(block_dir: Path) -> List[int]:
    ids: List[int] = []
    for p in block_dir.glob("radar_*.npy"):
        try:
            idx = int(p.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        required = [
            block_dir / f"optical_{idx}.npy",
            block_dir / f"label_{idx}.npy",
            block_dir / f"superpixel_{idx}.npy",
        ]
        if all(x.exists() for x in required):
            ids.append(idx)
    ids.sort()
    if not ids:
        raise RuntimeError(f"No complete radar/optical/label/superpixel blocks found in {block_dir}")
    return ids


def create_or_load_split(block_ids: Sequence[int]) -> Dict[str, List[int]]:
    if config.split_path.exists():
        with open(config.split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        return {k: [int(x) for x in v] for k, v in split.items()}

    ids = np.asarray(block_ids, dtype=np.int64)
    rng = np.random.default_rng(config.split_seed)
    ids = rng.permutation(ids)

    n = len(ids)
    n_train = int(round(n * config.train_ratio))
    n_val = int(round(n * config.val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    split = {
        "train": ids[:n_train].tolist(),
        "val": ids[n_train:n_train + n_val].tolist(),
        "test": ids[n_train + n_val:].tolist(),
    }

    config.output_path.mkdir(parents=True, exist_ok=True)
    with open(config.split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def compute_train_statistics(block_dir: Path, train_ids: Sequence[int]) -> Dict[str, np.ndarray]:
    if config.stats_path.exists():
        stats = np.load(config.stats_path)
        return {k: stats[k].astype(np.float32) for k in stats.files}

    radar_sum = np.zeros(config.radar_bands, dtype=np.float64)
    radar_sq_sum = np.zeros(config.radar_bands, dtype=np.float64)
    optical_sum = np.zeros(config.optical_bands, dtype=np.float64)
    optical_sq_sum = np.zeros(config.optical_bands, dtype=np.float64)
    pixel_count = 0

    for idx in tqdm(train_ids, desc="Computing TRAIN-only normalization statistics"):
        radar = np.load(block_dir / f"radar_{idx}.npy").astype(np.float64)
        optical = np.load(block_dir / f"optical_{idx}.npy").astype(np.float64)

        radar_sum += radar.sum(axis=(0, 1))
        radar_sq_sum += np.square(radar).sum(axis=(0, 1))
        optical_sum += optical.sum(axis=(0, 1))
        optical_sq_sum += np.square(optical).sum(axis=(0, 1))
        pixel_count += radar.shape[0] * radar.shape[1]

    radar_mean = radar_sum / pixel_count
    radar_var = np.maximum(radar_sq_sum / pixel_count - np.square(radar_mean), 1e-8)
    optical_mean = optical_sum / pixel_count
    optical_var = np.maximum(optical_sq_sum / pixel_count - np.square(optical_mean), 1e-8)

    payload = {
        "radar_mean": radar_mean.astype(np.float32),
        "radar_std": np.sqrt(radar_var).astype(np.float32),
        "optical_mean": optical_mean.astype(np.float32),
        "optical_std": np.sqrt(optical_var).astype(np.float32),
    }
    np.savez(config.stats_path, **payload)
    return payload


def compute_class_counts(block_dir: Path, ids: Sequence[int]) -> np.ndarray:
    counts = np.zeros(config.num_classes, dtype=np.int64)
    for idx in tqdm(ids, desc="Counting TRAIN labels"):
        label = np.load(block_dir / f"label_{idx}.npy").astype(np.int64)
        label = np.where(np.isin(label, config.valid_classes), label, config.ignore_index)
        unique, c = np.unique(label[label != config.ignore_index], return_counts=True)
        for cls, n in zip(unique, c):
            counts[int(cls)] += int(n)
    return counts


def inverse_frequency_weights_mean_one(class_counts: np.ndarray) -> np.ndarray:
    """ICF weights normalized so the mean weight across valid classes is 1.

    This normalization is consistent with manuscript-style values such as a much larger
    Suaeda weight than the dominant Phragmites class while preserving the ICF ratios.
    """
    weights = np.zeros(config.num_classes, dtype=np.float32)
    valid = np.asarray(config.valid_classes, dtype=np.int64)
    counts = class_counts[valid].astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"At least one valid class has zero training pixels: {class_counts}")
    raw = 1.0 / counts
    raw = raw / raw.mean()
    weights[valid] = raw.astype(np.float32)
    return weights


def _augment(
    radar: np.ndarray,
    optical: np.ndarray,
    label: np.ndarray,
    superpixel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Random horizontal/vertical flips + discrete 90-degree rotations.
    if random.random() < 0.5:
        radar = np.flip(radar, axis=1)
        optical = np.flip(optical, axis=1)
        label = np.flip(label, axis=1)
        superpixel = np.flip(superpixel, axis=1)
    if random.random() < 0.5:
        radar = np.flip(radar, axis=0)
        optical = np.flip(optical, axis=0)
        label = np.flip(label, axis=0)
        superpixel = np.flip(superpixel, axis=0)

    k = random.randint(0, 3)
    if k:
        radar = np.rot90(radar, k=k, axes=(0, 1))
        optical = np.rot90(optical, k=k, axes=(0, 1))
        label = np.rot90(label, k=k, axes=(0, 1))
        superpixel = np.rot90(superpixel, k=k, axes=(0, 1))

    return (
        np.ascontiguousarray(radar),
        np.ascontiguousarray(optical),
        np.ascontiguousarray(label),
        np.ascontiguousarray(superpixel),
    )


class RemoteSensingDataset(Dataset):
    def __init__(
        self,
        block_dir: str | Path,
        block_ids: Sequence[int],
        stats: Dict[str, np.ndarray],
        augment: bool = False,
    ) -> None:
        self.block_dir = Path(block_dir)
        self.block_ids = [int(x) for x in block_ids]
        self.stats = stats
        self.augment = augment

    def __len__(self) -> int:
        return len(self.block_ids)

    def __getitem__(self, i: int):
        idx = self.block_ids[i]
        radar = np.load(self.block_dir / f"radar_{idx}.npy").astype(np.float32)
        optical = np.load(self.block_dir / f"optical_{idx}.npy").astype(np.float32)
        label = np.load(self.block_dir / f"label_{idx}.npy").astype(np.int64)
        superpixel = np.load(self.block_dir / f"superpixel_{idx}.npy").astype(np.int64)

        if radar.shape[-1] != config.radar_bands:
            raise ValueError(f"Block {idx}: radar shape {radar.shape} does not match {config.radar_bands} bands")
        if optical.shape[-1] != config.optical_bands:
            raise ValueError(f"Block {idx}: optical shape {optical.shape} does not match {config.optical_bands} bands")

        label = np.where(np.isin(label, config.valid_classes), label, config.ignore_index)

        radar = (radar - self.stats["radar_mean"]) / self.stats["radar_std"]
        optical = (optical - self.stats["optical_mean"]) / self.stats["optical_std"]

        if self.augment:
            radar, optical, label, superpixel = _augment(radar, optical, label, superpixel)

        radar_t = torch.from_numpy(radar).permute(2, 0, 1).float()
        optical_t = torch.from_numpy(optical).permute(2, 0, 1).float()
        label_t = torch.from_numpy(label).long()
        superpixel_t = torch.from_numpy(superpixel).long()
        return radar_t, optical_t, label_t, superpixel_t, idx


def get_dataloaders(block_dir: str | Path | None = None):
    block_dir = Path(block_dir or config.output_dir)
    ids = _available_block_ids(block_dir)
    split = create_or_load_split(ids)
    stats = compute_train_statistics(block_dir, split["train"])
    class_counts = compute_class_counts(block_dir, split["train"])
    class_weights = inverse_frequency_weights_mean_one(class_counts)

    train_ds = RemoteSensingDataset(block_dir, split["train"], stats, augment=True)
    val_ds = RemoteSensingDataset(block_dir, split["val"], stats, augment=False)
    test_ds = RemoteSensingDataset(block_dir, split["test"], stats, augment=False)

    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, class_counts, class_weights, split, stats
