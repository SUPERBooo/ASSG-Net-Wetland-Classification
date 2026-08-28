from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # ---------- Raw data ----------
    radar_path: str = r"G:\H学术科研7\原始图像\radar_bands_image.tif"
    optical_path: str = r"G:\H学术科研7\原始图像\remaining_bands_image.tif"
    label_path: str = r"G:\H学术科研7\原始图像\Land_use_classification.tif"
    # Optical input: retain all 10 bands from remaining_bands_image.tif.
    # This is intentionally kept as the actual experimental setting.
    optical_band_indices: Tuple[int, ...] = tuple(range(10))

    # ---------- Output ----------
    output_dir: str = r"G:\H学术科研7\处理后图像_v2"
    checkpoint_dir: str = "checkpoints"
    prediction_name: str = "prediction.tif"

    # ---------- Label definition ----------
    num_classes: int = 6
    valid_classes: Tuple[int, ...] = (1, 2, 3, 4, 5)
    ignore_index: int = 0

    # ---------- Input dimensions ----------
    block_size: int = 32
    radar_bands: int = 5
    optical_bands: int = 4

    # ---------- Split ----------
    # Manuscript Table 1 implies Yancheng ≈ 70/15/15.
    # For Linhong, set train/val/test to ≈ 0.60/0.20/0.20 if you want to reproduce that table.
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # ---------- Training: manuscript-aligned defaults ----------
    batch_size: int = 32
    epochs: int = 100
    lr: float = 5e-3
    weight_decay: float = 0.0
    num_workers: int = 0 if os.name == "nt" else 4
    run_seeds: Tuple[int, ...] = (42,)

    # ---------- ASPM ----------
    feature_dim: int = 64
    aspm_gfe_channels: int = 16
    local_variance_kernel: int = 3
    eca_kernel_size: int = 3

    # ---------- SNIC ----------
    # These are explicit reproducibility parameters. If your historical experiment used different
    # SNIC settings, replace them with the exact original values before regenerating superpixels.
    snic_num_superpixels: int = 64
    snic_compactness: float = 10.0

    # ---------- AGSM: manuscript-aligned defaults ----------
    knn_k: int = 8
    agsm_temperature: float = 0.5
    agsm_dropedge: float = 0.2
    agsm_depth: int = 2
    sparsity_target: float = 0.5
    sparsity_lambda: float = 0.1
    graph_dropout: float = 0.2

    # ---------- GFM ----------
    gfm_temperature: float = 0.5

    # ---------- Misc ----------
    pin_memory: bool = True
    save_every_epoch_metrics: bool = True

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def metadata_path(self) -> Path:
        return self.output_path / "metadata.json"

    @property
    def stats_path(self) -> Path:
        return self.output_path / "dataset_stats.npz"

    @property
    def split_path(self) -> Path:
        return self.output_path / "split_manifest.json"

    @property
    def metrics_dir(self) -> Path:
        return self.output_path / "metrics"

    @property
    def confusion_dir(self) -> Path:
        return self.output_path / "confusion_matrices"

    @property
    def checkpoints_path(self) -> Path:
        return self.output_path / self.checkpoint_dir

    @property
    def pred_save_path(self) -> Path:
        return self.output_path / self.prediction_name

    def validate(self) -> None:
        if self.radar_bands != 5:
            raise ValueError("The manuscript describes a 5-band Sentinel-1 composite.")
        if self.optical_bands != 10:
            raise ValueError("This implementation is configured to use all 10 optical bands.")
        if len(self.optical_band_indices) != self.optical_bands:
            raise ValueError("optical_band_indices length must equal optical_bands.")
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-8:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.")
        if self.ignore_index in self.valid_classes:
            raise ValueError("ignore_index must not be one of valid_classes.")


config = Config()
config.validate()
