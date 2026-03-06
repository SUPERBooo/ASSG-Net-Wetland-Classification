# -------------------- dataloader.py --------------------
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from configs import config
from glob import glob
from typing import Tuple
from tqdm import tqdm


class RemoteSensingDataset(Dataset):
    def __init__(self, block_dir: str, augment: bool = False):
        self.block_dir = block_dir
        self.augment = augment  # 开关数据增强
        self._prepare_indices()

        # 如果是首次运行或重置了数据，重新计算统计量
        stats_path = os.path.join(self.block_dir, "dataset_stats.npz")
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.radar_mean = stats['radar_mean']
            self.radar_std = stats['radar_std']
            self.optical_mean = stats['optical_mean']
            self.optical_std = stats['optical_std']
            # 重新计算分布（因为文件变多了）
            self.class_counts = self._compute_class_distribution()
        else:
            self._compute_statistics()
            self.class_counts = self._compute_class_distribution()

        self._validate_labels()

    def _prepare_indices(self):
        radar_files = glob(os.path.join(self.block_dir, 'radar_*.npy'))
        self.block_indices = []
        for f in radar_files:
            try:
                idx = int(os.path.basename(f).split('_')[1].split('.')[0])
                if os.path.exists(os.path.join(self.block_dir, f'optical_{idx}.npy')) and \
                        os.path.exists(os.path.join(self.block_dir, f'label_{idx}.npy')):
                    self.block_indices.append(idx)
            except:
                continue
        self.block_indices = sorted(self.block_indices)
        print(f"数据集加载完毕，共 {len(self.block_indices)} 个样本")

    def _compute_statistics(self):
        radar_sum = np.zeros(config.radar_bands, dtype=np.float32)
        radar_sq_sum = np.zeros(config.radar_bands, dtype=np.float32)
        optical_sum = np.zeros(config.optical_bands, dtype=np.float32)
        optical_sq_sum = np.zeros(config.optical_bands, dtype=np.float32)
        pixel_count = 0

        print("正在计算数据统计量...")
        # 为了速度，如果是重叠切块，可以只采样部分数据计算均值
        sample_indices = self.block_indices[::10] if len(self.block_indices) > 2000 else self.block_indices

        for idx in tqdm(sample_indices, desc='进度'):
            radar = np.load(os.path.join(self.block_dir, f'radar_{idx}.npy')).astype(np.float32)
            optical = np.load(os.path.join(self.block_dir, f'optical_{idx}.npy')).astype(np.float32)
            radar_sum += radar.sum(axis=(0, 1))
            radar_sq_sum += (radar ** 2).sum(axis=(0, 1))
            optical_sum += optical.sum(axis=(0, 1))
            optical_sq_sum += (optical ** 2).sum(axis=(0, 1))
            pixel_count += radar.shape[0] * radar.shape[1]

        self.radar_mean = radar_sum / pixel_count
        self.radar_std = np.sqrt(np.maximum(radar_sq_sum / pixel_count - self.radar_mean ** 2, 1e-8))
        self.optical_mean = optical_sum / pixel_count
        self.optical_std = np.sqrt(np.maximum(optical_sq_sum / pixel_count - self.optical_mean ** 2, 1e-8))

        stats_path = os.path.join(self.block_dir, "dataset_stats.npz")
        np.savez(
            stats_path,
            radar_mean=self.radar_mean.astype(np.float32),
            radar_std=self.radar_std.astype(np.float32),
            optical_mean=self.optical_mean.astype(np.float32),
            optical_std=self.optical_std.astype(np.float32)
        )

    def _preprocess_label(self, label: np.ndarray) -> np.ndarray:
        return np.where(np.isin(label, list(config.valid_classes)), label, config.ignore_index)

    def _compute_class_distribution(self):
        class_counts = np.zeros(config.num_classes, dtype=np.int64)
        # 采样计算
        sample_indices = self.block_indices[::5] if len(self.block_indices) > 2000 else self.block_indices
        for idx in tqdm(sample_indices, desc='统计类别分布'):
            label = np.load(os.path.join(self.block_dir, f'label_{idx}.npy'))
            label = self._preprocess_label(label)
            valid_labels = label[label != config.ignore_index]
            unique, counts = np.unique(valid_labels, return_counts=True)
            for u, c in zip(unique, counts):
                if u < config.num_classes:
                    class_counts[u] += c
        return class_counts

    def _validate_labels(self):
        # 简单抽查
        print("\n验证标签合法性(抽查)...")
        idx = self.block_indices[0]
        label = np.load(os.path.join(self.block_dir, f'label_{idx}.npy'))
        processed = self._preprocess_label(label)
        print(f"样本 {idx} 标签唯一值: {np.unique(processed)}")

    def __len__(self) -> int:
        return len(self.block_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block_id = self.block_indices[idx]
        radar = np.load(os.path.join(self.block_dir, f'radar_{block_id}.npy')).astype(np.float32)
        optical = np.load(os.path.join(self.block_dir, f'optical_{block_id}.npy')).astype(np.float32)
        label = np.load(os.path.join(self.block_dir, f'label_{block_id}.npy')).astype(np.int64)
        label = self._preprocess_label(label)

        # 标准化
        radar = (radar - self.radar_mean) / self.radar_std
        optical = (optical - self.optical_mean) / self.optical_std

        # 超像素
        spx_path = os.path.join(config.superpixel_dir, f'spx_{block_id}.npy')
        if config.use_superpixels and os.path.exists(spx_path):
            spx = np.load(spx_path).astype(np.int32)
        else:
            h, w = label.shape
            spx = np.arange(h * w, dtype=np.int32).reshape(h, w)

        # ================= 数据增强 =================
        if self.augment:
            # 1. 随机水平翻转
            if np.random.rand() > 0.5:
                radar = np.flip(radar, axis=1)
                optical = np.flip(optical, axis=1)
                label = np.flip(label, axis=1)
                spx = np.flip(spx, axis=1)

            # 2. 随机垂直翻转
            if np.random.rand() > 0.5:
                radar = np.flip(radar, axis=0)
                optical = np.flip(optical, axis=0)
                label = np.flip(label, axis=0)
                spx = np.flip(spx, axis=0)

            # 3. 随机旋转 90度
            k = np.random.randint(0, 4)
            if k > 0:
                radar = np.rot90(radar, k, axes=(0, 1))
                optical = np.rot90(optical, k, axes=(0, 1))
                label = np.rot90(label, k, axes=(0, 1))
                spx = np.rot90(spx, k, axes=(0, 1))

        # 复制以消除负步长影响
        radar = radar.copy()
        optical = optical.copy()
        label = label.copy()
        spx = spx.copy()
        # ===========================================

        radar_tensor = torch.from_numpy(radar).permute(2, 0, 1).float()
        optical_tensor = torch.from_numpy(optical).permute(2, 0, 1).float()
        label_tensor = torch.from_numpy(label).long()
        spx_tensor = torch.from_numpy(spx).long()
        return radar_tensor, optical_tensor, label_tensor, spx_tensor


def get_dataloaders(block_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # 训练集开启增强，验证集关闭
    full_dataset_train = RemoteSensingDataset(block_dir, augment=True)
    full_dataset_val = RemoteSensingDataset(block_dir, augment=False)

    val_size = int(len(full_dataset_train) * config.val_ratio)
    train_size = len(full_dataset_train) - val_size

    # 拆分索引
    indices = torch.randperm(len(full_dataset_train)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader