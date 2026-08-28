from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from configs import config
from model import FusionModel


def load_checkpoint(model: FusionModel, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)


def predict(run_id: int = 1) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    checkpoint_path = config.checkpoints_path / f"best_model_run_{run_id}.pth"
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    stats = np.load(config.stats_path)
    radar_mean = stats["radar_mean"].astype(np.float32)
    radar_std = stats["radar_std"].astype(np.float32)
    optical_mean = stats["optical_mean"].astype(np.float32)
    optical_std = stats["optical_std"].astype(np.float32)

    with open(config.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    h = int(meta["height"])
    w = int(meta["width"])
    grid_rows = int(meta["grid_rows"])
    grid_cols = int(meta["grid_cols"])
    total_blocks = int(meta["total_blocks"])
    bs = int(meta["block_size"])

    pred_full = np.full((h, w), config.ignore_index, dtype=np.uint8)

    with torch.no_grad():
        for idx in tqdm(range(total_blocks), desc="Predicting"):
            radar_path = config.output_path / f"radar_{idx}.npy"
            optical_path = config.output_path / f"optical_{idx}.npy"
            sp_path = config.output_path / f"superpixel_{idx}.npy"
            if not (radar_path.exists() and optical_path.exists() and sp_path.exists()):
                raise FileNotFoundError(f"Missing block files for block {idx}")

            radar = np.load(radar_path).astype(np.float32)
            optical = np.load(optical_path).astype(np.float32)
            sp = np.load(sp_path).astype(np.int64)

            radar = (radar - radar_mean) / radar_std
            optical = (optical - optical_mean) / optical_std

            radar_t = torch.from_numpy(radar).permute(2, 0, 1).unsqueeze(0).to(device)
            optical_t = torch.from_numpy(optical).permute(2, 0, 1).unsqueeze(0).to(device)
            sp_t = torch.from_numpy(sp).unsqueeze(0).to(device)

            logits, _ = model(radar_t, optical_t, sp_t)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            gy = idx // grid_cols
            gx = idx % grid_cols
            y0, x0 = gy * bs, gx * bs
            y1, x1 = min(y0 + bs, h), min(x0 + bs, w)
            pred_full[y0:y1, x0:x1] = pred[: y1 - y0, : x1 - x0]

    with rasterio.open(config.label_path) as src:
        profile = src.profile.copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=config.ignore_index,
        compress="lzw",
        width=w,
        height=h,
    )

    config.pred_save_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(config.pred_save_path, "w", **profile) as dst:
        dst.write(pred_full, 1)
        dst.write_colormap(
            1,
            {
                0: (0, 0, 0),
                1: (255, 0, 0),
                2: (0, 255, 0),
                3: (0, 0, 255),
                4: (255, 255, 0),
                5: (128, 0, 128),
            },
        )

    unique, counts = np.unique(pred_full[pred_full != config.ignore_index], return_counts=True)
    print("Prediction class counts:", dict(zip(unique.tolist(), counts.tolist())))
    print("Saved:", config.pred_save_path)


if __name__ == "__main__":
    predict(run_id=1)
