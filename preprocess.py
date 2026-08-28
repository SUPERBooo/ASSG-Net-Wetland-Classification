from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from configs import config
from snic import snic_segment


def _replace_nan_per_band(data: np.ndarray, name: str) -> np.ndarray:
    data = data.astype(np.float32, copy=True)
    for band in range(data.shape[0]):
        mask = ~np.isfinite(data[band])
        if mask.any():
            valid = data[band][~mask]
            fill = float(valid.mean()) if valid.size else 0.0
            data[band][mask] = fill
            print(f"{name} band {band}: replaced {int(mask.sum())} invalid pixels")
    return data


def crop_images_and_build_snic() -> None:
    out_dir = config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(config.radar_path) as src_radar, \
         rasterio.open(config.optical_path) as src_optical, \
         rasterio.open(config.label_path) as src_label:

        radar = _replace_nan_per_band(src_radar.read(), "Radar")
        optical_all = _replace_nan_per_band(src_optical.read(), "Optical")
        labels = src_label.read(1).astype(np.int64)

        if radar.shape[0] != config.radar_bands:
            raise ValueError(
                f"Radar TIFF has {radar.shape[0]} bands, but manuscript/code expects {config.radar_bands}."
            )

        max_idx = max(config.optical_band_indices)
        if max_idx >= optical_all.shape[0]:
            raise ValueError(
                f"optical_band_indices={config.optical_band_indices} exceed TIFF band count={optical_all.shape[0]}."
            )
        optical = optical_all[list(config.optical_band_indices)]
        if optical.shape[0] != config.optical_bands:
            raise RuntimeError(f"Optical selection produced {optical.shape[0]} bands, expected {config.optical_bands}.")

        if src_radar.shape != src_optical.shape or src_radar.shape != src_label.shape:
            raise ValueError("Radar, optical, and label rasters must be co-registered and have identical H/W.")

        # Keep labels 1..5; all other values become ignore_index=0.
        labels = np.where(np.isin(labels, config.valid_classes), labels, config.ignore_index)

        height, width = src_radar.shape
        bs = config.block_size
        pad_h = (bs - height % bs) % bs
        pad_w = (bs - width % bs) % bs
        padded_h = height + pad_h
        padded_w = width + pad_w
        grid_rows = padded_h // bs
        grid_cols = padded_w // bs
        total_blocks = grid_rows * grid_cols

        radar_pad = np.pad(radar, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        optical_pad = np.pad(optical, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        label_pad = np.pad(
            labels,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=config.ignore_index,
        )

        profile = src_label.profile.copy()

    print(f"Image size: {height} x {width}")
    print(f"Grid: {grid_rows} x {grid_cols} = {total_blocks} blocks")

    pbar = tqdm(total=total_blocks, desc="Preprocess + SNIC")
    block_id = 0
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            y0, x0 = gy * bs, gx * bs
            radar_block = radar_pad[:, y0:y0 + bs, x0:x0 + bs].transpose(1, 2, 0)
            optical_block = optical_pad[:, y0:y0 + bs, x0:x0 + bs].transpose(1, 2, 0)
            label_block = label_pad[y0:y0 + bs, x0:x0 + bs]

            sp_map = snic_segment(
                radar_block,
                num_superpixels=config.snic_num_superpixels,
                compactness=config.snic_compactness,
            )

            np.save(out_dir / f"radar_{block_id}.npy", radar_block.astype(np.float32))
            np.save(out_dir / f"optical_{block_id}.npy", optical_block.astype(np.float32))
            np.save(out_dir / f"label_{block_id}.npy", label_block.astype(np.int64))
            np.save(out_dir / f"superpixel_{block_id}.npy", sp_map.astype(np.int32))

            block_id += 1
            pbar.update(1)
    pbar.close()

    metadata = {
        "height": height,
        "width": width,
        "padded_height": padded_h,
        "padded_width": padded_w,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "total_blocks": total_blocks,
        "block_size": bs,
        "radar_bands": config.radar_bands,
        "optical_bands": config.optical_bands,
        "optical_band_indices": list(config.optical_band_indices),
        "snic_num_superpixels": config.snic_num_superpixels,
        "snic_compactness": config.snic_compactness,
        "crs": str(profile.get("crs")),
        "transform": tuple(profile.get("transform")) if profile.get("transform") is not None else None,
    }
    with open(config.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved {total_blocks} blocks to {out_dir}")
    print(f"Metadata: {config.metadata_path}")


if __name__ == "__main__":
    crop_images_and_build_snic()
