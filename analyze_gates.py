from __future__ import annotations

"""Export ASPM/GFM gate statistics for manuscript mechanism analysis.

This script intentionally does not modify the network. It extracts evidence that can support:
1) ASPM scale-selection behaviour; 2) GFM modality reliability weighting.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configs import config
from dataloader import get_dataloaders
from model import FusionModel


def main(run_id: int = 1) -> None:
    _, _, test_loader, _, _, _, _ = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionModel().to(device)
    ckpt = torch.load(config.checkpoints_path / f"best_model_run_{run_id}.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gfm_by_class = defaultdict(list)
    small_by_class = defaultdict(list)
    large_by_class = defaultdict(list)

    with torch.no_grad():
        for radar, optical, labels, superpixels, _ in tqdm(test_loader, desc="Gate analysis"):
            radar = radar.to(device)
            optical = optical.to(device)
            labels = labels.to(device)
            superpixels = superpixels.to(device)

            logits, sparse_loss, aux = model(radar, optical, superpixels, return_aux=True)
            fusion_mask = aux["gfm"]["fusion_mask"]      # [B,C,H,W], CNN/MSI weight
            w_small = aux["aspm"]["aspm2"]["w_small"]  # [B,1,H,W]
            w_large = aux["aspm"]["aspm2"]["w_large"]

            # Reduce channel-wise fusion weights to one value per pixel.
            msi_weight = fusion_mask.mean(dim=1)
            for cls in config.valid_classes:
                mask = labels == cls
                if bool(mask.any()):
                    gfm_by_class[cls].append(float(msi_weight[mask].mean().cpu()))
                    small_by_class[cls].append(float(w_small[:, 0][mask].mean().cpu()))
                    large_by_class[cls].append(float(w_large[:, 0][mask].mean().cpu()))

    rows = []
    for cls in config.valid_classes:
        rows.append(
            {
                "class": cls,
                "msi_weight_mean": np.mean(gfm_by_class[cls]),
                "sar_weight_mean": 1.0 - np.mean(gfm_by_class[cls]),
                "aspm_small_scale_mean": np.mean(small_by_class[cls]),
                "aspm_large_scale_mean": np.mean(large_by_class[cls]),
            }
        )

    out = config.metrics_dir / f"gate_statistics_run_{run_id}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main(1)
