from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class _Cluster:
    count: int
    sum_y: float
    sum_x: float
    sum_feat: np.ndarray

    @property
    def cy(self) -> float:
        return self.sum_y / max(self.count, 1)

    @property
    def cx(self) -> float:
        return self.sum_x / max(self.count, 1)

    @property
    def mean_feat(self) -> np.ndarray:
        return self.sum_feat / max(self.count, 1)


def _normalize_features(image: np.ndarray) -> np.ndarray:
    """Per-band robust normalization for SNIC distance only."""
    image = image.astype(np.float32, copy=False)
    out = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        band = image[..., c]
        lo, hi = np.percentile(band, [2.0, 98.0])
        if hi <= lo:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def _seed_positions(h: int, w: int, target_k: int) -> List[Tuple[int, int]]:
    target_k = max(1, min(target_k, h * w))
    aspect = h / max(w, 1)
    n_rows = max(1, int(round(math.sqrt(target_k * aspect))))
    n_cols = max(1, int(math.ceil(target_k / n_rows)))

    ys = np.linspace(0, h - 1, n_rows + 2, dtype=np.float32)[1:-1]
    xs = np.linspace(0, w - 1, n_cols + 2, dtype=np.float32)[1:-1]
    seeds = [(int(round(y)), int(round(x))) for y in ys for x in xs]
    return seeds[:target_k]


def snic_segment(
    image: np.ndarray,
    num_superpixels: int = 64,
    compactness: float = 10.0,
) -> np.ndarray:
    """A compact NumPy implementation of the SNIC priority-queue region growing idea.

    Parameters
    ----------
    image : H x W x C ndarray
        Multi-band image used to construct superpixels. For ASSG-Net this should be
        the Sentinel-1 composite.
    num_superpixels : int
        Target number of seeds/superpixels per patch.
    compactness : float
        Relative weight of spatial distance.

    Returns
    -------
    labels : H x W int32 ndarray
        Contiguous superpixel IDs starting at 0.

    Notes
    -----
    SNIC implementations can differ slightly in seed placement and distance details.
    If the manuscript results were produced using an existing SNIC implementation
    (for example, a previously saved SNIC map), keep those exact maps for strict
    reproducibility instead of regenerating them with a different implementation.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape={image.shape}")

    h, w, c = image.shape
    feat = _normalize_features(image)
    seeds = _seed_positions(h, w, num_superpixels)
    k_actual = len(seeds)
    if k_actual == 0:
        return np.zeros((h, w), dtype=np.int32)

    spacing = math.sqrt((h * w) / k_actual)
    spatial_scale = compactness / max(spacing, 1e-6)

    labels = np.full((h, w), -1, dtype=np.int32)
    clusters: List[_Cluster] = []
    pq: List[Tuple[float, int, int, int]] = []

    for cid, (y, x) in enumerate(seeds):
        clusters.append(
            _Cluster(
                count=0,
                sum_y=0.0,
                sum_x=0.0,
                sum_feat=np.zeros(c, dtype=np.float64),
            )
        )
        heapq.heappush(pq, (0.0, cid, y, x))

    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def distance(cid: int, y: int, x: int) -> float:
        cluster = clusters[cid]
        if cluster.count == 0:
            cy, cx = seeds[cid]
            mu = feat[cy, cx]
        else:
            cy, cx = cluster.cy, cluster.cx
            mu = cluster.mean_feat
        df2 = float(np.sum((feat[y, x] - mu) ** 2))
        ds2 = float((y - cy) ** 2 + (x - cx) ** 2)
        return math.sqrt(df2 + (spatial_scale ** 2) * ds2)

    while pq:
        _, cid, y, x = heapq.heappop(pq)
        if labels[y, x] != -1:
            continue

        labels[y, x] = cid
        cluster = clusters[cid]
        cluster.count += 1
        cluster.sum_y += y
        cluster.sum_x += x
        cluster.sum_feat += feat[y, x].astype(np.float64)

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == -1:
                heapq.heappush(pq, (distance(cid, ny, nx), cid, ny, nx))

    # Extremely defensive fallback; a connected grid should normally assign every pixel.
    missing = np.argwhere(labels < 0)
    if len(missing):
        centers = np.array([(cl.cy, cl.cx) for cl in clusters], dtype=np.float32)
        for y, x in missing:
            cid = int(np.argmin(np.sum((centers - np.array([y, x])) ** 2, axis=1)))
            labels[y, x] = cid

    # Re-index only IDs actually used.
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(h, w).astype(np.int32)
