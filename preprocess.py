# -------------------- preprocess.py --------------------
import os
import json
import time
import numpy as np
from typing import Tuple, List

import rasterio
from configs import config

# SNIC 优先，若不可用自动回退 SLIC
try:
    from skimage.segmentation import snic as sk_snic

    HAS_SNIC = True
except Exception:
    HAS_SNIC = False
from skimage.segmentation import slic as sk_slic
from skimage.segmentation import mark_boundaries

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


def _scale01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi <= lo:
            out[..., c] = 0.0
        else:
            x = np.clip((ch - lo) / (hi - lo), 0, 1)
            out[..., c] = x
    return out


def _label_purity(spx: np.ndarray, label: np.ndarray, ignore_index: int) -> float:
    valid_mask = label != ignore_index
    if not valid_mask.any():
        return 0.0
    spx_ids = np.unique(spx)
    correct = 0
    total = int(valid_mask.sum())
    for sid in spx_ids:
        mask = (spx == sid) & valid_mask
        if not mask.any():
            continue
        vals, cnt = np.unique(label[mask], return_counts=True)
        major = vals[np.argmax(cnt)]
        correct += int((label[mask] == major).sum())
    return correct / max(total, 1)


def _edge_recall_proxy(spx: np.ndarray, label: np.ndarray, ignore_index: int) -> float:
    up = np.pad(spx, 1, 'edge')[:-2, 1:-1]
    dn = np.pad(spx, 1, 'edge')[2:, 1:-1]
    lt = np.pad(spx, 1, 'edge')[1:-1, :-2]
    rt = np.pad(spx, 1, 'edge')[1:-1, 2:]
    b_spx = (up != dn) | (lt != rt)

    lbl = (label != ignore_index).astype(np.int32)
    up = np.pad(lbl, 1, 'edge')[:-2, 1:-1]
    dn = np.pad(lbl, 1, 'edge')[2:, 1:-1]
    lt = np.pad(lbl, 1, 'edge')[1:-1, :-2]
    rt = np.pad(lbl, 1, 'edge')[1:-1, 2:]
    b_lbl = (up != dn) | (lt != rt)
    b_lbl_dil = binary_dilation(b_lbl, iterations=1)

    denom = int(b_spx.sum())
    if denom == 0:
        return 0.0
    return float((b_spx & b_lbl_dil).sum()) / float(denom)


def _run_superpixel_block_from_radar(radar: np.ndarray, n_segments: int, compactness: float,
                                     prefer_snic: bool = True) -> np.ndarray:
    img = _scale01(radar)
    seg = None
    if prefer_snic and HAS_SNIC and config.sp_method.lower() == "snic":
        try:
            seg = sk_snic(img, n_segments=int(n_segments), compactness=float(compactness),
                          start_label=0, channel_axis=-1)
        except TypeError:
            try:
                seg = sk_snic(img, n_segments=int(n_segments), compactness=float(compactness), start_label=0)
            except Exception:
                seg = None
    if seg is None:
        seg = sk_slic(img, n_segments=int(n_segments), compactness=float(compactness),
                      start_label=0, convert2lab=False, channel_axis=-1)
    ids = np.unique(seg)
    remap = {old: new for new, old in enumerate(ids.tolist())}
    seg = np.vectorize(remap.get)(seg).astype(np.int32)
    return seg


def _viz_spx(radar_block: np.ndarray, spx: np.ndarray, save_path: str):
    base = radar_block.mean(-1)
    vmin, vmax = float(np.percentile(base, 2)), float(np.percentile(base, 98))
    base = np.clip((base - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    vis = mark_boundaries(base, spx, color=(1, 0, 0), mode='outer')
    plt.figure(figsize=(3, 3));
    plt.axis('off');
    plt.imshow(vis, cmap='gray')
    plt.tight_layout();
    plt.savefig(save_path, dpi=200);
    plt.close()


def _safe_write_csv(df, target_csv: str, log_dir: str, retries: int = 4):
    import pandas as pd
    os.makedirs(log_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(target_csv))[0]
    for i in range(retries):
        tmp_path = os.path.join(log_dir, f"_{base}.tmp.{os.getpid()}.{i}.csv")
        try:
            df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
            try:
                os.replace(tmp_path, target_csv)
                return target_csv
            except PermissionError:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                time.sleep(0.5)
                continue
        except PermissionError:
            time.sleep(0.5)
            continue
        except Exception as e:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise e
    ts = time.strftime("%Y%m%d_%H%M%S")
    fallback = os.path.join(log_dir, f"{base}_{ts}.csv")
    df.to_csv(fallback, index=False, encoding="utf-8-sig")
    print(f"[提示] 目标文件被占用，已改为写入：{fallback}")
    return fallback


def _bayes_opt_choose_params(block_indices: List[int]) -> Tuple[int, float]:
    seg_min, seg_max = config.sp_segments_min, config.sp_segments_max
    comp_min, comp_max = config.sp_compact_min, config.sp_compact_max

    log_dir = os.path.join(config.superpixel_dir, "_bo_log")
    os.makedirs(log_dir, exist_ok=True)

    probe_blocks = block_indices[:max(0, config.sp_log_probe_blocks)]
    trial_rows = []

    def objective(params):
        n_segments = int(params[0]);
        compactness = float(params[1])
        purities, seg_nums, edges = [], [], []
        for idx in block_indices:
            radar = np.load(os.path.join(config.output_dir, f'radar_{idx}.npy')).astype(np.float32)
            label = np.load(os.path.join(config.output_dir, f'label_{idx}.npy')).astype(np.int64)
            spx = _run_superpixel_block_from_radar(
                radar, n_segments, compactness, prefer_snic=(config.sp_method.lower() == "snic")
            )
            purity = _label_purity(spx, label, config.ignore_index)
            seg_num = int(spx.max() + 1)
            purities.append(purity);
            seg_nums.append(seg_num)
            if config.sp_score_use_edge:
                edges.append(_edge_recall_proxy(spx, label, config.ignore_index))

        mean_purity = float(np.mean(purities)) if purities else 0.0
        mean_seg = float(np.mean(seg_nums)) if seg_nums else 0.0
        mean_edge = float(np.mean(edges)) if (config.sp_score_use_edge and edges) else 0.0

        mean_score = (
                config.sp_score_w_purity * mean_purity
                + (config.sp_score_w_edge * mean_edge if config.sp_score_use_edge else 0.0)
                - config.sp_score_lambda_segments * mean_seg
        )

        trial_rows.append({
            "iter": len(trial_rows) + 1,
            "n_segments": n_segments,
            "compactness": compactness,
            "mean_purity": mean_purity,
            "mean_edge": mean_edge,
            "mean_segments": mean_seg,
            "mean_score": mean_score
        })

        if config.sp_viz_every_iter and len(probe_blocks) > 0:
            for b in probe_blocks:
                rblk = np.load(os.path.join(config.output_dir, f'radar_{b}.npy')).astype(np.float32)
                spx_b = _run_superpixel_block_from_radar(
                    rblk, n_segments, compactness, prefer_snic=(config.sp_method.lower() == "snic")
                )
                save_p = os.path.join(log_dir, f"iter_{len(trial_rows):03d}_block_{b}.png")
                _viz_spx(rblk, spx_b, save_p)

        return -mean_score  # gp_minimize 最小化

    best_params = None
    if config.sp_use_bayes_opt:
        try:
            from skopt import gp_minimize
            from skopt.space import Integer, Real
            space = [Integer(seg_min, seg_max, name="n_segments"),
                     Real(comp_min, comp_max, name="compactness")]
            res = gp_minimize(
                objective, space,
                n_calls=config.sp_bo_iter,
                n_initial_points=min(8, max(2, config.sp_bo_iter // 2)),
                acq_func="EI",
                random_state=config.seed
            )
            best_params = (int(res.x[0]), float(res.x[1]))
        except Exception as e:
            print(f"[警告] 未能使用 scikit-optimize，改用随机搜索。错误：{e}")

    if best_params is None:
        best_val = float("inf")
        best = (min(32, seg_max), 10.0)
        rng = np.random.RandomState(config.seed)
        for _ in range(max(10, config.sp_bo_iter)):
            n_segments = int(rng.randint(seg_min, seg_max + 1))
            compactness = float(rng.uniform(comp_min, comp_max))
            val = objective([n_segments, compactness])
            if val < best_val:
                best_val = val
                best = (n_segments, compactness)
        best_params = best

    if config.sp_log_trials and len(trial_rows) > 0:
        try:
            import pandas as pd
            csv_path = os.path.join(log_dir, "bo_trace.csv")
            _safe_write_csv(pd.DataFrame(trial_rows), csv_path, log_dir, retries=4)
        except Exception as e:
            print(f"[警告] 记录 bo_trace.csv 失败：{e}")

    print(f"[BO] 选择到的超像素参数（Radar-only）：n_segments={best_params[0]}, compactness={best_params[1]:.2f}")
    return best_params


def crop_images():
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.superpixel_dir, exist_ok=True)

    with rasterio.open(config.radar_path) as src_radar, \
            rasterio.open(config.optical_path) as src_optical, \
            rasterio.open(config.label_path) as src_label:

        radar_data = src_radar.read().astype(np.float32)  # [Cr,H,W]
        optical_data = src_optical.read().astype(np.float32)  # [Co,H,W]
        label_data = src_label.read(1).astype(np.int64)  # [H,W]

        def process_nan(data, name):
            if np.isnan(data).any():
                print(f"检测到{name}数据中存在NaN值，正在处理...")
                for band in range(data.shape[0]):
                    mask = np.isnan(data[band])
                    if mask.any():
                        mean_val = np.nanmean(data[band])
                        data[band][mask] = mean_val
            return data

        radar_data = process_nan(radar_data, "雷达")
        optical_data = process_nan(optical_data, "光学")

        # ================= 1. 增强标签清洗逻辑 =================
        # 将原始0值(背景)和NoData(如2147483647)转为ignore_index
        label_data = np.where(label_data == 0, config.ignore_index, label_data)
        label_data = np.where(label_data == 2147483647, config.ignore_index, label_data)

        # 兜底：所有不在 valid_classes 里的值都视为无效
        valid_set = set(config.valid_classes)
        valid_set.add(config.ignore_index)
        mask_invalid = ~np.isin(label_data, list(valid_set))
        if mask_invalid.any():
            print(f"检测到 {mask_invalid.sum()} 个非法标签像素，已重置为0")
            label_data[mask_invalid] = config.ignore_index

        print("预处理后标签唯一值:", np.unique(label_data))
        # ========================================================

        height, width = src_radar.shape
        pad_height = (config.block_size - (height % config.block_size)) % config.block_size
        pad_width = (config.block_size - (width % config.block_size)) % config.block_size

        # 反射填充
        radar_padded = np.pad(radar_data, ((0, 0), (0, pad_height), (0, pad_width)), mode='reflect')
        optical_padded = np.pad(optical_data, ((0, 0), (0, pad_height), (0, pad_width)), mode='reflect')
        label_padded = np.pad(label_data, ((0, pad_height), (0, pad_width)),
                              mode='constant', constant_values=config.ignore_index)

        # ================= 2. 关键修改：重叠切块增加样本 =================
        valid_blocks = 0
        H, W = height + pad_height, width + pad_width

        # 使用步长=8 (BlockSize/4)，实现75%重叠，大幅增加样本量
        # 原始：stride = config.block_size (32)
        # 现在：stride = 8
        stride = 8
        print(f"正在进行重叠切块采样 (Stride={stride})...")

        for y in range(0, H - config.block_size + 1, stride):
            for x in range(0, W - config.block_size + 1, stride):
                rblk = radar_padded[:, y:y + config.block_size, x:x + config.block_size].transpose(1, 2, 0)
                oblk = optical_padded[:, y:y + config.block_size, x:x + config.block_size].transpose(1, 2, 0)
                lblk = label_padded[y:y + config.block_size, x:x + config.block_size]

                # 只有当块内包含有效标签时才保存（可选，防止全黑背景块过多）
                # 这里为了保证样本量，我们保存所有块，但在Dataloader里可以通过权重控制
                np.save(os.path.join(config.output_dir, f'radar_{valid_blocks}.npy'), rblk)
                np.save(os.path.join(config.output_dir, f'optical_{valid_blocks}.npy'), oblk)
                np.save(os.path.join(config.output_dir, f'label_{valid_blocks}.npy'), lblk)
                valid_blocks += 1

        print(f"有效数据块数量: {valid_blocks} (已大幅增加)")
        # ==============================================================

        # ========== Radar-only BO 选参 ==========
        if config.use_superpixels:
            block_indices = list(range(valid_blocks))
            # 随机采样一些块用于BO，因为现在块太多了
            if len(block_indices) > config.sp_bo_sample_blocks:
                rng = np.random.RandomState(config.seed)
                block_indices = rng.choice(block_indices, size=config.sp_bo_sample_blocks,
                                           replace=False).tolist()

            n_segments, compactness = _bayes_opt_choose_params(block_indices)

            # 保存参数
            param_path = os.path.join(config.superpixel_dir, "superpixel_params.json")
            with open(param_path, "w", encoding="utf-8") as f:
                json.dump({
                    "n_segments": int(n_segments),
                    "compactness": float(compactness),
                    "method": ("snic" if (HAS_SNIC and config.sp_method.lower() == "snic") else "slic"),
                    "radar_only": True
                }, f, ensure_ascii=False, indent=2)

            # ========== 生成超像素（注意：这里需要生成两部分）==========
            # 1. 生成“整图”的超像素 (用于 predict.py)
            print("[SPX] 正在整图生成超像素 ...")
            radar_full = radar_data.transpose(1, 2, 0)
            spx_full = _run_superpixel_block_from_radar(
                radar_full, n_segments, compactness, prefer_snic=(config.sp_method.lower() == "snic")
            )
            # 保存整图超像素
            np.save(os.path.join(config.superpixel_dir, "spx_full.npy"), spx_full)

            # 2. 生成“切块”的超像素 (用于 train.py)
            # 由于切块是重叠的，我们必须对每个切块单独生成超像素，或者从大图切出来
            # 为了简单且效果最好，建议对每个切块单独运行超像素算法，或者从Padding后的整图切

            # 这里采用从 Padding 后的整图切出的方式，保证一致性
            spx_padded = np.pad(spx_full, ((0, pad_height), (0, pad_width)), mode='edge')

            idx = 0
            # 使用与上面完全相同的循环和步长
            for y in range(0, H - config.block_size + 1, stride):
                for x in range(0, W - config.block_size + 1, stride):
                    spx_blk = spx_padded[y:y + config.block_size, x:x + config.block_size]
                    np.save(os.path.join(config.superpixel_dir, f"spx_{idx}.npy"), spx_blk.astype(np.int32))
                    idx += 1

            print(f"切块超像素已生成，保存到：{config.superpixel_dir}")


if __name__ == "__main__":
    crop_images()