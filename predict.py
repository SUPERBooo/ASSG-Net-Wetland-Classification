# -------------------- predict.py --------------------
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from model import FusionModel
from configs import config
import os
from tqdm import tqdm


def _gaussian_window(h, w, sigma_h, sigma_w):
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.float32)
    y = np.arange(h) - (h - 1) / 2.0
    x = np.arange(w) - (w - 1) / 2.0
    yy, xx = np.meshgrid(y, x, indexing='ij')
    win = np.exp(-(yy ** 2 / (2 * sigma_h ** 2) + xx ** 2 / (2 * sigma_w ** 2))).astype(np.float32)
    win /= win.max()
    return win


def _safe_remove(path):
    import time
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except PermissionError:
            time.sleep(0.3)
    return False


def _process_nan(data):
    """处理NaN值，防止预测出现噪点"""
    if np.isnan(data).any():
        for band in range(data.shape[2]):
            mask = np.isnan(data[:, :, band])
            if mask.any():
                mean_val = np.nanmean(data[:, :, band])
                if np.isnan(mean_val): mean_val = 0.0
                data[:, :, band][mask] = mean_val
    return data


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(config.model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    print("模型加载成功")

    os.makedirs(os.path.dirname(config.pred_save_path), exist_ok=True)

    stats_path = os.path.join(config.output_dir, "dataset_stats.npz")
    stats = np.load(stats_path)
    radar_mean = stats['radar_mean'].astype(np.float32)
    radar_std = stats['radar_std'].astype(np.float32)
    optical_mean = stats['optical_mean'].astype(np.float32)
    optical_std = stats['optical_std'].astype(np.float32)

    # 1. 动态获取全图尺寸
    with rasterio.open(config.radar_path) as src:
        H, W = src.shape
        print(f"检测到输入图像实际尺寸: 高={H}, 宽={W}")

    # 2. 验证并对齐超像素
    spx_full_path = os.path.join(config.superpixel_dir, "spx_full.npy")
    if not os.path.exists(spx_full_path):
        raise FileNotFoundError(f"未找到整图超像素：{spx_full_path}\n请先运行 preprocess.py 生成超像素。")

    spx_full = np.load(spx_full_path).astype(np.int64)
    # 尺寸对齐
    if spx_full.shape != (H, W):
        print(f"警告: 超像素尺寸 {spx_full.shape} 与当前图像尺寸 ({H}, {W}) 不一致! 正在自动对齐...")
        spx_aligned = np.zeros((H, W), dtype=np.int64)
        min_h = min(H, spx_full.shape[0])
        min_w = min(W, spx_full.shape[1])
        spx_aligned[:min_h, :min_w] = spx_full[:min_h, :min_w]
        spx_full = spx_aligned

    C = config.num_classes
    prob_acc = np.zeros((C, H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)

    ks = config.block_size
    stride = ks // 2
    gw_standard = _gaussian_window(ks, ks, sigma_h=ks / 6.0, sigma_w=ks / 6.0)

    with rasterio.open(config.radar_path) as src_radar, \
            rasterio.open(config.optical_path) as src_optical:

        y_positions = list(range(0, H, stride))
        x_positions = list(range(0, W, stride))

        pbar = tqdm(total=len(y_positions) * len(x_positions), desc='预测进度')
        with torch.no_grad():
            for y in y_positions:
                for x in x_positions:
                    y2 = min(y + ks, H)
                    x2 = min(x + ks, W)
                    h = y2 - y
                    w = x2 - x
                    if h <= 0 or w <= 0:
                        pbar.update(1);
                        continue

                    try:
                        radar = src_radar.read(window=Window(x, y, w, h)).transpose(1, 2, 0).astype(np.float32)
                        optical = src_optical.read(window=Window(x, y, w, h)).transpose(1, 2, 0).astype(np.float32)
                    except ValueError:
                        pbar.update(1);
                        continue

                    if radar.shape[0] == 0 or radar.shape[1] == 0:
                        pbar.update(1);
                        continue

                    # 处理 NaN
                    radar = _process_nan(radar)
                    optical = _process_nan(optical)

                    # 标准化
                    radar = (radar - radar_mean) / np.clip(radar_std, 1e-6, None)
                    optical = (optical - optical_mean) / np.clip(optical_std, 1e-6, None)

                    spx = spx_full[y:y2, x:x2]

                    radar_tensor = torch.from_numpy(radar).permute(2, 0, 1).unsqueeze(0).to(device)
                    optical_tensor = torch.from_numpy(optical).permute(2, 0, 1).unsqueeze(0).to(device)
                    spx_tensor = torch.from_numpy(spx).unsqueeze(0).to(device)

                    logits, _ = model(radar_tensor, optical_tensor, spx_tensor)
                    if logits.shape[2:] != (h, w):
                        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
                    prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    if h == ks and w == ks:
                        wmap = gw_standard
                    else:
                        wmap = _gaussian_window(h, w, sigma_h=ks / 6.0, sigma_w=ks / 6.0)

                    for c in range(C):
                        prob_acc[c, y:y2, x:x2] += prob[c] * wmap
                    weight_acc[y:y2, x:x2] += wmap
                    pbar.update(1)
        pbar.close()

    print("正在归一化结果...")
    prob_acc /= np.clip(weight_acc, 1e-6, None)

    # ================= 强力掩膜策略 =================
    print("正在应用掩膜并修正边缘分类...")
    with rasterio.open(config.label_path) as src_label:
        mask_data = src_label.read(1)
        # 尺寸对齐
        if mask_data.shape != (H, W):
            temp_mask = np.zeros((H, W), dtype=mask_data.dtype)
            h_min = min(mask_data.shape[0], H)
            w_min = min(mask_data.shape[1], W)
            temp_mask[:h_min, :w_min] = mask_data[:h_min, :w_min]
            mask_data = temp_mask

        # 1. 定义有效区域 (非背景0 且 非NoData)
        valid_area = (mask_data != config.ignore_index) & (mask_data != 2147483647)

        # 2. 在有效区域内，强制屏蔽背景类(0)
        prob_acc[0][valid_area] = -1.0

        # 3. 生成预测
        pred_full = prob_acc.argmax(axis=0).astype(np.uint8)

        # 4. 在无效区域，强制设为0
        pred_full[~valid_area] = config.ignore_index

    # 统计
    valid_pred = pred_full[pred_full != config.ignore_index]
    unique, counts = np.unique(valid_pred, return_counts=True)
    print("\n最终预测结果类别统计:")
    for cls, cnt in zip(unique, counts):
        print(f"类别 {cls}: {cnt} 像素")

    # 保存
    tif_path = config.pred_save_path
    extensions_to_clean = ["", ".ovr", ".aux.xml", ".aux", ".tfw"]
    for ext in extensions_to_clean:
        target_file = tif_path + ext
        if os.path.exists(target_file):
            if not _safe_remove(target_file):
                print(f"【错误】无法删除文件: {target_file} (被占用)，请关闭相关软件！")
                return

    with rasterio.open(config.radar_path) as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.uint8, count=1, compress='lzw',
            nodata=config.ignore_index, width=W, height=H
        )
        with rasterio.open(tif_path, 'w', **profile) as dst:
            dst.write(pred_full.astype(np.uint8), 1)
            dst.write_colormap(1, {
                0: (0, 0, 0), 1: (255, 255, 0), 2: (255, 0, 0),
                3: (0, 255, 2), 4: (0, 0, 255), 5: (192, 192, 192)
            })
    print(f"预测结果已保存至 {tif_path}")


if __name__ == "__main__":
    predict()