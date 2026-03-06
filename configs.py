# -------------------- configs.py --------------------
import os
from typing import Tuple, Optional

class Config:
    # 输入路径
    radar_path: str = r"D:\临洪ASSG\M学术科研1\原始图像\radar_bands_image.tif"
    optical_path: str = r"D:\临洪ASSG\M学术科研1\原始图像\remaining_bands_image.tif"
    label_path: str = r"D:\临洪ASSG\M学术科研1\原始图像\Land_use_classification.tif"

    # 输出路径
    output_dir: str = r"D:\临洪ASSG\M学术科研1\处理后图像"
    model_save_path: str = os.path.join(os.path.dirname(__file__), "checkpoints/best_model.pth")
    pred_save_path: str = r"D:\临洪ASSG\M学术科研1\预测结果\prediction.tif"

    # 指标/混淆矩阵
    metrics_dir: str = os.path.join(output_dir, "metrics")
    confusion_dir: str = os.path.join(output_dir, "confusion_matrices")

    # 训练参数
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 0 if os.name == 'nt' else 4
    seed: int = 42

    # 数据参数
    num_classes: int = 6
    valid_classes: Tuple[int, ...] = (1, 2, 3, 4, 5)
    ignore_index: int = 0
    block_size: int = 32
    radar_bands: int = 5
    optical_bands: int = 10

    # 预测/整图尺寸（已更新为你提供的真实尺寸）
    original_height: int = 623
    original_width: int = 463
    # 下面的块数只是估算，不影响程序运行
    num_blocks_x: int = 15
    total_blocks: int = 300

    # -------- ASPM --------
    aspm2_kernel_small = (3, 1)  # (kernel, dilation)
    aspm2_kernel_large = (3, 3)
    gate_down: int = 4
    gate_channels: Optional[int] = None  # None -> max(C_in//2, 16)
    gate_use_proxy: bool = True
    aspm2_dropout: float = 0.1

    # -------- 超像素（整图生成）+ 贝叶斯优化 ----------
    use_superpixels: bool = True
    superpixel_dir: str = os.path.join(output_dir, "superpixels")
    sp_method: str = "snic"

    # BO
    sp_use_bayes_opt: bool = True
    sp_bo_sample_blocks: int = 200
    sp_bo_iter: int = 25

    # 搜索范围
    sp_segments_min: int = 8
    sp_segments_max: int = 64
    sp_compact_min: float = 5.0
    sp_compact_max: float = 30.0

    # 评分项权重
    sp_score_lambda_segments: float = 0.002
    sp_score_w_purity: float = 1.0
    sp_score_use_edge: bool = True
    sp_score_w_edge: float = 0.2

    # BO 可视化/记录
    sp_log_trials: bool = True
    sp_log_probe_blocks: int = 5
    sp_viz_every_iter: bool = True

    # -------- LightAGSM（可学习阈值/稀疏率）--------
    agsm_hidden_dim: int = 64
    agsm_k: int = 6
    agsm_rho: float = 0.30
    agsm_temperature: float = 0.5
    agsm_dropedge_p: float = 0.2
    agsm_reg_lambda: float = 0.2

config = Config()
