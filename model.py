# -------------------- model.py（ASPM + AGSM + 轻量门控融合，含短残差gamma 与尺寸对齐修复） --------------------
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_scatter import scatter_add
from configs import config

def GN(c: int) -> nn.GroupNorm:
    groups = 32 if c >= 32 else max(1, c // 2)
    return nn.GroupNorm(num_groups=groups, num_channels=c)

# ========== 两分支 ASPM（像素级门控，加入短残差 + gamma） ==========
class TwoBranchASPM_PixelGate(nn.Module):
    """
    两分支 ASPM（像素级门控） + 短残差（带可学习缩放 gamma）
    - 残差相加位置：PW(1×1)→GN→ReLU 之后，ECA/Dropout 之前
    - 若 C_in!=C_out，用 1×1 Conv(+GN) 做投影残差
    - gamma 初始化为 0，使模块初始近似恒等映射，更稳
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        # 小/大感受野的 depthwise 卷积
        k_s, d_s = config.aspm2_kernel_small
        pad_s = ((k_s - 1) // 2) * d_s
        self.dw_small = nn.Conv2d(in_channels, in_channels, k_s,
                                  padding=pad_s, dilation=d_s,
                                  groups=in_channels, bias=False)

        k_l, d_l = config.aspm2_kernel_large
        pad_l = ((k_l - 1) // 2) * d_l
        self.dw_large = nn.Conv2d(in_channels, in_channels, k_l,
                                  padding=pad_l, dilation=d_l,
                                  groups=in_channels, bias=False)

        # 像素级门控：下采样提取门控底座
        Cg = config.gate_channels or max(in_channels // 2, 16)
        stride = config.gate_down
        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels, Cg, kernel_size=3, stride=stride, padding=1, bias=False),
            GN(Cg),
            nn.ReLU(inplace=True),
        )

        # 门控 logits（可拼接代理特征 edge/var）
        proxy_in = Cg + (2 if config.gate_use_proxy else 0)
        self.g2 = nn.Sequential(
            nn.Conv2d(proxy_in, Cg, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cg, 2, kernel_size=1, bias=True)
        )

        self.post_gn = GN(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_gn = GN(out_channels)
        self.dropout = nn.Dropout(config.aspm2_dropout)

        # 轻量通道注意（ECA）
        self.eca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # ===== 残差分支与可学习缩放 =====
        if in_channels != out_channels:
            self.res_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                GN(out_channels)
            )
        else:
            self.res_proj = nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始为 0

    # ------ 工具：Sobel 与 局部方差（用于代理特征） ------
    @staticmethod
    def _sobel(x: torch.Tensor) -> torch.Tensor:
        weight_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=x.device, dtype=x.dtype)
        weight_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, weight_x, padding=1)
        gy = F.conv2d(x, weight_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    @staticmethod
    def _local_var(x: torch.Tensor, k=5) -> torch.Tensor:
        mean = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)
        mean2 = F.avg_pool2d(x * x, kernel_size=k, stride=1, padding=k//2)
        return torch.relu(mean2 - mean * mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两个感受野
        y_small = self.dw_small(x)
        y_large = self.dw_large(x)

        # 门控底座（下采样）
        g1 = self.g1(x)  # [B, Cg, Hg, Wg]

        # —— 代理分支尺寸对齐到 g1 并拼接 ——
        if config.gate_use_proxy:
            m = x.mean(dim=1, keepdim=True)
            m_low = F.avg_pool2d(m, kernel_size=config.gate_down, stride=config.gate_down, ceil_mode=True)
            edge = self._sobel(m_low)
            var = self._local_var(m_low, k=5)
            if edge.shape[-2:] != g1.shape[-2:]:
                edge = F.interpolate(edge, size=g1.shape[-2:], mode='bilinear', align_corners=False)
                var  = F.interpolate(var,  size=g1.shape[-2:], mode='bilinear', align_corners=False)
            g1 = torch.cat([g1, edge, var], dim=1)

        # 低分辨率 logits → 上采样 → Softmax 得像素级权重
        logits_low = self.g2(g1)
        logits = F.interpolate(logits_low, size=x.shape[-2:], mode='bilinear', align_corners=False)
        w_map = torch.softmax(logits, dim=1)  # [B,2,H,W]

        # 加权融合
        y = w_map[:, 0:1] * y_small + w_map[:, 1:2] * y_large
        y = self.post_gn(y)
        y = F.relu(y, inplace=True)

        # 1×1 到 out_ch
        out = self.pw(y)
        out = self.pw_gn(out)
        out = F.relu(out, inplace=True)

        # ===== 短残差 + gamma（ECA/Dropout 之前）=====
        res = self.res_proj(x)
        out = res + self.gamma * out

        # ECA
        y_avg = F.adaptive_avg_pool2d(out, 1)                         # [B,C,1,1]
        y_gate = self.eca(y_avg.squeeze(-1).transpose(-1, -2))         # [B,1,C]
        y_gate = torch.sigmoid(y_gate.transpose(-1, -2).unsqueeze(-1)) # → [B,C,1,1]
        out = out * y_gate

        return self.dropout(out)

class ASPM(TwoBranchASPM_PixelGate):
    pass

# ========== 最优版 LightAGSM ==========
class LightAGSM(nn.Module):
    def __init__(self, in_dim, hidden_dim, k=8, rho=0.3, temperature=0.7, dropedge_p=0.1):
        super().__init__()
        self.k = k
        self.rho = rho
        self.temperature = temperature
        self.dropedge_p = dropedge_p

        self.q1 = nn.Linear(in_dim, hidden_dim)
        self.k1 = nn.Linear(in_dim, hidden_dim)
        self.v1 = nn.Linear(in_dim, hidden_dim)
        self.res1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.do1 = nn.Dropout(0.2)

        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.k2 = nn.Linear(hidden_dim, hidden_dim)
        self.v2 = nn.Linear(hidden_dim, hidden_dim)
        self.res2 = nn.Identity()
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.do2 = nn.Dropout(0.2)

        # 将阈值初值置为负，利于早期稀疏
        self.tau = nn.Parameter(torch.tensor(-0.5))

    @staticmethod
    def _build_mutual_knn(pos: torch.Tensor, k: int):
        N = pos.size(0)
        if N <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=pos.device)
        dist2 = torch.cdist(pos, pos, p=2)
        diag = torch.arange(N, device=pos.device)
        dist2[diag, diag] = float('inf')
        k_eff = min(k, max(N - 1, 1))
        knn = torch.topk(-dist2, k=k_eff, dim=1).indices
        row = torch.arange(N, device=pos.device).unsqueeze(1).repeat(1, k_eff).reshape(-1)
        col = knn.reshape(-1)
        A = torch.zeros((N, N), device=pos.device, dtype=torch.bool)
        A[row, col] = True
        mutual = A & A.t()
        idx = mutual.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=pos.device)
        edge_index = idx.t().contiguous()
        return edge_index

    def _attention_layer(self, x, edge_index, q_lin, k_lin, v_lin, res_lin, bn, do, training=True):
        if edge_index.numel() == 0:
            out = res_lin(x)
            out = F.relu(bn(out), inplace=True)
            out = do(out)
            return out, torch.zeros(0, device=x.device)
        row, col = edge_index
        q = q_lin(x); k = k_lin(x); v = v_lin(x)
        d = q.size(-1)
        e = (q[row] * k[col]).sum(-1) / (d ** 0.5)
        g = torch.sigmoid((e - self.tau) / max(self.temperature, 1e-6))
        if training and self.dropedge_p > 0:
            keep = (torch.rand_like(g) > self.dropedge_p).float()
            g = g * keep
        den = scatter_add(g, row, dim=0, dim_size=x.size(0))[row].clamp_min(1e-6)
        alpha = g / den
        msg = v[col] * alpha.unsqueeze(-1)
        agg = scatter_add(msg, row, dim=0, dim_size=x.size(0))
        out = res_lin(x) + agg
        out = F.relu(bn(out), inplace=True)
        out = do(out)
        return out, g

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        edge_index = self._build_mutual_knn(pos, self.k)
        x1, g1 = self._attention_layer(x, edge_index, self.q1, self.k1, self.v1, self.res1, self.bn1, self.do1, self.training)
        x2, g2 = self._attention_layer(x1, edge_index, self.q2, self.k2, self.v2, self.res2, self.bn2, self.do2, self.training)
        g_all = torch.cat([g1, g2], dim=0) if g1.numel() > 0 else g2
        return x2, g_all

    def sparse_reg_loss(self, g: torch.Tensor):
        if g.numel() == 0:
            return torch.tensor(0.0, device=self.tau.device)
        return torch.abs(g.mean() - self.rho)

# ========== CNN 分支 ==========
class CNNBranch(nn.Module):
    def __init__(self, in_channels: int = 10, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ASPM(in_channels, 64),
            ASPM(64, 64),
            nn.Conv2d(64, out_dim, 3, padding=1, bias=False),
            GN(out_dim),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ========== 基于超像素图的 GCN 分支（集成 LightAGSM） ==========
class GCNBranch(nn.Module):
    def __init__(self, in_channels: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.agsm = LightAGSM(
            in_dim=in_channels, hidden_dim=hidden_dim,
            k=config.agsm_k, rho=config.agsm_rho,
            temperature=config.agsm_temperature,
            dropedge_p=config.agsm_dropedge_p
        )

    @staticmethod
    def _cluster_mean(x_hw_c: torch.Tensor, spx: torch.Tensor, num_nodes: int):
        C = x_hw_c.size(1)
        out = torch.zeros((num_nodes, C), device=x_hw_c.device, dtype=x_hw_c.dtype)
        out.index_add_(0, spx, x_hw_c)
        counts = torch.bincount(spx, minlength=num_nodes).clamp(min=1).unsqueeze(1).to(x_hw_c.dtype)
        out = out / counts
        return out

    @staticmethod
    def _cluster_centroid(spx: torch.Tensor, H: int, W: int, num_nodes: int):
        device = spx.device
        idx = spx.view(-1)
        ys = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).reshape(-1).to(torch.float32)
        xs = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).reshape(-1).to(torch.float32)
        sum_y = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        sum_x = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        sum_y.index_add_(0, idx, ys)
        sum_x.index_add_(0, idx, xs)
        cnts = torch.bincount(idx, minlength=num_nodes).clamp(min=1).to(torch.float32)
        cy = sum_y / cnts
        cx = sum_x / cnts
        cx = cx / max(W - 1, 1)
        cy = cy / max(H - 1, 1)
        pos = torch.stack([cx, cy], dim=1)
        return pos

    @staticmethod
    def _broadcast_to_pixels(node_feat: torch.Tensor, spx: torch.Tensor, H: int, W: int):
        x_hw_c = node_feat[spx]                  # [H*W, C]
        x = x_hw_c.view(H, W, -1).permute(2, 0, 1).contiguous()  # [C,H,W]
        return x

    def forward(self, x: torch.Tensor, spx_map: torch.Tensor):
        B, C, H, W = x.shape
        outputs = []
        reg_losses = []
        for b in range(B):
            xb = x[b]            # [Cr,H,W]
            spx = spx_map[b]     # [H,W]
            spx_ids = spx.view(-1)
            unique_ids, new_ids = torch.unique(spx_ids, sorted=True, return_inverse=True)
            num_nodes = unique_ids.numel()

            x_hw_c = xb.permute(1, 2, 0).reshape(-1, C)  # [H*W, Cr]
            node_feat = self._cluster_mean(x_hw_c, new_ids, num_nodes)   # [N, Cr]
            pos = self._cluster_centroid(new_ids, H, W, num_nodes)       # [N,2]

            node_out, g_all = self.agsm(node_feat, pos)                  # [N, hidden], [E_all]
            reg_loss = self.agsm.sparse_reg_loss(g_all)
            reg_losses.append(reg_loss)

            feat_hw = self._broadcast_to_pixels(node_out, new_ids, H, W)  # [hidden,H,W]
            outputs.append(feat_hw.unsqueeze(0))

        feat = torch.cat(outputs, dim=0)  # [B, hidden, H, W]
        reg_loss_total = torch.stack(reg_losses).mean() if len(reg_losses) > 0 else torch.tensor(0.0, device=x.device)
        return feat, reg_loss_total

# ========== 轻量门控融合（替代原自注意力融合） ==========
class GatedFusion(nn.Module):
    def __init__(self, in_dim=64, reduction=4, temperature=1.0):
        super().__init__()
        c = in_dim
        r = max(1, reduction)

        # 通道门（SE 风格）
        self.ch_gate = nn.Sequential(
            nn.Conv2d(2*c, c//r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//r, c, 1, bias=True)
        )
        # 空间门（轻量：1x1 降维 + 深度卷积）
        self.spa_pre = nn.Conv2d(2*c, c, 1, bias=False)
        self.spa_dw  = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=True)

        # 门组合的可学习权重和温度
        self.wc  = nn.Parameter(torch.tensor(1.0))
        self.ws  = nn.Parameter(torch.tensor(1.0))
        self.temp = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

        # 融合后细化（Depthwise + Pointwise）
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),  # DWConv
            nn.Conv2d(c, c, 1, bias=False),                       # PWConv
            GN(c),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat_gcn: torch.Tensor, feat_cnn: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_gcn, feat_cnn], dim=1)  # [B,2C,H,W]

        # 通道门
        gap = F.adaptive_avg_pool2d(x, 1)                 # [B,2C,1,1]
        m_c = torch.sigmoid(self.ch_gate(gap) / torch.clamp(self.temp, min=1e-6))  # [B,C,1,1]

        # 空间门
        s  = self.spa_pre(x)                               # [B,C,H,W]
        m_s = torch.sigmoid(self.spa_dw(s) / torch.clamp(self.temp, min=1e-6))     # [B,C,H,W]

        # 组合门并融合
        m = torch.sigmoid(self.wc * (m_c - 0.5) + self.ws * (m_s - 0.5))           # [B,C,H,W]
        fused = m * feat_gcn + (1.0 - m) * feat_cnn                                 # [B,C,H,W]

        # 细化
        out = self.refine(fused)
        return out

# ========== 融合与分类 ==========
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNBranch(in_channels=config.radar_bands, hidden_dim=config.agsm_hidden_dim)
        self.cnn = CNNBranch(in_channels=config.optical_bands, out_dim=config.agsm_hidden_dim)
        self.attention_fusion = GatedFusion(in_dim=config.agsm_hidden_dim, reduction=4, temperature=1.0)
        self.classifier = nn.Sequential(
            nn.Conv2d(config.agsm_hidden_dim, config.agsm_hidden_dim, 3, padding=1, bias=False),
            GN(config.agsm_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(config.agsm_hidden_dim, config.num_classes, 1)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, radar: torch.Tensor, optical: torch.Tensor, spx_map: torch.Tensor):
        gcn_feat, reg_loss = self.gcn(radar, spx_map)  # [B,64,H,W], scalar
        cnn_feat = self.cnn(optical)                   # [B,64,H,W]
        fused_feat = self.attention_fusion(gcn_feat, cnn_feat)
        logits = self.classifier(fused_feat)
        return logits, reg_loss

if __name__ == "__main__":
    B = 2
    dummy_radar = torch.randn(B, config.radar_bands, config.block_size, config.block_size)
    dummy_optical = torch.randn(B, config.optical_bands, config.block_size, config.block_size)
    spx = torch.zeros(B, config.block_size, config.block_size, dtype=torch.long)
    spx[:, :, :16] = 0
    spx[:, :, 16:] = 1
    model = FusionModel()
    out, reg = model(dummy_radar, dummy_optical, spx)
    print("输出维度:", out.shape, "正则:", float(reg))
