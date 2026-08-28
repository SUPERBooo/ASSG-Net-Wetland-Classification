from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import config


class ECABlock(nn.Module):
    """Efficient Channel Attention used after ASPM fusion."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2)  # [B,1,C]
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class HeterogeneityProxy(nn.Module):
    """Sobel edge magnitude + local variance, computed directly from the input feature map."""

    def __init__(self, variance_kernel: int = 3):
        super().__init__()
        if variance_kernel % 2 == 0:
            raise ValueError("variance_kernel must be odd")
        self.variance_kernel = variance_kernel

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # A channel-mean structural map avoids introducing extra learned parameters into the prior.
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-8)

        k = self.variance_kernel
        mean = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)
        mean_sq = F.avg_pool2d(gray.square(), kernel_size=k, stride=1, padding=k // 2)
        var = torch.clamp(mean_sq - mean.square(), min=0.0)
        return edge, var


class ASPM(nn.Module):
    """Physically-/heterogeneity-aware scale perception module from the manuscript.

    Two depthwise branches (dilation 1 and 3) are selected pixel-wise using a Softmax gate
    driven by Sobel edge and local variance proxies. The fused representation is projected,
    scaled by a learnable gamma, added to a residual projection, and refined by ECA.
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.proxy = HeterogeneityProxy(config.local_variance_kernel)

        self.small = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
        )
        self.large = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
        )

        g = config.aspm_gfe_channels
        self.gfe = nn.Sequential(
            nn.Conv2d(2, g, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4 if g % 4 == 0 else 1, g),
            nn.ReLU(inplace=True),
            nn.Conv2d(g, 2, kernel_size=1, bias=True),
        )

        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.eca = ECABlock(out_channels, config.eca_kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        edge, var = self.proxy(x)
        proxy = torch.cat([edge, var], dim=1)
        weights = torch.softmax(self.gfe(proxy), dim=1)
        w_small = weights[:, 0:1]
        w_large = weights[:, 1:2]

        f_small = self.small(x)
        f_large = self.large(x)
        fused = w_small * f_small + w_large * f_large

        out = self.residual(x) + self.gamma * self.project(fused)
        out = self.eca(out)
        aux = {
            "edge_proxy": edge.detach(),
            "variance_proxy": var.detach(),
            "w_small": w_small.detach(),
            "w_large": w_large.detach(),
        }
        return out, aux


class CNNBranch(nn.Module):
    def __init__(self, in_channels: int = 4, out_dim: int = 64):
        super().__init__()
        self.aspm1 = ASPM(in_channels, out_dim)
        self.aspm2 = ASPM(out_dim, out_dim)
        self.head = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        x, aux1 = self.aspm1(x)
        x, aux2 = self.aspm2(x)
        return self.head(x), {"aspm1": aux1, "aspm2": aux2}


def mutual_knn_graph(positions: torch.Tensor, k: int) -> torch.Tensor:
    """Build a directed mutual-kNN candidate graph from superpixel centroids.

    positions: [N,2]. For every mutual pair i<->j, both directed edges are returned.
    """
    n = positions.size(0)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=positions.device)

    k = min(k, n - 1)
    dist = torch.cdist(positions, positions, p=2)
    dist.fill_diagonal_(float("inf"))
    knn = torch.topk(dist, k=k, largest=False, dim=1).indices

    nbr = torch.zeros((n, n), dtype=torch.bool, device=positions.device)
    rows = torch.arange(n, device=positions.device).unsqueeze(1).expand(-1, k)
    nbr[rows, knn] = True
    mutual = nbr & nbr.t()

    src, dst = torch.where(mutual)
    return torch.stack([src, dst], dim=0)


def superpixel_pool(
    radar: torch.Tensor,
    sp_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool pixel features to superpixel nodes and compute normalized centroids.

    Returns
    -------
    node_features : [N,C]
    positions : [N,2] in [-1,1]
    inverse : [H*W], mapping every pixel to a contiguous node id
    """
    c, h, w = radar.shape
    flat_sp = sp_map.reshape(-1)
    _, inverse = torch.unique(flat_sp, sorted=True, return_inverse=True)
    n = int(inverse.max().item()) + 1

    pixel_feat = radar.permute(1, 2, 0).reshape(-1, c)
    node_sum = torch.zeros((n, c), device=radar.device, dtype=radar.dtype)
    node_sum.index_add_(0, inverse, pixel_feat)

    count = torch.zeros((n,), device=radar.device, dtype=radar.dtype)
    count.index_add_(0, inverse, torch.ones_like(inverse, dtype=radar.dtype))
    node_feat = node_sum / count.clamp_min(1.0).unsqueeze(1)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=radar.device, dtype=radar.dtype),
        torch.arange(w, device=radar.device, dtype=radar.dtype),
        indexing="ij",
    )
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    coord_sum = torch.zeros((n, 2), device=radar.device, dtype=radar.dtype)
    coord_sum.index_add_(0, inverse, coords)
    centers = coord_sum / count.clamp_min(1.0).unsqueeze(1)
    if h > 1:
        centers[:, 0] = centers[:, 0] / (h - 1) * 2.0 - 1.0
    if w > 1:
        centers[:, 1] = centers[:, 1] / (w - 1) * 2.0 - 1.0
    return node_feat, centers, inverse


class AGSMBlock(nn.Module):
    """Attention-based adaptive graph sparsity block.

    q/k similarity -> sigmoid topology gate with learnable threshold tau and temperature T
    -> DropEdge -> weighted V aggregation -> residual -> LayerNorm/ReLU/Dropout.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.tau = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.graph_dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if edge_index.numel() == 0:
            out = self.dropout(F.relu(self.norm(x)))
            zero = x.new_zeros(())
            return out, zero, {
                "mean_gate": zero.detach(),
                "retained_ratio": zero.detach(),
                "tau": self.tau.detach(),
            }

        src, dst = edge_index
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        score = (q[src] * k[dst]).sum(dim=1) / math.sqrt(self.dim)
        gate = torch.sigmoid((score - self.tau) / config.agsm_temperature)

        # Differentiable density target (anti-over-smoothing sparsity constraint).
        sparse_loss = (gate.mean() - config.sparsity_target).square()

        effective_gate = gate
        if self.training and config.agsm_dropedge > 0:
            keep = torch.rand_like(gate) >= config.agsm_dropedge
            # Prevent an accidental all-edge drop in a tiny graph.
            if not bool(keep.any()):
                keep[torch.argmax(gate)] = True
            effective_gate = gate * keep.to(gate.dtype)

        msg = v[src] * effective_gate.unsqueeze(1)
        agg = torch.zeros_like(v)
        agg.index_add_(0, dst, msg)
        denom = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, effective_gate)
        agg = agg / denom.clamp_min(1e-6).unsqueeze(1)

        out = self.norm(x + agg)
        out = self.dropout(F.relu(out))

        diagnostics = {
            "mean_gate": gate.mean().detach(),
            "retained_ratio": (gate >= 0.5).float().mean().detach(),
            "tau": self.tau.detach(),
        }
        return out, sparse_loss, diagnostics


class GCNBranch(nn.Module):
    """SNIC-superpixel graph branch using centroid mutual-kNN + stacked AGSM blocks."""

    def __init__(self, in_channels: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([AGSMBlock(hidden_dim) for _ in range(config.agsm_depth)])

    def _forward_one(
        self,
        radar: torch.Tensor,
        sp_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        c, h, w = radar.shape
        node_feat, centers, inverse = superpixel_pool(radar, sp_map)
        x = self.input_proj(node_feat)
        edge_index = mutual_knn_graph(centers, config.knn_k)

        sparse_losses: List[torch.Tensor] = []
        diagnostics: List[Dict[str, torch.Tensor]] = []
        for block in self.blocks:
            x, sparse_loss, diag = block(x, edge_index)
            sparse_losses.append(sparse_loss)
            diagnostics.append(diag)

        pixel_feat = x[inverse].reshape(h, w, -1).permute(2, 0, 1).contiguous()
        sparse_loss = torch.stack(sparse_losses).mean() if sparse_losses else radar.new_zeros(())
        return pixel_feat, sparse_loss, diagnostics

    def forward(
        self,
        radar: torch.Tensor,
        superpixel_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Dict[str, torch.Tensor]]]]:
        outputs = []
        sparse_losses = []
        batch_diagnostics = []
        for b in range(radar.size(0)):
            feat, loss, diag = self._forward_one(radar[b], superpixel_maps[b])
            outputs.append(feat)
            sparse_losses.append(loss)
            batch_diagnostics.append(diag)
        return (
            torch.stack(outputs, dim=0),
            torch.stack(sparse_losses).mean(),
            batch_diagnostics,
        )


class GatedFusionModule(nn.Module):
    """Dual-level modality gate from the manuscript figure.

    Global channel gate m_c: Cx1x1
    Local spatial gate m_s: CxHxW
    Final m = sigmoid((w_c * m_c_logit + w_s * m_s_logit) / T)
    F = m * F_cnn + (1-m) * F_gcn
    followed by depthwise + pointwise refinement.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        joint = channels * 2
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(joint, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(joint, channels, kernel_size=1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

        self.wc = nn.Parameter(torch.tensor(1.0))
        self.ws = nn.Parameter(torch.tensor(1.0))
        # Positive learnable shared temperature. Softplus(logit) is used in forward.
        init_t = max(float(config.gfm_temperature), 1e-3)
        self.temperature_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(init_t))))

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        f_cnn: torch.Tensor,
        f_gcn: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        joint = torch.cat([f_cnn, f_gcn], dim=1)
        mc_logit = self.channel_gate(joint)
        ms_logit = self.spatial_gate(joint)
        temperature = F.softplus(self.temperature_raw) + 1e-4
        mask = torch.sigmoid((self.wc * mc_logit + self.ws * ms_logit) / temperature)

        fused = mask * f_cnn + (1.0 - mask) * f_gcn
        out = self.refine(fused)
        aux = {
            "fusion_mask": mask.detach(),
            "cnn_weight_mean": mask.mean().detach(),
            "gcn_weight_mean": (1.0 - mask).mean().detach(),
            "temperature": temperature.detach(),
        }
        return out, aux


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch(config.optical_bands, config.feature_dim)
        self.gcn = GCNBranch(config.radar_bands, config.feature_dim)
        self.gfm = GatedFusionModule(config.feature_dim)
        self.classifier = nn.Sequential(
            nn.Conv2d(config.feature_dim, config.feature_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(config.feature_dim, config.num_classes, kernel_size=1),
        )

    def forward(
        self,
        radar: torch.Tensor,
        optical: torch.Tensor,
        superpixel_maps: torch.Tensor,
        return_aux: bool = False,
    ):
        cnn_feat, aspm_aux = self.cnn(optical)
        gcn_feat, sparse_loss, graph_aux = self.gcn(radar, superpixel_maps)
        fused, gfm_aux = self.gfm(cnn_feat, gcn_feat)
        logits = self.classifier(fused)

        if not return_aux:
            return logits, sparse_loss

        aux = {
            "aspm": aspm_aux,
            "graph": graph_aux,
            "gfm": gfm_aux,
        }
        return logits, sparse_loss, aux


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    b, h, w = 2, config.block_size, config.block_size
    radar = torch.randn(b, config.radar_bands, h, w)
    optical = torch.randn(b, config.optical_bands, h, w)
    # Simple synthetic superpixel map for shape testing.
    sp = torch.arange(h * w).reshape(h, w) // 16
    sp = sp.unsqueeze(0).repeat(b, 1, 1)

    model = FusionModel()
    logits, sparse_loss, aux = model(radar, optical, sp, return_aux=True)
    print("logits:", logits.shape)
    print("sparse_loss:", float(sparse_loss.detach()))
    print("parameters:", count_parameters(model), f"({count_parameters(model)/1e6:.3f} M)")
