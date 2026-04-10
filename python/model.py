"""
model_v4.py  —  CSIPose với TCN backbone (bidirectional)
═════════════════════════════════════════════════════════
Thay Transformer (478K params, cần >5K diverse samples) bằng
Temporal Convolutional Network (TCN) với dilated bidirectional conv.

Tại sao TCN phù hợp hơn với dữ liệu này:
  1. Inductive bias tốt hơn: CSI signal là local-temporal (motion trong 2s cửa sổ)
     → dilated conv capture multi-scale pattern mà không cần attention
  2. Ít tham số hơn (~160K vs 478K) → ít overfit trên ~400 train samples
  3. Stable training: không có attention collapse hay vanishing gradient
     như Transformer trên sequence ngắn
  4. Receptive field cố định và có thể tính toán trước:
     dilation=[1,2,4,8] với kernel=3 → RF = 2*(1+2+4+8)*(3-1) = 60 frames
     Bidirectional (non-causal) → RF hiệu dụng gấp đôi so với causal
     Window cố định 40 frame → không cần causal constraint

Anti mean-pose-collapse:
  - Temporal diff features: concat [x, Δx] → model thấy motion signal
  - Residual prediction: learn mean_pose + predict deviation
  - Center frame extraction: predict pose tại frame giữa window

Kiến trúc:
    Input (B, T=40, F_alive)
      ↓ Compute Δx (temporal diff), concat → (B, T, 2*F_alive)
      ↓ Linear projection (2*F_alive → C)
      ↓ TCN Block × 4 (dilation 1,2,4,8 — bidirectional, residual)
      ↓ Center frame extraction
      ↓ Pose head → residual (17,2) + learnable mean_pose (17,2)
      ↓ Vis head: Linear(C→17)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional Dilated Conv Block (non-causal — full window is available)
# ─────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """
    Residual TCN block (bidirectional — same padding):
        Conv1d → GroupNorm → GELU → Dropout
        Conv1d → GroupNorm → GELU → Dropout
        + residual (với 1×1 conv nếu channel thay đổi)

    Non-causal vì input là fixed window — toàn bộ temporal context đã có sẵn.
    Same padding giữ nguyên sequence length và tận dụng RF cả 2 hướng.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        # padding='same' equivalent: pad both sides equally
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.drop  = nn.Dropout(dropout)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.res(x)
        out = self.drop(F.gelu(self.norm1(self.conv1(x))))
        out = self.drop(F.gelu(self.norm2(self.conv2(out))))
        return out + residual


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class CSIPoseModelV4(nn.Module):
    """
    Args:
        feature_dim : số feature sau FeatureCleaner (thường ~546)
        tcn_channels: số channel trong TCN (default 64 — đủ capacity)
        dilations   : list dilation cho từng TCN block
        dropout     : dropout trong TCN block
    """
    def __init__(self,
                 feature_dim : int   = 546,
                 tcn_channels: int   = 64,
                 dilations   : list  = [1, 2, 4, 8],
                 dropout     : float = 0.2):
        super().__init__()

        # Input projection: 2*F_alive (original + temporal diff) → tcn_channels
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, tcn_channels),
            nn.LayerNorm(tcn_channels),
            nn.GELU(),
        )

        # TCN stack
        tcn_layers = []
        in_ch = tcn_channels
        for d in dilations:
            tcn_layers.append(TCNBlock(in_ch, tcn_channels,
                                       kernel_size=3,
                                       dilation=d,
                                       dropout=dropout))
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*tcn_layers)

        # Learnable mean pose — initialized to COCO standing pose center
        # Model predicts residuals from this, breaking mean-pose collapse
        default_mean = torch.tensor([
            [0.50,0.10],[0.49,0.09],[0.51,0.09],[0.47,0.10],[0.53,0.10],
            [0.44,0.28],[0.56,0.28],[0.40,0.42],[0.60,0.42],
            [0.38,0.55],[0.62,0.55],[0.46,0.58],[0.54,0.58],
            [0.46,0.75],[0.54,0.75],[0.46,0.92],[0.54,0.92],
        ], dtype=torch.float32)
        self.mean_pose = nn.Parameter(default_mean, requires_grad=True)

        # Pose head — outputs residual (17*2)
        self.pose_head = nn.Sequential(
            nn.Linear(tcn_channels, tcn_channels * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(tcn_channels * 2, 17 * 2),
        )

        # Visibility head (nhẹ hơn)
        self.vis_head = nn.Sequential(
            nn.Linear(tcn_channels, 32),
            nn.GELU(),
            nn.Linear(32, 17),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize pose head final layer to small values
        # so initial output ≈ mean_pose (residual ≈ 0)
        nn.init.normal_(self.pose_head[-1].weight, std=0.01)
        nn.init.zeros_(self.pose_head[-1].bias)

    def forward(self, x):
        # x: (B, T=40, F_alive)
        B, T, C_in = x.shape

        # Compute temporal difference features (motion signal)
        dx = x[:, 1:, :] - x[:, :-1, :]                # (B, T-1, F)
        dx = F.pad(dx, (0, 0, 1, 0))                    # (B, T, F) — pad first frame with 0
        x_aug = torch.cat([x, dx], dim=-1)               # (B, T, 2F)

        # Project features
        h = self.input_proj(x_aug)                 # (B, T, C)

        # TCN — cần (B, C, T)
        h = h.transpose(1, 2)                      # (B, C, T)
        h = self.tcn(h)                            # (B, C, T)
        h = h.transpose(1, 2)                      # (B, T, C)

        # Take center frame (pose target corresponds to window center)
        center = T // 2
        global_h = h[:, center, :]                 # (B, C)

        # Residual prediction + learnable mean pose
        residual  = self.pose_head(global_h).view(B, 17, 2)
        pose      = self.mean_pose.unsqueeze(0) + residual      # (B, 17, 2)
        vis_logit = self.vis_head(global_h)                      # (B, 17)

        return pose, vis_logit

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Receptive field calculator (debug utility)
# ─────────────────────────────────────────────────────────────────────────────

def compute_receptive_field(dilations: list, kernel_size: int = 3) -> int:
    """Tính receptive field của TCN stack."""
    rf = 1
    for d in dilations:
        rf += 2 * (kernel_size - 1) * d    # × 2 vì 2 conv/block
    return rf


if __name__ == '__main__':
    # Quick sanity check
    feature_dim = 546   # sau FeatureCleaner
    model = CSIPoseModelV4(feature_dim=feature_dim)
    print(f"Model params: {model.count_params():,}")
    print(f"Receptive field: {compute_receptive_field([1,2,4,8])} frames "
          f"(window=40 frames)")

    x = torch.randn(4, 40, feature_dim)
    pose, vis = model(x)
    print(f"Output: pose={tuple(pose.shape)}, vis={tuple(vis.shape)}")
    print("✅ Forward pass OK")