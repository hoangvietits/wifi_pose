"""
model_v2.py  —  CSIPoseDualBranchNet
═════════════════════════════════════════════════════════════════════════
Mô hình được thiết kế theo ý tưởng từ:

  "DensePose From WiFi"  (Geng, Huang, De la Torre — arXiv:2301.00250)
  CMU, 2022.

─────────────────────────────────────────────────────────────────────────
Ý tưởng cốt lõi kế thừa từ bài báo:
  §3.1 Phase Sanitization  →  sanitize_phase_sequence() bên dưới
  §3.2 Modality Translation Network  →  hai nhánh encoder độc lập cho
        amplitude và phase, sau đó fuse thành feature map chung.
        Bài báo minh chứng (ablation, Table 5): dùng phase sanitized
        tăng AP thêm +0.8 điểm so với chỉ dùng amplitude.
  §3.3 Two-branch head  →  Giữ hai đầu ra (pose + visibility)

─────────────────────────────────────────────────────────────────────────
Thích ứng cho hệ thống ESP32 (thay vì 3×3 antenna pairs của bài báo):
  • Input  : (B, T=40, F_alive) cửa sổ thời gian 40 frame (~2s @ 20Hz)
             thay vì 5 frame tĩnh × 30 subcarrier × 3×3 pairs
  • Output : 17 keypoint (x,y) + visibility — không dự đoán UV dense map
  • 3 node ESP32 độc lập (không phải 3×3 cổng antenna)
  • Cross-modal attention thay cho concat+MLP thuần túy của bài báo
    (phù hợp hơn vì chúng ta có T=40 frames temporal context)
  • Bidirectional dilated TCN thay cho ResNet-FPN (không có spatial map)

─────────────────────────────────────────────────────────────────────────
Kiến trúc tổng quan:

  Input (B, T, F_alive)
       │
       ├── [split theo alive_mask & n_sub]
       │        │                   │
       │   amp_feat             ph_feat
       │  (B,T,n_amp)          (B,T,n_ph)
       │        │                   │
       │  AmpEncoder(MLP)     PhEncoder(MLP)
       │  (B,T,C)              (B,T,C)
       │        └───CrossModalAttn───┘
       │                   │
       │         FusionMLP → (B,T,C)
       │                   │
       │     TempDiff: [x, Δx] → diff_proj → (B,T,C)
       │                   │
       │       TCN × 4  (dilation 1,2,4,8, non-causal)
       │                   │
       │  center = h[:, T//2, :]   →  (B,C)
       │               ┌──┴──┐
       │         pose_head  vis_head
       │         (B,17,2)   (B,17)
       │
       └── pose = mean_pose (learnable) + pose_residual

─────────────────────────────────────────────────────────────────────────
Cách sử dụng trong train.py:

  from model_v2 import CSIPoseDualBranchNet, build_model_v2

  # Tạo model từ FeatureCleaner đã fit
  model = build_model_v2(feature_cleaner)
  # → tự động đọc n_alive, n_sub, alive_mask

─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ══════════════════════════════════════════════════════════════════════════════
# §3.1  Phase Sanitization  (tiền xử lý — không nằm trong model graph)
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_phase_sequence(phase: np.ndarray) -> np.ndarray:
    """
    Xử lý pha CSI theo Section 3.1 của "DensePose From WiFi" (Geng et al., 2022).

    Bài báo chứng minh 3 bước sanitization cần thiết để loại bỏ:
      - Phase wrapping (arctan range [-π, π])
      - Random temporal jitter giữa các consecutive sample
      - Global phase drift linear theo subcarrier index

    Bước 1: Unwrap theo trục subcarrier (loại discontinuity)
    Bước 2: Median filter theo trục thời gian (loại random spike)
    Bước 3: Linear fitting per-frame (loại slope drift)

    Args:
        phase : (T, n_sub)  — giá trị pha thô, mỗi hàng là 1 frame,
                              mỗi cột là 1 subcarrier.  Đơn vị: radian.
    Returns:
        (T, n_sub)  — pha đã sanitize, temporally consistent.

    Cách dùng trong pipeline:
        # Trong _build_flat_feature() hoặc align_csi.py:
        for node_id in raw:
            amp, ph = raw[node_id]
            ph_clean = sanitize_phase_sequence(ph.reshape(-1, n_sub))
            # ph_clean thay thế ph trước khi build flat feature
    """
    from scipy.signal import medfilt

    # Bước 1: Unwrap dọc theo chiều subcarrier (axis=1)
    unwrapped = np.unwrap(phase, axis=1)           # (T, n_sub)

    # Bước 2: Median filter 5-frame theo chiều thời gian (axis=0)
    # medfilt yêu cầu kernel_size phải là số lẻ
    filtered = medfilt(unwrapped, kernel_size=(5, 1))   # (T, n_sub)

    # Bước 3: Per-frame linear fitting để loại slope tuyến tính theo subcarrier
    #   α₁ = (φ[F-1] - φ[0]) / (2π·F)
    #   α₀ = mean(φ)
    #   φ̂[f] = φ[f] - (α₁·f + α₀)
    T_len, F_sub = filtered.shape
    f_idx = np.arange(F_sub, dtype=np.float64)
    sanitized = np.empty_like(filtered)

    for t in range(T_len):
        row  = filtered[t]
        alpha1 = (row[-1] - row[0]) / (2.0 * math.pi * F_sub)
        alpha0 = row.mean()
        sanitized[t] = row - (alpha1 * f_idx + alpha0)

    return sanitized.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class TCNBlock(nn.Module):
    """
    Bidirectional dilated TCN block (giống CSIPoseModelV4 — tái sử dụng).

    Non-causal (same padding) vì input là fixed window — receptive field:
        RF = 1 + Σ  2·(kernel_size−1)·dilation
    Với [1,2,4,8], kernel=3  →  RF = 61 frames > window T=40.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = dilation * (3 - 1) // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 3, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, dilation=dilation, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.drop  = nn.Dropout(dropout)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.norm1(self.conv1(x))))
        h = self.drop(F.gelu(self.norm2(self.conv2(h))))
        return h + self.res(x)


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention giữa amplitude và phase features.

    Cải tiến so với cơ chế của bài báo (simple concat + MLP):
      Bài báo: concat([A_feat, P_feat]) → MLP (one-way, lossy)
      Ở đây : A ← attend(A queries, P keys/values)   (amp xem pha)
               P ← attend(P queries, A keys/values)   (pha xem amp)
    Cả hai modality cập nhật lẫn nhau trước khi fuse.

    Lý do phù hợp hơn cho bài toán của chúng ta:
      - T=40 frames tạo ra temporal context phong phú cho attention
      - Pha và biên độ mang thông tin bổ sung nhau (ablation §4.6 trong
        bài báo, phase giúp tăng AP đều ở mọi threshold)
      - Cross-attention giúp model biết pha nào liên quan đến biên độ nào
        (ví dụ: subcarrier bị ảnh hưởng bởi chuyển động sẽ có Δphase tương ứng)

    Args:
        dim     : Chiều feature C
        n_heads : Số attention heads (nên là ước số của dim)
        dropout : Attention dropout
    """
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ph_to_amp = nn.MultiheadAttention(dim, n_heads,
                                               batch_first=True,
                                               dropout=dropout)
        self.amp_to_ph = nn.MultiheadAttention(dim, n_heads,
                                               batch_first=True,
                                               dropout=dropout)
        self.norm_amp = nn.LayerNorm(dim)
        self.norm_ph  = nn.LayerNorm(dim)

    def forward(self, amp: torch.Tensor,
                ph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            amp, ph : (B, T, C)
        Returns:
            amp_out, ph_out : (B, T, C)  — enriched by cross-modal context
        """
        # Amplitude queries phase context: "dựa trên pha, biên độ này nghĩa là gì?"
        amp_ctx, _ = self.amp_to_ph(amp, ph, ph)
        amp_out = self.norm_amp(amp + amp_ctx)

        # Phase queries amplitude context: "biên độ này ảnh hưởng đến pha như thế nào?"
        ph_ctx, _  = self.ph_to_amp(ph,  amp, amp)
        ph_out  = self.norm_ph(ph + ph_ctx)

        return amp_out, ph_out


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════

# COCO-17 standing pose prior (normalized [0,1])
_COCO_MEAN_POSE = [
    [0.50, 0.10], [0.49, 0.09], [0.51, 0.09],   # nose, l-eye, r-eye
    [0.47, 0.10], [0.53, 0.10],                  # l-ear, r-ear
    [0.44, 0.28], [0.56, 0.28],                  # l-sho, r-sho
    [0.40, 0.42], [0.60, 0.42],                  # l-elb, r-elb
    [0.38, 0.55], [0.62, 0.55],                  # l-wri, r-wri
    [0.46, 0.58], [0.54, 0.58],                  # l-hip, r-hip
    [0.46, 0.75], [0.54, 0.75],                  # l-kne, r-kne
    [0.46, 0.92], [0.54, 0.92],                  # l-ank, r-ank
]


class CSIPoseDualBranchNet(nn.Module):
    """
    WiFi CSI → 17-joint Human Pose Estimation.

    Inspired by Modality Translation Network (§3.2) trong "DensePose From WiFi":
      - Dual MLP encoders (amplitude branch + phase branch)
      - Feature fusion qua Cross-Modal Attention
      - Temporal context qua Bidirectional Dilated TCN
      - Residual pose prediction từ learnable mean_pose

    So sánh với CSIPoseModelV4:
    ┌────────────────────────┬──────────────────┬──────────────────┐
    │ Thành phần             │  V4 (hiện tại)   │  V2 (mô hình này)│
    ├────────────────────────┼──────────────────┼──────────────────┤
    │ Encoder                │ Single flat MLP  │ Dual amp/phase   │
    │ Phase vs Amp           │ Flat concat      │ Separate paths   │
    │ Modality fusion        │ N/A (flat)       │ Cross-attention  │
    │ Temporal diff          │ ✓ (Δx concat)    │ ✓ (giữ nguyên)  │
    │ Temporal model         │ TCN × 4          │ TCN × 4 (rộng hơn│
    │ Pose output            │ Residual         │ Residual         │
    │ Visibility             │ ✓                │ ✓                │
    │ Phase sanitization     │ ✗                │ utility function │
    └────────────────────────┴──────────────────┴──────────────────┘

    Args:
        n_alive    : Số feature sau FeatureCleaner (F_alive)
        n_sub      : Số subcarrier thực tế/node (từ FeatureCleaner.n_sub)
        alive_mask : (F_raw,) bool array từ FeatureCleaner.alive_mask.
                     Dùng để xác định feature nào là amplitude, phase nào.
                     Nếu None, giả định nửa đầu là amp.
        n_nodes    : Số ESP32 node (default=3)
        T          : Window size (default=40)
        C          : Hidden channel dimension (default=64)
        n_joints   : Số keypoint (default=17, COCO)
        dilations  : Dilation schedule cho TCN
        dropout    : Dropout rate
    """

    def __init__(
        self,
        n_alive    : int,
        n_sub      : int,
        alive_mask : Optional[np.ndarray] = None,
        n_nodes    : int   = 3,
        T          : int   = 40,
        C          : int   = 64,
        n_joints   : int   = 17,
        dilations  : List[int] = [1, 2, 4, 8],
        dropout    : float = 0.2,
    ):
        super().__init__()
        self.T        = T
        self.n_joints = n_joints
        self.C        = C

        # ── Xây dựng index amp/phase từ alive_mask + n_sub ────────────────────
        # Feature layout (F_raw = n_nodes × 2 × n_sub):
        #   [amp_node0 | ph_node0 | amp_node1 | ph_node1 | amp_node2 | ph_node2]
        #    <-n_sub-> <-n_sub->   ...
        F_raw = n_nodes * 2 * n_sub
        is_amp_raw = np.zeros(F_raw, dtype=bool)
        for node in range(n_nodes):
            start = node * 2 * n_sub
            is_amp_raw[start : start + n_sub] = True  # amplitude trước, phase sau

        if alive_mask is not None:
            # alive_mask có thể là (F_raw,) hoặc đã bị crop nếu F_raw != len(alive_mask)
            mask = alive_mask
            if len(mask) != F_raw:
                # Fallback nếu kích thước không khớp (e.g., old checkpoint với padding 128)
                is_amp_raw_compat = np.zeros(len(mask), dtype=bool)
                step = len(mask) // (n_nodes * 2)
                for node in range(n_nodes):
                    s = node * 2 * step
                    is_amp_raw_compat[s : s + step] = True
                is_amp_raw = is_amp_raw_compat

            alive_idx    = np.where(mask)[0]
            is_amp_alive = is_amp_raw[alive_idx]  # (F_alive,) bool
        else:
            # Fallback: nửa đầu là amp (tương thích với flat layout cũ)
            is_amp_alive = np.arange(n_alive) < (n_alive // 2)

        amp_positions = np.where(is_amp_alive)[0].astype(np.int64)
        ph_positions  = np.where(~is_amp_alive)[0].astype(np.int64)

        self.register_buffer('amp_idx', torch.tensor(amp_positions))
        self.register_buffer('ph_idx',  torch.tensor(ph_positions))

        n_amp = len(amp_positions)
        n_ph  = len(ph_positions)

        # ── 1. Amplitude branch encoder (§3.2 first MLP encoder) ─────────────
        self.amp_encoder = nn.Sequential(
            nn.Linear(n_amp, C * 2), nn.LayerNorm(C * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(C * 2, C),     nn.LayerNorm(C),      nn.GELU(),
        )

        # ── 2. Phase branch encoder (§3.2 second MLP encoder) ────────────────
        self.ph_encoder = nn.Sequential(
            nn.Linear(n_ph,  C * 2), nn.LayerNorm(C * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(C * 2, C),     nn.LayerNorm(C),      nn.GELU(),
        )

        # ── 3. Cross-modal attention (cải tiến so với bài báo) ────────────────
        self.cross_attn = CrossModalAttention(C, n_heads=4)

        # ── 4. Fusion MLP (§3.2: concatenate → MLP → feature map) ────────────
        self.fusion = nn.Sequential(
            nn.Linear(C * 2, C * 2), nn.LayerNorm(C * 2), nn.GELU(),
            nn.Linear(C * 2, C),     nn.LayerNorm(C),
        )

        # ── 5. Temporal differential projection ──────────────────────────────
        #   Giữ lại từ V4: cung cấp motion signal rõ ràng cho model
        self.diff_proj = nn.Sequential(
            nn.Linear(C * 2, C), nn.LayerNorm(C), nn.GELU(),
        )

        # ── 6. Bidirectional dilated TCN ──────────────────────────────────────
        C_tcn = C
        self.tcn_in = nn.Linear(C, C_tcn)
        self.tcn    = nn.ModuleList(
            [TCNBlock(C_tcn, C_tcn, dilation=d, dropout=dropout) for d in dilations]
        )
        self.tcn_out = nn.Sequential(
            nn.Linear(C_tcn, C), nn.LayerNorm(C), nn.GELU(),
        )

        # ── 7. Pose head — residual prediction từ learnable mean_pose ─────────
        default_mean = torch.tensor(_COCO_MEAN_POSE, dtype=torch.float32)
        self.mean_pose = nn.Parameter(default_mean, requires_grad=True)

        self.pose_head = nn.Sequential(
            nn.Linear(C, C * 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(C * 2, n_joints * 2),
        )

        # ── 8. Visibility head ────────────────────────────────────────────────
        self.vis_head = nn.Sequential(
            nn.Linear(C, C // 2), nn.GELU(),
            nn.Linear(C // 2, n_joints),
        )

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────

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
        # Residual head khởi tạo gần 0 → pose ban đầu ≈ mean_pose
        nn.init.normal_(self.pose_head[-1].weight, std=0.01)
        nn.init.zeros_(self.pose_head[-1].bias)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, T, F_alive)  — z-normalized features từ FeatureCleaner
                Layout gốc (trước alive_mask):
                  [amp_n0|ph_n0 | amp_n1|ph_n1 | amp_n2|ph_n2]

        Returns:
            pose      : (B, n_joints, 2)  — tọa độ keypoint normalized [0,1]
            vis_logit : (B, n_joints)     — logit visibility (sigmoid → prob)
        """
        B, T, _ = x.shape

        # ── Tách biên độ và pha theo precomputed index ────────────────────────
        amp = x[:, :, self.amp_idx]   # (B, T, n_amp)
        ph  = x[:, :, self.ph_idx]    # (B, T, n_ph)

        # ── Dual branch encoding (per-frame MLP, shared across T) ─────────────
        amp_feat = self.amp_encoder(amp)   # (B, T, C)
        ph_feat  = self.ph_encoder(ph)     # (B, T, C)

        # ── Cross-modal attention  ─────────────────────────────────────────────
        #   Bài báo: concat → MLP (one-pass)
        #   Ở đây : mỗi modality attend vào modality kia trước khi fuse
        amp_feat, ph_feat = self.cross_attn(amp_feat, ph_feat)

        # ── Fusion (tương tự §3.2: concat → MLP → feature representation) ─────
        fused = self.fusion(torch.cat([amp_feat, ph_feat], dim=-1))  # (B, T, C)

        # ── Temporal differential features (từ V4 — giữ lại vì hiệu quả) ──────
        delta = fused - torch.roll(fused, 1, dims=1)
        delta[:, 0] = 0
        h = self.diff_proj(torch.cat([fused, delta], dim=-1))        # (B, T, C)

        # ── Bidirectional dilated TCN ──────────────────────────────────────────
        h = self.tcn_in(h).transpose(1, 2)   # (B, C_tcn, T)
        for block in self.tcn:
            h = block(h)
        h = self.tcn_out(h.transpose(1, 2))  # (B, T, C)

        # ── Lấy frame giữa window (pose target = center frame) ────────────────
        center = h[:, T // 2]   # (B, C)

        # ── Pose head: residual prediction từ learnable mean_pose ─────────────
        pose_residual = self.pose_head(center).view(B, self.n_joints, 2)
        pose          = self.mean_pose.unsqueeze(0) + pose_residual   # (B, J, 2)

        # ── Visibility ─────────────────────────────────────────────────────────
        vis_logit = self.vis_head(center)   # (B, J)

        return pose, vis_logit

    # ─────────────────────────────────────────────────────────────────────────

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def receptive_field(dilations: List[int] = [1, 2, 4, 8],
                        kernel_size: int = 3) -> int:
        """Receptive field của TCN stack."""
        return 1 + sum(2 * (kernel_size - 1) * d for d in dilations)


# ══════════════════════════════════════════════════════════════════════════════
# Factory function — tạo model từ FeatureCleaner đã fit
# ══════════════════════════════════════════════════════════════════════════════

def build_model_v2(
    feature_cleaner,
    C          : int   = 64,
    n_nodes    : int   = 3,
    T          : int   = 40,
    dilations  : List[int] = [1, 2, 4, 8],
    dropout    : float = 0.2,
) -> CSIPoseDualBranchNet:
    """
    Tạo CSIPoseDualBranchNet từ FeatureCleaner đã được fit().

    Args:
        feature_cleaner : FeatureCleaner instance sau khi đã gọi fit()
                          (hoặc load từ checkpoint)
        C               : Hidden channel dim (default=64)
        n_nodes         : Số ESP32 node (default=3)
        T               : Window size (default=40)
        dilations       : TCN dilation schedule
        dropout         : Dropout rate

    Returns:
        CSIPoseDualBranchNet model (chưa train, ở chế độ training).

    Ví dụ:
        from csi_pose_dataset import FeatureCleaner, build_datasets
        from model_v2 import build_model_v2

        ds_train, ds_val, fc = build_datasets('saved2/')
        model = build_model_v2(fc)
        print(f"Model params: {model.count_params():,}")
        print(f"Receptive field: {model.receptive_field()} frames")
    """
    return CSIPoseDualBranchNet(
        n_alive    = feature_cleaner.n_alive,
        n_sub      = feature_cleaner.n_sub,
        alive_mask = feature_cleaner.alive_mask,
        n_nodes    = n_nodes,
        T          = T,
        C          = C,
        dilations  = dilations,
        dropout    = dropout,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Quick sanity check (python model_v2.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("─" * 60)
    print("CSIPoseDualBranchNet  —  sanity check")
    print("─" * 60)

    # Simulate FeatureCleaner output
    n_sub   = 55
    n_nodes = 3
    F_raw   = n_nodes * 2 * n_sub          # 330
    # Giả lập alive_mask: 5% kênh chết
    rng = np.random.default_rng(42)
    alive_mask = rng.random(F_raw) > 0.05  # ~314 alive

    n_alive = int(alive_mask.sum())
    print(f"  F_raw={F_raw}, n_alive={n_alive}, n_sub={n_sub}")

    model = CSIPoseDualBranchNet(
        n_alive=n_alive, n_sub=n_sub, alive_mask=alive_mask
    )
    print(f"  Tổng tham số  : {model.count_params():,}")
    print(f"  Receptive field: {model.receptive_field()} frames (window=40)")

    # Forward pass
    B, T = 4, 40
    x = torch.randn(B, T, n_alive)
    pose, vis = model(x)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  pose   : {tuple(pose.shape)}  (expected {B}×17×2)")
    print(f"  vis    : {tuple(vis.shape)}   (expected {B}×17)")

    # Kiểm tra amp/ph split
    n_amp = len(model.amp_idx)
    n_ph  = len(model.ph_idx)
    print(f"  amp_idx: {n_amp} features  |  ph_idx: {n_ph} features")
    assert n_amp + n_ph == n_alive, "amp + ph phải bằng n_alive"

    print("✅ Forward pass OK")
    print()

    # So sánh tham số với V4
    from model import CSIPoseModelV4
    v4 = CSIPoseModelV4(feature_dim=n_alive)
    print(f"  So sánh params:")
    print(f"    CSIPoseModelV4       : {v4.count_params():>8,}")
    print(f"    CSIPoseDualBranchNet : {model.count_params():>8,}")
