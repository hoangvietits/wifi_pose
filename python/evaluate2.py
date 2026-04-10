"""
eval_v2.py  —  Offline Evaluation & Report Charts  (CSIPoseModel)
══════════════════════════════════════════════════════════════════════════
Chạy sau khi training xong để tạo bộ biểu đồ đầy đủ cho báo cáo đồ án.

Biểu đồ được tạo:
  1. loss_curves.png         — Train/Val loss (từ file log nếu có)
  2. skeleton_comparison.png — GT vs Prediction (COCO-17 chuẩn)
  3. joint_heatmap.png       — MPJPE từng joint + bản đồ cơ thể
  4. visibility_matrix.png   — Confusion matrix + per-joint accuracy
  5. pck_curve.png           — PCK @ nhiều ngưỡng threshold
  6. error_distribution.png  — Phân phối lỗi MPJPE toàn bộ dataset
  7. report_summary.png      — Tổng hợp tất cả cho báo cáo

COCO-17 keypoints:
  0  nose     1  left_eye    2  right_eye   3  left_ear   4  right_ear
  5  left_shoulder  6  right_shoulder  7  left_elbow  8  right_elbow
  9  left_wrist  10 right_wrist  11 left_hip  12 right_hip
  13 left_knee  14 right_knee  15 left_ankle  16 right_ankle

Cách dùng:
  python eval_v2.py \\
      --checkpoint checkpoints/best_csi_pose_model.pt \\
      --data-dir   saved2/ \\
      --val-sessions 3 \\
      --out        eval_results/

  # Dùng toàn bộ dataset (không phân session):
  python eval_v2.py --checkpoint checkpoints/best_csi_pose_model.pt \\
                    --data-dir saved2/ --all-sessions
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from csi_pose_dataset import build_datasets, FeatureCleaner
from model import CSIPoseModelV4

# ══════════════════════════════════════════════════════════════════
# COCO-17 metadata  (dùng chung toàn file)
# ══════════════════════════════════════════════════════════════════
JOINT_NAMES = [
    'Nose', 'L.Eye', 'R.Eye', 'L.Ear', 'R.Ear',
    'L.Shoulder', 'R.Shoulder', 'L.Elbow', 'R.Elbow',
    'L.Wrist', 'R.Wrist', 'L.Hip', 'R.Hip',
    'L.Knee', 'R.Knee', 'L.Ankle', 'R.Ankle',
]

JOINT_COLORS = [
    '#FF6B6B','#FF6B6B','#FF6B6B','#FF6B6B','#FF6B6B',   # head
    '#4ECDC4','#45B7D1',                                    # shoulders
    '#96CEB4','#96CEB4',                                    # elbows
    '#FFEAA7','#FFEAA7',                                    # wrists
    '#DDA0DD','#DA70D6',                                    # hips
    '#87CEEB','#6495ED',                                    # knees
    '#FFA07A','#FF8C69',                                    # ankles
]

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# Vị trí chuẩn cho body map
BODY_POS = np.array([
    [0.50,0.06],[0.44,0.04],[0.56,0.04],[0.38,0.06],[0.62,0.06],
    [0.33,0.22],[0.67,0.22],[0.24,0.38],[0.76,0.38],
    [0.18,0.52],[0.82,0.52],[0.38,0.56],[0.62,0.56],
    [0.35,0.74],[0.65,0.74],[0.33,0.92],[0.67,0.92],
], dtype=np.float32)

DARK_BG  = '#0D0D0D'
PANEL_BG = '#1A1A2E'
TEXT_CLR = '#DDDDDD'
GRID_CLR = '#2A2A4A'


# ══════════════════════════════════════════════════════════════════
# Skeleton drawing helper
# ══════════════════════════════════════════════════════════════════
def _draw_skel(ax, kps, vis=None, is_pred=False, err=None, title=''):
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.10, -0.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, color=TEXT_CLR, fontsize=8, fontweight='bold', pad=3)

    v = np.ones(17) if vis is None else np.array(vis)
    color_bone = '#FF4444' if is_pred else '#44FF88'

    for a, b in COCO_SKELETON:
        if v[a] > 0.3 and v[b] > 0.3:
            ax.plot([kps[a,0], kps[b,0]], [kps[a,1], kps[b,1]],
                    color=color_bone, lw=2.2, alpha=0.85,
                    solid_capstyle='round')

    cmap_e = plt.cm.RdYlGn_r
    norm_e = Normalize(0, 0.15)
    for j in range(17):
        if v[j] < 0.1: continue
        c = cmap_e(norm_e(err[j])) if err is not None else JOINT_COLORS[j]
        ax.scatter(kps[j,0], kps[j,1], c=[c], s=55, zorder=5,
                   edgecolors='white', linewidths=0.5,
                   alpha=float(np.clip(v[j], 0.4, 1.0)))


# ══════════════════════════════════════════════════════════════════
# Inference — thu thập predictions toàn bộ val set
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_gt, all_pred, all_vgt, all_vpred = [], [], [], []
    for x, y_true, v_true in loader:
        x = x.to(device)
        y_pred, v_logits = model(x)
        v_pred = torch.sigmoid(v_logits)
        all_gt.append(y_true.numpy())
        all_pred.append(y_pred.cpu().numpy())
        all_vgt.append(v_true.numpy())
        all_vpred.append(v_pred.cpu().numpy())

    gt    = np.concatenate(all_gt,    axis=0)   # (N,17,2)
    pred  = np.concatenate(all_pred,  axis=0)
    vgt   = np.concatenate(all_vgt,   axis=0)   # (N,17)
    vpred = np.concatenate(all_vpred, axis=0)

    # Per-sample per-joint Euclidean error
    err   = np.linalg.norm(gt - pred, axis=-1)  # (N,17)
    return gt, pred, vgt, vpred, err


# ══════════════════════════════════════════════════════════════════
# 1. Skeleton comparison
# ══════════════════════════════════════════════════════════════════
def plot_skeleton_comparison(gt, pred, vgt, vpred, err, out_dir,
                              n_samples=8):
    """n_samples ảnh GT + Pred sắp xếp theo lỗi (xấu → tốt)."""
    mean_err = err.mean(axis=1)  # (N,)

    # Lấy mẫu: 1/3 lỗi cao, 1/3 trung bình, 1/3 lỗi thấp
    n3 = n_samples // 3
    idx_hi  = np.argsort(mean_err)[-n3:][::-1]
    idx_mid = np.argsort(np.abs(mean_err - np.median(mean_err)))[:n3]
    idx_lo  = np.argsort(mean_err)[:n3]
    idxs    = list(idx_hi) + list(idx_mid) + list(idx_lo)
    idxs    = idxs[:n_samples]

    cols = n_samples
    fig  = plt.figure(figsize=(cols * 2.6, 6.0), facecolor=DARK_BG)
    gs   = gridspec.GridSpec(2, cols, hspace=0.05, wspace=0.04,
                              top=0.90, bottom=0.02)
    fig.suptitle('Skeleton: Ground Truth  vs  Prediction  (COCO-17)',
                 color='white', fontsize=11, fontweight='bold')

    labels_top = (['High Error']*n3 +
                  ['Mid Error']*n3 +
                  ['Low Error']*(n_samples - 2*n3))

    for col, (si, lbl) in enumerate(zip(idxs, labels_top)):
        e_i = float(mean_err[si])
        ax_gt   = fig.add_subplot(gs[0, col])
        ax_pred = fig.add_subplot(gs[1, col])

        _draw_skel(ax_gt,   gt[si],   vgt[si],
                   is_pred=False,
                   title=f'GT [{lbl}]' if col % n3 == 0 else 'GT')
        _draw_skel(ax_pred, pred[si], vpred[si],
                   is_pred=True, err=err[si],
                   title=f'Pred e={e_i:.3f}')

    # Colorbar lỗi joint
    sm = ScalarMappable(cmap='RdYlGn_r', norm=Normalize(0, 0.15))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='vertical',
                        fraction=0.010, pad=0.008, shrink=0.85)
    cbar.set_label('Joint Error (normalized)', color=TEXT_CLR, fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR, fontsize=7)

    path = os.path.join(out_dir, 'skeleton_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ skeleton_comparison.png")
    return path


# ══════════════════════════════════════════════════════════════════
# 2. Joint Error Heatmap
# ══════════════════════════════════════════════════════════════════
def plot_joint_heatmap(err, out_dir):
    """err: (N,17)"""
    mpjpe = err.mean(axis=0)  # (17,) mean over samples

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              facecolor='#111111',
                              gridspec_kw={'width_ratios': [1, 1.3]})
    fig.suptitle('Per-Joint MPJPE (Mean Error per Keypoint)',
                 color='white', fontsize=12, fontweight='bold')

    # ── Bar chart ──
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    sidx   = np.argsort(mpjpe)[::-1]
    colors = plt.cm.RdYlGn_r(Normalize()(mpjpe[sidx]))
    bars   = ax.barh([JOINT_NAMES[i] for i in sidx],
                     mpjpe[sidx], color=colors,
                     edgecolor='#333333', linewidth=0.5)
    ax.set_xlabel('MPJPE (normalized coords)', color='#AAAAAA', fontsize=9)
    ax.set_title('Error per Joint (sorted)', color='white', fontsize=10)
    ax.tick_params(colors='#CCCCCC', labelsize=8)
    ax.spines[:].set_color('#333355')
    ax.grid(axis='x', color=GRID_CLR, linewidth=0.7)
    # Overall mean line
    overall = mpjpe.mean()
    ax.axvline(overall, color='#FFD700', lw=1.5, ls='--',
               label=f'Overall MPJPE={overall:.4f}')
    ax.legend(facecolor='#222233', labelcolor='white', fontsize=8)
    for bar, idx in zip(bars, sidx):
        ax.text(bar.get_width() + overall*0.02,
                bar.get_y() + bar.get_height()/2,
                f'{mpjpe[idx]:.4f}', va='center', ha='left',
                color=TEXT_CLR, fontsize=7)

    # ── Body map ──
    ax2 = axes[1]
    ax2.set_facecolor(DARK_BG)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(1.05, -0.05)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Error trên bản đồ cơ thể', color='white', fontsize=10)

    for a, b in COCO_SKELETON:
        ax2.plot([BODY_POS[a,0], BODY_POS[b,0]],
                 [BODY_POS[a,1], BODY_POS[b,1]],
                 color='#2A2A4A', lw=2.5, zorder=1)

    norm2 = Normalize(mpjpe.min(), mpjpe.max())
    sizes = 350 + 2200 * norm2(mpjpe)
    sc = ax2.scatter(BODY_POS[:,0], BODY_POS[:,1],
                     c=mpjpe, cmap='RdYlGn_r', norm=norm2,
                     s=sizes, zorder=5, edgecolors='white', lw=0.8)

    for j in range(17):
        dx = 0.08 if BODY_POS[j,0] >= 0.50 else -0.08
        ax2.text(BODY_POS[j,0]+dx, BODY_POS[j,1],
                 JOINT_NAMES[j], fontsize=6.5, color=TEXT_CLR,
                 va='center', ha='left' if dx > 0 else 'right')

    cbar = fig.colorbar(sc, ax=ax2, orientation='vertical',
                        fraction=0.03, pad=0.02)
    cbar.set_label('MPJPE', color='white', fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, 'joint_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"  ✓ joint_heatmap.png  (overall MPJPE={overall:.4f})")
    return path


# ══════════════════════════════════════════════════════════════════
# 3. Visibility Confusion Matrix
# ══════════════════════════════════════════════════════════════════
def plot_visibility(vgt, vpred, out_dir, threshold=0.5):
    vgt_b   = (vgt   >= threshold).astype(int)
    vpred_b = (vpred >= threshold).astype(int)

    acc_j   = (vgt_b == vpred_b).mean(axis=0)   # (17,)
    overall = acc_j.mean()

    tp = int(((vgt_b==1)&(vpred_b==1)).sum())
    fp = int(((vgt_b==0)&(vpred_b==1)).sum())
    fn = int(((vgt_b==1)&(vpred_b==0)).sum())
    tn = int(((vgt_b==0)&(vpred_b==0)).sum())
    cm = np.array([[tp,fn],[fp,tn]])

    prec = tp / (tp+fp+1e-8)
    rec  = tp / (tp+fn+1e-8)
    f1   = 2*prec*rec/(prec+rec+1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              facecolor='#111111')
    fig.suptitle(
        f'Visibility Accuracy  (Overall={overall:.1%} | '
        f'P={prec:.3f} R={rec:.3f} F1={f1:.3f})',
        color='white', fontsize=11, fontweight='bold')

    # ── Per-joint bar ──
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    sidx = np.argsort(acc_j)
    ax.barh([JOINT_NAMES[i] for i in sidx],
            acc_j[sidx],
            color=plt.cm.RdYlGn(acc_j[sidx]),
            edgecolor='#333333', lw=0.5)
    ax.axvline(overall, color='#FFD700', lw=1.5, ls='--',
               label=f'Overall {overall:.1%}')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Accuracy', color='#AAAAAA', fontsize=9)
    ax.set_title('Per-Joint Visibility Accuracy', color='white', fontsize=10)
    ax.tick_params(colors='#CCCCCC', labelsize=8)
    ax.spines[:].set_color('#333355')
    ax.grid(axis='x', color=GRID_CLR, lw=0.7)
    ax.legend(facecolor='#222233', labelcolor='white', fontsize=8)
    for i, idx in enumerate(sidx):
        ax.text(min(acc_j[idx]+0.01, 1.0), i,
                f'{acc_j[idx]:.1%}', va='center', ha='left',
                color=TEXT_CLR, fontsize=7)

    # ── Confusion matrix ──
    ax2 = axes[1]
    ax2.set_facecolor('#111111')
    im = ax2.imshow(cm, cmap='Blues', aspect='auto')
    cell_labels = [['TP','FN'],['FP','TN']]
    row_lbl = ['Actual: Visible','Actual: Not Visible']
    col_lbl = ['Pred: Visible','Pred: Not Visible']

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v   = cm[i,j]
            pct = v/total*100
            clr = 'white' if v < cm.max()*0.55 else '#0D0D0D'
            ax2.text(j, i,
                     f'{cell_labels[i][j]}\n{v:,}\n({pct:.1f}%)',
                     ha='center', va='center',
                     color=clr, fontsize=11, fontweight='bold')

    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(col_lbl, color='#CCCCCC', fontsize=9)
    ax2.set_yticklabels(row_lbl, color='#CCCCCC', fontsize=9)
    ax2.set_title('Confusion Matrix — Visibility', color='white', fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.04)

    plt.tight_layout()
    path = os.path.join(out_dir, 'visibility_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"  ✓ visibility_matrix.png  (acc={overall:.1%} F1={f1:.3f})")
    return path


# ══════════════════════════════════════════════════════════════════
# 4. PCK Curve
# ══════════════════════════════════════════════════════════════════
def plot_pck_curve(err, out_dir):
    """
    PCK (Percentage of Correct Keypoints) @ threshold τ.
    τ = fraction of normalized coord distance (0–0.2).
    """
    thresholds = np.linspace(0.0, 0.20, 100)
    pck_all    = []
    pck_joints = np.zeros((17, len(thresholds)))

    for ti, tau in enumerate(thresholds):
        correct = (err <= tau)                     # (N,17) bool
        pck_all.append(correct.mean())
        pck_joints[:, ti] = correct.mean(axis=0)   # (17,)

    # Nhóm body parts
    groups = {
        'Head'     : [0,1,2,3,4],
        'Arms'     : [5,6,7,8,9,10],
        'Legs'     : [11,12,13,14,15,16],
    }
    group_colors = {'Head':'#FF6B6B', 'Arms':'#4ECDC4', 'Legs':'#87CEEB'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              facecolor='#111111')
    fig.suptitle('PCK — Percentage of Correct Keypoints',
                 color='white', fontsize=12, fontweight='bold')

    # ── Left: overall + groups ──
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.plot(thresholds, pck_all, color='#FFD700', lw=2.5,
            label='Overall', zorder=5)
    for gname, jidxs in groups.items():
        g_pck = np.array([err[:, jidxs].mean(axis=1) <= tau
                          for tau in thresholds]).mean(axis=1)
        ax.plot(thresholds, g_pck,
                color=group_colors[gname], lw=1.8,
                linestyle='--', label=gname, alpha=0.9)

    # Reference lines
    for ref_tau in [0.05, 0.10, 0.15]:
        idx   = np.argmin(np.abs(thresholds - ref_tau))
        ref_v = pck_all[idx]
        ax.axvline(ref_tau, color='#555566', lw=0.8, ls=':')
        ax.text(ref_tau + 0.001, 0.05,
                f'τ={ref_tau}\n{ref_v:.1%}',
                color='#AAAAAA', fontsize=7.5)

    ax.set_xlabel('Threshold τ (normalized distance)', color='#AAAAAA')
    ax.set_ylabel('PCK (%)', color='#AAAAAA')
    ax.set_title('PCK Overall + Body Groups', color='white', fontsize=10)
    ax.legend(facecolor='#222233', labelcolor='white', fontsize=8)
    ax.tick_params(colors='#CCCCCC')
    ax.spines[:].set_color('#333355')
    ax.grid(color=GRID_CLR, lw=0.7)
    ax.set_xlim(0, 0.20); ax.set_ylim(0, 1.05)

    # ── Right: per-joint PCK @ τ=0.10 ──
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    tau10_idx  = np.argmin(np.abs(thresholds - 0.10))
    pck_at_10  = pck_joints[:, tau10_idx]           # (17,)
    sidx       = np.argsort(pck_at_10)
    colors_bar = plt.cm.RdYlGn(pck_at_10[sidx])

    ax2.barh([JOINT_NAMES[i] for i in sidx],
             pck_at_10[sidx], color=colors_bar,
             edgecolor='#333333', lw=0.5)
    ax2.axvline(pck_at_10.mean(), color='#FFD700', lw=1.5, ls='--',
                label=f'Mean={pck_at_10.mean():.1%}')
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel('PCK @ τ=0.10', color='#AAAAAA', fontsize=9)
    ax2.set_title('Per-Joint PCK @ τ=0.10', color='white', fontsize=10)
    ax2.tick_params(colors='#CCCCCC', labelsize=8)
    ax2.spines[:].set_color('#333355')
    ax2.grid(axis='x', color=GRID_CLR, lw=0.7)
    ax2.legend(facecolor='#222233', labelcolor='white', fontsize=8)
    for i, idx in enumerate(sidx):
        ax2.text(min(pck_at_10[idx]+0.01, 1.0), i,
                 f'{pck_at_10[idx]:.1%}', va='center', ha='left',
                 color=TEXT_CLR, fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, 'pck_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    pck10 = np.array(pck_all)[tau10_idx]
    print(f"  ✓ pck_curve.png  (PCK@0.10={pck10:.1%})")
    return path


# ══════════════════════════════════════════════════════════════════
# 5. Error Distribution
# ══════════════════════════════════════════════════════════════════
def plot_error_distribution(err, out_dir):
    """Phân phối MPJPE per-sample và per-joint."""
    sample_err = err.mean(axis=1)   # (N,)
    joint_err  = err.mean(axis=0)   # (17,)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              facecolor='#111111')
    fig.suptitle('Error Distribution', color='white',
                 fontsize=12, fontweight='bold')

    # ── Per-sample histogram ──
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    n, bins, patches = ax.hist(sample_err, bins=60,
                                color='#4ECDC4', edgecolor='#0D0D0D',
                                linewidth=0.4, alpha=0.85)
    # Tô màu theo giá trị
    norm_h = Normalize(bins[0], bins[-1])
    for p, left in zip(patches, bins[:-1]):
        p.set_facecolor(plt.cm.RdYlGn_r(norm_h(left + (bins[1]-bins[0])/2)))

    ax.axvline(sample_err.mean(), color='#FFD700', lw=2,
               label=f'Mean={sample_err.mean():.4f}')
    ax.axvline(np.median(sample_err), color='#FF6B6B', lw=1.5,
               ls='--', label=f'Median={np.median(sample_err):.4f}')
    ax.set_xlabel('MPJPE per sample', color='#AAAAAA')
    ax.set_ylabel('Count', color='#AAAAAA')
    ax.set_title('Sample-level Error Distribution', color='white', fontsize=10)
    ax.tick_params(colors='#CCCCCC')
    ax.spines[:].set_color('#333355')
    ax.grid(color=GRID_CLR, lw=0.7, axis='y')
    ax.legend(facecolor='#222233', labelcolor='white', fontsize=8)

    # Stats text
    p25, p75 = np.percentile(sample_err, [25, 75])
    ax.text(0.97, 0.97,
            f'N={len(sample_err)}\n'
            f'std={sample_err.std():.4f}\n'
            f'P25={p25:.4f}\nP75={p75:.4f}',
            transform=ax.transAxes,
            va='top', ha='right', color=TEXT_CLR,
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', fc='#1A1A2E', ec='#333355'))

    # ── Per-joint box plot ──
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    data_bp = [err[:, j] for j in range(17)]
    bp = ax2.boxplot(data_bp, vert=False, patch_artist=True,
                     widths=0.55,
                     medianprops=dict(color='#FFD700', lw=2),
                     whiskerprops=dict(color='#AAAAAA'),
                     capprops=dict(color='#AAAAAA'),
                     flierprops=dict(marker='.', color='#555566',
                                     markersize=2, alpha=0.4))
    norm_bp = Normalize(joint_err.min(), joint_err.max())
    for j, patch in enumerate(bp['boxes']):
        patch.set_facecolor(plt.cm.RdYlGn_r(norm_bp(joint_err[j])))
        patch.set_alpha(0.8)

    ax2.set_yticks(range(1, 18))
    ax2.set_yticklabels(JOINT_NAMES, fontsize=8, color='#CCCCCC')
    ax2.set_xlabel('Error (normalized)', color='#AAAAAA')
    ax2.set_title('Per-Joint Error Distribution (Box Plot)',
                  color='white', fontsize=10)
    ax2.tick_params(colors='#CCCCCC')
    ax2.spines[:].set_color('#333355')
    ax2.grid(axis='x', color=GRID_CLR, lw=0.7)

    plt.tight_layout()
    path = os.path.join(out_dir, 'error_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"  ✓ error_distribution.png")
    return path


# ══════════════════════════════════════════════════════════════════
# 6. Report summary (ghép các ảnh)
# ══════════════════════════════════════════════════════════════════
def make_report_summary(paths: dict, metrics: dict, out_dir: str):
    """Ghép skeleton + heatmap + visibility + pck thành 1 trang."""
    order = ['skeleton_comparison', 'joint_heatmap',
             'visibility_matrix',   'pck_curve',
             'error_distribution']

    imgs = [(k, plt.imread(paths[k]))
            for k in order if k in paths and os.path.exists(paths[k])]

    if not imgs:
        return None

    n = len(imgs)
    fig, axes = plt.subplots(n, 1, figsize=(15, 5.5 * n),
                              facecolor='#0A0A0A')
    fig.suptitle(
        f'Evaluation Report  |  '
        f'MPJPE={metrics["mpjpe"]:.4f}  '
        f'PCK@0.10={metrics["pck10"]:.1%}  '
        f'Vis-Acc={metrics["vis_acc"]:.1%}',
        color='white', fontsize=13, fontweight='bold', y=1.002)

    ax_list = axes if n > 1 else [axes]
    for ax, (label, img) in zip(ax_list, imgs):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout(pad=0.4)
    path = os.path.join(out_dir, 'report_summary.png')
    plt.savefig(path, dpi=120, bbox_inches='tight',
                facecolor='#0A0A0A')
    plt.close()
    print(f"  ✓ report_summary.png")
    return path


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Offline Evaluation — CSIPoseModel")
    p.add_argument('--checkpoint',    required=True,
                   help='Path checkpoint .pt')
    p.add_argument('--data-dir',      default='saved2',
                   help='Thư mục chứa aligned_*.npz')
    p.add_argument('--cleaner',       default='checkpoints_v4/feature_cleaner',
                   help='Path to FeatureCleaner .npz (without extension)')
    p.add_argument('--chunk-size',     type=int, default=100)
    p.add_argument('--stride',         type=int, default=5)
    p.add_argument('--edge-buf',       type=int, default=15)
    p.add_argument('--out',           default='eval_results',
                   help='Thư mục output biểu đồ')
    p.add_argument('--batch-size',    type=int, default=32)
    p.add_argument('--n-skeleton',    type=int, default=8,
                   help='Số mẫu skeleton hiển thị (mặc định: 8)')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== eval_v2  ===")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_dir}")
    print(f"  Cleaner    : {args.cleaner}")
    print(f"  Output     : {args.out}")
    print(f"  Device     : {device}\n")

    # ── Load dataset (dùng V4 pipeline) ──
    _, val_ds, cleaner = build_datasets(
        data_dir      = args.data_dir,
        chunk_size    = args.chunk_size,
        stride        = args.stride,
        edge_buf      = args.edge_buf,
        val_every     = 4,
        cleaner_path  = args.cleaner,
        augment_train = False,
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    print(f"  Val dataset: {len(val_ds)} samples\n")

    # ── Load model ──
    model = CSIPoseModelV4(feature_dim=cleaner.n_alive).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"  Model loaded ✓\n")

    # ── Inference ──
    print("  Running inference...")
    gt, pred, vgt, vpred, err = run_inference(model, loader, device)
    print(f"  Done. gt={gt.shape}  err={err.shape}\n")

    # ── Metrics ──
    mpjpe   = float(err.mean())
    thresholds = np.linspace(0, 0.20, 100)
    pck_vals   = [(err <= tau).mean() for tau in thresholds]
    tau10_idx  = np.argmin(np.abs(thresholds - 0.10))
    pck10      = pck_vals[tau10_idx]
    vis_acc    = ((vgt >= 0.5).astype(int) == (vpred >= 0.5).astype(int)).mean()

    print(f"  ══ Metrics ══")
    print(f"  Overall MPJPE  : {mpjpe:.4f}")
    print(f"  PCK @ τ=0.05   : {pck_vals[np.argmin(np.abs(thresholds-0.05))]:.1%}")
    print(f"  PCK @ τ=0.10   : {pck10:.1%}")
    print(f"  PCK @ τ=0.15   : {pck_vals[np.argmin(np.abs(thresholds-0.15))]:.1%}")
    print(f"  Visibility Acc : {vis_acc:.1%}")
    print()

    # Per-joint MPJPE
    print(f"  {'Joint':<16} {'MPJPE':>8}")
    print(f"  {'─'*25}")
    for j in range(17):
        print(f"  {JOINT_NAMES[j]:<16} {err[:,j].mean():>8.4f}")
    print()

    metrics = dict(mpjpe=mpjpe, pck10=pck10, vis_acc=vis_acc)

    # ── Vẽ biểu đồ ──
    print("  Đang vẽ biểu đồ...")
    paths = {}

    paths['skeleton_comparison'] = plot_skeleton_comparison(
        gt, pred, vgt, vpred, err, args.out, n_samples=args.n_skeleton)

    paths['joint_heatmap'] = plot_joint_heatmap(err, args.out)

    paths['visibility_matrix'] = plot_visibility(vgt, vpred, args.out)

    paths['pck_curve'] = plot_pck_curve(err, args.out)

    paths['error_distribution'] = plot_error_distribution(err, args.out)

    paths['report_summary'] = make_report_summary(paths, metrics, args.out)

    print(f"\n  ✅ Tất cả biểu đồ đã lưu vào: {args.out}/")
    print(f"\n  Tóm tắt kết quả:")
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │ MPJPE          : {mpjpe:.4f}         │")
    print(f"  │ PCK @ τ=0.10   : {pck10:.1%}          │")
    print(f"  │ Visibility Acc : {vis_acc:.1%}          │")
    print(f"  └─────────────────────────────────┘")


if __name__ == '__main__':
    main()