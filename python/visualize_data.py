"""
visualize_data.py  —  Visualize dữ liệu CSI-Pose trước huấn luyện
═══════════════════════════════════════════════════════════════════
Tạo báo cáo trực quan để:
  1. Hiểu cấu trúc & ý nghĩa dữ liệu (giải thích cho người chưa biết)
  2. Kiểm tra chất lượng dữ liệu (sẵn sàng train hay chưa)

Cách dùng:
    python visualize_data.py
    python visualize_data.py --data saved2/aligned_20260407_140052.npz
    python visualize_data.py --data saved2 saved3       (nhiều thư mục)
    python visualize_data.py --out-dir my_report

Output: thư mục chứa ~10 biểu đồ PNG + report_summary.txt
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import pearsonr

# ── COCO-17 ──────────────────────────────────────────────────
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

JOINT_NAMES = [
    'Nose', 'L.Eye', 'R.Eye', 'L.Ear', 'R.Ear',
    'L.Shoulder', 'R.Shoulder', 'L.Elbow', 'R.Elbow',
    'L.Wrist', 'R.Wrist', 'L.Hip', 'R.Hip',
    'L.Knee', 'R.Knee', 'L.Ankle', 'R.Ankle',
]

JOINT_GROUPS = {
    'Head':  [0, 1, 2, 3, 4],
    'Upper': [5, 6, 7, 8, 9, 10],
    'Lower': [11, 12, 13, 14, 15, 16],
}

GROUP_COLORS = {
    'Head':  '#FF6B6B',
    'Upper': '#4ECDC4',
    'Lower': '#45B7D1',
}

DARK_BG    = '#0D1117'
PANEL_BG   = '#161B22'
TEXT_COLOR  = '#C9D1D9'
ACCENT     = '#58A6FF'
GRID_COLOR = '#21262D'


def load_data(paths):
    """Load 1 hoặc nhiều file .npz, trả về dict gộp."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, 'aligned_*.npz'))))
        elif os.path.isfile(p):
            files.append(p)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file aligned_*.npz trong {paths}")

    all_X, all_y, all_vis, all_ts = [], [], [], []
    file_info = []
    offset = 0.0
    total_orig = 0

    # If many files, subsample each to stay under budget
    MAX_TOTAL = 15000  # max samples to keep in memory

    for f in files:
        d = np.load(f)
        X = d['X']
        y = d['y']

        vis_key = 'visibility' if 'visibility' in d else 'vis' if 'vis' in d else None
        vis = d[vis_key] if vis_key else np.ones((len(X), 17), dtype=np.float32)

        ts_key = 'timestamps' if 'timestamps' in d else 'ts' if 'ts' in d else None
        if ts_key:
            ts = d[ts_key].astype(np.float64)
        else:
            ts = np.arange(len(X), dtype=np.float64) / 20.0

        n_orig = len(X)
        total_orig += n_orig
        file_info.append({
            'path': f,
            'n_samples': n_orig,
            'duration': ts[-1] - ts[0],
        })
        print(f"  Loaded {f}: {n_orig} samples, {ts[-1]-ts[0]:.0f}s")

        # Subsample evenly (keep temporal order)
        budget_per_file = MAX_TOTAL // max(len(files), 1)
        if n_orig > budget_per_file:
            step = n_orig // budget_per_file
            idx = np.arange(0, n_orig, step)[:budget_per_file]
            X   = X[idx]
            y   = y[idx]
            vis = vis[idx]
            ts  = ts[idx]

        # Rebase timestamp
        ts_rebased = ts - ts[0] + offset
        offset = ts_rebased[-1] + 5.0  # 5s gap giữa sessions

        all_X.append(X)
        all_y.append(y)
        all_vis.append(vis)
        all_ts.append(ts_rebased)

    if total_orig > MAX_TOTAL:
        kept = sum(len(x) for x in all_X)
        print(f"  (Subsampled {total_orig:,} → {kept:,} samples to fit in memory)")

    return {
        'X': np.concatenate(all_X),
        'y': np.concatenate(all_y),
        'vis': np.concatenate(all_vis),
        'ts': np.concatenate(all_ts),
        'files': file_info,
    }


# ══════════════════════════════════════════════════════════════
# PLOT 1 — Data Overview (architecture diagram)
# ══════════════════════════════════════════════════════════════

def plot_data_overview(data, out_dir):
    """Sơ đồ giải thích cấu trúc dữ liệu."""
    X, y, vis = data['X'], data['y'], data['vis']
    N, T, F = X.shape

    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    fig.suptitle('DATA OVERVIEW — Cấu trúc dữ liệu CSI → Pose',
                 color='white', fontsize=16, fontweight='bold', y=0.97)

    # Layout: 3 cột
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    # ── (0,0) Tổng quan dataset ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('1. Tổng quan Dataset', color=ACCENT, fontsize=13, fontweight='bold', pad=10)

    info_lines = [
        ('Tổng samples', f'{N:,}'),
        ('Window size (T)', f'{T} frames ≈ {T/20:.1f}s'),
        ('CSI features (F)', f'{F}'),
        ('', f'= 3 nodes × 2 (amp+phase) × 128 sub'),
        ('Keypoints', '17 joints (COCO format)'),
        ('Output dim', f'17 × 2 + 17 vis = 51'),
        ('Sessions', f'{len(data["files"])}'),
        ('Tổng thời lượng', f'{sum(f["duration"] for f in data["files"]):.0f}s'),
    ]
    for i, (label, value) in enumerate(info_lines):
        yp = 9.0 - i * 1.1
        if label:
            ax.text(0.3, yp, label + ':', color='#8B949E', fontsize=9,
                    va='center', fontfamily='monospace')
            ax.text(5.5, yp, value, color=TEXT_COLOR, fontsize=9.5,
                    va='center', fontweight='bold', fontfamily='monospace')
        else:
            ax.text(5.5, yp, value, color='#6E7681', fontsize=8,
                    va='center', fontfamily='monospace')

    # ── (0,1) CSI Input heatmap (1 sample) ──
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('2. CSI Input — 1 mẫu (X[0])', color=ACCENT, fontsize=13,
                 fontweight='bold', pad=10)

    sample = X[N // 2]  # lấy mẫu giữa
    im = ax.imshow(sample.T, aspect='auto', cmap='viridis',
                   extent=[0, T, F, 0], interpolation='nearest')
    ax.set_xlabel('Time frames (T=40)', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Features (768)', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # Đánh dấu 3 node regions
    for i, (start, label, color) in enumerate([
        (0,   'Node1 (amp+phase)', '#FF6B6B'),
        (256, 'Node2 (amp+phase)', '#4ECDC4'),
        (512, 'Node3 (amp+phase)', '#45B7D1'),
    ]):
        ax.axhline(y=start, color=color, linewidth=1.5, linestyle='--', alpha=0.7)
        ax.text(T + 0.5, start + 64, label, color=color, fontsize=7,
                va='center', fontweight='bold')

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.12)
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

    # ── (0,2) Keypoint skeleton (1 sample) ──
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('3. Pose Output — COCO-17 Skeleton', color=ACCENT, fontsize=13,
                 fontweight='bold', pad=10)

    kp = y[N // 2]  # (17, 2)
    v  = vis[N // 2]

    # Vẽ skeleton
    for (i, j) in COCO_SKELETON:
        if v[i] > 0.3 and v[j] > 0.3:
            group = 'Head' if (i <= 4 or j <= 4) else 'Upper' if (i <= 10 or j <= 10) else 'Lower'
            ax.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]],
                    color=GROUP_COLORS[group], linewidth=2.5, alpha=0.8)

    for k in range(17):
        if v[k] > 0.3:
            group = 'Head' if k <= 4 else 'Upper' if k <= 10 else 'Lower'
            ax.scatter(kp[k, 0], kp[k, 1], c=GROUP_COLORS[group],
                       s=60, zorder=5, edgecolors='white', linewidth=0.5)
            ax.annotate(JOINT_NAMES[k], (kp[k, 0], kp[k, 1]),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=6, color='#8B949E')

    ax.set_xlim(-0.05, 1.05); ax.set_ylim(1.05, -0.05)
    ax.set_xlabel('x (normalized)', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('y (normalized)', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.set_aspect('equal')

    # Legend
    for gname, gcol in GROUP_COLORS.items():
        ax.scatter([], [], c=gcol, s=40, label=gname)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc='lower right')

    # ── (1,0) Pipeline diagram ──
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(PANEL_BG)
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title('4. Pipeline thu thập dữ liệu', color=ACCENT, fontsize=13,
                 fontweight='bold', pad=10)

    steps = [
        ('ESP32 ×3\n(WiFi CSI)', '#FF6B6B'),
        ('UDP → PC\n(20 Hz)', '#FFA07A'),
        ('Amp + Phase\nnormalize', '#FFD93D'),
        ('Sliding\nwindow (40)', '#4ECDC4'),
        ('Align với\nYOLO pose', '#45B7D1'),
        ('Lưu .npz\n(X, y, vis)', '#58A6FF'),
    ]
    for i, (text, color) in enumerate(steps):
        x = 0.5 + i * 1.6
        rect = FancyBboxPatch((x - 0.6, 3.5), 1.2, 3,
                               boxstyle="round,pad=0.15",
                               facecolor=color, alpha=0.2,
                               edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 5, text, ha='center', va='center',
                color=color, fontsize=7.5, fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 0.75, 5), xytext=(x + 0.55, 5),
                        arrowprops=dict(arrowstyle='->', color='#8B949E', lw=1.5))

    # ── (1,1) Feature distribution ──
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('5. Phân bố CSI features', color=ACCENT, fontsize=13,
                 fontweight='bold', pad=10)

    # Lấy mẫu để vẽ nhanh
    idx = np.random.choice(N, min(500, N), replace=False)
    X_sample = X[idx].reshape(-1, F)

    # Amplitude (node 1) vs Phase (node 1)
    amp_mean = X_sample[:, :128].mean(axis=1)
    ph_mean  = X_sample[:, 128:256].mean(axis=1)
    ax.scatter(amp_mean, ph_mean, c='#4ECDC4', s=3, alpha=0.3)
    ax.set_xlabel('Mean Amplitude (Node1)', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Mean Phase (Node1)', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # ── (1,2) Visibility stats ──
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('6. Visibility trung bình mỗi joint', color=ACCENT, fontsize=13,
                 fontweight='bold', pad=10)

    mean_vis = vis.mean(axis=0)
    colors_bar = ['#FF6B6B' if v < 0.5 else '#FFD93D' if v < 0.75 else '#4ECDC4'
                  for v in mean_vis]
    bars = ax.barh(range(17), mean_vis, color=colors_bar, alpha=0.85, height=0.7)
    ax.set_yticks(range(17))
    ax.set_yticklabels(JOINT_NAMES, fontsize=8)
    ax.set_xlabel('Mean Visibility', color=TEXT_COLOR, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color='#FF6B6B', linestyle='--', alpha=0.5, label='Threshold')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.savefig(os.path.join(out_dir, '01_data_overview.png'), dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 01_data_overview.png')


# ══════════════════════════════════════════════════════════════
# PLOT 2 — CSI Temporal Patterns
# ══════════════════════════════════════════════════════════════

def plot_csi_temporal(data, out_dir):
    """Visualize tín hiệu CSI theo thời gian — cho thấy CSI 'nhìn thấy' chuyển động."""
    X, y, ts = data['X'], data['y'], data['ts']
    N, T, F = X.shape

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), facecolor=DARK_BG, sharex=True)
    fig.suptitle('CSI TEMPORAL PATTERNS — CSI thay đổi khi người di chuyển',
                 color='white', fontsize=15, fontweight='bold', y=0.97)
    fig.subplots_adjust(hspace=0.25, left=0.07, right=0.95, top=0.91, bottom=0.06)

    # Lấy 500 frames liên tiếp
    start = N // 4
    end   = min(start + 500, N)
    t_rel = ts[start:end] - ts[start]

    # (a) CSI amplitude trung bình theo thời gian (3 nodes)
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    for node, (s, e, color, label) in enumerate([
        (0,   128, '#FF6B6B', 'Node 1'),
        (256, 384, '#4ECDC4', 'Node 2'),
        (512, 640, '#45B7D1', 'Node 3'),
    ]):
        amp_mean = X[start:end, -1, s:e].mean(axis=1)  # last frame of window
        ax.plot(t_rel, amp_mean, color=color, linewidth=0.8, alpha=0.8, label=label)
    ax.set_ylabel('Mean Amplitude', color=TEXT_COLOR, fontsize=10)
    ax.set_title('(a) CSI Amplitude — 3 ESP32 Nodes', color=ACCENT, fontsize=11)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # (b) CSI variance trong window → proxy cho motion
    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    csi_var = X[start:end].std(axis=1).mean(axis=1)  # std over T, mean over F
    ax.fill_between(t_rel, 0, csi_var, color='#FFD93D', alpha=0.4)
    ax.plot(t_rel, csi_var, color='#FFD93D', linewidth=0.8)
    ax.set_ylabel('CSI Variance', color=TEXT_COLOR, fontsize=10)
    ax.set_title('(b) CSI Biến động trong window — cao = người đang di chuyển',
                 color=ACCENT, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # (c) Keypoint Y (hip) movement
    ax = axes[2]
    ax.set_facecolor(PANEL_BG)
    for ji, jname, color in [(0, 'Nose', '#FF6B6B'), (11, 'L.Hip', '#4ECDC4'),
                              (15, 'L.Ankle', '#45B7D1')]:
        ax.plot(t_rel, y[start:end, ji, 1], color=color, linewidth=0.8,
                alpha=0.8, label=jname)
    ax.set_ylabel('Y position', color=TEXT_COLOR, fontsize=10)
    ax.set_title('(c) Keypoint Y — chuyển động dọc (Ground Truth từ YOLO)',
                 color=ACCENT, fontsize=11)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # (d) Keypoint velocity
    ax = axes[3]
    ax.set_facecolor(PANEL_BG)
    kp_vel = np.zeros(end - start)
    kp_vel[1:] = np.linalg.norm(
        np.diff(y[start:end, :, :2], axis=0), axis=-1
    ).mean(axis=-1)
    ax.fill_between(t_rel, 0, kp_vel, color='#C084FC', alpha=0.3)
    ax.plot(t_rel, kp_vel, color='#C084FC', linewidth=0.8)
    ax.set_ylabel('Keypoint Velocity', color=TEXT_COLOR, fontsize=10)
    ax.set_xlabel('Time (seconds)', color=TEXT_COLOR, fontsize=10)
    ax.set_title('(d) Tốc độ di chuyển keypoints — cao = motion mạnh',
                 color=ACCENT, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    fig.savefig(os.path.join(out_dir, '02_csi_temporal.png'), dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 02_csi_temporal.png')


# ══════════════════════════════════════════════════════════════
# PLOT 3 — CSI Feature Heatmap (dead channels, correlation)
# ══════════════════════════════════════════════════════════════

def plot_feature_analysis(data, out_dir):
    """Phân tích feature: dead channels, channel importance, correlation."""
    X, y = data['X'], data['y']
    N, T, F = X.shape

    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    fig.suptitle('FEATURE ANALYSIS — Chất lượng CSI features',
                 color='white', fontsize=15, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    # ── (0,0) Channel activation heatmap ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(a) Channel STD — Dead channel detection', color=ACCENT,
                 fontsize=11, fontweight='bold')

    # Mean abs value per feature (across all samples, last frame)
    feat_std = X[:, -1, :].std(axis=0)  # (768,)
    feat_2d = feat_std.reshape(6, 128)  # 6 rows: amp1, ph1, amp2, ph2, amp3, ph3
    im = ax.imshow(feat_2d, aspect='auto', cmap='inferno',
                   interpolation='nearest')
    ax.set_yticks(range(6))
    ax.set_yticklabels(['N1 Amp', 'N1 Phase', 'N2 Amp', 'N2 Phase',
                        'N3 Amp', 'N3 Phase'], fontsize=8)
    ax.set_xlabel('Subcarrier index (128)', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cb.set_label('STD', color=TEXT_COLOR, fontsize=9)

    # ── (0,1) Dead vs alive channels ──
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(b) Dead vs Alive Channels', color=ACCENT,
                 fontsize=11, fontweight='bold')

    dead_thresh = 0.001
    feat_mean = X.mean(axis=(0, 1))
    n_dead  = (feat_mean < dead_thresh).sum()
    n_alive = F - n_dead

    colors_pie = ['#FF6B6B', '#4ECDC4']
    sizes = [n_dead, n_alive]
    labels = [f'Dead ({n_dead})\nmean < {dead_thresh}',
              f'Alive ({n_alive})\nUsed for training']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 9, 'color': TEXT_COLOR})
    for t in autotexts:
        t.set_color('white')
        t.set_fontweight('bold')

    # ── (0,2) Feature variance distribution ──
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(c) Feature Variance Distribution', color=ACCENT,
                 fontsize=11, fontweight='bold')

    alive_mask = feat_mean >= dead_thresh
    alive_std = feat_std[alive_mask]
    ax.hist(alive_std, bins=50, color='#4ECDC4', alpha=0.7, edgecolor='#2A9D8F')
    ax.axvline(np.median(alive_std), color='#FFD93D', linestyle='--',
               label=f'Median={np.median(alive_std):.4f}')
    ax.set_xlabel('Feature STD', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Count', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ── (1,0) Top correlated features with keypoints ──
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(d) Feature ↔ Keypoint Correlation (top 20)',
                 color=ACCENT, fontsize=11, fontweight='bold')

    # Compute correlation of each alive feature with hip-y
    idx_sample = np.random.choice(N, min(2000, N), replace=False)
    target = y[idx_sample, 11, 1]  # L.Hip y-coord
    alive_idx = np.where(alive_mask)[0]

    corrs = np.zeros(len(alive_idx))
    for i, fi in enumerate(alive_idx):
        feat_vals = X[idx_sample, -1, fi]
        if np.std(feat_vals) > 1e-8:
            corrs[i] = abs(pearsonr(feat_vals, target)[0])

    top20 = np.argsort(corrs)[-20:][::-1]
    ax.barh(range(20), corrs[top20], color='#58A6FF', alpha=0.8, height=0.6)
    ax.set_yticks(range(20))

    def feat_label(idx):
        fi = alive_idx[idx]
        node = fi // 256 + 1
        is_phase = (fi % 256) >= 128
        sub = fi % 128
        return f'N{node} {"Ph" if is_phase else "Amp"}[{sub}]'

    ax.set_yticklabels([feat_label(i) for i in top20], fontsize=7)
    ax.set_xlabel('|Pearson r| with L.Hip Y', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.invert_yaxis()

    # ── (1,1) Correlation matrix per node ──
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(e) CSI ↔ Keypoints Corr. (per node × joint)',
                 color=ACCENT, fontsize=11, fontweight='bold')

    # 3 nodes × 5 major joints, correlation
    major_joints = [0, 5, 11, 13, 15]
    major_names  = ['Nose', 'L.Sho', 'L.Hip', 'L.Knee', 'L.Ank']
    corr_matrix = np.zeros((6, len(major_joints)))  # 3 nodes × 2 axes

    for ni, (amp_s, ph_s) in enumerate([(0, 128), (256, 384), (512, 640)]):
        for ji_idx, ji in enumerate(major_joints):
            for ai, (a_start, label) in enumerate([(amp_s, 'amp')]):
                feats = X[idx_sample, -1, a_start:a_start + 128].mean(axis=1)
                for axis in range(2):
                    target_j = y[idx_sample, ji, axis]
                    if np.std(feats) > 1e-8 and np.std(target_j) > 1e-8:
                        r = abs(pearsonr(feats, target_j)[0])
                    else:
                        r = 0
                    corr_matrix[ni * 2 + axis, ji_idx] = r

    im = ax.imshow(corr_matrix, cmap='YlOrRd', vmin=0, vmax=0.7, aspect='auto')
    ax.set_xticks(range(len(major_joints)))
    ax.set_xticklabels(major_names, fontsize=8, rotation=30)
    ax.set_yticks(range(6))
    ax.set_yticklabels(['N1→x', 'N1→y', 'N2→x', 'N2→y', 'N3→x', 'N3→y'], fontsize=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for i in range(6):
        for j in range(len(major_joints)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if corr_matrix[i, j] > 0.3 else '#555')
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)

    # ── (1,2) Temporal autocorrelation ──
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(f) Temporal Autocorrelation of CSI',
                 color=ACCENT, fontsize=11, fontweight='bold')

    # Autocorrelation of a representative feature
    n_lags = min(200, N // 2)
    rep_feat = X[:min(2000, N), -1, alive_idx[0]]
    rep_feat = (rep_feat - rep_feat.mean()) / (rep_feat.std() + 1e-8)
    autocorr = np.correlate(rep_feat, rep_feat, 'full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr[:n_lags] / autocorr[0]

    lags_sec = np.arange(n_lags) / 20.0  # 20 Hz
    ax.plot(lags_sec, autocorr, color='#C084FC', linewidth=1.0)
    ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)
    ax.axhline(y=1.96 / np.sqrt(len(rep_feat)), color='#FF6B6B',
               linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(y=-1.96 / np.sqrt(len(rep_feat)), color='#FF6B6B',
               linestyle='--', alpha=0.5)

    # Decorrelation time
    decor_idx = np.where(autocorr < 0.5)[0]
    if len(decor_idx) > 0:
        decor_sec = decor_idx[0] / 20.0
        ax.axvline(x=decor_sec, color='#FFD93D', linestyle='--', alpha=0.7,
                   label=f'Decorrelation ≈ {decor_sec:.1f}s')

    ax.set_xlabel('Lag (seconds)', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Autocorrelation', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    fig.savefig(os.path.join(out_dir, '03_feature_analysis.png'), dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 03_feature_analysis.png')


# ══════════════════════════════════════════════════════════════
# PLOT 4 — Keypoint Distribution & Quality
# ══════════════════════════════════════════════════════════════

def plot_keypoint_quality(data, out_dir):
    """Phân bố keypoints, diversity, visibility, anomalies."""
    y, vis = data['y'], data['vis']
    N = len(y)

    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    fig.suptitle('KEYPOINT QUALITY — Kiểm tra Ground Truth',
                 color='white', fontsize=15, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.96, top=0.90, bottom=0.06)

    # ── (0,0) Keypoint heatmap overlay (all samples) ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(a) Keypoint Distribution Heatmap', color=ACCENT,
                 fontsize=11, fontweight='bold')

    idx = np.random.choice(N, min(3000, N), replace=False)
    for group, joints in JOINT_GROUPS.items():
        for j in joints:
            mask = vis[idx, j] > 0.3
            ax.scatter(y[idx[mask], j, 0], y[idx[mask], j, 1],
                       c=GROUP_COLORS[group], s=1, alpha=0.05)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(1.05, -0.05)
    ax.set_xlabel('x (normalized)', color=TEXT_COLOR)
    ax.set_ylabel('y (normalized)', color=TEXT_COLOR)
    ax.set_aspect('equal')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # Legend
    for g, c in GROUP_COLORS.items():
        ax.scatter([], [], c=c, s=30, label=g)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc='lower right')

    # ── (0,1) Per-joint STD (diversity) ──
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(b) Keypoint Diversity (STD per joint)',
                 color=ACCENT, fontsize=11, fontweight='bold')

    std_x = y[:, :, 0].std(axis=0)
    std_y = y[:, :, 1].std(axis=0)

    x_pos = np.arange(17)
    width = 0.35
    ax.bar(x_pos - width / 2, std_x, width, color='#FF6B6B', alpha=0.8, label='STD x')
    ax.bar(x_pos + width / 2, std_y, width, color='#4ECDC4', alpha=0.8, label='STD y')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n[:5] for n in JOINT_NAMES], fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('STD', color=TEXT_COLOR, fontsize=10)
    ax.axhline(y=0.02, color='#FF6B6B', linestyle='--', alpha=0.5,
               label='Min threshold (0.02)')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # ── (0,2) Sample skeletons (random 6) ──
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(c) Random Skeleton Samples (6 mẫu)',
                 color=ACCENT, fontsize=11, fontweight='bold')

    sample_idx = np.random.choice(N, 6, replace=False)
    alphas = [0.9, 0.7, 0.55, 0.45, 0.35, 0.25]
    cmap = plt.cm.Set2
    for si, (sidx, alpha) in enumerate(zip(sample_idx, alphas)):
        kp = y[sidx]
        v = vis[sidx]
        color = cmap(si / 6)
        for (i, j) in COCO_SKELETON:
            if v[i] > 0.3 and v[j] > 0.3:
                ax.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]],
                        color=color, linewidth=1.5, alpha=alpha)
        for k in range(17):
            if v[k] > 0.3:
                ax.scatter(kp[k, 0], kp[k, 1], c=[color], s=15, alpha=alpha, zorder=5)

    ax.set_xlim(-0.05, 1.05); ax.set_ylim(1.05, -0.05)
    ax.set_aspect('equal')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # ── (1,0) Keypoint range box plot ──
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(d) Keypoint Coordinate Ranges',
                 color=ACCENT, fontsize=11, fontweight='bold')

    bp = ax.boxplot([y[:, j, 1] for j in range(17)],
                    vert=True, patch_artist=True, widths=0.6)
    for i, box in enumerate(bp['boxes']):
        group = 'Head' if i <= 4 else 'Upper' if i <= 10 else 'Lower'
        box.set_facecolor(GROUP_COLORS[group])
        box.set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color(TEXT_COLOR)
    for flier in bp['fliers']:
        flier.set(marker='.', markersize=1, alpha=0.3)

    ax.set_xticklabels([n[:4] for n in JOINT_NAMES], fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Y coordinate', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # ── (1,1) Consecutive frame distance (jitter check) ──
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(e) Frame-to-Frame Jitter (outlier detection)',
                 color=ACCENT, fontsize=11, fontweight='bold')

    frame_dist = np.linalg.norm(np.diff(y[:, :, :2], axis=0), axis=-1).mean(axis=1)
    ax.hist(frame_dist, bins=100, color='#C084FC', alpha=0.7, edgecolor='#9B59B6')
    p95 = np.percentile(frame_dist, 95)
    p99 = np.percentile(frame_dist, 99)
    ax.axvline(p95, color='#FFD93D', linestyle='--',
               label=f'P95 = {p95:.4f}')
    ax.axvline(p99, color='#FF6B6B', linestyle='--',
               label=f'P99 = {p99:.4f}')
    n_outlier = (frame_dist > p99).sum()
    ax.set_xlabel('Mean joint displacement per frame', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Count', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.text(0.95, 0.95, f'Outliers (>P99): {n_outlier}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color='#FF6B6B')

    # ── (1,2) Visibility matrix (time × joint) ──
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(f) Visibility over Time (500 samples)',
                 color=ACCENT, fontsize=11, fontweight='bold')

    start = N // 3
    vis_slice = vis[start:start + 500]
    im = ax.imshow(vis_slice.T, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(17))
    ax.set_yticklabels(JOINT_NAMES, fontsize=7)
    ax.set_xlabel('Sample index', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    cb.set_label('Visibility', color=TEXT_COLOR, fontsize=9)

    fig.savefig(os.path.join(out_dir, '04_keypoint_quality.png'), dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 04_keypoint_quality.png')


# ══════════════════════════════════════════════════════════════
# PLOT 5 — Train/Val Split & Readiness Check
# ══════════════════════════════════════════════════════════════

def plot_train_readiness(data, out_dir):
    """Kiểm tra dữ liệu sẵn sàng train: split, leakage, balance."""
    X, y, vis, ts = data['X'], data['y'], data['vis'], data['ts']
    N, T, F = X.shape

    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    fig.suptitle('TRAINING READINESS CHECK — Dữ liệu sẵn sàng huấn luyện?',
                 color='white', fontsize=15, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    # ── Simulate train/val split (interleaved chunk strategy) ──
    chunk_size = 100
    stride     = 5
    edge_buf   = 15
    val_every  = 4

    n_chunks = N // chunk_size
    train_idx, val_idx = [], []
    for c in range(n_chunks):
        cs = c * chunk_size
        ce = cs + chunk_size
        is_val = (c % val_every == 0)
        inner = list(range(cs + edge_buf, ce - edge_buf, stride))
        if is_val:
            val_idx.extend(inner)
        else:
            train_idx.extend(inner)

    n_train = len(train_idx)
    n_val   = len(val_idx)

    # ── (0,0) Split visualization ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(a) Train / Val Split (interleaved chunks)',
                 color=ACCENT, fontsize=11, fontweight='bold')

    # Show chunk coloring
    chunk_colors = []
    for c in range(n_chunks):
        cs = c * chunk_size
        ce = cs + chunk_size
        is_val = (c % val_every == 0)
        color = '#FF6B6B' if is_val else '#4ECDC4'
        ax.barh(0, chunk_size, left=cs, color=color, alpha=0.6, height=0.5)
    ax.set_xlim(0, N)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Sample index', color=TEXT_COLOR, fontsize=10)
    ax.set_yticks([])
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # Legend + stats
    ax.barh([], [], color='#4ECDC4', alpha=0.6, label=f'Train: {n_train} samples')
    ax.barh([], [], color='#FF6B6B', alpha=0.6, label=f'Val: {n_val} samples')
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc='upper right')

    # ── (0,1) Distribution comparison ──
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(b) Train vs Val — Y distribution',
                 color=ACCENT, fontsize=11, fontweight='bold')

    if n_train > 0 and n_val > 0:
        train_y = y[train_idx, 11, 1]  # L.Hip y
        val_y   = y[val_idx, 11, 1]
        bins = np.linspace(0, 1, 40)
        ax.hist(train_y, bins=bins, alpha=0.6, color='#4ECDC4', label='Train', density=True)
        ax.hist(val_y, bins=bins, alpha=0.6, color='#FF6B6B', label='Val', density=True)
        ax.set_xlabel('L.Hip Y coordinate', color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel('Density', color=TEXT_COLOR, fontsize=10)
        ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    # ── (0,2) Readiness scorecard ──
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(PANEL_BG)
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title('(c) Readiness Checklist', color=ACCENT, fontsize=13,
                 fontweight='bold', pad=10)

    # Compute checks
    checks = []

    # 1. Enough data
    ok = n_train >= 500
    checks.append(('Train samples ≥ 500', f'{n_train}', ok))

    # 2. Val size
    ok = n_val >= 100
    checks.append(('Val samples ≥ 100', f'{n_val}', ok))

    # 3. No NaN/Inf
    ok = not (np.isnan(X).any() or np.isinf(X).any())
    checks.append(('No NaN/Inf in X', 'Clean' if ok else 'DIRTY', ok))

    # 4. Keypoints in [0,1]
    ok = y.min() >= -0.01 and y.max() <= 1.01
    checks.append(('Keypoints in [0,1]', f'[{y.min():.3f}, {y.max():.3f}]', ok))

    # 5. Keypoint diversity
    mean_std = (y[:, :, 0].std(axis=0).mean() + y[:, :, 1].std(axis=0).mean()) / 2
    ok = mean_std > 0.03
    checks.append(('Pose diversity (std > 0.03)', f'{mean_std:.4f}', ok))

    # 6. Visibility
    vis_ok = vis.mean()
    ok = vis_ok > 0.5
    checks.append(('Mean visibility > 0.5', f'{vis_ok:.3f}', ok))

    # 7. Dead channels
    feat_mean = X.mean(axis=(0, 1))
    n_alive = (feat_mean >= 0.001).sum()
    ok = n_alive >= 300
    checks.append(('Alive features ≥ 300', f'{n_alive}/{F}', ok))

    # 8. Sample rate
    rate = N / max(ts[-1] - ts[0], 1)
    ok = rate >= 10
    checks.append(('Sample rate ≥ 10 Hz', f'{rate:.1f} Hz', ok))

    pass_count = sum(1 for _, _, ok in checks)
    total_count = len(checks)

    for i, (label, value, ok) in enumerate(checks):
        yp = 9.0 - i * 1.05
        icon = '✓' if ok else '✗'
        icon_color = '#4ECDC4' if ok else '#FF6B6B'
        ax.text(0.3, yp, icon, color=icon_color, fontsize=14, va='center', fontweight='bold')
        ax.text(1.0, yp, label, color=TEXT_COLOR, fontsize=9, va='center')
        ax.text(8.5, yp, value, color=icon_color if not ok else '#8B949E',
                fontsize=9, va='center', ha='right', fontweight='bold')

    # Overall verdict
    verdict_y = 9.0 - len(checks) * 1.05 - 0.5
    if pass_count == total_count:
        ax.text(5, verdict_y, f'✓ READY ({pass_count}/{total_count})',
                ha='center', va='center', fontsize=14, color='#4ECDC4', fontweight='bold')
    else:
        ax.text(5, verdict_y, f'⚠ {total_count - pass_count} ISSUE(S) ({pass_count}/{total_count})',
                ha='center', va='center', fontsize=14, color='#FFD93D', fontweight='bold')

    # ── (1,0) Temporal leakage diagram ──
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(d) Temporal Leakage Prevention',
                 color=ACCENT, fontsize=11, fontweight='bold')

    # Show 4 chunks + edge buffers
    for c in range(min(8, n_chunks)):
        cs = c * chunk_size
        ce = cs + chunk_size
        is_val = (c % val_every == 0)
        base_color = '#FF6B6B' if is_val else '#4ECDC4'

        # Main region
        ax.barh(0, chunk_size - 2 * edge_buf, left=cs + edge_buf,
                color=base_color, alpha=0.7, height=0.5)
        # Edge buffers (gray = discarded)
        ax.barh(0, edge_buf, left=cs, color='#333', alpha=0.5, height=0.5)
        ax.barh(0, edge_buf, left=ce - edge_buf, color='#333', alpha=0.5, height=0.5)

        # Label
        label = 'Val' if is_val else 'Train'
        ax.text(cs + chunk_size / 2, 0, label, ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

    ax.set_xlim(0, min(8, n_chunks) * chunk_size)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Sample index', color=TEXT_COLOR, fontsize=10)
    ax.set_yticks([])
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.text(0.5, -0.7, f'Gray = edge buffer ({edge_buf} samples) → prevents temporal leakage',
            color='#8B949E', fontsize=8, ha='center', transform=ax.transAxes)

    # ── (1,1) CSI spectrogram (single channel over time) ──
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(PANEL_BG)
    ax.set_title('(e) CSI Signal — Single subcarrier over time',
                 color=ACCENT, fontsize=11, fontweight='bold')

    # Pick alive channel with highest variance
    alive_mask = feat_mean >= 0.001
    alive_indices = np.where(alive_mask)[0]
    variances = X[:, -1, :].var(axis=0)
    best_ch = alive_indices[np.argmax(variances[alive_indices])]

    seg = X[:min(1000, N), -1, best_ch]
    t_seg = ts[:min(1000, N)] - ts[0]
    ax.plot(t_seg, seg, color='#58A6FF', linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Time (s)', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Feature value', color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)

    node_id = best_ch // 256 + 1
    sub_id  = best_ch % 128
    ftype   = 'Phase' if (best_ch % 256) >= 128 else 'Amplitude'
    ax.text(0.02, 0.95, f'Node{node_id} {ftype}[{sub_id}]',
            transform=ax.transAxes, color='#FFD93D', fontsize=9,
            va='top', fontweight='bold')

    # ── (1,2) Per-session summary ──
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(PANEL_BG)
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title('(f) Per-session Summary', color=ACCENT, fontsize=13,
                 fontweight='bold', pad=10)

    headers = ['File', 'Samples', 'Duration']
    for col, h in enumerate(headers):
        ax.text(0.5 + col * 3.2, 9.0, h, color=ACCENT, fontsize=9,
                fontweight='bold', va='center')
    ax.axhline(y=8.5, xmin=0.03, xmax=0.97, color=GRID_COLOR, linewidth=1)

    for i, info in enumerate(data['files']):
        yp = 8.0 - i * 1.2
        fname = os.path.basename(info['path'])
        ax.text(0.5, yp, fname, color=TEXT_COLOR, fontsize=8, va='center',
                fontfamily='monospace')
        ax.text(3.7, yp, f'{info["n_samples"]:,}', color=TEXT_COLOR, fontsize=9,
                va='center', fontweight='bold')
        ax.text(6.9, yp, f'{info["duration"]:.0f}s', color=TEXT_COLOR, fontsize=9,
                va='center', fontweight='bold')

    # Total
    total_samples = sum(f['n_samples'] for f in data['files'])
    total_dur     = sum(f['duration'] for f in data['files'])
    yp = 8.0 - len(data['files']) * 1.2 - 0.5
    ax.axhline(y=yp + 0.4, xmin=0.03, xmax=0.97, color=GRID_COLOR, linewidth=1)
    ax.text(0.5, yp, 'TOTAL', color=ACCENT, fontsize=9, va='center', fontweight='bold')
    ax.text(3.7, yp, f'{total_samples:,}', color=ACCENT, fontsize=10,
            va='center', fontweight='bold')
    ax.text(6.9, yp, f'{total_dur:.0f}s', color=ACCENT, fontsize=10,
            va='center', fontweight='bold')

    fig.savefig(os.path.join(out_dir, '05_train_readiness.png'), dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print('  ✓ 05_train_readiness.png')

    return checks


# ══════════════════════════════════════════════════════════════
# Text summary report
# ══════════════════════════════════════════════════════════════

def write_summary(data, checks, out_dir):
    X, y, vis, ts = data['X'], data['y'], data['vis'], data['ts']
    N, T, F = X.shape

    lines = []
    lines.append('=' * 60)
    lines.append('  BÁO CÁO CHẤT LƯỢNG DỮ LIỆU CSI-POSE')
    lines.append('=' * 60)
    lines.append('')

    lines.append('1. TỔNG QUAN')
    lines.append(f'   Số samples       : {N:,}')
    lines.append(f'   Window size      : {T} frames (~{T/20:.1f}s)')
    lines.append(f'   Feature dim      : {F} (3 nodes × 2 × 128 subcarriers)')
    lines.append(f'   Keypoint joints  : 17 (COCO format)')
    lines.append(f'   Sessions         : {len(data["files"])}')
    lines.append(f'   Total duration   : {sum(f["duration"] for f in data["files"]):.0f}s')
    lines.append('')

    lines.append('2. READINESS CHECKS')
    for label, value, ok in checks:
        icon = 'PASS' if ok else 'FAIL'
        lines.append(f'   [{icon}] {label}: {value}')
    pass_count = sum(1 for _, _, ok in checks)
    lines.append(f'   → {pass_count}/{len(checks)} passed')
    lines.append('')

    lines.append('3. DATA RANGES')
    lines.append(f'   X  : [{X.min():.4f}, {X.max():.4f}]  mean={X.mean():.4f}  std={X.std():.4f}')
    lines.append(f'   y  : [{y.min():.4f}, {y.max():.4f}]')
    lines.append(f'   vis: [{vis.min():.4f}, {vis.max():.4f}]  mean={vis.mean():.3f}')
    lines.append('')

    feat_mean = X.mean(axis=(0, 1))
    n_dead = (feat_mean < 0.001).sum()
    lines.append('4. FEATURE HEALTH')
    lines.append(f'   Dead channels    : {n_dead}/{F} (mean < 0.001)')
    lines.append(f'   Alive channels   : {F - n_dead}/{F}')
    lines.append(f'   NaN in X         : {np.isnan(X).sum()}')
    lines.append(f'   Inf in X         : {np.isinf(X).sum()}')
    lines.append('')

    lines.append('5. KEYPOINT DIVERSITY')
    for j in range(17):
        sx = y[:, j, 0].std()
        sy = y[:, j, 1].std()
        flag = ' ⚠' if (sx < 0.02 and sy < 0.02) else ''
        lines.append(f'   {JOINT_NAMES[j]:12s}  std_x={sx:.4f}  std_y={sy:.4f}{flag}')
    lines.append('')

    lines.append('6. FILES')
    for info in data['files']:
        lines.append(f'   {info["path"]}')
        lines.append(f'     samples={info["n_samples"]:,}  duration={info["duration"]:.0f}s')
    lines.append('')
    lines.append('=' * 60)

    path = os.path.join(out_dir, 'report_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ report_summary.txt')
    # Print to console too
    print('\n'.join(lines))


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Visualize CSI-Pose data before training')
    parser.add_argument('--data', nargs='+',
                        default=['saved2', 'saved3'],
                        help='Paths to .npz files or directories')
    parser.add_argument('--out-dir', default='data_report',
                        help='Output directory for plots')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 55)
    print('DATA VISUALIZATION — CSI Pose Dataset')
    print('=' * 55)
    print(f'Output: {out_dir}/')
    print()

    # Load
    print('[1/5] Loading data...')
    data = load_data(args.data)
    print(f'  Total: {len(data["X"]):,} samples from {len(data["files"])} files')
    print()

    # Plot 1
    print('[2/5] Data overview & structure...')
    plot_data_overview(data, out_dir)

    # Plot 2
    print('[3/5] CSI temporal patterns...')
    plot_csi_temporal(data, out_dir)

    # Plot 3
    print('[4/5] Feature analysis...')
    plot_feature_analysis(data, out_dir)

    # Plot 4
    print('[5/5] Keypoint quality & train readiness...')
    plot_keypoint_quality(data, out_dir)
    checks = plot_train_readiness(data, out_dir)

    # Summary
    print()
    write_summary(data, checks, out_dir)
    print(f'\nDone! Mở thư mục {out_dir}/ để xem kết quả.')


if __name__ == '__main__':
    main()
