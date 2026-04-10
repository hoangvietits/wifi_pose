"""
train_v4.py  —  CSIPose Training (Interleaved Chunk Split + TCN)
═══════════════════════════════════════════════════════════════════
Thay đổi so với train2.py:

SPLIT STRATEGY:
  Cũ: 3-fold LOSO → val loss spike, model collapse về mean pose
  Mới: Interleaved Chunk Split — tất cả sessions đều có mặt trong
       cả train lẫn val, split theo thời gian trong session với gap buffer

MODEL:
  Cũ: Conv1d + AdaptiveAvgPool + Transformer (478K params)
  Mới: Input proj + TCN stack dilations=[1,2,4,8] (~160K params)

DATA PIPELINE:
  - FeatureCleaner: loại 222 kênh chết, z-norm per-feature
  - Stride=20 frames (giảm redundancy 59× → ~4.5×)
  - Mixup augmentation (alpha=0.3)

TRAINING:
  - OneCycleLR thay CosineAnnealingLR (ổn định hơn trên dataset nhỏ)
  - Gradient clipping 2.0 (chặt hơn)
  - EMA (Exponential Moving Average) model weights
  - Early stopping 20 epoch patience
  - Log MPJPE (metric thực sự) song song với loss

METRIC:
  - Val MPJPE (mean per-joint position error) — thang đo chính
  - Target: 0.06-0.10 normalized (~6-10% of image width) = acceptable prototype
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import math

from csi_pose_dataset import build_datasets, mixup_collate
from model_v2 import build_model_v2
# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CFG = dict(
    DATA_DIR      = 'saved3',  # thư mục chứa aligned_*.npz
    FILE_PATTERN  = 'aligned_*.npz',

    # Split (giảm chunk/edge/stride để tận dụng nhiều data hơn)
    CHUNK_SIZE    = 100,      # ~3.3s per chunk
    STRIDE        = 5,        # stride giữa window
    EDGE_BUF      = 15,       # buffer biên chunk (~0.5s mỗi bên)
    VAL_EVERY     = 4,        # 1/4 chunk = val

    # Model
    TCN_CHANNELS  = 64,
    DILATIONS     = [1, 2, 4, 8],
    DROPOUT       = 0.2,

    # Training
    BATCH_SIZE    = 32,
    EPOCHS        = 150,
    LR            = 3e-4,
    WEIGHT_DECAY  = 1e-3,
    GRAD_CLIP     = 2.0,
    EMA_DECAY     = 0.995,

    # Loss weights
    W_BONE        = 0.15,
    W_DIV         = 0.15,     # diversity loss weight (anti mean-pose collapse)
    VIS_THRESH    = 0.35,

    # Mixup
    MIXUP_ALPHA   = 0.3,

    # Early stopping
    PATIENCE      = 25,

    # Output
    RESULTS_DIR   = 'results_v5',
    CKPT_DIR      = 'checkpoints_v5',
    PLOT_EVERY    = 10,
    SAVE_EVERY    = 10,
)

# ═══════════════════════════════════════════════════════════════════
# COCO-17 metadata
# ═══════════════════════════════════════════════════════════════════
JOINT_NAMES = [
    'Nose','L.Eye','R.Eye','L.Ear','R.Ear',
    'L.Sho','R.Sho','L.Elb','R.Elb','L.Wri','R.Wri',
    'L.Hip','R.Hip','L.Kne','R.Kne','L.Ank','R.Ank'
]
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
BONE_PAIRS = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),(5,11),(6,12),
]

# ═══════════════════════════════════════════════════════════════════
# Loss — Wing Loss + Bone + Diversity (anti mean-pose collapse)
# ═══════════════════════════════════════════════════════════════════

def wing_loss_fn(pred, gt, w=0.1, epsilon=2.0):
    """
    Wing Loss for landmark regression (Feng et al., 2018).
    - |x| < w: log term → large gradient for small errors (breaks mean collapse)
    - |x| >= w: L1 → robust to outliers
    MSE has gradient ∝ error → vanishing near mean → model stalls at mean pose.
    Wing Loss has ~constant gradient for small errors → keeps pushing.
    """
    diff = pred - gt
    abs_diff = diff.abs()
    C = w - w * math.log(1 + w / epsilon)
    loss = torch.where(
        abs_diff < w,
        w * torch.log(1 + abs_diff / epsilon),
        abs_diff - C
    )
    return loss


def pose_loss(pred, gt, vis, w_bone=0.15, w_div=0.15, vis_thresh=0.35):
    w = (vis > vis_thresh).float().unsqueeze(-1)       # (B, 17, 1)
    n_vis = w.sum().clamp(min=1.0)

    # Wing Loss instead of MSE
    loss_wing = (w * wing_loss_fn(pred, gt)).sum() / n_vis

    # Bone length consistency
    bone_loss = torch.tensor(0.0, device=pred.device)
    for a, b in BONE_PAIRS:
        pred_l = (pred[:,a] - pred[:,b]).norm(dim=-1)
        gt_l   = (gt[:,a]   - gt[:,b]  ).norm(dim=-1)
        pv     = ((vis[:,a] > vis_thresh) & (vis[:,b] > vis_thresh)).float()
        np_    = pv.sum().clamp(min=1.0)
        bone_loss += (pv * (pred_l - gt_l).abs()).sum() / np_
    bone_loss /= len(BONE_PAIRS)

    # Diversity loss: penalize when pred variance << GT variance
    # This explicitly prevents mean-pose collapse
    pred_std = pred.std(dim=0)                          # (17, 2)
    gt_std   = gt.std(dim=0).detach()                   # (17, 2)
    # Per-joint: pred should have at least 50% of GT's variance
    div_loss = F.relu(gt_std * 0.5 - pred_std).mean()

    return loss_wing + w_bone * bone_loss + w_div * div_loss


def compute_mpjpe(pred, gt, vis, vis_thresh=0.35):
    """Mean Per-Joint Position Error (normalized coords)."""
    mask = vis > vis_thresh                              # (B, 17)
    err  = (pred - gt).norm(dim=-1)                     # (B, 17)
    if mask.sum() == 0:
        return float('nan')
    return (err * mask.float()).sum().item() / mask.float().sum().item()


# ═══════════════════════════════════════════════════════════════════
# EMA helper
# ═══════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model, decay=0.995):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay     = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()


# ═══════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════

def plot_loss_curves(train_l, val_l, train_mpjpe, val_mpjpe, epoch, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   facecolor='#0A0A0A')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1A1A1A')
        ax.tick_params(colors='#AAAAAA')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        for s in ('top','right'):
            ax.spines[s].set_visible(False)

    epochs = range(1, len(train_l) + 1)
    ax1.plot(epochs, train_l, '#4ECDC4', lw=1.5, label='Train loss')
    ax1.plot(epochs, val_l,   '#FF6B6B', lw=1.5, label='Val loss')
    ax1.set_title('Loss', color='white', fontsize=11)
    ax1.legend(facecolor='#2A2A2A', edgecolor='#555', labelcolor='white')
    ax1.set_xlabel('Epoch', color='#AAAAAA')

    ax2.plot(epochs, train_mpjpe, '#4ECDC4', lw=1.5, label='Train MPJPE')
    ax2.plot(epochs, val_mpjpe,   '#FF6B6B', lw=1.5, label='Val MPJPE')
    ax2.axhline(0.08, color='#FFD700', lw=1.0, ls='--', alpha=0.7, label='Target 0.08')
    ax2.set_title('MPJPE (normalized)', color='white', fontsize=11)
    ax2.legend(facecolor='#2A2A2A', edgecolor='#555', labelcolor='white')
    ax2.set_xlabel('Epoch', color='#AAAAAA')

    plt.tight_layout(pad=1.5)
    path = os.path.join(out_dir, f'curves_epoch_{epoch:03d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='#0A0A0A')
    plt.close()
    return path


def draw_skeleton_mini(ax, kp, vis, color):
    ax.set_facecolor('#111')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(1.05, -0.05)
    ax.axis('off')
    for a, b in COCO_SKELETON:
        if vis[a] > 0.3 and vis[b] > 0.3:
            ax.plot([kp[a,0],kp[b,0]], [kp[a,1],kp[b,1]],
                    color=color, lw=2.0, alpha=0.85)
    visible = vis > 0.3
    ax.scatter(kp[visible,0], kp[visible,1],
               c=color, s=18, zorder=5)


def plot_skeletons(model, val_loader, device, epoch, out_dir, n_samples=6):
    # Trong plot_skeletons, sau khi có y_gt và y_pred
    # gt = y_gt[0].cpu().numpy()    # (17, 2)
    # pr = y_pred[0].cpu().numpy()  # (17, 2)

    # print("GT  x range:", gt[:,0].min().round(3), "→", gt[:,0].max().round(3))
    # print("GT  y range:", gt[:,1].min().round(3), "→", gt[:,1].max().round(3))
    # print("Pred x range:", pr[:,0].min().round(3), "→", pr[:,0].max().round(3))
    # print("Pred y range:", pr[:,1].min().round(3), "→", pr[:,1].max().round(3))

    # # Kiểm tra xem head (joint 0) có nằm trên không
    # print("Nose y (GT):", gt[0,1].round(3), " — phải gần 0 nếu y↓")
    # print("Ankle y (GT):", gt[15,1].round(3), " — phải gần 1 nếu y↓")
    model.eval()
    x, y_gt, vis = next(iter(val_loader))
    with torch.no_grad():
        y_pred, _ = model(x[:n_samples].to(device))
    y_pred = y_pred.cpu().numpy()
    y_gt   = y_gt[:n_samples].numpy()
    vis_np = vis[:n_samples].numpy()

    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6),
                             facecolor='#0A0A0A')
    for i in range(n_samples):
        draw_skeleton_mini(axes[0,i], y_gt[i],   vis_np[i], '#44FF88')
        draw_skeleton_mini(axes[1,i], y_pred[i],  vis_np[i], '#FF4444')
        axes[0,i].set_title(f'GT #{i}',   color='white', fontsize=8)
        axes[1,i].set_title(f'Pred #{i}', color='white', fontsize=8)

    plt.suptitle(f'Skeleton comparison — Epoch {epoch}',
                 color='white', fontsize=11, y=1.01)
    plt.tight_layout(pad=0.4)
    path = os.path.join(out_dir, f'skel_epoch_{epoch:03d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='#0A0A0A')
    plt.close()
    return path


def plot_per_joint_mpjpe(model, val_loader, device, epoch, out_dir):
    model.eval()
    joint_errs = np.zeros(17)
    joint_vis  = np.zeros(17)

    with torch.no_grad():
        for x, y_gt, vis in val_loader:
            y_pred, _ = model(x.to(device))
            err = (y_pred.cpu() - y_gt).norm(dim=-1).numpy()   # (B, 17)
            mask = (vis > 0.35).numpy()
            joint_errs += (err * mask).sum(0)
            joint_vis  += mask.sum(0)

    per_joint = np.where(joint_vis > 0, joint_errs / joint_vis, np.nan)

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0A0A0A')
    ax.set_facecolor('#1A1A1A')
    colors = ['#FF6B6B' if e > 0.10 else '#4ECDC4' for e in per_joint]
    bars = ax.bar(JOINT_NAMES, per_joint, color=colors, width=0.7)
    ax.axhline(0.08, color='#FFD700', lw=1.2, ls='--', alpha=0.8, label='Target 0.08')
    ax.set_title(f'Per-joint MPJPE — Epoch {epoch}', color='white', fontsize=11)
    ax.tick_params(axis='x', rotation=45, colors='#AAAAAA', labelsize=8)
    ax.tick_params(axis='y', colors='#AAAAAA')
    for s in ('top','right'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.legend(facecolor='#2A2A2A', edgecolor='#555', labelcolor='white')
    plt.tight_layout()
    path = os.path.join(out_dir, f'perjoint_epoch_{epoch:03d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='#0A0A0A')
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Strategy: Interleaved Chunk Split (chunk={CFG['CHUNK_SIZE']}, "
          f"stride={CFG['STRIDE']}, val_every={CFG['VAL_EVERY']})\n")

    os.makedirs(CFG['RESULTS_DIR'], exist_ok=True)
    os.makedirs(CFG['CKPT_DIR'],    exist_ok=True)

    # ── Build datasets ──────────────────────────────────────────────
    print("Building datasets...")
    train_ds, val_ds, cleaner = build_datasets(
        data_dir      = CFG['DATA_DIR'],
        file_pattern  = CFG['FILE_PATTERN'],
        chunk_size    = CFG['CHUNK_SIZE'],
        stride        = CFG['STRIDE'],
        edge_buf      = CFG['EDGE_BUF'],
        val_every     = CFG['VAL_EVERY'],
        save_cleaner  = os.path.join(CFG['CKPT_DIR'], 'feature_cleaner'),
        augment_train = True,
    )

    feature_dim = cleaner.n_alive
    print(f"\nFeature dim after cleaning: {feature_dim}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size  = CFG['BATCH_SIZE'],
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = mixup_collate(alpha=CFG['MIXUP_ALPHA']),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = CFG['BATCH_SIZE'],
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True,
    )

    # ── Model ───────────────────────────────────────────────────────
    torch.manual_seed(42)
    model = build_model_v2(
        feature_cleaner = cleaner,
        C               = CFG['TCN_CHANNELS'],
        dilations       = CFG['DILATIONS'],
        dropout         = CFG['DROPOUT'],
    ).to(device)
    ema = EMA(model, decay=CFG['EMA_DECAY'])

    n_params = model.count_params()
    print(f"\nModel params: {n_params:,}")

    # ── Optimizer + Scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG['LR'],
        weight_decay = CFG['WEIGHT_DECAY'],
    )

    # OneCycleLR: warmup 10% → peak → cosine decay
    # Ổn định hơn CosineAnnealingLR trên dataset nhỏ
    total_steps = CFG['EPOCHS'] * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr       = CFG['LR'],
        total_steps  = total_steps,
        pct_start    = 0.10,
        anneal_strategy = 'cos',
        final_div_factor = 100,
    )

    # ── Training state ──────────────────────────────────────────────
    best_val_mpjpe = float('inf')
    no_improve     = 0
    train_losses, val_losses     = [], []
    train_mpjpes, val_mpjpes     = [], []
    save_path = os.path.join(CFG['CKPT_DIR'], 'best_model.pt')
    ema_save  = os.path.join(CFG['CKPT_DIR'], 'best_ema.pt')

    print(f"\n{'Epoch':>6} | {'Train L':>9} | {'Val L':>9} | "
          f"{'Train MPJPE':>11} | {'Val MPJPE':>11} | {'LR':>8}")
    print('─' * 70)

    for epoch in range(1, CFG['EPOCHS'] + 1):

        # ── Train ────────────────────────────────────────────────────
        model.train()
        t_loss_sum = 0.0
        t_mpjpe_sum, t_mpjpe_n = 0.0, 0

        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:3d}/{CFG['EPOCHS']}",
                    leave=False)
        for x, y_true, v_true in pbar:
            x      = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            v_true = v_true.to(device, non_blocking=True)

            y_pred, v_logits = model(x)

            loss_p = pose_loss(y_pred, y_true, v_true,
                               w_bone=CFG['W_BONE'], w_div=CFG['W_DIV'],
                               vis_thresh=CFG['VIS_THRESH'])
            loss_v = F.binary_cross_entropy_with_logits(v_logits, v_true)
            loss   = loss_p + 0.2 * loss_v

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['GRAD_CLIP'])
            optimizer.step()
            scheduler.step()
            ema.update(model)

            t_loss_sum += loss.item()
            mpjpe = compute_mpjpe(y_pred.detach(), y_true, v_true)
            if not np.isnan(mpjpe):
                t_mpjpe_sum += mpjpe
                t_mpjpe_n   += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             mpjpe=f'{mpjpe:.4f}')

        avg_train_loss  = t_loss_sum / len(train_loader)
        avg_train_mpjpe = t_mpjpe_sum / max(t_mpjpe_n, 1)

        # ── Validation (dùng EMA model) ──────────────────────────────
        ema.ema_model.eval()
        v_loss_sum = 0.0
        v_mpjpe_sum, v_mpjpe_n = 0.0, 0

        with torch.no_grad():
            for x, y_true, v_true in val_loader:
                x      = x.to(device, non_blocking=True)
                y_true = y_true.to(device, non_blocking=True)
                v_true = v_true.to(device, non_blocking=True)

                y_pred, v_logits = ema.ema_model(x)
                loss_p = pose_loss(y_pred, y_true, v_true,
                                   w_bone=CFG['W_BONE'], w_div=CFG['W_DIV'],
                                   vis_thresh=CFG['VIS_THRESH'])
                loss_v = F.binary_cross_entropy_with_logits(v_logits, v_true)
                v_loss_sum += (loss_p + 0.2 * loss_v).item()

                mpjpe = compute_mpjpe(y_pred, y_true, v_true)
                if not np.isnan(mpjpe):
                    v_mpjpe_sum += mpjpe
                    v_mpjpe_n   += 1

        avg_val_loss  = v_loss_sum / len(val_loader)
        avg_val_mpjpe = v_mpjpe_sum / max(v_mpjpe_n, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_mpjpes.append(avg_train_mpjpe)
        val_mpjpes.append(avg_val_mpjpe)

        cur_lr = scheduler.get_last_lr()[0]
        mark   = ''
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            torch.save(ema.state_dict(),   ema_save)
            mark = '  ← best'
        else:
            no_improve += 1

        print(f"{epoch:6d} | {avg_train_loss:9.5f} | {avg_val_loss:9.5f} | "
              f"{avg_train_mpjpe:11.5f} | {avg_val_mpjpe:11.5f} | "
              f"{cur_lr:8.2e}{mark}")

        # ── Plots ─────────────────────────────────────────────────────
        if epoch % CFG['PLOT_EVERY'] == 0 or epoch == CFG['EPOCHS']:
            ep_dir = os.path.join(CFG['RESULTS_DIR'], f'epoch_{epoch:03d}')
            os.makedirs(ep_dir, exist_ok=True)
            plot_loss_curves(train_losses, val_losses,
                             train_mpjpes, val_mpjpes, epoch, ep_dir)
            plot_skeletons(ema.ema_model, val_loader, device, epoch, ep_dir)
            plot_per_joint_mpjpe(ema.ema_model, val_loader, device, epoch, ep_dir)
            print(f"  ↳ Plots → {ep_dir}/")

        # ── Checkpoint ────────────────────────────────────────────────
        if epoch % CFG['SAVE_EVERY'] == 0:
            ckpt = os.path.join(CFG['CKPT_DIR'], f'epoch_{epoch:03d}.pt')
            torch.save({
                'epoch'     : epoch,
                'model'     : model.state_dict(),
                'ema'       : ema.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'val_mpjpe' : avg_val_mpjpe,
                'cfg'       : CFG,
            }, ckpt)

        # ── Early stopping ────────────────────────────────────────────
        if no_improve >= CFG['PATIENCE']:
            print(f"\n⏹  Early stopping tại epoch {epoch} "
                  f"(val MPJPE không cải thiện {CFG['PATIENCE']} epoch)")
            break

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"Training xong!")
    print(f"Best Val MPJPE : {best_val_mpjpe:.5f}")
    print(f"Target range   : 0.06 – 0.10 (normalized)")
    print(f"Model saved    : {save_path}")
    print(f"EMA saved      : {ema_save}")
    print(f"{'═'*55}")