"""
csi_pose_dataset_v4.py  —  Interleaved Chunk Split + Feature Cleaning
═══════════════════════════════════════════════════════════════════════
Giải quyết 3 vấn đề chính của v3:

1. DATA LEAKAGE: stride cũ 0.67 frame (98% overlap) → stride 20 frames (~4.5×
   decorrelation time). Tất cả window trong cùng 1 chunk đảm bảo không overlap
   với window của chunk lân cận (gap biên 40 frames = 1.3s).

2. SPLIT STRATEGY: Bỏ 3-fold LOSO (val loss spike vì distribution shift quá lớn
   giữa sessions). Thay bằng INTERLEAVED CHUNK SPLIT:
   - Mỗi session chia thành 44 chunk × 200 windows (~6.7s/chunk)
   - Chunk index % 4 == 0 → val (25%), còn lại → train (75%)
   - Min gap train↔val = ~5s (decorrelated hoàn toàn)
   - Tất cả 3 session đều có mặt trong cả train lẫn val
     → model thấy mọi recording condition khi train
     → val đại diện cho distribution thực, không bị domain shift

3. FEATURE CLEANING: Loại bỏ 222/768 kênh chết (mean < 0.001),
   rồi z-norm per-feature để các kênh có scale đồng đều.
   Dead mask được fit trên TRAIN SET, áp dụng cho val (tránh leakage).

Kết quả: ~396 train + ~132 val windows (3 sessions, stride=20)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import os
from typing import Optional, Tuple

FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# ─────────────────────────────────────────────────────────────────────────────
# Feature cleaner — fit trên train, transform trên cả train + val
# ─────────────────────────────────────────────────────────────────────────────

class FeatureCleaner:
    """
    Bước 1: Loại kênh chết (mean < dead_thresh trên tập fit)
    Bước 2: Z-normalization per surviving feature
    """
    def __init__(self, dead_thresh: float = 0.001):
        self.dead_thresh = dead_thresh
        self.alive_mask  : Optional[np.ndarray] = None   # (F,) bool
        self.mean_       : Optional[np.ndarray] = None   # (F_alive,)
        self.std_        : Optional[np.ndarray] = None   # (F_alive,)
        self.n_alive     : int = 0
        self.n_sub       : int = 0    # subcarrier count dùng khi thu (để inference biết pad bao nhiêu)

    def fit(self, X: np.ndarray) -> 'FeatureCleaner':
        """X: (N, T, F)  — F = N_LINKS * 2 * n_sub"""
        feat_mean = X.mean(axis=(0, 1))                  # (F,)
        self.alive_mask = feat_mean >= self.dead_thresh
        self.n_alive    = int(self.alive_mask.sum())
        # Suy ra n_sub từ feature dimension (F = 3 * 2 * n_sub)
        self.n_sub      = X.shape[2] // 6

        X_alive = X[:, :, self.alive_mask]               # (N, T, F_alive)
        flat    = X_alive.reshape(-1, self.n_alive)
        self.mean_ = flat.mean(axis=0)
        self.std_  = flat.std(axis=0).clip(min=1e-6)
        print(f"  FeatureCleaner: {X.shape[2]} → {self.n_alive} features "
              f"({X.shape[2] - self.n_alive} dead channels removed)")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, T, F) → (N, T, F_alive) z-normed"""
        assert self.alive_mask is not None, "Call fit() first"
        X_alive = X[:, :, self.alive_mask]
        return ((X_alive - self.mean_) / self.std_).astype(np.float32)

    def save(self, path: str):
        np.savez(path,
                 alive_mask=self.alive_mask,
                 mean=self.mean_,
                 std=self.std_,
                 n_sub=np.int32(self.n_sub))

    @classmethod
    def load(cls, path: str) -> 'FeatureCleaner':
        fc = cls()
        data = np.load(path)
        fc.alive_mask = data['alive_mask']
        fc.mean_      = data['mean']
        fc.std_       = data['std']
        fc.n_alive    = int(fc.alive_mask.sum())
        fc.n_sub      = int(data['n_sub']) if 'n_sub' in data else (len(fc.alive_mask) // 6)
        return fc


# ─────────────────────────────────────────────────────────────────────────────
# Build interleaved chunk indices
# ─────────────────────────────────────────────────────────────────────────────

def build_interleaved_indices(
    n_windows    : int,
    chunk_size   : int = 200,    # ~6.7s @ 30fps stride
    stride       : int = 20,     # stride giữa các window trong chunk
    edge_buf     : int = 40,     # buffer biên chunk để tránh leakage
    val_every    : int = 4,      # chunk % val_every == 0 → val
    file_offset  : int = 0,      # offset để tránh trùng index khi concat nhiều file
) -> Tuple[list, list]:
    """
    Trả về (train_indices, val_indices) là danh sách index vào mảng gốc.

    Layout ví dụ (chunk_size=200, val_every=4):
      chunk 0 → VAL   (index 0..199)
      chunk 1 → train (200..399)
      chunk 2 → train (400..599)
      chunk 3 → train (600..799)
      chunk 4 → VAL   (800..999)
      ...

    Gap giữa train và val: edge_buf windows ở mỗi biên chunk
    → Min temporal gap ≈ 2 × edge_buf × dt ≈ 2 × 40 × 34ms ≈ 2.7s
    """
    n_chunks    = n_windows // chunk_size
    train_idx   = []
    val_idx     = []

    for c in range(n_chunks):
        cs = c * chunk_size
        ce = cs + chunk_size
        indices = list(range(
            cs + edge_buf,
            ce - edge_buf - 40,    # 40 = window width
            stride
        ))
        if c % val_every == 0:
            val_idx.extend([i + file_offset for i in indices])
        else:
            train_idx.extend([i + file_offset for i in indices])

    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CSIPoseDatasetV4(Dataset):
    """
    Args:
        X_clean  : (N_total, T, F_alive) — đã qua FeatureCleaner.transform()
        y_all    : (N_total, 17, 2)
        vis_all  : (N_total, 17)
        indices  : list of int — subset indices (train hoặc val)
        augment  : bool
    """

    def __init__(self,
                 X_clean  : np.ndarray,
                 y_all    : np.ndarray,
                 vis_all  : np.ndarray,
                 indices  : list,
                 augment  : bool = False):
        self.X       = X_clean[indices]
        self.y       = y_all[indices]
        self.vis     = vis_all[indices]
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        v = torch.from_numpy(self.vis[idx]).float()

        if self.augment:
            # Gaussian noise (nhẹ hơn vì đã z-norm)
            x = x + torch.randn_like(x) * 0.03

            # Time jitter ±3 frame
            shift = int(np.random.randint(-3, 4))
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

            # Amplitude scaling (z-norm rồi thì scale ở feature level)
            x = x * float(np.random.uniform(0.92, 1.08))

            # Multiplicative feature noise (thay vì zero-out sau z-norm)
            if np.random.rand() < 0.5:
                noise = torch.empty(x.shape[-1]).uniform_(0.90, 1.10)
                x = x * noise

            # Mixup-lite: blend với 1 sample ngẫu nhiên khác trong batch
            # (được xử lý ở level collate_fn nếu muốn full mixup)

            # Horizontal flip (50%)
            if np.random.rand() < 0.5:
                y = y[FLIP_MAP]
                v = v[FLIP_MAP]
                y[:, 0] = 1.0 - y[:, 0]

            # Scale jitter keypoint
            scale  = float(np.random.uniform(0.95, 1.05))
            center = y.mean(dim=0, keepdim=True)
            y      = torch.clamp((y - center) * scale + center, 0.0, 1.0)

        return x, y, v


# ─────────────────────────────────────────────────────────────────────────────
# Factory function — build train/val dataset từ nhiều file
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    data_dir     : str  = 'saved3',
    file_pattern : str  = 'aligned_*.npz',
    chunk_size   : int  = 200,
    stride       : int  = 10,
    edge_buf     : int  = 40,
    val_every    : int  = 4,
    cleaner_path : Optional[str] = None,   # None = fit mới, str = load từ file
    save_cleaner : Optional[str] = None,   # lưu cleaner để dùng lại lúc inference
    augment_train: bool = True,
    verbose      : bool = True,
) -> Tuple['CSIPoseDatasetV4', 'CSIPoseDatasetV4', 'FeatureCleaner']:
    """
    Returns: (train_dataset, val_dataset, feature_cleaner)
    """
    files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy '{file_pattern}' trong '{data_dir}'")

    X_list, y_list, vis_list = [], [], []
    train_idx_all, val_idx_all = [], []
    offset = 0

    for f in files:
        if verbose:
            print(f"  Loading {os.path.basename(f)} ...")
        d       = np.load(f)
        X_f     = d['X'].astype(np.float32)
        y_f     = d['y'][:, :, :2].astype(np.float32)
        vis_f   = d.get('visibility', d.get('vis',
                    np.ones((len(X_f), 17), dtype=np.float32))).astype(np.float32)

        t_idx, v_idx = build_interleaved_indices(
            n_windows   = len(X_f),
            chunk_size  = chunk_size,
            stride      = stride,
            edge_buf    = edge_buf,
            val_every   = val_every,
            file_offset = offset,
        )
        train_idx_all.extend(t_idx)
        val_idx_all.extend(v_idx)

        X_list.append(X_f)
        y_list.append(y_f)
        vis_list.append(vis_f)
        offset += len(X_f)

    X_all   = np.concatenate(X_list,   axis=0)
    y_all   = np.concatenate(y_list,   axis=0)
    vis_all = np.concatenate(vis_list, axis=0)

    if verbose:
        print(f"\n  Raw dataset: {len(X_all)} total windows")
        print(f"  Train indices: {len(train_idx_all)} | Val indices: {len(val_idx_all)}")

    # Feature cleaning
    if cleaner_path and os.path.exists(cleaner_path + '.npz'):
        cleaner = FeatureCleaner.load(cleaner_path + '.npz')
        if verbose:
            print(f"  Loaded FeatureCleaner from {cleaner_path}.npz "
                  f"({cleaner.n_alive} alive features)")
    else:
        cleaner = FeatureCleaner()
        cleaner.fit(X_all[train_idx_all])   # fit CHỈ trên train set

    X_clean = cleaner.transform(X_all)     # apply trên toàn bộ

    if save_cleaner:
        cleaner.save(save_cleaner)
        if verbose:
            print(f"  Saved FeatureCleaner → {save_cleaner}.npz")

    train_ds = CSIPoseDatasetV4(X_clean, y_all, vis_all, train_idx_all, augment=augment_train)
    val_ds   = CSIPoseDatasetV4(X_clean, y_all, vis_all, val_idx_all,   augment=False)

    if verbose:
        print(f"\n✅ Dataset ready:")
        print(f"   Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
        print(f"   Feature dim: {X_all.shape[2]} → {cleaner.n_alive} (after cleaning)")
        print(f"   Window shape: {X_clean[0].shape}")

    return train_ds, val_ds, cleaner


# ─────────────────────────────────────────────────────────────────────────────
# Mixup collate fn (optional — dùng trong DataLoader)
# ─────────────────────────────────────────────────────────────────────────────

def mixup_collate(alpha: float = 0.3):
    """
    Trả về collate_fn áp dụng Mixup CHỈ trên CSI input x.
    Không blend keypoints y vì tạo ra pose phi vật lý.
    alpha: tham số Beta distribution (0.3 = mixup nhẹ, phù hợp dataset nhỏ)
    """
    def _collate(batch):
        x_list, y_list, v_list = zip(*batch)
        x  = torch.stack(x_list)
        y  = torch.stack(y_list)
        v  = torch.stack(v_list)

        if np.random.rand() < 0.5:
            lam  = float(np.random.beta(alpha, alpha))
            lam  = max(lam, 1 - lam)  # đảm bảo lam >= 0.5 → giữ label chính
            perm = torch.randperm(len(x))
            x    = lam * x  + (1 - lam) * x[perm]
            # y, v giữ nguyên từ sample chính (lam >= 0.5)

        return x, y, v
    return _collate