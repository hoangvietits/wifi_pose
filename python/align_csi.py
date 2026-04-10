"""
align_csi.py  (v6 — video_offset tự động + MAX_LAG tighter)
════════════════════════════════════════════════════════════════════

Thay đổi so với v5:

  FIX A — video_offset tự động:
    Đọc key 'video_offset' từ csi_raw_*.npz (được test.py v5 lưu).
    Nếu file cũ không có key này → fallback về --video-offset như cũ.
    Không cần đoán mò nữa.

  FIX B — MAX_LAG_SEC giảm từ 300ms → 150ms:
    Sau khi CSI rate ~20 Hz (dedup fix), mỗi sample cách nhau 50ms.
    searchsorted sai lệch tối đa ±25ms → 150ms đủ rộng mà vẫn chặt.

Output: saved3/aligned_YYYYMMDD_HHMMSS.npz
  X          : (N, 40, 768)
  y          : (N, 17, 2)
  visibility : (N, 17)
  timestamps : (N,)

Cách dùng:
  # File mới (test.py v5): offset tự động
  python align_csi.py \\
      --video ruview_sessions/video_*.mp4 \\
      --csi   ruview_sessions/csi_raw_*.npz \\
      --out   saved3/

  # File cũ (không có video_offset trong npz): truyền tay
  python align_csi.py --video x.mp4 --csi x.npz --video-offset 6.0
"""

import argparse
import os
import time
import numpy as np
import cv2
from datetime import datetime
from queue import Queue
from threading import Thread

try:
    from scipy.ndimage import median_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠  scipy chưa cài → Hampel chậm hơn. pip install scipy")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("⚠  ultralytics chưa cài. pip install ultralytics")

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

WIN_SIZE    = 40
MAX_LAG_SEC = 0.150          # FIX B: giảm từ 300ms → 150ms
MIN_VIS     = 0.35
MIN_JOINTS  = 9
YOLO_CONF   = 0.45
BATCH_SIZE  = 8
TARGET_FPS  = 0

OUTLIER_JUMP_THRESH = 0.20
MIN_CSI_RATE_HZ     = 15.0

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

JOINT_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle',
]


# ══════════════════════════════════════════════════════════════
# Hampel filter
# ══════════════════════════════════════════════════════════════
def clean_csi_window(window: np.ndarray,
                     win_size: int = 7,
                     n_sigmas: float = 3.0) -> np.ndarray:
    if SCIPY_OK:
        med  = median_filter(window, size=(win_size, 1), mode='nearest')
        mad  = 1.4826 * median_filter(
                   np.abs(window - med), size=(win_size, 1), mode='nearest')
        mask = (mad > 1e-8) & (np.abs(window - med) > n_sigmas * mad)
        out  = window.copy()
        out[mask] = med[mask]
        return out
    else:
        def _h1d(x):
            x = x.copy(); k = win_size // 2
            for i in range(len(x)):
                lo, hi = max(0,i-k), min(len(x),i+k+1)
                w = x[lo:hi]; m = np.median(w)
                mad = 1.4826*np.median(np.abs(w-m))
                if mad > 0 and np.abs(x[i]-m) > n_sigmas*mad:
                    x[i] = m
            return x
        return np.apply_along_axis(_h1d, 0, window)


# ══════════════════════════════════════════════════════════════
# Frame reader thread
# ══════════════════════════════════════════════════════════════
def _reader_worker(video_path: str, q: Queue):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            q.put(None); break
        q.put((idx, frame))
        idx += 1
    cap.release()


# ══════════════════════════════════════════════════════════════
# YOLO batch processing
# ══════════════════════════════════════════════════════════════
def _process_batch(batch, yolo_model, W, H, use_fp16):
    frames  = [b[1] for b in batch]
    results = yolo_model(frames, verbose=False, conf=YOLO_CONF, half=use_fp16)
    out = []
    for i, (fidx, frame, window_clean, t_video) in enumerate(batch):
        r = results[i]
        if (r.keypoints is None or r.keypoints.xy is None
                or len(r.keypoints.xy) == 0):
            out.append(None); continue

        if r.boxes is not None and len(r.boxes.xywh) > 1:
            areas = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]
            best  = int(areas.argmax().item())
        else:
            best = 0

        if best >= len(r.keypoints.xy):
            out.append(None); continue

        kps_xy   = r.keypoints.xy[best].cpu().numpy().astype(np.float32)
        kps_conf = r.keypoints.conf[best].cpu().numpy().astype(np.float32)

        if kps_xy.shape[0] != 17:
            out.append(None); continue

        kps_norm = kps_xy.copy()
        kps_norm[:, 0] /= W
        kps_norm[:, 1] /= H

        if np.any(kps_norm < -0.05) or np.any(kps_norm > 1.05):
            out.append(None); continue
        kps_norm = np.clip(kps_norm, 0.0, 1.0)

        n_vis = int(np.sum(kps_conf > MIN_VIS))
        if n_vis < MIN_JOINTS:
            out.append(('low_vis',)); continue

        out.append((window_clean, kps_norm, kps_conf, t_video))
    return out


# ══════════════════════════════════════════════════════════════
# Load CSI — FIX A: đọc video_offset từ npz
# ══════════════════════════════════════════════════════════════
def load_csi(csi_path: str, override_offset: float = None):
    """
    Trả về (csi_ts, csi_feat, video_offset).

    video_offset ưu tiên:
      1. override_offset (--video-offset CLI) nếu được truyền
      2. key 'video_offset' trong npz (test.py v5)
      3. 0.0 fallback
    """
    data = np.load(csi_path)
    ts   = data['timestamps'].astype(np.float64)
    feat = data['features'].astype(np.float32)

    print(f"  CSI : {len(ts)} frames | {feat.shape}")
    print(f"        t={ts[0]:.3f} → {ts[-1]:.3f}  ({ts[-1]-ts[0]:.1f}s)")
    print(f"        feat [{feat.min():.3f}, {feat.max():.3f}]")

    if len(ts) >= 2:
        actual_rate = len(ts) / (ts[-1] - ts[0])
        print(f"        rate: {actual_rate:.1f} Hz", end='')
        if actual_rate < MIN_CSI_RATE_HZ:
            print(f"  ⚠  thấp")
        else:
            print("  ✓")

    bad = np.isnan(feat).any(1) | np.isinf(feat).any(1)
    if bad.sum():
        print(f"  ⚠  {bad.sum()} frames NaN/Inf")

    # FIX A
    if override_offset is not None:
        video_offset = override_offset
        print(f"  video_offset : {video_offset:+.3f}s  (CLI override)")
    elif 'video_offset' in data:
        video_offset = float(data['video_offset'])
        print(f"  video_offset : {video_offset:+.3f}s  ✓ (tự động từ npz)")
    else:
        video_offset = 0.0
        print(f"  video_offset : {video_offset:+.3f}s  "
              f"⚠  không có trong npz — dùng --video-offset nếu cần")

    return ts, feat, video_offset


# ══════════════════════════════════════════════════════════════
# Align chính
# ══════════════════════════════════════════════════════════════
def align(video_path, csi_ts, csi_feat, out_dir,
          yolo_model, use_fp16, skip_fps,
          video_offset, outlier_thresh):

    cap_p = cv2.VideoCapture(video_path)
    fps   = cap_p.get(cv2.CAP_PROP_FPS)
    total = int(cap_p.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap_p.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap_p.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_p.release()

    print(f"\n  Video: {total} frames @ {fps:.1f}fps | {W}×{H} | {total/fps:.1f}s")

    skip_every = max(1, round(fps/skip_fps)) if skip_fps > 0 else 1
    if skip_every > 1:
        print(f"  Skip: 1/{skip_every} (~{fps/skip_every:.1f}fps)")

    t0 = csi_ts[0] + video_offset
    print(f"  t0 = csi_ts[0]{video_offset:+.3f} = {t0:.3f}")
    print(f"  MAX_LAG = {MAX_LAG_SEC*1000:.0f}ms")

    fq = Queue(maxsize=256)
    reader = Thread(target=_reader_worker, args=(video_path, fq), daemon=True)
    reader.start()

    X_list, y_list, vis_list, ts_list = [], [], [], []
    n_np = n_lag = n_lv = n_sv = n_sk = n_jump = 0
    pending  = []
    t_print  = time.time()
    prev_kps = None

    def _flush():
        nonlocal n_np, n_lv, n_sv, n_jump, prev_kps
        if not pending: return
        res_list = _process_batch(pending, yolo_model, W, H, use_fp16)
        for res in res_list:
            if res is None:
                n_np += 1; prev_kps = None
            elif isinstance(res, tuple) and len(res) == 1:
                n_lv += 1
            elif isinstance(res, tuple) and len(res) == 4:
                wc, kn, kc, tv = res
                if prev_kps is not None:
                    vis_mask = kc > MIN_VIS
                    if vis_mask.sum() >= 4:
                        delta = np.linalg.norm(
                            kn[vis_mask] - prev_kps[vis_mask], axis=-1
                        ).mean()
                        if delta > outlier_thresh:
                            n_jump += 1; continue
                prev_kps = kn.copy()
                X_list.append(wc)
                y_list.append(kn.copy())
                vis_list.append(kc)
                ts_list.append(tv)
                n_sv += 1
            else:
                n_np += 1
        pending.clear()

    frame_idx = -1
    while True:
        item = fq.get()
        if item is None:
            _flush(); break

        frame_idx, frame = item

        if skip_every > 1 and frame_idx % skip_every != 0:
            n_sk += 1; continue

        t_vid = t0 + frame_idx / fps
        ic    = int(np.clip(np.searchsorted(csi_ts, t_vid), 0, len(csi_ts)-1))

        if abs(csi_ts[ic] - t_vid) > MAX_LAG_SEC:
            n_lag += 1; continue
        if ic < WIN_SIZE:
            continue

        wr = csi_feat[ic - WIN_SIZE : ic].copy()
        if np.isnan(wr).any() or np.isinf(wr).any():
            continue

        wc = clean_csi_window(wr)
        pending.append((frame_idx, frame, wc, t_vid))

        if len(pending) >= BATCH_SIZE:
            _flush()

        now = time.time()
        if now - t_print >= 5.0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  [{pct:5.1f}%] f={frame_idx}/{total} "
                  f"saved={n_sv} no_person={n_np} "
                  f"lag={n_lag} low_vis={n_lv} jump={n_jump}")
            t_print = now

    reader.join(timeout=5)

    print(f"\n  ── Kết quả ──")
    print(f"  Tổng frames  : {frame_idx+1}")
    print(f"  Skip         : {n_sk}")
    print(f"  Saved        : {n_sv}")
    print(f"  No person    : {n_np}")
    print(f"  Lag lớn      : {n_lag}")
    print(f"  Low vis      : {n_lv}")
    print(f"  Outlier jump : {n_jump}")

    if n_lag > max(n_sv, 1) * 2:
        print(f"\n  ⚠  n_lag >> n_sv — offset có thể sai!")
        print(f"     Thử: --video-offset {video_offset+2:.1f}  "
              f"hoặc  --video-offset {video_offset-2:.1f}")

    if not X_list:
        print("\n  ❌ Không có sample")
        return None

    X   = np.array(X_list,   dtype=np.float32)
    y   = np.array(y_list,   dtype=np.float32)
    vis = np.array(vis_list, dtype=np.float32)
    ts  = np.array(ts_list,  dtype=np.float64)

    print(f"\n  ── Kiểm tra ──")
    print(f"  X   {X.shape}  [{X.min():.4f}, {X.max():.4f}]")
    print(f"  y   {y.shape}  [{y.min():.4f}, {y.max():.4f}]")
    print(f"  vis {vis.shape}")

    assert y.max() <= 1.01,       f"❌ y max={y.max():.3f}"
    assert not np.isnan(X).any(), "❌ X có NaN"
    assert not np.isnan(y).any(), "❌ y có NaN"

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir,
        f"aligned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez_compressed(fname, X=X, y=y, visibility=vis, timestamps=ts)

    print(f"\n  ✅ {fname}")
    print(f"     {n_sv} samples | X{X.shape} | y{y.shape}")
    return fname


# ══════════════════════════════════════════════════════════════
# Fix dataset cũ
# ══════════════════════════════════════════════════════════════
def fix_existing_dataset(npz_path, frame_w=1280, frame_h=720):
    data = np.load(npz_path)
    y    = data['y'].astype(np.float32)
    y2   = y[:, :, :2].copy()
    if y2.max() > 2.0:
        y2[:, :, 0] /= frame_w
        y2[:, :, 1] /= frame_h
        y2 = np.clip(y2, 0.0, 1.0)
        out  = npz_path.replace('.npz', '_fixed.npz')
        keys = {k: data[k] for k in data.files}
        keys['y'] = y2
        np.savez_compressed(out, **keys)
        print(f"  ✅ {out}")
    else:
        print("  y đã normalize.")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    global BATCH_SIZE
    p = argparse.ArgumentParser(
        description="Align CSI+Video → npz v6 (offset tự động)")
    p.add_argument('--video',          default=None)
    p.add_argument('--csi',            default=None)
    p.add_argument('--out',            default='saved3')
    p.add_argument('--yolo-model',     default='yolo11n-pose.pt')
    p.add_argument('--batch-size',     type=int,   default=BATCH_SIZE)
    p.add_argument('--fp16',           action='store_true')
    p.add_argument('--target-fps',     type=int,   default=TARGET_FPS)
    p.add_argument('--video-offset',   type=float, default=None,
                   help='Ghi đè video_offset (giây). Mặc định: đọc tự động từ npz.')
    p.add_argument('--outlier-thresh', type=float, default=OUTLIER_JUMP_THRESH)
    p.add_argument('--fix-only',       default=None)
    p.add_argument('--fix-w',          type=int,   default=1280)
    p.add_argument('--fix-h',          type=int,   default=720)
    args = p.parse_args()

    if args.fix_only:
        fix_existing_dataset(args.fix_only, args.fix_w, args.fix_h)
        return

    if not YOLO_OK:
        print("pip install ultralytics"); return
    if not args.video or not args.csi:
        p.error("Cần --video và --csi")
    for f in [args.video, args.csi]:
        if not os.path.exists(f):
            print(f"❌ Không tìm thấy: {f}"); return

    use_fp16 = args.fp16
    batch_sz = args.batch_size

    if TORCH_OK and torch.cuda.is_available():
        free       = torch.cuda.mem_get_info()[0] / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM {total_vram:.1f}GB free {free:.1f}GB")
        if free < 1.5 and batch_sz > 4:
            batch_sz = 4; print("⚠  VRAM thấp → batch=4")
    else:
        print("⚠  CPU mode"); use_fp16 = False

    BATCH_SIZE = batch_sz

    print(f"\n=== align_csi v6  (offset-auto | MAX_LAG=150ms) ===")
    print(f"  video  : {args.video}")
    print(f"  csi    : {args.csi}")
    print(f"  out    : {args.out}\n")

    print("Nạp YOLO...")
    yolo = YOLO(args.yolo_model)

    print("Nạp CSI...")
    csi_ts, csi_feat, video_offset = load_csi(args.csi, args.video_offset)

    t0 = time.time()
    align(args.video, csi_ts, csi_feat, args.out,
          yolo, use_fp16, args.target_fps,
          video_offset, args.outlier_thresh)
    e = time.time() - t0
    print(f"\nTổng: {e:.1f}s ({e/60:.1f} phút)")


if __name__ == '__main__':
    main()