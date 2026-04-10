"""
test.py  (v5 — Router-Reflector | dedup | ts-fix | video_start_ts)
───────────────────────────────────────────────────────────────────
Fix so với v4:

  FIX 1 — Dedup (từ v4):
    Chỉ lưu khi có frame CSI mới thực sự → ~20 Hz, không duplicate.

  FIX 2 — Timestamp đúng (từ v4):
    Trả về mean(frames[-1].timestamp) của 3 node thay vì time.time().

  FIX 3 (MỚI) — Lưu video_start_ts:
    Ghi lại time.time() đúng tại thời điểm video_writer.write(frame) đầu tiên.
    Lưu vào csi_raw_*.npz dưới key 'video_start_ts'.
    align_csi.py đọc giá trị này để tính video_offset chính xác,
    không cần đoán mò bằng --video-offset nữa.

    video_offset = video_start_ts - csi_ts[0]
    (dương = video bắt đầu sau CSI, thường 5-7s)
"""

import argparse
import time
import os
import numpy as np
import cv2
from datetime import datetime
import threading
import logging

from csi_mesh import CSIMeshAggregator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

N_LINKS = 3


# ════════════════════════════════════════════════════════════
# _get_raw_per_node — Thu raw CSI không normalize, không pad
# ════════════════════════════════════════════════════════════
_last_frame_ts = {1: 0.0, 2: 0.0, 3: 0.0}

def _get_raw_per_node(mesh: CSIMeshAggregator):
    """
    Trả về (csi_ts, raw_per_node) hoặc (None, None).

    raw_per_node: dict {node_id: (amp_float32, ph_float32)} với amp/ph
    là mảng có độ dài THỰC TẾ từ gói tin (không padding cố định).
    Node không có data → value=None.

    amp: sqrt(real^2 + imag^2)  [float32, range ~0..181 với int8 input]
    ph : arctan2(imag, real)    [float32, radians ∈ (−π, π)]
    """
    global _last_frame_ts

    alive = mesh.get_alive_processors()
    if not alive:
        return None, None

    proc_map      = {nid: proc for nid, proc in alive}
    raw_per_node  = {}
    frame_ts_list = []
    has_new       = False

    for node_id in [1, 2, 3]:
        proc = proc_map.get(node_id)
        if proc is None:
            raw_per_node[node_id] = None
            continue

        frames = proc._snapshot(n=20)
        if not frames:
            raw_per_node[node_id] = None
            continue

        latest_ts = frames[-1].timestamp
        if latest_ts > _last_frame_ts[node_id]:
            has_new = True
            _last_frame_ts[node_id] = latest_ts
        frame_ts_list.append(latest_ts)

        amp = frames[-1].amplitude.copy()   # actual len, float32
        ph  = frames[-1].phase.copy()       # actual len, radians
        raw_per_node[node_id] = (amp, ph)

    if not has_new:
        return None, None

    csi_ts = float(np.mean(frame_ts_list)) if frame_ts_list else time.time()
    return csi_ts, raw_per_node


def _build_flat_feature(raw_per_node: dict, n_sub: int) -> np.ndarray:
    """
    Từ raw_per_node + n_sub, tạo feature vector phẳng (N_LINKS * 2 * n_sub,).
    Layout: [amp_node1 | ph_node1 | amp_node2 | ph_node2 | amp_node3 | ph_node3]
    Node thiếu → block 0.
    """
    feat = np.zeros(N_LINKS * 2 * n_sub, dtype=np.float32)
    for j, node_id in enumerate([1, 2, 3]):
        nd = raw_per_node.get(node_id)
        if nd is None:
            continue
        amp, ph = nd
        n = min(len(amp), n_sub)
        off = j * 2 * n_sub
        feat[off       : off + n]        = amp[:n]
        feat[off + n_sub : off + n_sub + n] = ph[:n]
    return feat


def extract_csi_frame(mesh: CSIMeshAggregator, n_sub: int = 64):
    """
    Wrapper cho inference — trả (csi_ts, feat) hoặc (None, None).
    n_sub phải khớp với giá trị lưu trong feature_cleaner.npz (fc.n_sub).
    Không áp dụng normalize [0,1] — để FeatureCleaner z-norm xử lý.
    """
    csi_ts, raw = _get_raw_per_node(mesh)
    if raw is None:
        return None, None
    return csi_ts, _build_flat_feature(raw, n_sub)


# ════════════════════════════════════════════════════════════
# CSIVideoCollector
# ════════════════════════════════════════════════════════════
class CSIVideoCollector:
    def __init__(self, output_dir='ruview_sessions', camera_id=0, buffer_size=1000):
        self.output_dir  = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.video_path   = os.path.join(output_dir, f"video_{self.session_id}.mp4")
        self.csi_raw_path = os.path.join(output_dir, f"csi_raw_{self.session_id}.npz")

        self.mesh        = CSIMeshAggregator(buffer_size=buffer_size)
        self.csi_buffer  = []   # list of (ts, raw_per_node_dict)

        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fps_video    = 30.0
        self.video_writer = None
        self.frame_width  = 1280
        self.frame_height = 720

        # FIX 3: timestamp khi video_writer.write() đầu tiên được gọi
        self.video_start_ts = None

        self.running      = False
        self.video_thread = None

    def start(self):
        self.mesh.start()
        print("Đợi ESP32 (5s)...", end='', flush=True)
        time.sleep(5)

        node_status = self.mesh.get_node_status()
        n_online    = sum(node_status.values())
        print(f" {n_online}/3 nodes online")
        for label, ok in node_status.items():
            print(f"  {'✓' if ok else '✗'} {label}")

        ret, test_frame = self.cap.read()
        if ret:
            self.frame_height, self.frame_width = test_frame.shape[:2]
            print(f"Camera: {self.frame_width}×{self.frame_height}")
        else:
            print("⚠  Không đọc được frame test, dùng 1280×720")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_path, fourcc, self.fps_video,
            (self.frame_width, self.frame_height))

        self.running      = True
        self.video_thread = threading.Thread(
            target=self._video_loop, daemon=True)
        self.video_thread.start()

        # Reset dedup state
        global _last_frame_ts
        _last_frame_ts = {1: 0.0, 2: 0.0, 3: 0.0}

        print(f"\n[Session] {self.session_id}")
        print(f"[Session] Lưu raw CSI (amp+phase thực tế, không normalize)")
        print("[Session] Nhấn Q trong cửa sổ camera để dừng\n")

    def _video_loop(self):
        first_frame = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # FIX 3: ghi lại timestamp chính xác của frame VIDEO đầu tiên
            if first_frame:
                self.video_start_ts = time.time()
                first_frame = False
                logger.info(f"[Video] First frame written at {self.video_start_ts:.3f}")

            self.video_writer.write(frame)

            alive        = self.mesh.get_alive_processors()
            n_alive      = len(alive)
            status_color = (0, 255, 0) if n_alive == 3 else (0, 165, 255)
            display      = frame.copy()
            cv2.putText(display, f"CSI nodes: {n_alive}/3",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)
            cv2.putText(display, f"Samples: {len(self.csi_buffer)}",
                        (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.putText(display, "Nhấn Q để dừng",
                        (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('CSI Collector', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

            time.sleep(0.001)

    def collect_csi(self, duration=600):
        start = time.time()
        print(f"[Collector] Thu raw CSI trong {duration}s  "
              f"(dedup ON, lưu amp+phase thực tế không normalize)...")

        while self.running and (time.time() - start < duration):
            ts, raw = _get_raw_per_node(self.mesh)
            if raw is not None:
                self.csi_buffer.append((ts, raw))

            elapsed = time.time() - start
            n = len(self.csi_buffer)
            if n > 0 and n % 200 == 0:
                csi_rate = n / max(elapsed, 1)
                alive_count = len(self.mesh.get_alive_processors())
                print(f"\r[{elapsed:5.1f}s/{duration}s] "
                      f"samples={n}  rate={csi_rate:.1f}Hz  "
                      f"nodes={alive_count}/3   ",
                      end='', flush=True)

            time.sleep(0.001)

        print(f"\n[Collector] Xong: {len(self.csi_buffer)} CSI samples")

        if len(self.csi_buffer) > 10:
            ts_arr      = np.array([t for t, _ in self.csi_buffer])
            actual_rate = len(ts_arr) / (ts_arr[-1] - ts_arr[0])
            dt          = np.diff(ts_arr)
            print(f"  CSI rate : {actual_rate:.1f} Hz  (expected ~20)")
            print(f"  Gap max  : {dt.max()*1000:.0f} ms")
            if actual_rate > 50:
                print("  ⚠  Rate vẫn cao — dedup chưa hoạt động?")
            elif actual_rate < 5:
                print("  ⚠  Rate quá thấp — ESP32 mất kết nối?")
            else:
                print("  ✓  Rate hợp lý")

    def stop(self):
        print("\n[Stop] Dừng...")
        self.running = False

        if self.video_thread:
            self.video_thread.join(timeout=3)

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.mesh.stop()

        if self.csi_buffer:
            # ── Tìm max_sub: độ dài subcarrier lớn nhất trên tất cả node/frame ──
            max_sub = 1
            for _ts, raw in self.csi_buffer:
                for node_id in [1, 2, 3]:
                    nd = raw.get(node_id)
                    if nd is not None:
                        max_sub = max(max_sub, len(nd[0]))

            feat_dim = N_LINKS * 2 * max_sub

            # ── Xây dựng feature array (N, feat_dim) ──
            ts_arr   = np.array([t for t, _ in self.csi_buffer], dtype=np.float64)
            feat_arr = np.zeros((len(self.csi_buffer), feat_dim), dtype=np.float32)
            for i, (_ts, raw) in enumerate(self.csi_buffer):
                feat_arr[i] = _build_flat_feature(raw, max_sub)

            dur = ts_arr[-1] - ts_arr[0]

            if self.video_start_ts is not None:
                video_offset = self.video_start_ts - ts_arr[0]
            else:
                video_offset = 0.0
                print("⚠  video_start_ts không được ghi lại — offset = 0")

            np.savez_compressed(
                self.csi_raw_path,
                timestamps       = ts_arr,
                features         = feat_arr,
                n_sub            = np.int32(max_sub),
                video_start_ts   = np.float64(self.video_start_ts or 0.0),
                video_offset     = np.float64(video_offset),
            )
            print(f"✓ CSI raw: {self.csi_raw_path}")
            print(f"  shape        : {feat_arr.shape}  (N × {N_LINKS}×2×{max_sub})")
            print(f"  n_sub        : {max_sub}  (subcarrier thực tế, không padding 128)")
            print(f"  amp range    : [{feat_arr.min():.1f}, {feat_arr.max():.1f}]  (raw, chưa normalize)")
            print(f"  duration     : {dur:.1f}s")
            print(f"  avg rate     : {len(ts_arr)/dur:.1f} Hz")
            print(f"  video_offset : {video_offset:+.3f}s  "
                  f"← align_csi.py sẽ dùng tự động")
        else:
            print("⚠  Không có CSI data")

        print(f"✓ Video: {self.video_path}")
        print(f"\n→ Bước tiếp theo:")
        print(f"  python align_csi.py \\")
        print(f"      --video {self.video_path} \\")
        print(f"      --csi   {self.csi_raw_path} \\")
        print(f"      --out   saved3/")
        print(f"  (video_offset được đọc tự động từ file npz)")


def main():
    parser = argparse.ArgumentParser(
        description="CSI + Video Collector v5")
    parser.add_argument('--duration', type=int, default=300)
    parser.add_argument('--camera',   type=int, default=0)
    parser.add_argument('--output',   type=str, default='ruview_sessions')
    parser.add_argument('--buffer',   type=int, default=1000)
    args = parser.parse_args()

    print("=" * 58)
    print("  CSI Collector v6  (raw CSI, dynamic n_sub, no normalize)")
    print("=" * 58)
    print(f"  Duration : {args.duration}s")
    print(f"  Camera   : {args.camera}")
    print(f"  Output   : {args.output}/")
    print(f"  Lưu raw amp+phase (n_sub tính tự động từ dữ liệu thực tế)")
    print()

    collector = CSIVideoCollector(
        output_dir  = args.output,
        camera_id   = args.camera,
        buffer_size = args.buffer,
    )

    try:
        collector.start()
        collector.collect_csi(duration=args.duration)
    except KeyboardInterrupt:
        print("\n\n[Dừng bởi người dùng]")
    finally:
        collector.stop()


if __name__ == "__main__":
    main()