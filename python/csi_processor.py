"""
csi_processor.py  (v3 — Router-Reflector Mode)
───────────────────────────────────────────────
Xử lý tín hiệu CSI thô từ 1 link ESP32 ↔ Router.

Thay đổi so với v2 (mesh node-to-node):
  - tx_mac giờ là MAC của ROUTER (không còn là MAC node khác)
  - Không cần phân biệt 6 link nữa — chỉ còn 3 link (1 per node)
  - N_SUB tăng lên 128 để tận dụng đầy đủ subcarrier từ router response
    (router thường trả 802.11n HT40 → nhiều subcarrier hơn)
  - is_alive timeout nới từ 3s → 5s (router response ổn định hơn node-to-node)

Pipeline:
    bytes UDP → parse → amplitude/phase → Hampel filter → Bandpass → FFT
              → breathing_bpm / heart_bpm / motion_score / presence_score
"""

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from scipy.signal import butter, filtfilt


# ─────────────────────────────────────────────────────────────
# Số subcarrier chuẩn hóa
# Router response (HT40) thường trả 114 subcarrier.
# Dùng 128 để không mất data, pad nếu thiếu.
# ─────────────────────────────────────────────────────────────
N_SUB = 128


def _pad_or_trim(arr: np.ndarray, length: int = N_SUB) -> np.ndarray:
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)), constant_values=0.0)


# ─────────────────────────────────────────────────────────────
# Data class cho 1 frame CSI
# ─────────────────────────────────────────────────────────────

@dataclass
class CSIFrame:
    timestamp : float
    rx_node   : int          # node ESP32 thu (1/2/3)
    tx_mac    : str          # MAC router (BSSID)
    rssi      : int          # dBm
    amplitude : np.ndarray   # |H|, shape (N_SUB,)
    phase     : np.ndarray   # ∠H, shape (N_SUB,), radian


# ─────────────────────────────────────────────────────────────
# CSIProcessor
# ─────────────────────────────────────────────────────────────

class CSIProcessor:
    """
    Xử lý CSI cho 1 link (ESP32 node ↔ Router).

    Trong router-reflector mode, mỗi node ESP32 có đúng 1 processor
    → tổng 3 processor thay vì 6 như trước.

    Thread-safety: buffer được bảo vệ bởi lock.
    """

    SAMPLE_RATE = 20.0   # Hz — ESP32 ping mỗi 50ms

    def __init__(self, rx_node: int, tx_mac: str, buffer_size: int = 300):
        self.rx_node    = rx_node
        self.tx_mac     = tx_mac
        self._buffer    : deque[CSIFrame] = deque(maxlen=buffer_size)
        self._lock      = threading.Lock()
        self._last_seen = 0.0

    # ─────────────────────────────────────────────────────────
    # Snapshot buffer an toàn
    # ─────────────────────────────────────────────────────────

    def _snapshot(self, n: Optional[int] = None) -> list:
        with self._lock:
            frames = list(self._buffer)
        if n is not None:
            frames = frames[-n:]
        return frames

    # ─────────────────────────────────────────────────────────
    # Parse UDP packet
    # ─────────────────────────────────────────────────────────

    def parse_and_add(self, raw: bytes) -> Optional[CSIFrame]:
        """
        Parse UDP packet từ firmware ESP32 (router-reflector mode).

        Packet format:
          [0]     node_id    uint8
          [1:7]   router_mac 6 bytes  ← thay đổi: giờ là MAC router
          [7]     rssi+128   uint8
          [8]     rate       uint8
          [9]     csi_len_hi uint8
          [10]    csi_len_lo uint8
          [11:]   csi_data   int8 pairs (real, imag, ...)
        """
        if len(raw) < 12:
            return None

        node_id = raw[0]
        tx_mac  = ':'.join(f'{b:02x}' for b in raw[1:7])
        rssi    = int(raw[7]) - 128
        csi_len = (raw[9] << 8) | raw[10]

        payload = raw[11 : 11 + csi_len]
        if len(payload) < 4:
            return None

        csi_int8  = np.frombuffer(payload, dtype=np.int8).astype(np.float32)
        real_part = csi_int8[0::2]
        imag_part = csi_int8[1::2]

        min_len   = min(len(real_part), len(imag_part))
        real_part = real_part[:min_len]
        imag_part = imag_part[:min_len]

        amplitude_raw = np.sqrt(real_part**2 + imag_part**2)
        phase_raw     = np.arctan2(imag_part, real_part)

        # Lưu đúng độ dài thực tế từ gói tin — không pad cố định sang N_SUB.
        # Padding theo max_sub sẽ được thực hiện tập trung lúc lưu file.
        amplitude = amplitude_raw.astype(np.float32)
        phase     = phase_raw.astype(np.float32)

        frame = CSIFrame(
            timestamp = time.time(),
            rx_node   = node_id,
            tx_mac    = tx_mac,
            rssi      = rssi,
            amplitude = amplitude,
            phase     = phase,
        )

        with self._lock:
            self._buffer.append(frame)
            self._last_seen = frame.timestamp

        return frame

    # ─────────────────────────────────────────────────────────
    # Hampel filter
    # ─────────────────────────────────────────────────────────

    def _hampel_filter(self, signal: np.ndarray,
                       window: int = 5,
                       threshold: float = 3.0) -> np.ndarray:
        result = signal.copy()
        n = len(signal)
        for i in range(window, n - window):
            local = signal[i - window : i + window + 1]
            med   = np.median(local)
            mad   = np.median(np.abs(local - med))
            if mad == 0:
                continue
            if np.abs(signal[i] - med) > threshold * 1.4826 * mad:
                result[i] = med
        return result

    # ─────────────────────────────────────────────────────────
    # Bandpass filter
    # ─────────────────────────────────────────────────────────

    def _bandpass(self, signal: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        if len(signal) < 20:
            return signal
        nyq = self.SAMPLE_RATE / 2.0
        lo  = max(low_hz  / nyq, 1e-4)
        hi  = min(high_hz / nyq, 0.999)
        if lo >= hi:
            return signal
        b, a = butter(4, [lo, hi], btype='band')
        return filtfilt(b, a, signal)

    # ─────────────────────────────────────────────────────────
    # FFT peak
    # ─────────────────────────────────────────────────────────

    def _fft_peak_hz(self, signal: np.ndarray, low_hz: float, high_hz: float) -> Optional[float]:
        n      = len(signal)
        window = np.hanning(n)
        fft    = np.abs(np.fft.rfft(signal * window))
        freqs  = np.fft.rfftfreq(n, d=1.0 / self.SAMPLE_RATE)
        mask   = (freqs >= low_hz) & (freqs <= high_hz)
        if not mask.any():
            return None
        return float(freqs[mask][np.argmax(fft[mask])])

    # ─────────────────────────────────────────────────────────
    # Mean amplitude series
    # ─────────────────────────────────────────────────────────

    def _mean_amplitude_series(self, frames: list) -> Optional[np.ndarray]:
        if len(frames) < 30:
            return None
        amps = np.array([f.amplitude for f in frames], dtype=np.float32)
        return amps.mean(axis=1)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        """Router response ổn định hơn — nới timeout lên 5s."""
        with self._lock:
            last = self._last_seen
        return (time.time() - last) < 5.0

    def detect_motion(self) -> float:
        frames = self._snapshot(n=20)
        if len(frames) < 5:
            return 0.0
        amplitudes = np.array([f.amplitude for f in frames], dtype=np.float32)
        return float(np.mean(np.var(amplitudes, axis=0)))

    def presence_score(self) -> float:
        frames = self._snapshot()
        if len(frames) < 30:
            return 0.0

        short_frames = frames[-20:]
        all_amps   = np.array([f.amplitude for f in frames],       dtype=np.float32)
        short_amps = np.array([f.amplitude for f in short_frames], dtype=np.float32)

        motion    = float(np.mean(np.var(short_amps, axis=0)))
        long_var  = float(np.var(all_amps.mean(axis=1)))
        short_var = float(np.var(short_amps.mean(axis=1)))
        ratio     = (short_var / long_var) if long_var > 1e-6 else 1.0

        raw_score = (min(motion / 3.0, 1.0) * 0.6 +
                     min(ratio  / 5.0, 1.0) * 0.4)
        return round(min(raw_score, 1.0), 3)

    def extract_breathing_rate(self) -> Optional[float]:
        frames = self._snapshot()
        raw    = self._mean_amplitude_series(frames)
        if raw is None:
            return None
        cleaned  = self._hampel_filter(raw)
        filtered = self._bandpass(cleaned, low_hz=0.1, high_hz=0.5)
        peak_hz  = self._fft_peak_hz(filtered, low_hz=0.1, high_hz=0.5)
        if peak_hz is None:
            return None
        return round(peak_hz * 60.0, 1)

    def extract_heart_rate(self) -> Optional[float]:
        frames = self._snapshot()
        if len(frames) < 100:
            return None
        raw = self._mean_amplitude_series(frames)
        if raw is None:
            return None
        cleaned  = self._hampel_filter(raw)
        filtered = self._bandpass(cleaned, low_hz=0.8, high_hz=2.0)
        peak_hz  = self._fft_peak_hz(filtered, low_hz=0.8, high_hz=2.0)
        if peak_hz is None:
            return None
        return round(peak_hz * 60.0, 1)

    def get_subcarrier_profile(self) -> Optional[np.ndarray]:
        frames = self._snapshot(n=30)
        if len(frames) < 10:
            return None
        amps = np.array([f.amplitude for f in frames], dtype=np.float32)
        return np.mean(amps, axis=0)

    def get_rssi_avg(self) -> Optional[int]:
        frames = self._snapshot(n=20)
        if not frames:
            return None
        return int(np.mean([f.rssi for f in frames]))

    def get_summary(self) -> dict:
        frames       = self._snapshot()
        short_frames = frames[-20:]

        motion = 0.0
        if len(short_frames) >= 5:
            short_amps = np.array([f.amplitude for f in short_frames], dtype=np.float32)
            motion = float(np.mean(np.var(short_amps, axis=0)))

        pres = 0.0
        if len(frames) >= 30:
            all_amps   = np.array([f.amplitude for f in frames],       dtype=np.float32)
            short_amps = np.array([f.amplitude for f in frames[-20:]], dtype=np.float32)
            long_var   = float(np.var(all_amps.mean(axis=1)))
            short_var  = float(np.var(short_amps.mean(axis=1)))
            ratio      = (short_var / long_var) if long_var > 1e-6 else 1.0
            raw_score  = (min(motion / 3.0, 1.0) * 0.6 +
                          min(ratio  / 5.0, 1.0) * 0.4)
            pres = round(min(raw_score, 1.0), 3)

        breathing = None
        mean_series = self._mean_amplitude_series(frames)
        if mean_series is not None:
            cleaned  = self._hampel_filter(mean_series)
            filtered = self._bandpass(cleaned, 0.1, 0.5)
            hz       = self._fft_peak_hz(filtered, 0.1, 0.5)
            if hz:
                breathing = round(hz * 60.0, 1)

        heart = None
        if len(frames) >= 100 and mean_series is not None:
            cleaned  = self._hampel_filter(mean_series)
            filtered = self._bandpass(cleaned, 0.8, 2.0)
            hz       = self._fft_peak_hz(filtered, 0.8, 2.0)
            if hz:
                heart = round(hz * 60.0, 1)

        rssi_avg = None
        if frames:
            rssi_avg = int(np.mean([f.rssi for f in frames[-20:]]))

        with self._lock:
            last = self._last_seen

        return {
            'rx_node'        : self.rx_node,
            'tx_mac'         : self.tx_mac,   # router MAC
            'frame_count'    : len(frames),
            'is_alive'       : (time.time() - last) < 5.0,
            'rssi_avg'       : rssi_avg,
            'motion_level'   : round(motion, 3),
            'presence_score' : pres,
            'breathing_bpm'  : breathing,
            'heart_bpm'      : heart,
        }

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._last_seen = 0.0

    def __repr__(self):
        with self._lock:
            n = len(self._buffer)
        return (f"CSIProcessor(node={self.rx_node}, router={self.tx_mac}, "
                f"frames={n}, alive={self.is_alive})")