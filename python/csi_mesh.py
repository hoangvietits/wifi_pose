"""
csi_mesh.py  (v3 — Router-Reflector Mode)
──────────────────────────────────────────
Lắng nghe UDP từ 3 ESP32 (port 5005/5006/5007).
Mỗi node thu CSI từ ROUTER — không còn thu lẫn nhau.

Thay đổi so với v2:
  - 3 link thay vì 6 (1 processor per node)
  - link_key = (rx_node_id,) — không cần tx_mac trong key nữa vì
    mỗi node chỉ có 1 nguồn CSI (router)
  - Vẫn giữ tx_mac trong packet để verify router MAC nếu cần debug

Sơ đồ:
    ESP32-S3 (node1) :5005  →  CSI từ router  →  Processor 1
    ESP32-Classic (2) :5006 →  CSI từ router  →  Processor 2
    ESP32-C3 (node3) :5007  →  CSI từ router  →  Processor 3
"""

import socket
import threading
import time
import logging
from typing import Optional

from csi_processor import CSIProcessor

logger = logging.getLogger(__name__)


class CSIMeshAggregator:
    """
    Quản lý 3 UDP stream từ 3 ESP32.
    Mỗi node → 1 CSIProcessor (thu từ router).

    Sử dụng:
        mesh = CSIMeshAggregator()
        mesh.start()
        alive = mesh.get_alive_processors()  # list of (node_id, processor)
    """

    NODE_PORTS = {
        1: 5005,   # ESP32-S3
        2: 5006,   # ESP32-Classic
        3: 5007,   # ESP32-C3
    }

    NODE_LABELS = {
        1: 'ESP32-S3',
        2: 'ESP32-Classic',
        3: 'ESP32-C3',
    }

    def __init__(self, buffer_size: int = 300):
        self.buffer_size = buffer_size

        # key: rx_node_id (int) → CSIProcessor
        # Đơn giản hơn v2: không cần tx_mac trong key vì chỉ có 1 tx (router)
        self.processors: dict[int, CSIProcessor] = {}
        self._lock = threading.Lock()

        self._packet_count: dict[int, int] = {}
        self._error_count:  dict[int, int] = {}
        self._start_time = 0.0
        self._running    = False

    # ─────────────────────────────────────────────────────────
    # Start / Stop
    # ─────────────────────────────────────────────────────────

    def start(self):
        self._running    = True
        self._start_time = time.time()

        for node_id, port in self.NODE_PORTS.items():
            self._packet_count[node_id] = 0
            self._error_count[node_id]  = 0
            t = threading.Thread(
                target = self._udp_listener,
                args   = (node_id, port),
                daemon = True,
                name   = f"udp_{self.NODE_LABELS[node_id]}",
            )
            t.start()
            logger.info(f"[Mesh] {self.NODE_LABELS[node_id]} listening on UDP :{port}")

        print(f"[Mesh] Started — router-reflector mode, ports {list(self.NODE_PORTS.values())}")

    def stop(self):
        self._running = False

    # ─────────────────────────────────────────────────────────
    # UDP Listener (1 thread / node)
    # ─────────────────────────────────────────────────────────

    def _udp_listener(self, node_id: int, port: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        sock.bind(('0.0.0.0', port))

        label = self.NODE_LABELS[node_id]
        logger.debug(f"[{label}] Socket bound to :{port}")

        while self._running:
            try:
                data, addr = sock.recvfrom(1024)
                self._handle_packet(node_id, data, addr)
                self._packet_count[node_id] += 1
            except socket.timeout:
                continue
            except Exception as e:
                self._error_count[node_id] += 1
                logger.warning(f"[{label}] Packet error: {e}")

        sock.close()
        logger.info(f"[{label}] Listener stopped")

    # ─────────────────────────────────────────────────────────
    # Packet Handler
    # ─────────────────────────────────────────────────────────

    def _handle_packet(self, rx_node: int, data: bytes, addr):
        """
        Parse packet và route đến CSIProcessor của node này.
        Tạo processor mới nếu chưa có (lần đầu nhận từ node này).
        """
        if len(data) < 12:
            return

        # Đọc router MAC từ packet để log (bytes 1–6)
        router_mac = ':'.join(f'{b:02x}' for b in data[1:7])

        with self._lock:
            if rx_node not in self.processors:
                proc = CSIProcessor(
                    rx_node     = rx_node,
                    tx_mac      = router_mac,
                    buffer_size = self.buffer_size,
                )
                self.processors[rx_node] = proc
                label = self.NODE_LABELS.get(rx_node, f"Node{rx_node}")
                logger.info(
                    f"[Mesh] New link: {label} ← router {router_mac} "
                    f"(from {addr[0]})"
                )
            processor = self.processors[rx_node]

        processor.parse_and_add(data)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def get_alive_processors(self) -> list:
        """
        Danh sách (node_id, processor) của các node đang gửi CSI.

        Returns:
            list of (int, CSIProcessor) — sorted by node_id
        """
        with self._lock:
            items = list(self.processors.items())
        return [(nid, proc) for nid, proc in sorted(items) if proc.is_alive]

    def get_status(self) -> dict:
        alive = self.get_alive_processors()

        motion_scores   = [proc.detect_motion()          for _, proc in alive]
        presence_scores = [proc.presence_score()         for _, proc in alive]
        breathing_vals  = [b for _, proc in alive
                           if (b := proc.extract_breathing_rate()) is not None]
        heart_vals      = [h for _, proc in alive
                           if (h := proc.extract_heart_rate())     is not None]

        presence = any(s > 0.4 for s in presence_scores)

        elapsed  = max(time.time() - self._start_time, 1)
        pkt_rate = {
            self.NODE_LABELS.get(nid, f"Node{nid}"): round(cnt / elapsed, 1)
            for nid, cnt in self._packet_count.items()
        }

        per_link = {}
        for nid, proc in alive:
            label = self.NODE_LABELS.get(nid, f"Node{nid}")
            per_link[label] = proc.get_summary()

        return {
            'active_links'   : len(alive),
            'total_links'    : len(self.processors),
            'presence'       : presence,
            'motion_level'   : round(float(sum(motion_scores) / max(len(motion_scores), 1)), 3),
            'presence_score' : round(float(sum(presence_scores) / max(len(presence_scores), 1)), 3),
            'breathing_bpm'  : round(sum(breathing_vals) / len(breathing_vals), 1)
                               if breathing_vals else None,
            'heart_bpm'      : round(sum(heart_vals) / len(heart_vals), 1)
                               if heart_vals else None,
            'per_link'       : per_link,
            'packet_rate'    : pkt_rate,
            'uptime_s'       : round(elapsed),
        }

    def get_node_status(self) -> dict:
        """Trạng thái từng node: label → bool connected."""
        with self._lock:
            connected = set(self.processors.keys())
            # Chỉ tính là connected nếu processor is_alive
            alive_nodes = {nid for nid, proc in self.processors.items()
                           if proc.is_alive}
        return {
            label: (node_id in alive_nodes)
            for node_id, label in self.NODE_LABELS.items()
        }

    def calibrate(self):
        """Reset tất cả processor — gọi khi muốn làm mới baseline."""
        with self._lock:
            for proc in self.processors.values():
                proc.clear()
        logger.info("[Mesh] All processors calibrated")
        print("[Mesh] Calibrated — buffers cleared")

    def __repr__(self):
        return (f"CSIMeshAggregator("
                f"nodes={len(self.processors)}, "
                f"alive={len(self.get_alive_processors())})")