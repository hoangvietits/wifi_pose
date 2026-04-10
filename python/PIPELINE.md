# WiFi CSI–Based Human Pose Estimation Pipeline

**Tác giả:** Võ Hồ Hoàng Việt  

**Cơ sở:**  Ước lượng tư thế người sử dụng tín hiệu WiFi CSI  

**Ngày cập nhật:** 08/04/2026

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)  
2. [Phần cứng và thu nhận tín hiệu](#2-phần-cứng-và-thu-nhận-tín-hiệu)  
3. [Pipeline thu thập dữ liệu](#3-pipeline-thu-thập-dữ-liệu)  
4. [Tiền xử lý và căn chỉnh dữ liệu](#4-tiền-xử-lý-và-căn-chỉnh-dữ-liệu)  
5. [Biểu diễn đặc trưng](#5-biểu-diễn-đặc-trưng)  
6. [Kiến trúc mô hình](#6-kiến-trúc-mô-hình)  
7. [Hàm mất mát](#7-hàm-mất-mát)  
8. [Chiến lược huấn luyện](#8-chiến-lược-huấn-luyện)  
9. [Đánh giá mô hình](#9-đánh-giá-mô-hình)  
10. [Suy luận thời gian thực](#10-suy-luận-thời-gian-thực)  
11. [Kết quả thực nghiệm](#11-kết-quả-thực-nghiệm)  
12. [Điểm mạnh](#12-điểm-mạnh)  
13. [Hạn chế và thách thức](#13-hạn-chế-và-thách-thức)  
14. [Hướng phát triển](#14-hướng-phát-triển)  
15. [Cấu trúc tệp dự án](#15-cấu-trúc-tệp-dự-án)

---

## 1. Tổng quan hệ thống

Hệ thống ước lượng tư thế người (**Human Pose Estimation — HPE**) dựa trên **Channel State Information (CSI)** của tín hiệu WiFi, không sử dụng camera. Thay vì phân tích hình ảnh, hệ thống phân tích cách cơ thể người phản xạ và hấp thụ sóng vô tuyến để suy luận ra tư thế bộ xương người với 17 keypoint theo chuẩn COCO.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TỔNG QUAN PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Router WiFi]──RF──[ESP32 Node 1/2/3]──UDP──[Python Server]    │
│                                                                   │
│         ┌──────────────────────────────────────┐                 │
│         │    Thu thập dữ liệu (test.py)         │                 │
│         │   CSI raw + Video → .npz              │                 │
│         └──────────────┬─────────────────────── ┘                │
│                        ↓                                          │
│         ┌──────────────────────────────────────┐                 │
│         │    Căn chỉnh & Gán nhãn (align_csi)  │                 │
│         │   CSI ↔ YOLO keypoints + timestamp   │                 │
│         └──────────────┬─────────────────────── ┘                │
│                        ↓                                          │
│         ┌──────────────────────────────────────┐                 │
│         │    Huấn luyện (train.py)              │                 │
│         │   FeatureCleaner + TCN Model          │                 │
│         └──────────────┬─────────────────────── ┘                │
│                        ↓                                          │
│         ┌──────────────────────────────────────┐                 │
│         │    Suy luận (realtime_eval.py)        │                 │
│         │   CSI → FeatureCleaner → Model → Pose│                 │
│         └──────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

**Đặc điểm nổi bật:**
- **Không xâm phạm quyền riêng tư**: Không dùng camera trong giai đoạn suy luận (camera chỉ dùng để tạo nhãn lúc thu dữ liệu).
- **Hoạt động xuyên vật cản**: WiFi xuyên tường, cho phép theo dõi người ở phòng khác.
- **Chi phí thấp**: Sử dụng ESP32 (< 100.000 VNĐ/node) và router WiFi thông thường.

---

## 2. Phần cứng và thu nhận tín hiệu

### 2.1 Cấu hình mạng — Router-Reflector Mode

```
          [Router WiFi / Access Point]
                  ↑↓ Beacon/Probe
         ┌────────┴──────────┐
    ESP32-S3              ESP32-C3
    (Node 1, :5005)       (Node 3, :5007)
         └────────┬──────────┘
             ESP32-classic
             (Node 2, :5006)
```

Mỗi ESP32 định kỳ gửi **ping packet** tới router. Router phản hồi, và ESP32 đọc **CSI** (Channel State Information) từ ACK phản hồi của router. CSI phản ánh trạng thái của kênh vô tuyến, bao gồm cả ảnh hưởng phản xạ/hấp thụ từ cơ thể người.

| Thành phần | Thông số |
|---|---|
| Node 1 | ESP32-S3, UDP port 5005 |
| Node 2 | ESP32-Classic, UDP port 5006 |
| Node 3 | ESP32-C3, UDP port 5007 |
| Giao thức | 802.11n HT20 |
| Tần số thu | ~20 Hz (1 ping / 50ms) |
| OFDM subcarrier | ~52–55 subcarrier/node (thực tế, không padding) |

### 2.2 Cấu trúc gói tin UDP

```
Byte [0]     : node_id (uint8)
Byte [1:7]   : router MAC / BSSID (6 bytes)
Byte [7]     : RSSI + 128 (uint8)
Byte [8]     : rate (uint8)
Byte [9:11]  : csi_len (uint16 big-endian)
Byte [11:]   : CSI data (int8 pairs: real, imag, ...)
```

### 2.3 Trích xuất biên độ và pha

Từ phần `csi_data` (int8 pairs), với mỗi subcarrier $k$:

$$H_k = \text{real}_k + j \cdot \text{imag}_k$$

$$\text{amplitude}_k = |H_k| = \sqrt{\text{real}_k^2 + \text{imag}_k^2}$$

$$\text{phase}_k = \angle H_k = \arctan2(\text{imag}_k, \text{real}_k)$$

> **Thay đổi v6**: Không còn padding cố định lên 128 subcarrier. Giá trị biên độ và pha được lưu với **độ dài thực tế** từ gói tin. Độ dài lớn nhất qua các frame được dùng làm `max_sub` khi xây dựng feature vector, giảm thiểu dead channels từ 29% xuống < 1%.

---

## 3. Pipeline thu thập dữ liệu

### 3.1 Luồng thu thập (`test.py`)

```
[ESP32 Node 1/2/3] ──UDP──→ CSIMeshAggregator
                                    │
                    ┌───────────────┴───────────────┐
                    │ _get_raw_per_node()            │
                    │   dedup: chỉ lưu frame mới    │
                    │   lưu (amp, ph) độ dài thực tế│
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ Video capture (camera thread)  │
                    │   Ghi lại video_start_ts       │
                    └───────────────┬───────────────┘
                                    │ stop()
                    ┌───────────────┴───────────────┐
                    │ Tính max_sub từ toàn session   │
                    │ _build_flat_feature(raw, max_sub)│
                    │ Lưu csi_raw_YYYYMMDD.npz       │
                    │   keys: timestamps, features,  │
                    │         n_sub, video_offset     │
                    └───────────────────────────────┘
```

**Cơ chế dedup**: Chỉ lưu frame khi `latest_ts > _last_frame_ts[node_id]`, ngăn duplicate khi polling nhanh hơn CSI rate.

### 3.2 Định dạng file `csi_raw_*.npz`

| Key | Shape | Mô tả |
|---|---|---|
| `timestamps` | (N,) float64 | Unix timestamp tại mỗi CSI frame |
| `features` | (N, 3×2×n_sub) float32 | Raw amp + phase, chưa normalize |
| `n_sub` | scalar int32 | Số subcarrier thực tế (max qua tất cả node×frame) |
| `video_start_ts` | scalar float64 | Timestamp frame video đầu tiên |
| `video_offset` | scalar float64 | `video_start_ts - timestamps[0]` (giây) |

### 3.3 Tổ chức vùng thu

Đặt 3 node ESP32 tạo thành tam giác xung quanh vùng hoạt động:

```
         [Camera] (ghi label)
              |
         [Node 2]  ← phía trước, ngang tầm người (1–1.2m)
        /         \
  [Node 1]       [Node 3]
  (góc trái)     (góc phải)
  ~45° so với người
```

---

## 4. Tiền xử lý và căn chỉnh dữ liệu

### 4.1 Quy trình căn chỉnh (`align_csi.py`)

```
Video .mp4  ──YOLO11n-pose──→ Keypoints (17×2) + Visibility (17)
                                        │
CSI csi_raw_*.npz ─────────────────────┤
  timestamps                            │
  features (N, F)               ┌──────┴──────┐
  n_sub                         │ Temporal    │
  video_offset ─────────────────│ alignment   │
                                │searchsorted │
                                └──────┬──────┘
                                       │ |lag| < 150ms
                                       ↓
                            ┌──────────────────┐
                            │ Hampel filter    │
                            │ (Outlier loại bỏ)│
                            └──────────────────┘
                                       │
                                       ↓
                            aligned_YYYYMMDD.npz
                              X: (N, 40, F)
                              y: (N, 17, 2)
                              visibility: (N, 17)
                              timestamps: (N,)
```

**Tham số quan trọng:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `WIN_SIZE` | 40 frame | Cửa sổ thời gian (~2 giây tại 20 Hz) |
| `MAX_LAG_SEC` | 0.150 s | Độ lệch timestamp tối đa cho phép giữa CSI và video |
| `MIN_VIS` | 0.35 | Ngưỡng YOLO confidence để chấp nhận keypoint |
| `MIN_JOINTS` | 9/17 | Số keypoint tối thiểu phải nhìn thấy |
| `YOLO_CONF` | 0.45 | Ngưỡng confidence YOLO person detection |

### 4.2 Nhãn keypoint — Chuẩn COCO-17

```
         0(Nose)
    1(L.Eye) 2(R.Eye)
   3(L.Ear)   4(R.Ear)
 5(L.Sho.)     6(R.Sho.)
   7(L.Elb.) 8(R.Elb.)
  9(L.Wri.) 10(R.Wri.)
11(L.Hip)     12(R.Hip)
  13(L.Kne.) 14(R.Kne.)
 15(L.Ank.) 16(R.Ank.)
```

Tọa độ được **chuẩn hóa về [0, 1]** theo kích thước frame ảnh, trục y hướng xuống.

---

## 5. Biểu diễn đặc trưng

### 5.1 Feature vector thô

Với `n_sub` subcarrier thực tế, feature vector 1 frame:

$$\mathbf{f} = \underbrace{[\underbrace{\text{amp}_1^{(1)}, \ldots, \text{amp}_{n_{sub}}^{(1)}}_{\text{Node 1 amp}}, \underbrace{\text{ph}_1^{(1)}, \ldots, \text{ph}_{n_{sub}}^{(1)}}_{\text{Node 1 phase}}]}_{\text{Node 1}}, \underbrace{[\ldots]}_{\text{Node 2}}, \underbrace{[\ldots]}_{\text{Node 3}}$$

$$\dim(\mathbf{f}) = 3 \times 2 \times n_{sub} \approx 330 \text{ (với } n_{sub} \approx 55\text{)}$$

> **So sánh với phiên bản cũ**: Trước đây padding cố định lên 128 subcarrier → 768 chiều, trong đó 222 chiều (29%) luôn bằng 0. Phiên bản mới chỉ còn ~330 chiều với < 1% dead channels.

### 5.2 FeatureCleaner

Được áp dụng sau khi tạo aligned dataset:

```
Input X: (N, T=40, F=330)
        │
        ▼ Bước 1: Loại dead channels
        │  alive_mask = (mean(X, axis=(0,1)) >= 0.001)  →  F_alive ≤ F
        │
        ▼ Bước 2: Z-normalization per feature
        │  X_alive = X[:, :, alive_mask]
        │  z = (X_alive - mean) / std
        │
Output: (N, T=40, F_alive)  —  z-normalized, không dead channels
```

**Quan trọng**: `FeatureCleaner.fit()` chỉ được gọi **trên train set**. Val/test set dùng `transform()` với `mean_` và `std_` từ train set, tránh data leakage.

Sau khi fit, cleaner lưu thêm `n_sub` — số subcarrier lúc thu — để inference biết cần pad feature vector đến chiều nào:

```python
# Lưu
np.savez('feature_cleaner.npz', alive_mask=..., mean=..., std=..., n_sub=np.int32(55))

# Tải khi inference
fc = FeatureCleaner.load('feature_cleaner.npz')
feat = build_flat_feature(raw_per_node, n_sub=fc.n_sub)
feat_clean = fc.transform(feat[None, None, :])[0, 0]   # z-normalized
```

### 5.3 Temporal difference features

Trong `model.forward()`:

$$\Delta x_t = x_t - x_{t-1}, \quad \Delta x_0 = \mathbf{0}$$

Input vào mô hình: $[\mathbf{x}, \Delta \mathbf{x}]$ → chiều tăng gấp đôi: $2 \times F_{alive}$

Mục đích: Cung cấp rõ ràng **tín hiệu chuyển động** cho mô hình, bổ sung vào tín hiệu biên độ tĩnh vốn có ít tương quan với tư thế (max Pearson r = 0.35 từ phân tích thực nghiệm).

---

## 6. Kiến trúc mô hình

### 6.1 Tổng quan — CSIPoseModelV4

```
Input: (B, T=40, F_alive)
         │
         ├── Temporal diff: Δx = x[1:] - x[:-1], pad first → (B, T, F_alive)
         │
         ▼
   Concat([x, Δx]) → (B, T, 2·F_alive)
         │
         ▼
   input_proj: Linear(2·F_alive → C=64) + LayerNorm + GELU
         │
         ▼ reshape (B, C, T)
   TCN Block (dilation=1)  ─────┐
   TCN Block (dilation=2)       │ residual
   TCN Block (dilation=4)       │ connections
   TCN Block (dilation=8)  ─────┘
         │
         ▼ reshape (B, T, C)
   Center frame: h = output[:, T//2, :]  →  (B, C=64)
         │
         ├─────────────────────────────┐
         ▼                             ▼
   pose_head:                    vis_head:
   Linear(64→128)+GELU           Linear(64→32)+GELU
   +Dropout(0.3)                 Linear(32→17)
   Linear(128→17×2)              │
         │                       ▼
         ▼                  vis_logit: (B,17)
   residual: (B,17,2)
         │
         ▼
   pose = mean_pose (learnable) + residual → (B,17,2)
```

### 6.2 TCN Block — Bidirectional Dilated Conv

```python
class TCNBlock(nn.Module):
    # pad = dilation × (kernel_size - 1) // 2  →  same padding (non-causal)
    conv1: Conv1d(in_ch, out_ch, kernel=3, dilation=d, padding=pad)
    conv2: Conv1d(out_ch, out_ch, kernel=3, dilation=d, padding=pad)
    norm1, norm2: GroupNorm
    drop: Dropout(0.2)
    res: Conv1d(in_ch, out_ch, 1) if in_ch ≠ out_ch else Identity
```

**Receptive field** của stack 4 block [1, 2, 4, 8]:

$$RF = 1 + \sum_{d \in \{1,2,4,8\}} 2 \cdot (3-1) \cdot d = 1 + 4 \cdot 2 \cdot 15 = 61 \text{ frames}$$

$RF = 61 > T = 40$ → mỗi vị trí thời gian có thể "nhìn" toàn bộ cửa sổ.

Với same padding (non-causal), RF hiệu dụng là **song hướng** — mô hình sử dụng thông tin từ cả quá khứ lẫn tương lai trong window.

### 6.3 Cơ chế anti-collapse trung bình

Mô hình cũ (MSE + AvgPool) thường hội tụ về **mean pose** (tư thế đứng thẳng tay buông) do gradient MSE nhỏ khi lỗi nhỏ. Ba cơ chế được áp dụng để phá vỡ hiện tượng này:

| Cơ chế | Mô tả |
|---|---|
| **Temporal diff** | Feature $\Delta x$ cung cấp tín hiệu chuyển động trực tiếp |
| **Residual prediction** | Model học deviation từ `mean_pose` learnable, không học absolute position |
| **Diversity loss** | Phạt khi `std(pred) < 0.5 × std(gt)` — buộc model phải dự đoán đa dạng |

**Kết quả**: STD ratio (pred/gt) tăng từ 0.16 → 0.84 sau khi áp dụng ba cơ chế.

### 6.4 Thông số mô hình

| Thông số | Giá trị |
|---|---|
| Tổng tham số | ~185,301 |
| TCN channels (C) | 64 |
| Dilations | [1, 2, 4, 8] |
| Kernel size | 3 |
| Dropout | 0.2 (TCN), 0.3 (pose head) |
| Window size (T) | 40 frames (~2s) |
| Input features | ~546 (sau FeatureCleaner, cấu hình 128-pad cũ) |

---

## 7. Hàm mất mát

### 7.1 Wing Loss

Thay thế MSE để cải thiện gradient cho lỗi nhỏ:

$$L_{Wing}(x) = \begin{cases} w \ln\!\left(1 + \frac{|x|}{\epsilon}\right) & \text{nếu } |x| < w \\ |x| - C & \text{nếu } |x| \geq w \end{cases}$$

$$C = w - w \ln\!\left(1 + \frac{w}{\epsilon}\right)$$

Với $w = 0.1$, $\epsilon = 2.0$. Gradient trong vùng $|x| < w$:

$$\frac{\partial L_{Wing}}{\partial x} = \frac{w}{\epsilon + |x|}$$

Không bằng 0 khi $x \to 0$, khác với MSE ($\frac{\partial L_{MSE}}{\partial x} = 2x \to 0$). Điều này buộc mô hình tiếp tục tối ưu ngay cả khi sai số nhỏ, ngăn hội tụ về mean pose.

### 7.2 Bone Length Consistency Loss

$$L_{bone} = \frac{1}{|B|} \sum_{(a,b) \in B} \frac{\sum_i v_{ia} v_{ib} \left| \|\hat{p}_{ia} - \hat{p}_{ib}\| - \|p_{ia} - p_{ib}\| \right|}{\sum_i v_{ia} v_{ib}}$$

Với $B$ là tập 12 cặp xương cố định theo cấu trúc xương người.

### 7.3 Diversity Loss

$$L_{div} = \frac{1}{J \cdot 2} \sum_{j=1}^{J} \sum_{c \in \{x,y\}} \max\!\left(0,\ 0.5 \cdot \sigma_{jc}^{GT} - \sigma_{jc}^{pred}\right)$$

Phạt khi phương sai dự đoán của mỗi joint nhỏ hơn 50% phương sai của nhãn.

### 7.4 Tổng hàm mất mát

$$L = L_{Wing} + 0.15 \cdot L_{bone} + 0.15 \cdot L_{div}$$

---

## 8. Chiến lược huấn luyện

### 8.1 Phân chia dữ liệu — Interleaved Chunk Split

Dữ liệu theo chuỗi thời gian có độ tương quan cao. Phân chia ngẫu nhiên 80/20 sẽ gây **data leakage** (window train và val overlapping). Thay vào đó:

```
Session (N windows)
│
├─ Chunk 0  → VAL    (index 0..chunk_size-1)
├─ Chunk 1  → Train  (index chunk_size..2×chunk_size-1)
├─ Chunk 2  → Train
├─ Chunk 3  → Train
├─ Chunk 4  → VAL    ← val_every=4
│  ...
```

Mỗi chunk có **buffer biên** `edge_buf = 15` frames ở 2 đầu không được sử dụng, đảm bảo **khoảng cách tối thiểu** giữa window train và val ≥ $2 \times edge\_buf \times dt \approx 1.5\text{s}$ (vượt decorrelation time ~0.1s).

**Tham số:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `CHUNK_SIZE` | 100 | Số window mỗi chunk |
| `STRIDE` | 5 | Bước nhảy giữa các window trong chunk |
| `EDGE_BUF` | 15 | Buffer biên chunk |
| `VAL_EVERY` | 4 | 1/4 chunk → val (~25%) |

### 8.2 Data Augmentation

Áp dụng **chỉ** trên train set:

| Augmentation | Chi tiết |
|---|---|
| Gaussian noise | $\mathcal{N}(0, 0.03)$ cộng vào feature (sau z-norm) |
| Time jitter | Roll ±3 frames theo trục thời gian |
| Amplitude scaling | Nhân toàn bộ feature với $\mathcal{U}(0.92, 1.08)$ |
| Multiplicative noise | Nhân per-feature với $\mathcal{U}(0.90, 1.10)$ (xác suất 50%) |
| Horizontal flip | Hoán vị left-right keypoints + mirror tọa độ x (xác suất 50%) |
| Scale jitter | Scale keypoint quanh centroid với $\mathcal{U}(0.95, 1.05)$ |
| Mixup | Kết hợp tuyến tính hai CSI windows với $\text{Beta}(0.3, 0.3)$ |

### 8.3 Optimizer và Schedule

```
Optimizer : AdamW (lr=3e-4, weight_decay=1e-3)
Scheduler : OneCycleLR (epochs=150, pct_start=0.1)
            Warmup 10% → peak → cosine decay → min_lr = lr/10
Grad clip : 2.0 (norm clipping)
```

### 8.4 Exponential Moving Average (EMA)

$$\theta_{EMA} \leftarrow 0.995 \cdot \theta_{EMA} + 0.005 \cdot \theta_{model}$$

Checkpoint tốt nhất lưu `best_ema.pt` — sử dụng EMA weights cho inference.

### 8.5 Early Stopping

Patience = 25 epoch, theo dõi **val MPJPE** (không phải loss).

---

## 9. Đánh giá mô hình

### 9.1 Metrics

**Mean Per-Joint Position Error (MPJPE)** — metric chính:

$$MPJPE = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{J} \sum_{j=1}^{J} \|\hat{p}_{ij} - p_{ij}\|_2$$

Với tọa độ chuẩn hóa [0,1], MPJPE = 0.10 tương đương 10% chiều rộng/chiều cao ảnh.

**Percentage of Correct Keypoints (PCK)**:

$$PCK@\alpha = \frac{\# \{i,j : \|\hat{p}_{ij} - p_{ij}\|_2 < \alpha \cdot d_{torso}\}}{N \cdot J}$$

Với $d_{torso} = \|p_{left\_shoulder} - p_{right\_hip}\|$.

**Visibility Accuracy**: Độ chính xác phân loại nhị phân keypoint có/không nhìn thấy (BCEWithLogitsLoss).

### 9.2 Hai chế độ đánh giá (`realtime_eval.py`)

**Chế độ dataset** (`--mode dataset`):
- Load val set từ `aligned_*.npz`
- Chạy inference toàn bộ val set 1 lần (batch=64)
- Hiển thị từng sample tuần tự với phím điều hướng
- Hiển thị GT (xanh) vs Prediction (cam) side-by-side

**Chế độ real-time** (`--mode realtime`):
- Kết nối trực tiếp với ESP32 qua UDP
- YOLO11n-pose làm ground truth real-time từ camera
- Hiển thị skeleton dự đoán với latency < 50ms

---

## 10. Suy luận thời gian thực

### 10.1 Pipeline inference

```
[ESP32 UDP] → CSIMeshAggregator
                      │
                      ↓ extract_csi_frame(n_sub=fc.n_sub)
               raw feat: (N_LINKS × 2 × n_sub,)
                      │
                      ↓ sliding window deque (maxlen=40)
               window: (40, N_LINKS × 2 × n_sub)
                      │
                      ↓ FeatureCleaner.transform()
               clean:  (1, 40, F_alive)
                      │
                      ↓ CSIPoseModelV4.forward()
               pose:   (1, 17, 2)  vis: (1, 17)
                      │
                      ↓ np.clip(pose, 0, 1)
               Skeleton overlay trên camera frame
```

### 10.2 Latency

| Thành phần | Thời gian ước tính |
|---|---|
| CSI sampling (1 frame) | 50 ms (20 Hz) |
| FeatureCleaner.transform() | < 1 ms |
| Model inference (GPU) | ~12 ms |
| YOLO detection (ground truth) | ~15–20 ms |
| OpenCV display | ~5 ms |
| **Tổng latency** | **~70–80 ms** |

### 10.3 Nhất quán giữa train và inference

| Bước | Lúc thu/train | Lúc inference |
|---|---|---|
| Feature dim | 3 × 2 × `n_sub` (tự động từ data) | 3 × 2 × `fc.n_sub` (từ FeatureCleaner) |
| Normalization | FeatureCleaner z-norm | FeatureCleaner z-norm (cùng mean, std) |
| Normalize [0,1] per-frame | **Không** | **Không** |
| Window size | 40 frames | 40 frames (deque) |
| n_sub location | Lưu trong csi_raw_*.npz | Đọc từ feature_cleaner.npz |

> **Vấn đề trong phiên bản cũ**: `extract_csi_frame()` áp dụng min-max normalize [0,1] trên window ngắn 20 frames (local scope), trong khi FeatureCleaner fit trên toàn session (global scope) → scale không nhất quán → dự đoán sai. Phiên bản hiện tại lưu raw values, để FeatureCleaner xử lý nhất quán.

---

## 11. Kết quả thực nghiệm

### 11.1 Dữ liệu huấn luyện

| | Thông số |
|---|---|
| Tổng số sessions | 5 sessions |
| Dữ liệu raw (sau align) | ~26,688 windows × (40, 768) |
| Train / Val | ~1,568 / ~532 samples |
| Sampling rate | ~29 Hz (trước dedup), ~20 Hz (sau dedup) |
| Đa dạng tư thế | Thấp (1 người, chủ yếu đứng/ngồi) |

### 11.2 Kết quả (val set, `checkpoints_v4/best_ema.pt`)

| Metric | Giá trị |
|---|---|
| MPJPE | **0.1230** (normalized) |
| PCK@0.10 | **46.8%** |
| PCK@0.15 | **70.5%** |
| Visibility Accuracy | **89.3%** |

### 11.3 Phân tích tương quan CSI–Pose

Từ phân tích trên tập dữ liệu *pre-training* (saved2):

| | Giá trị |
|---|---|
| Max Pearson r (CSI subcarrier ↔ keypoint coord) | 0.35 |
| STD ratio pred/gt (trục x) | 0.84 (sau anti-collapse fix) |
| Pearson r (pred ↔ gt, trục y) | 0.72 |
| Pearson r (pred ↔ gt, trục x) | ~0.08 (yếu) |
| Decorrelation time | ~0.1 s (2 frames) |
| Ankle visibility | ~65% |
| Ear visibility | ~55–60% |

> Node 1 gần như không đóng góp vào top-20 subcarrier có tương quan cao nhất — cho thấy vị trí và loại ESP32 ảnh hưởng đáng kể đến chất lượng tín hiệu.

---

## 12. Điểm mạnh

1. **Bảo vệ quyền riêng tư**: Suy luận hoàn toàn từ tín hiệu WiFi — không lưu trữ hay xử lý hình ảnh người dùng.

2. **Xuyên vật cản**: Tín hiệu WiFi 2.4/5 GHz xuyên tường gỗ/thạch cao, cho phép ứng dụng trong nhà thông minh mà không cần camera mỗi phòng.

3. **Chi phí thấp**: ~300.000 VNĐ phần cứng (3 ESP32 + router WiFi sẵn có), so với ~5–50 triệu VNĐ cho camera depth (RealSense, Kinect).

4. **Pipeline tự khép kín**: Từ thu dữ liệu → gán nhãn tự động (YOLO) → huấn luyện → suy luận, không cần annotation thủ công.

5. **Kiến trúc nhỏ gọn**: ~185K tham số, inference 12ms trên GPU thông thường, khả thi triển khai trên Raspberry Pi 4 (~80ms).

6. **Pipeline nhất quán**: FeatureCleaner đảm bảo z-normalization giống nhau giữa train và inference — tránh scale mismatch.

7. **Anti-collapse hiệu quả**: Kết hợp Wing Loss + temporal diff + residual prediction tăng STD ratio từ 0.16 lên 0.84, phá vỡ hoàn toàn hiện tượng hội tụ mean pose.

---

## 13. Hạn chế và thách thức

### 13.1 Hạn chế về phần cứng

| Hạn chế | Nguyên nhân | Tác động |
|---|---|---|
| Không phân biệt trái/phải | 3 node không đủ spatial resolution | Pearson r trục x ≈ 0.08 |
| Chỉ 52–55 subcarrier thực (không phải 256) | ESP32 dùng HT20 OFDM, không phải HT40 như doc claim | Feature dimension thực tế thấp hơn thiết kế |
| Biên độ pha không ổn định | ESP32 không có clock synchronization | Phase data có quantization artifacts |
| Node 1 (ESP32-S3) yếu hơn | Vị trí/hướng ăng-ten khác | Đóng góp ít vào top correlated features |

### 13.2 Hạn chế về dữ liệu

- **Ít đa dạng**: 5 sessions với ~1 người, tư thế chủ yếu là đứng/ngồi cơ bản.
- **Nhãn phụ thuộc YOLO**: Nếu YOLO sai (ánh sáng yếu, người bị khuất), nhãn huấn luyện sai theo.
- **Phụ thuộc môi trường**: Model huấn luyện trong phòng A sẽ kém chính xác trong phòng B do multipath propagation khác nhau.
- **1 người**: Chưa kiểm thử trên nhiều người hay người ngồi ghế/nằm.

### 13.3 Hạn chế về mô hình

- **Không temporal**: Center frame prediction không nắm bắt được chuỗi chuyển động liên tục.
- **Visibility head yếu**: BCELoss trên visibility thường học threshold rất cao, không phân biệt tốt keypoint mờ.
- **Trục x gần như không dự đoán được** với cấu hình 3 node dọc theo 1 mặt phẳng.

---

## 14. Hướng phát triển

### 14.1 Cải thiện phần cứng (tác động cao, mức độ khó: Thấp)

- **Thêm node thứ 4** ở phía đối diện camera → cải thiện giải quyết không gian trục x.
- **Bố trí tam giác đều** xung quanh vùng hoạt động thay vì 1 phía.
- **Nâng cấp lên ESP32-S3 (HT40)** để lấy nhiều subcarrier hơn thực sự.

### 14.2 Tăng và đa dạng hóa dữ liệu (tác động cao, mức độ khó: Thấp)

- **Thu 15+ sessions** (hiện tại: 5) với đa dạng tư thế: giơ tay, cúi người, ngồi, đi lại.
- **Thu với 2–3 người** khác nhau để tăng generalization.
- **Nhóm thu có chủ đích**: chuyển động chậm (để model học mapping rõ ràng), tư thế tĩnh đa dạng, chuyển động tự nhiên.
- **Dùng depth camera** (Intel RealSense) làm label thay YOLO để cải thiện chất lượng nhãn 3D.

### 14.3 Cải thiện kiến trúc mô hình (tác động cao, mức độ khó: Trung bình)

- **Temporal pose prediction**: Thay center frame extraction bằng predict toàn bộ T frames với LSTM decoder — nhất quán hơn theo thời gian.
- **Multi-scale feature fusion**: Kết hợp output từ nhiều TCN block thay vì chỉ block cuối.
- **Attention trên subcarrier**: Cho model học subcarrier nào quan trọng hơn thay vì xử lý flat.
- **Graph Convolutional Network (GCN)** cho pose head — khai thác cấu trúc skeleton trong không gian output.

### 14.4 Cải thiện pipeline (tác động trung bình, mức độ khó: Thấp)

- **CSI phase sanitization**: Áp dụng linear interpolation để loại bỏ phase jump trước khi lưu raw.
- **Session-adaptive normalization**: Cập nhật running mean/std của FeatureCleaner online thay vì chỉ dùng training statistics.
- **Multi-resolution windows**: Thử nghiệm window 20/40/80 frame và ensemble.

### 14.5 Không khuyến nghị (theo phân tích thực nghiệm)

- **STFT features**: Phân tích thực nghiệm cho thấy raw CSI > STFT cho bài toán pose (max |r|: 0.582 vs 0.393). TCN đã học pattern tần số bên trong.
- **Transfer learning từ MM-Fi**: Domain gap phần cứng (Intel NIC vs ESP32, 180 vs 330 features) quá lớn. Input projection layer phải thay mới, làm mất toàn bộ pre-trained knowledge từ lớp đó.

---

## 15. Cấu trúc tệp dự án

```
wifi_pose_project/python/
│
├── ── PHẦN CỨNG / THU LIỆU ──────────────────────────────────────
│
├── csi_processor.py      Xử lý raw UDP packet → CSIFrame(amp, phase)
│                         Hampel filter, bandpass, motion detection
│                         [THAY ĐỔI v6]: Không padding cố định lên 128
│
├── csi_mesh.py           Quản lý 3 UDP listener (port 5005/5006/5007)
│                         CSIMeshAggregator → list of (node_id, CSIProcessor)
│
├── test.py               Thu thập session: CSI + video đồng bộ
│                         _get_raw_per_node(): lưu raw amp/phase thực tế
│                         stop(): tính max_sub, build flat feature, lưu .npz
│
├── ── DỮ LIỆU / TIỀN XỬ LÝ ─────────────────────────────────────
│
├── align_csi.py          Căn chỉnh CSI ↔ video/YOLO
│                         Đọc video_offset tự động từ csi_raw_*.npz
│                         Output: aligned_*.npz (X, y, visibility, timestamps)
│
├── mediapipe_pose.py     Wrapper MediaPipe pose (dùng trong data_collector_2.py)
│
├── visualize_data.py     5 biểu đồ phân tích chất lượng dữ liệu
│                         Kiểm tra readiness trước khi train
│
├── ── MÔ HÌNH / HUẤN LUYỆN ──────────────────────────────────────
│
├── model.py              CSIPoseModelV4
│                         TCN bidirectional, temporal diff, residual pose
│
├── csi_pose_dataset.py   FeatureCleaner (fit/transform/save/load + n_sub)
│                         CSIPoseDatasetV4, build_datasets(), mixup_collate()
│                         Interleaved chunk split
│
├── train.py              Pipeline huấn luyện đầy đủ
│                         Wing Loss + Bone + Diversity, EMA, OneCycleLR
│
├── ── ĐÁNH GIÁ ──────────────────────────────────────────────────
│
├── evaluate2.py          Offline evaluation + charts (báo cáo)
│                         7 biểu đồ: skeleton, heatmap, PCK, visibility...
│
├── realtime_eval.py      2 chế độ đánh giá:
│                         --mode realtime: camera + ESP32 real-time
│                         --mode dataset: val set navigation với phím
│
├── ── DỮ LIỆU ───────────────────────────────────────────────────
│
├── ruview_sessions/      csi_raw_*.npz + video_*.mp4 (raw sessions)
├── saved2/               aligned_*.npz (training data v2)
├── saved/                aligned_*.npz (training data v1)
├── data/                 aligned_*.npz (data khác)
│
├── checkpoints_v4/
│   ├── best_ema.pt       Best checkpoint (EMA weights, ~185K params)
│   └── feature_cleaner.npz  alive_mask, mean, std, n_sub
│
└── eval_results_v4/      Biểu đồ đánh giá (PNG)
```

---

## Tài liệu tham khảo

1. **Wing Loss**: Feng, Z. et al. (2018). *Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks*. CVPR 2018.

2. **Temporal Convolutional Network**: Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling*. arXiv:1803.01271.

3. **MM-Fi Dataset**: Yang, J. et al. (2023). *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing*. NeurIPS 2023.

4. **COCO Keypoint Format**: Lin, T. Y. et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014.

5. **WiFi CSI Human Sensing**: Wang, W. et al. (2019). *CSI-based Human Pose Estimation*. IEEE Transactions on Mobile Computing.

6. **ESP32 CSI Tool**: [https://github.com/ESP32-CSI](https://github.com/espressif/esp-csi) — Espressif Systems ESP-CSI firmware.

7. **YOLO11n-pose**: Jocher, G. et al. (2023). *Ultralytics YOLO*. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).

---

*Tài liệu này được tạo tự động từ codebase và kết quả thực nghiệm.*  
*Cập nhật lần cuối: 08/04/2026*
