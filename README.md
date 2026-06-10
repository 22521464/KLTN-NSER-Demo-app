# Hệ thống Nhận diện Cảm xúc Tiêu cực trong Giọng nói
**Speech Emotion Recognition (SER) — Ứng dụng cho Call Center / Hotline An ninh**

---

## Giới thiệu

Dự án xây dựng hệ thống nhận diện cảm xúc trong giọng nói (Speech Emotion Recognition) phục vụ giám sát cuộc gọi tại các trung tâm chăm sóc khách hàng và đường dây an ninh. Hệ thống tự động phân tích file ghi âm cuộc gọi, phát hiện các cảm xúc tiêu cực và phát cảnh báo để hỗ trợ nhân viên can thiệp kịp thời.

**Luận văn tốt nghiệp — Khoa Công nghệ Thông tin**

---

## Các cảm xúc nhận diện

| Cảm xúc | Phân loại |
|---------|-----------|
| Angry (Tức giận) | Tiêu cực — Cảnh báo |
| Disgust (Ghê tởm) | Tiêu cực — Cảnh báo |
| Fear (Sợ hãi) | Tiêu cực — Cảnh báo |
| Sad (Buồn bã) | Tiêu cực — Cảnh báo |
| Neutral (Trung tính) | Bình thường |

---

## Các mô hình AI

| Mô hình | Loại | Accuracy | Mô tả |
|---------|------|----------|-------|
| **Wav2Vec2** | HuggingFace Transformer | 93.64% | Fine-tuned Wav2Vec 2.0, xử lý raw audio end-to-end |
| **DNN** | Keras / MFCC | 92.01% | Dense Neural Network, inference nhanh |
| **CNN-LSTM** | Keras / MFCC | 91.44% | CNN + Bidirectional LSTM, nắm bắt đặc trưng cục bộ và tuần tự |
| **HuBERT** | HuggingFace Transformer | 91.33% | Fine-tuned HuBERT, đặc trưng ngữ âm sâu |
| **Transformer** | Keras / MFCC | 90.86% | Self-Attention Encoder, 3 blocks |
| **Conformer** | Keras / MFCC | 87.27% | Conformer (Self-Attention + Depthwise Conv) |

**Dataset:** RAVDESS — 864 file gốc, tăng cường dữ liệu lên 4320 mẫu (5 phiên bản/file).
**Features:** MFCC 40 chiều, sampling rate 16kHz.

---

## Chức năng Demo App

- **Phân tích cuộc gọi** — Upload file WAV/MP3, chọn mô hình, nhận kết quả cảm xúc kèm confidence
- **Cảnh báo tự động** — Highlight màu đỏ khi phát hiện cảm xúc tiêu cực vượt ngưỡng confidence
- **Phân phối xác suất** — Hiển thị xác suất dự đoán cho tất cả 5 cảm xúc
- **Ngưỡng cảnh báo tùy chỉnh** — Điều chỉnh mức confidence tối thiểu để kích hoạt cảnh báo (30–90%)
- **Lịch sử phân tích** — Lưu log toàn bộ kết quả, lọc theo cảm xúc/model/trạng thái cảnh báo
- **Xuất báo cáo** — Tải xuống log dạng CSV

---

## Cấu trúc thư mục

```
Demo app/
├── app.py                        # Ứng dụng Streamlit chính
├── requirements.txt              # Thư viện cần thiết
├── README.md
├── sample_audios/                # File âm thanh mẫu (2 file/cảm xúc)
│   ├── angry_01.wav
│   ├── angry_02.wav
│   └── ...
├── models dnn/                   # DNN model
│   ├── best_dnn_model_m01.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── models cnn lstm/              # CNN-LSTM model
│   ├── best_cnn_lstm_m01.h5
│   ├── scaler_cnn_lstm.pkl
│   └── label_encoder.pkl
├── models transformer/           # Transformer model
│   ├── best_transformer_model_m01.h5
│   ├── scaler_transformer_m01.pkl
│   └── label_encoder_transformer_m01.pkl
├── models conformer/             # Conformer model
│   ├── best_conformer_model_m01.h5
│   ├── scaler_conformer_m01.pkl
│   └── label_encoder_conformer_m01.pkl
├── models hubert/                # HuBERT (load từ HuggingFace Hub)
│   └── best_model/
└── models wav2vec2/              # Wav2Vec2 (load từ HuggingFace Hub)
    └── final_model/
```

> **Lưu ý:** `models hubert/` và `models wav2vec2/` không được đưa vào Git (quá lớn). App tự động tải từ HuggingFace Hub nếu không có local.

---

## Cài đặt và chạy

### Yêu cầu
- Python 3.9+
- pip

### 1. Clone repo

```bash
git clone https://github.com/22521464/KLTN-NSER-Demo-app.git
cd KLTN-NSER-Demo-app
```

### 2. Tạo môi trường ảo và cài thư viện

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# hoặc: .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`.

---

## Lần đầu chạy

- **DNN, CNN-LSTM, Transformer, Conformer** — load từ local, không cần internet.
- **HuBERT, Wav2Vec2** — nếu không có folder `models hubert/` và `models wav2vec2/`, app sẽ tự động tải từ HuggingFace Hub (`TienGiang/ser-hubert-ravdess`, `TienGiang/ser-wav2vec2-ravdess`). Lần đầu cần internet và có thể mất vài phút.

---

## Sử dụng

1. Mở app, chọn mô hình ở thanh bên trái
2. Vào tab **Phân tích cuộc gọi** → Upload file WAV hoặc MP3
3. Nhấn **Phân tích cảm xúc**
4. Xem kết quả: cảm xúc dự đoán, confidence, phân phối xác suất
5. Kết quả được tự động lưu vào tab **Lịch sử cảnh báo**

---

## Công nghệ sử dụng

- **Frontend:** Streamlit
- **Deep Learning:** TensorFlow/Keras, PyTorch, HuggingFace Transformers
- **Audio Processing:** Librosa
- **ML Utilities:** scikit-learn, joblib
