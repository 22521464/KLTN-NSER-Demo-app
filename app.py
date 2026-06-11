import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["METAL_DEVICE_ORDINAL"] = ""

import streamlit as st
import numpy as np
import librosa
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
SR = 16000
N_MFCC = 40
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}
LOG_FILE = Path(__file__).parent / "alert_log.csv"
BASE_DIR = Path(__file__).parent

MODEL_OPTIONS = {
    "DNN": {
        "model_dir": "models dnn",
        "model_file": "best_dnn_model_m01.h5",
        "scaler_file": "scaler.pkl",
        "encoder_file": "label_encoder.pkl",
        "accuracy": "92.71%",
        "description": "Dense Neural Network — nhẹ, inference nhanh (~ms)",
        "input_format": "flat",
        "model_type": "keras",
    },
    "CNN-LSTM": {
        "model_dir": "models cnn lstm",
        "model_file": "best_cnn_lstm_m01.h5",
        "scaler_file": "scaler_cnn_lstm.pkl",
        "encoder_file": "label_encoder.pkl",
        "accuracy": "91.44%",
        "description": "CNN + Bidirectional LSTM — nắm bắt đặc trưng cục bộ và tuần tự",
        "input_format": "3d_last",
        "model_type": "keras",
    },
    "Conformer": {
        "model_dir": "models conformer",
        "model_file": "best_conformer_model_m01.h5",
        "scaler_file": "scaler_conformer_m01.pkl",
        "encoder_file": "label_encoder_conformer_m01.pkl",
        "accuracy": "87.27%",
        "description": "Conformer — Self-Attention + Depthwise Conv, kiến trúc hybrid mạnh cho audio",
        "input_format": "3d_last",
        "model_type": "keras",
        "load_direct": True,
    },
    "Transformer": {
        "model_dir": "models transformer",
        "model_file": "best_transformer_model_m01.h5",
        "scaler_file": "scaler_transformer_m01.pkl",
        "encoder_file": "label_encoder_transformer_m01.pkl",
        "accuracy": "90.86%",
        "description": "Transformer (Self-Attention) — 3 encoder blocks, MFCC features",
        "input_format": "3d_first",
        "model_type": "keras",
    },
    "HuBERT": {
        "model_dir": "models hubert/best_model",
        "hf_repo": "TienGiang/ser-hubert-ravdess",
        "accuracy": "91.33%",
        "description": "HuBERT fine-tuned — Transformer xử lý raw audio, đặc trưng ngữ âm sâu",
        "model_type": "huggingface",
    },
    "Wav2Vec2": {
        "model_dir": "models wav2vec2/final_model",
        "hf_repo": "TienGiang/ser-wav2vec2-ravdess",
        "accuracy": "93.64%",
        "description": "Wav2Vec 2.0 fine-tuned — Transformer end-to-end, accuracy cao nhất",
        "model_type": "huggingface",
    },
}

EMOTION_CONFIG = {
    "angry":   {"color": "#FF4B4B", "alert": True},
    "disgust": {"color": "#FF8C00", "alert": True},
    "fear":    {"color": "#FF6B6B", "alert": True},
    "sad":     {"color": "#4A90D9", "alert": True},
    "neutral": {"color": "#28A745", "alert": False},
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SER Security System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .alert-box {
        background: linear-gradient(135deg, #FF4B4B22, #FF4B4B11);
        border: 2px solid #FF4B4B;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .safe-box {
        background: linear-gradient(135deg, #28A74522, #28A74511);
        border: 2px solid #28A745;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .model-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid #444;
        margin-bottom: 0.5rem;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,75,75,0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255,75,75,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,75,75,0); }
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
def _load_weights_from_keras3_h5(model, h5path):
    import h5py, numpy as np
    with h5py.File(h5path, "r") as f:
        mw = f["model_weights"]
        for layer in model.layers:
            if not layer.weights or layer.name not in mw:
                continue
            top = mw[layer.name]
            if layer.name in top:
                base = top[layer.name]
            elif "sequential" in top and layer.name in top["sequential"]:
                base = top["sequential"][layer.name]
            else:
                continue
            weights = []
            for w in layer.weights:
                parts = w.name.replace(":0", "").split("/")[1:]
                node = base
                ok = True
                for p in parts:
                    if p in node:
                        node = node[p]
                    elif p + ":0" in node:
                        node = node[p + ":0"]
                    else:
                        ok = False
                        break
                if ok:
                    weights.append(np.array(node))
            if weights:
                layer.set_weights(weights)


def _build_dnn(num_classes=5):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    return Sequential([
        Input(shape=(40,)),
        Dense(512, activation="relu"), BatchNormalization(), Dropout(0.4),
        Dense(256, activation="relu"), BatchNormalization(), Dropout(0.4),
        Dense(128, activation="relu"), Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])


def _build_conformer(num_classes=5):
    from tensorflow.keras import layers, Model, Input
    import tensorflow as tf
    inp = Input(shape=(40, 1), name="input_1")
    x = layers.Dense(128, name="dense")(inp)
    for i in range(3):
        ln_base = i * 5
        d_base  = i * 4 + 1
        def ln(off, _lb=ln_base):
            n = _lb + off
            return "layer_normalization" if n == 0 else f"layer_normalization_{n}"
        def dn(off, _db=d_base):
            n = _db + off
            return "dense_1" if n == 1 else f"dense_{n}"
        mha_n = "multi_head_attention" if i == 0 else f"multi_head_attention_{i}"
        c1_n  = "conv1d"               if i == 0 else f"conv1d_{i*2}"
        c2_n  = f"conv1d_{i*2+1}"
        sc_n  = "separable_conv1d"     if i == 0 else f"separable_conv1d_{i}"
        bn_n  = "batch_normalization"  if i == 0 else f"batch_normalization_{i}"
        ff1 = layers.LayerNormalization(name=ln(0))(x)
        ff1 = layers.Dense(256, activation="relu", name=dn(0))(ff1)
        ff1 = layers.Dropout(0.1)(ff1)
        ff1 = layers.Dense(128, name=dn(1))(ff1)
        ff1 = layers.Dropout(0.1)(ff1)
        x = layers.Add()([x, ff1 * 0.5])
        mhsa = layers.LayerNormalization(name=ln(1))(x)
        mhsa = layers.MultiHeadAttention(num_heads=4, key_dim=32, name=mha_n)(mhsa, mhsa)
        mhsa = layers.Dropout(0.1)(mhsa)
        x = layers.Add()([x, mhsa])
        conv = layers.LayerNormalization(name=ln(2))(x)
        conv = layers.Conv1D(256, 1, name=c1_n)(conv)
        conv_a, conv_b = tf.split(conv, 2, axis=-1)
        conv = conv_a * tf.math.sigmoid(conv_b)
        conv = layers.SeparableConv1D(128, 7, padding="same", name=sc_n)(conv)
        conv = layers.BatchNormalization(name=bn_n)(conv)
        conv = layers.Conv1D(128, 1, name=c2_n)(conv)
        conv = layers.Dropout(0.1)(conv)
        x = layers.Add()([x, conv])
        ff2 = layers.LayerNormalization(name=ln(3))(x)
        ff2 = layers.Dense(256, activation="relu", name=dn(2))(ff2)
        ff2 = layers.Dropout(0.1)(ff2)
        ff2 = layers.Dense(128, name=dn(3))(ff2)
        ff2 = layers.Dropout(0.1)(ff2)
        x = layers.Add()([x, ff2 * 0.5])
        x = layers.LayerNormalization(name=ln(4))(x)
    x = layers.GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    x = layers.Dense(128, activation="relu", name="dense_13")(x)
    x = layers.BatchNormalization(name="batch_normalization_3")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="dense_14")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dense_15")(x)
    return Model(inp, outputs)


def _build_transformer(num_classes=5):
    from tensorflow.keras import layers, Model, Input
    inp = Input(shape=(1, 40), name="input_1")
    x = inp
    for i in range(3):
        mha_name = "multi_head_attention" if i == 0 else f"multi_head_attention_{i}"
        ln1_name = f"layer_normalization_{i * 2}"     if i > 0 else "layer_normalization"
        ln2_name = f"layer_normalization_{i * 2 + 1}" if i > 0 else "layer_normalization_1"
        d1_name  = "dense"   if i == 0 else f"dense_{i * 2}"
        d2_name  = "dense_1" if i == 0 else f"dense_{i * 2 + 1}"
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, name=mha_name)(x, x)
        attn = layers.Dropout(0.1)(attn)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization(name=ln1_name)(x)
        ff = layers.Dense(256, activation="relu", name=d1_name)(x)
        ff = layers.Dense(40, name=d2_name)(ff)
        ff = layers.Dropout(0.1)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization(name=ln2_name)(x)
    x = layers.GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    x = layers.Dense(128, activation="relu", name="dense_6")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dense_7")(x)
    return Model(inp, outputs)


def _build_cnn_lstm(num_classes=5):
    from tensorflow.keras import layers, Model, Input as KInput
    inp = KInput(shape=(40, 1), name="input")
    x = layers.Conv1D(64, 3, activation="relu", padding="same", name="conv1d_1")(inp)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.MaxPooling1D(2, name="maxpool_1")(x)
    x = layers.Dropout(0.25, name="dropout_1")(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling1D(2, name="maxpool_2")(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)
    x = layers.Conv1D(256, 3, activation="relu", padding="same", name="conv1d_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(0.25, name="dropout_3")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), name="bilstm_1")(x)
    x = layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3), name="bilstm_2")(x)
    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_final")(x)
    x = layers.Dropout(0.5, name="dropout_dense")(x)
    x = layers.Dense(128, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.3, name="dropout_final")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inp, out)


@st.cache_resource
def load_all_artifacts():
    import tensorflow as tf

    loaded = {}
    errors = []

    keras_cfg = next(c for c in MODEL_OPTIONS.values() if c["model_type"] == "keras")
    encoder = joblib.load(BASE_DIR / keras_cfg["model_dir"] / keras_cfg["encoder_file"])
    encoder_cache = {}

    for name, cfg in MODEL_OPTIONS.items():
        models_dir = BASE_DIR / cfg["model_dir"]

        if cfg["model_type"] == "huggingface":
            source = str(models_dir) if models_dir.exists() else cfg.get("hf_repo")
            if not source:
                errors.append(str(models_dir))
                loaded[name] = None
                continue
            try:
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
                fe = AutoFeatureExtractor.from_pretrained(source)
                hf_model = AutoModelForAudioClassification.from_pretrained(source)
                hf_model.eval()
                loaded[name] = {"model": hf_model, "feature_extractor": fe}
            except Exception as e:
                errors.append(f"{name}: {e}")
                loaded[name] = None
        else:
            model_path = models_dir / cfg["model_file"]
            scaler_path = models_dir / cfg["scaler_file"]
            missing = [str(p) for p in [model_path, scaler_path] if not p.exists()]
            if missing:
                errors.extend(missing)
                loaded[name] = None
                continue
            enc_path = models_dir / cfg["encoder_file"]
            enc = joblib.load(enc_path) if enc_path.exists() else encoder
            encoder_cache[name] = enc
            num_classes = len(enc.classes_)
            if cfg.get("load_direct"):
                model = tf.keras.models.load_model(str(model_path), compile=False)
            elif name == "CNN-LSTM":
                model = _build_cnn_lstm(num_classes)
            elif name == "Transformer":
                model = _build_transformer(num_classes)
            else:
                model = _build_dnn(num_classes)
            if not cfg.get("load_direct"):
                _load_weights_from_keras3_h5(model, str(model_path))
            loaded[name] = {"model": model, "scaler": joblib.load(scaler_path), "encoder": enc}

    return loaded, encoder, errors


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)


# ── Predict ────────────────────────────────────────────────────────────────────
def predict_emotion(audio_bytes: bytes, model_name: str, artifacts: dict, encoder):
    import io

    cfg = MODEL_OPTIONS[model_name]
    art = artifacts[model_name]
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

    if cfg["model_type"] == "huggingface":
        import torch
        fe = art["feature_extractor"]
        hf_model = art["model"]
        max_len = SR * 5
        audio = audio[:max_len]
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = hf_model(**inputs).logits
        probs_arr = torch.softmax(logits, dim=-1)[0].numpy()
        id2label = hf_model.config.id2label
        probs = {id2label[i]: float(probs_arr[i]) * 100 for i in range(len(probs_arr))}
        idx = int(probs_arr.argmax())
        label = id2label[idx]
        confidence = float(probs_arr[idx]) * 100
    else:
        model = art["model"]
        scaler = art["scaler"]
        enc = art.get("encoder", encoder)
        feat = extract_features(audio, SR).reshape(1, -1)
        feat_sc = scaler.transform(feat)
        fmt = cfg.get("input_format", "flat")
        if fmt == "3d_last":
            feat_sc = feat_sc.reshape(feat_sc.shape[0], feat_sc.shape[1], 1)
        elif fmt == "3d_first":
            feat_sc = feat_sc.reshape(feat_sc.shape[0], 1, feat_sc.shape[1])
        probs_arr = model(feat_sc, training=False).numpy()[0]
        idx = int(np.argmax(probs_arr))
        label = enc.classes_[idx]
        confidence = float(probs_arr[idx]) * 100
        probs = {enc.classes_[i]: float(probs_arr[i]) * 100 for i in range(len(probs_arr))}

    return label, confidence, probs


def predict_ensemble(audio_bytes: bytes, artifacts: dict, encoder, loaded_models: list):
    all_class_probs: dict = {}
    model_count = 0
    for model_name in loaded_models:
        if artifacts.get(model_name) is None:
            continue
        try:
            _, _, probs = predict_emotion(audio_bytes, model_name, artifacts, encoder)
            for cls, p in probs.items():
                all_class_probs.setdefault(cls, []).append(p)
            model_count += 1
        except Exception:
            continue
    if not all_class_probs:
        raise ValueError("Không có mô hình nào hoạt động")
    avg_probs = {cls: float(np.mean(vals)) for cls, vals in all_class_probs.items()}
    best_label = max(avg_probs, key=avg_probs.get)
    confidence = avg_probs[best_label]
    return best_label, confidence, avg_probs, model_count


# ── Render result helper ───────────────────────────────────────────────────────
def render_result(label: str, confidence: float, all_probs: dict, model_label: str, threshold: int) -> bool:
    is_alert = label in NEGATIVE_EMOTIONS and confidence >= threshold
    if is_alert:
        st.markdown(f"""
        <div class="alert-box">
            <h2 style="color:#FF4B4B; margin:0">CANH BAO</h2>
            <h1 style="color:#FF4B4B; font-size:3rem; margin:0.2rem 0">{label.upper()}</h1>
            <h3 style="color:#FF4B4B; margin:0">Confidence: {confidence:.1f}%</h3>
            <small style="color:#FF4B4B; opacity:0.8">Model: {model_label}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        note = ""
        if label in NEGATIVE_EMOTIONS:
            note = f"<br><small style='color:#888'>(Confidence {confidence:.1f}% &lt; ngưỡng {threshold}%)</small>"
        st.markdown(f"""
        <div class="safe-box">
            <h2 style="color:#28A745; margin:0">BINH THUONG</h2>
            <h1 style="font-size:3rem; margin:0.2rem 0">{label.upper()}</h1>
            <h3 style="color:#28A745; margin:0">Confidence: {confidence:.1f}%{note}</h3>
            <small style="color:#28A745; opacity:0.8">Model: {model_label}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Phân phối xác suất:**")
    for emo, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        highlight = " — Dự đoán" if emo == label else ""
        st.markdown(f"**{emo.capitalize()}**{highlight}")
        st.progress(prob / 100, text=f"{prob:.1f}%")
    return is_alert


# ── Log helpers ────────────────────────────────────────────────────────────────
def append_log(filename: str, model_name: str, emotion: str, confidence: float, is_alert: bool):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "model": model_name,
        "emotion": emotion,
        "confidence_pct": round(confidence, 1),
        "alert": "YES" if is_alert else "NO",
    }
    df_new = pd.DataFrame([row])
    if LOG_FILE.exists():
        df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_FILE, index=False)


def load_log() -> pd.DataFrame:
    cols = ["timestamp", "filename", "model", "emotion", "confidence_pct", "alert"]
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE, on_bad_lines="skip")
    return pd.DataFrame(columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>Hệ thống Nhận diện Cảm xúc Tiêu cực</h1>
    <p style="margin:0; opacity:0.8; font-size:1.1rem;">
        Speech Emotion Recognition — Call Center / Hotline An ninh
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Trạng thái hệ thống")
    artifacts, encoder, errors = load_all_artifacts()

    if errors:
        for e in errors:
            st.error(f"Thiếu file: `{os.path.basename(e)}`")

    loaded_models = [name for name, v in artifacts.items() if v is not None]
    if not loaded_models:
        st.stop()

    st.success(f"Đã tải: {', '.join(loaded_models)}")
    st.info(f"Nhãn: {', '.join(encoder.classes_)}")

    st.divider()

    # Ensemble toggle
    st.markdown("### Chế độ phân tích")
    use_ensemble = st.checkbox(
        "Ensemble — dùng tất cả mô hình",
        value=False,
        help="Trung bình xác suất từ tất cả mô hình đã tải để tăng độ chính xác",
    )

    if use_ensemble:
        selected_model = None
        model_label = f"Ensemble ({len(loaded_models)} models)"
        st.info(f"Sẽ chạy: {', '.join(loaded_models)}")
    else:
        st.markdown("#### Chọn mô hình đơn")
        selected_model = st.radio(
            "Mô hình nhận diện",
            options=loaded_models,
            label_visibility="collapsed",
        )
        cfg = MODEL_OPTIONS[selected_model]
        st.markdown(f"""
        <div class="model-card">
            <b>{selected_model}</b><br>
            <small style="color:#aaa">{cfg['description']}</small><br>
            <small>Accuracy: <b>{cfg['accuracy']}</b></small>
        </div>
        """, unsafe_allow_html=True)
        model_label = selected_model

    st.divider()
    st.markdown("### Cảm xúc cần cảnh báo")
    for emo in sorted(NEGATIVE_EMOTIONS):
        st.markdown(f"- **{emo.capitalize()}**")

    st.divider()
    confidence_threshold = st.slider(
        "Ngưỡng confidence (%)", min_value=30, max_value=90, value=50, step=5,
        help="Chỉ alert khi confidence >= ngưỡng này"
    )

# Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Phân tích cuộc gọi", "Lịch sử cảnh báo"])

# ── Tab 1: Analysis ────────────────────────────────────────────────────────────
with tab1:
    sub1, sub2, sub3 = st.tabs(["Tải file đơn", "Ghi âm trực tiếp", "Xử lý hàng loạt"])

    # ── Sub-tab 1: Single upload ───────────────────────────────────────────────
    with sub1:
        col_upload, col_result = st.columns([1, 1], gap="large")

        with col_upload:
            st.markdown("#### Upload file ghi âm")
            st.caption(f"Chế độ: **{model_label}**")
            uploaded = st.file_uploader(
                "Chọn file WAV hoặc MP3",
                type=["wav", "mp3"],
                help="Hỗ trợ WAV / MP3, mono hoặc stereo. Hệ thống tự chuyển sang 16kHz.",
                key="single_upload",
            )

            if uploaded:
                st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
                st.caption(f"`{uploaded.name}` — {uploaded.size / 1024:.1f} KB")
                analyze_btn = st.button("Phân tích cảm xúc", type="primary", use_container_width=True, key="btn_single")

        with col_result:
            st.markdown("#### Kết quả phân tích")

            if uploaded and analyze_btn:
                with st.spinner(f"Đang phân tích bằng {model_label}..."):
                    try:
                        audio_bytes = uploaded.read()
                        if use_ensemble:
                            label, confidence, all_probs, n_models = predict_ensemble(
                                audio_bytes, artifacts, encoder, loaded_models
                            )
                            display_model = f"Ensemble ({n_models} models)"
                        else:
                            label, confidence, all_probs = predict_emotion(
                                audio_bytes, selected_model, artifacts, encoder
                            )
                            display_model = selected_model

                        is_alert = render_result(label, confidence, all_probs, display_model, confidence_threshold)
                        append_log(uploaded.name, display_model, label, confidence, is_alert)

                        if is_alert:
                            st.toast(f"CẢNH BÁO — {display_model}: {label.upper()} ({confidence:.1f}%)")
                        else:
                            st.toast(f"{display_model}: {label.upper()}")

                    except Exception as e:
                        st.error(f"Lỗi khi phân tích: {e}")
            else:
                st.info("Upload file WAV/MP3 và nhấn **Phân tích cảm xúc** để bắt đầu.")

    # ── Sub-tab 2: Microphone ──────────────────────────────────────────────────
    with sub2:
        col_mic, col_mic_result = st.columns([1, 1], gap="large")

        with col_mic:
            st.markdown("#### Ghi âm trực tiếp")
            st.caption(f"Chế độ: **{model_label}**")
            st.info("Nhấn nút mic bên dưới để bắt đầu ghi âm, nhấn lại để dừng.")
            audio_input = st.audio_input("Ghi âm giọng nói", key="mic_input")

            if audio_input:
                st.caption(f"Thời lượng ghi âm: {audio_input.size / 1024:.1f} KB")
                analyze_mic_btn = st.button(
                    "Phân tích cảm xúc", type="primary", use_container_width=True, key="btn_mic"
                )

        with col_mic_result:
            st.markdown("#### Kết quả phân tích")

            if audio_input and analyze_mic_btn:
                with st.spinner(f"Đang phân tích bằng {model_label}..."):
                    try:
                        audio_bytes = audio_input.read()
                        if use_ensemble:
                            label, confidence, all_probs, n_models = predict_ensemble(
                                audio_bytes, artifacts, encoder, loaded_models
                            )
                            display_model = f"Ensemble ({n_models} models)"
                        else:
                            label, confidence, all_probs = predict_emotion(
                                audio_bytes, selected_model, artifacts, encoder
                            )
                            display_model = selected_model

                        is_alert = render_result(label, confidence, all_probs, display_model, confidence_threshold)
                        append_log("mic_recording", display_model, label, confidence, is_alert)

                        if is_alert:
                            st.toast(f"CANH BAO — {display_model}: {label.upper()} ({confidence:.1f}%)")
                        else:
                            st.toast(f"{display_model}: {label.upper()}")

                    except Exception as e:
                        st.error(f"Lỗi khi phân tích: {e}")
            else:
                st.info("Ghi âm xong và nhấn **Phân tích cảm xúc** để bắt đầu.")

    # ── Sub-tab 3: Batch ───────────────────────────────────────────────────────
    with sub3:
        st.markdown("#### Xử lý hàng loạt")
        st.caption(f"Chế độ: **{model_label}** — Upload nhiều file, phân tích tất cả cùng lúc")

        uploaded_batch = st.file_uploader(
            "Chọn nhiều file WAV / MP3",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            help="Có thể chọn nhiều file cùng lúc (Ctrl+Click hoặc Shift+Click)",
            key="batch_upload",
        )

        if uploaded_batch:
            st.caption(f"Đã chọn: **{len(uploaded_batch)} file**")
            analyze_batch_btn = st.button(
                f"Phân tích tất cả {len(uploaded_batch)} file",
                type="primary",
                use_container_width=True,
                key="btn_batch",
            )

            if analyze_batch_btn:
                results = []
                progress_bar = st.progress(0, text="Đang phân tích...")
                status_text = st.empty()

                for i, f in enumerate(uploaded_batch):
                    status_text.caption(f"Đang xử lý: `{f.name}` ({i+1}/{len(uploaded_batch)})")
                    try:
                        audio_bytes = f.read()
                        if use_ensemble:
                            label, confidence, all_probs, n_models = predict_ensemble(
                                audio_bytes, artifacts, encoder, loaded_models
                            )
                            display_model = f"Ensemble ({n_models} models)"
                        else:
                            label, confidence, all_probs = predict_emotion(
                                audio_bytes, selected_model, artifacts, encoder
                            )
                            display_model = selected_model

                        is_alert = label in NEGATIVE_EMOTIONS and confidence >= confidence_threshold
                        append_log(f.name, display_model, label, confidence, is_alert)
                        results.append({
                            "File": f.name,
                            "Cảm xúc": label.capitalize(),
                            "Confidence (%)": round(confidence, 1),
                            "Cảnh báo": "YES" if is_alert else "NO",
                            "Model": display_model,
                        })
                    except Exception as e:
                        results.append({
                            "File": f.name,
                            "Cảm xúc": "Lỗi",
                            "Confidence (%)": 0.0,
                            "Cảnh báo": "NO",
                            "Model": display_model,
                        })

                    progress_bar.progress((i + 1) / len(uploaded_batch), text=f"{i+1}/{len(uploaded_batch)} file")

                status_text.empty()
                progress_bar.empty()

                df_batch = pd.DataFrame(results)
                n_alert = (df_batch["Cảnh báo"] == "YES").sum()
                n_ok = len(df_batch) - n_alert

                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng file", len(df_batch))
                m2.metric("Cảnh báo", n_alert, delta=None)
                m3.metric("Bình thường", n_ok)

                st.dataframe(
                    df_batch,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Cảnh báo": st.column_config.TextColumn(
                            "Cảnh báo",
                            help="YES = cảm xúc tiêu cực vượt ngưỡng",
                        ),
                        "Confidence (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

                csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Tải xuống kết quả (CSV)",
                    data=csv_bytes,
                    file_name=f"batch_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        else:
            st.info("Upload nhiều file WAV/MP3 và nhấn **Phân tích tất cả** để bắt đầu.")

# ── Tab 2: History ─────────────────────────────────────────────────────────────
with tab2:
    df_log = load_log()

    if df_log.empty:
        st.info("Chưa có lịch sử phân tích.")
    else:
        total = len(df_log)
        alerts = (df_log["alert"] == "YES").sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng cuộc gọi", total)
        col2.metric("Cảnh báo", alerts)
        col3.metric("Tỷ lệ cảnh báo", f"{alerts/total*100:.1f}%")
        col4.metric("Bình thường", total - alerts)

        st.divider()

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            filter_alert = st.selectbox("Lọc cảnh báo", ["Tất cả", "Chỉ cảnh báo", "Không cảnh báo"])
        with filter_col2:
            filter_emotion = st.selectbox("Lọc cảm xúc", ["Tất cả"] + list(encoder.classes_))
        with filter_col3:
            model_col = "model" if "model" in df_log.columns else None
            if model_col:
                filter_model = st.selectbox("Lọc model", ["Tất cả"] + list(MODEL_OPTIONS.keys()) + ["Ensemble"])
            else:
                filter_model = "Tất cả"

        df_display = df_log.copy()
        if filter_alert == "Chỉ cảnh báo":
            df_display = df_display[df_display["alert"] == "YES"]
        elif filter_alert == "Không cảnh báo":
            df_display = df_display[df_display["alert"] == "NO"]
        if filter_emotion != "Tất cả":
            df_display = df_display[df_display["emotion"] == filter_emotion]
        if filter_model != "Tất cả" and model_col:
            if filter_model == "Ensemble":
                df_display = df_display[df_display["model"].str.startswith("Ensemble")]
            else:
                df_display = df_display[df_display["model"] == filter_model]

        st.dataframe(
            df_display.sort_values("timestamp", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Tải xuống log (CSV)",
            data=csv_bytes,
            file_name=f"alert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        if st.button("Xóa toàn bộ lịch sử", type="secondary"):
            LOG_FILE.unlink(missing_ok=True)
            st.rerun()
