import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import cv2
import pywt
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern as sk_lbp
import altair as alt

# --- Paths (adjust if needed) ---
ART_DIR = "model_files"
FP_PATH = f"{ART_DIR}/scanner_fingerprints.pkl"
ORDER_NPY = f"{ART_DIR}/fp_keys.npy"
MODEL_PATH = f"{ART_DIR}/scanner_hybrid.keras"
ENCODER_PATH = f"{ART_DIR}/hybrid_label_encoder.pkl"
SCALER_PATH = f"{ART_DIR}/hybrid_feat_scaler.pkl"
IMG_SIZE = (256, 256)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FP_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
    return model, le, scaler, scanner_fps, fp_keys

model, le, scaler, scanner_fps, fp_keys = load_artifacts()

def preprocess_residual_pywt_bytes(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    return (img - den).astype(np.float32)

def corr2d(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = fftshift(fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K + 1)
    feats = []
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        feats.append(float(mag[mask].mean() if np.any(mask) else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = (img - img.min()) / (rng + 1e-8) if rng >= 1e-12 else np.zeros_like(img)
    g8 = (g * 255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins + 1), density=True)
    return hist.tolist()

def make_feats_from_res(res):
    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(res, K=6)
    v_lbp = lbp_hist_safe(res, P=8, R=1.0)
    feats = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
    return scaler.transform(feats)

st.set_page_config(page_title="🖨️ Scanner Model Classifier", layout="wide")

st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🖨️ Scanner Model Classifier</h1>", unsafe_allow_html=True)
st.markdown("Upload a scanned image to identify which scanner model it came from.")

if "log" not in st.session_state:
    st.session_state["log"] = []

uploaded_file = st.file_uploader("📤 Choose an image file (jpg, png, tif, etc.)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    with st.spinner("🔍 Analyzing image..."):
        try:
            residual = preprocess_residual_pywt_bytes(uploaded_file)
            x_img = np.expand_dims(residual, axis=(0, -1))
            x_feat = make_feats_from_res(residual)
            prob = model.predict([x_img, x_feat], verbose=0).ravel()
            top_idx = int(np.argmax(prob))
            label = le.classes_[top_idx]
            confidence = float(prob[top_idx] * 100.0)

            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.success(f"**Predicted Scanner:** {label}  🎯 with confidence: {confidence:.2f}%")

            # Show confidence bar chart for top 5 predictions
            top5_idx = prob.argsort()[-5:][::-1]
            top5_labels = [le.classes_[i] for i in top5_idx]
            top5_confidences = prob[top5_idx] * 100
            
            df_conf = pd.DataFrame({
                "Scanner Model": top5_labels,
                "Confidence (%)": top5_confidences
            })

            chart = alt.Chart(df_conf).mark_bar(color="#4B8BBE").encode(
                x=alt.X("Confidence (%):Q", scale=alt.Scale(domain=[0,100])),
                y=alt.Y("Scanner Model:N", sort="-x")
            ).properties(height=200, width=600)

            st.altair_chart(chart)

            # Log prediction
            st.session_state["log"].append({
                "filename": uploaded_file.name,
                "predicted_scanner": label,
                "confidence_percent": f"{confidence:.2f}"
            })

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

if st.session_state["log"]:
    st.markdown("---")
    st.subheader("Prediction Log")
    df_log = pd.DataFrame(st.session_state["log"])
    st.dataframe(df_log, use_container_width=True)
    csv = df_log.to_csv(index=False)
    st.download_button(
        label="⬇️ Download prediction log as CSV",
        data=csv,
        file_name="prediction_log.csv",
        mime="text/csv"
    )

