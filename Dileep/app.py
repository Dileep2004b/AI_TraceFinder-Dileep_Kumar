import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import pywt
import io
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern as sk_lbp

# Paths to your model and artifacts
ART_DIR = "model_files"
FP_PATH = f"{ART_DIR}/scanner_fingerprints.pkl"
ORDER_NPY = f"{ART_DIR}/fp_keys.npy"
MODEL_PATH = f"{ART_DIR}/scanner_hybrid.keras"
ENCODER_PATH = f"{ART_DIR}/hybrid_label_encoder.pkl"
SCALER_PATH = f"{ART_DIR}/hybrid_feat_scaler.pkl"

IMG_SIZE = (256, 256)

# Load model and artifacts only once
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FP_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
    return model, le, scaler, scanner_fps, fp_keys

model, le, scaler, scanner_fps, fp_keys = load_resources()

def preprocess_residual_pywt_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img_array, 'haar')
    cH.fill(0)
    cV.fill(0)
    cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    residual = (img_array - den).astype(np.float32)
    return residual

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
        feats.append(float(mag[mask].mean()) if np.any(mask) else 0.0)
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

# Streamlit app starts here
st.title("Scanner Model Prediction")

uploaded_file = st.file_uploader("Upload scanned image (jpg/png/tiff):", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    residual = preprocess_residual_pywt_bytes(img_bytes)
    x_img = residual[np.newaxis, :, :, np.newaxis]
    x_feat = make_feats_from_res(residual)
    prob = model.predict([x_img, x_feat], verbose=0).ravel()
    top_idx = int(np.argmax(prob))
    label = le.classes_[top_idx]
    confidence = prob[top_idx] * 100
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted Scanner Model: **{label}** with confidence {confidence:.2f}%")

