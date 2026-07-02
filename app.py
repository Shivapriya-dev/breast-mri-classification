import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)

os.makedirs("models", exist_ok=True)

MODEL_FILES = {
    "efficientnet_b0_breast_mri22.keras": "1NeQOG2GH-fFshHCjkS2_W1n3iBJml5vT",
    "convnext_tiny_breast_mri.keras": "17XgJQuG49fmcpIsrIDp6vZLtzsLVkjLZ",
    "resnet50updated_breast_mri.keras": "1ZJ6rp4uf7ttOegXEKGiSWw2KYiQQDkgi",
    "densenet121.keras": "1ocRG_Qb-lG6K-AuQPGPZTaX7mc68QZX7",
}

def download_models():
    for filename, file_id in MODEL_FILES.items():
        filepath = os.path.join("models", filename)

        if not os.path.exists(filepath):
            st.info(f"Downloading {filename}... Please wait.")
            success = gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                filepath,
                quiet=False
            )
            st.write("Downloaded:", success)
            st.write("Exists:", os.path.exists(filepath))


download_models()

@st.cache_resource
def load_models():
    return {
        "EfficientNetB0": tf.keras.models.load_model("models/efficientnet_b0_breast_mri22.keras"),
        "ResNet50": tf.keras.models.load_model("models/resnet50updated_breast_mri.keras"),
        "DenseNet121": tf.keras.models.load_model("models/densenet121.keras"),
        "ConvNeXtTiny": tf.keras.models.load_model("models/convnext_tiny_breast_mri.keras")
    }

models = load_models()

st.set_page_config(page_title="Breast MRI AI", layout="centered")

# ===============================
# CUSTOM AWARENESS STYLE
# ===============================
st.markdown("""
<style>

/* Soft pink background */
.stApp {
    background-color: #fff5f8;
}

.main-container {
    max-width: 720px;
    margin: 30px auto;
    padding: 0px;
    background-color: white;
    border-radius: 25px;
    box-shadow: 0 12px 35px rgba(214, 51, 132, 0.08);
    overflow: hidden;
}

/* Header section inside card */
.header-section {
    padding: 35px 30px 20px 30px;
    text-align: center;
    background: linear-gradient(135deg, #fff0f5, #ffe6f0);
}

/* Title */
.main-title {
    color: #d63384;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Subtitle */
.subtitle {
    color: #666;
    font-size: 15px;
}

/* Body content area */
.content-section {
    padding: 30px 35px 40px 35px;
}

/* Result card */
.result-card {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #888;
    margin-top: 40px;
}

/* Remove default top spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
            
.block-container {
    padding-top: 0rem;
    padding-bottom: 2rem;
}
footer {visibility: hidden;}

/* Remove Streamlit header */
header {visibility: hidden;}

/* Remove top spacing */
.block-container {
    padding-top: 0rem !important;
}
/* Hide Streamlit header completely */
header[data-testid="stHeader"] {
    display: none;
}

/* Remove extra padding from top */
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}                 
.block-container {
    padding-top: 0.5rem;
}
section.main > div:first-child {
    padding-top: 0rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("### 💗 About This Tool")
    st.write("AI-powered breast MRI classification system designed to assist early detection research.")
    st.markdown("---")
    st.info("This tool is not a substitute for medical diagnosis.")
    st.markdown("Created for awareness & academic research.")

# ===============================
# MAIN CONTAINER START
# ===============================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-section">
    <div class="main-title">Breast MRI Classification System</div>
    <div class="subtitle">Supporting early detection through AI assistance</div>
</div>
""", unsafe_allow_html=True)

# Content Section
st.markdown('<div class="content-section">', unsafe_allow_html=True)

st.markdown("🌸 Please select a model and upload an MRI image below.")

st.markdown("---")

# ===============================
# MODEL SELECTION
# ===============================
selected_model_name = st.selectbox("Select AI Model", list(models.keys()))
model = models[selected_model_name]

uploaded_file = st.file_uploader("Upload Breast MRI Image", type=["jpg", "png", "jpeg"])

# ===============================
# PREDICTION
# ===============================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width="stretch")

    img = np.array(image)
    img = cv2.resize(img, IMG_SIZE)
    img_preprocessed = preprocess_input(img)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

    with st.spinner("Analyzing image with care..."):
        prediction = model.predict(img_preprocessed)[0][0]

    if prediction >= 0.5:
        label = "Malignant"
        confidence = prediction
        bg_color = "#f8d7da"
        text_color = "#842029"
    else:
        label = "Benign"
        confidence = 1 - prediction
        bg_color = "#d4edda"
        text_color = "#0f5132"

    st.markdown(f"""
    <div class="result-card" style="background-color:{bg_color}; color:{text_color};">
        {label} <br>
        Confidence: {confidence*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    if st.checkbox("Show Grad-CAM Explanation"):
        heatmap = make_gradcam_heatmap(img_preprocessed, model)

        if heatmap is not None:
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed = heatmap * 0.4 + img

            st.image(superimposed.astype("uint8"),
                     caption="Grad-CAM Visualization",
                     width="stretch")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close content-section
st.markdown("</div>", unsafe_allow_html=True)  # close main-container
# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer">
💗 Breast Cancer Awareness AI Tool <br>
© 2026 | Developed for academic research and awareness.
</div>
""", unsafe_allow_html=True)