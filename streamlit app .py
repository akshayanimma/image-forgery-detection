"""
app.py - Streamlit Community Cloud entry point.

Self-contained: doesn't import from dataset.py/model.py, since the deploy
only needs this file + best_model.pth + requirements.txt.
"""

import os
import time

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown

# --- Download the model checkpoint from Google Drive if not already present ---
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "1MHf68utb_vzEKE4w3GWOxUm1dgaD5DW1"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (first run only)..."):
        url = f"https://drive.google.com/uc?id={"1MHf68utb_vzEKE4w3GWOxUm1dgaD5DW1"}"
        gdown.download(url=url, output=MODEL_PATH, quiet=False)

# --- Page setup ---
st.set_page_config(
    page_title="Image Forgery Detector",
    page_icon="🔍",
    layout="centered",
)

# A little custom CSS to make it feel less like a bare default Streamlit page.
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9aa0a6;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-authentic {
        background-color: #16a34a33;
        color: #4ade80;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-tampered {
        background-color: #dc262633;
        color: #f87171;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🔍 Image Forgery Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Fine-tuned ResNet-18 · CASIA v2 dataset · 88.8% test accuracy</div>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About this project"):
    st.write(
        "This model was fine-tuned from an ImageNet-pretrained ResNet-18 to "
        "classify images as **authentic** or **digitally tampered** "
        "(spliced / copy-moved), trained and evaluated on the CASIA v2 "
        "benchmark dataset. It's a proof-of-concept for identity/document "
        "fraud detection use cases, not a production-grade detector."
    )

# --- Model definition (same architecture used in training) ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class_names = {0: "Authentic", 1: "Tampered"}

# --- UI: upload + predict ---
uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png"],
    help="Works best on document/photo-style images similar to CASIA v2.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        time.sleep(0.4)  # small pause so the spinner is visible rather than an instant flash
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()

    with col2:
        badge_class = "badge-authentic" if pred_idx == 0 else "badge-tampered"
        icon = "✅" if pred_idx == 0 else "⚠️"
        st.markdown(
            f'<span class="{badge_class}">{icon} {class_names[pred_idx]}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Confidence", f"{confidence:.1%}")

        st.write("**Class probabilities**")
        st.progress(float(probs[0]), text=f"Authentic — {probs[0]:.1%}")
        st.progress(float(probs[1]), text=f"Tampered — {probs[1]:.1%}")

    if pred_idx == 1 and confidence < 0.65:
        st.info(
            "The model flagged this as tampered but with relatively low "
            "confidence — worth a closer manual look."
        )
else:
    st.caption("Upload a JPG or PNG to get a prediction.")
