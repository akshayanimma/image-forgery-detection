"""
app.py - Streamlit Community Cloud entry point.

Self-contained: doesn't import from dataset.py/model.py, since the deploy
only needs this file + best_model.pth + requirements.txt.
"""

import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown

# --- Download the model checkpoint from Google Drive if not already present ---
# (GitHub's web upload has a 25MB limit; ResNet-18 checkpoints are usually
# ~44MB, so we keep the file in Drive and fetch it here instead.)
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "1MHf68utb_vzEKE4w3GWOxUm1dgaD5DW1"  # see instructions below for how to get this

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model (first run only)..."):
        url = f"https://drive.google.com/uc?id={"1MHf68utb_vzEKE4w3GWOxUm1dgaD5DW1"}"
        gdown.download(url=url, output=MODEL_PATH, quiet=False, fuzzy=True)

# --- Page setup ---
st.set_page_config(page_title="Image Forgery Detector", page_icon="🔍")
st.title("🔍 Image Forgery Detector")
st.write(
    "Fine-tuned ResNet-18 trained on the CASIA v2 dataset to detect "
    "digitally tampered (spliced / copy-moved) images vs. authentic ones. "
    "Test accuracy: 88.8%."
)

# --- Model definition (same architecture used in training) ---
@st.cache_resource  # loads the model only once, not on every interaction
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Same preprocessing used during training/evaluation ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class_names = {0: "Authentic", 1: "Tampered"}

# --- UI: upload + predict ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = probs.argmax().item()
    st.subheader(f"Prediction: **{class_names[pred_idx]}**")
    st.write(f"Authentic: {probs[0]:.2%}")
    st.write(f"Tampered: {probs[1]:.2%}")
