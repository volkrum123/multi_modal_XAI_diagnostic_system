from pathlib import Path
import pandas as pd
import torch
import streamlit as st
from PIL import Image

from src.config import load_config
from src.model import MultiModalPneumoniaModel
from src.utils import get_device
from src.feedback import save_feedback
from torchvision import transforms
from captum.attr import LayerGradCam, LayerAttribution
import numpy as np

st.set_page_config(page_title="Multi-Modal XAI Pneumonia Prototype", layout="wide")

cfg = load_config()
device = get_device()
checkpoint = Path(cfg["output"]["checkpoint"])

st.title("Multi-Modal XAI Pneumonia Diagnostic Prototype")
st.caption("Research prototype — not for clinical diagnosis.")

@st.cache_resource
def load_model():
    if not checkpoint.exists():
        return None
    bundle = torch.load(checkpoint, map_location=device)
    model = MultiModalPneumoniaModel(
        pretrained=False,
        metadata_dim=cfg["model"]["metadata_dim"],
        image_embedding_dim=cfg["model"]["image_embedding_dim"],
        metadata_embedding_dim=cfg["model"]["metadata_embedding_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model

model = load_model()

if model is None:
    st.warning("No trained checkpoint found. Train the model before using the application.")
    st.stop()

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
age = st.number_input("Patient age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", ["F", "M"])
view = st.selectbox("View position", ["PA", "AP"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input CXR", width=500)

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x = transform(image).unsqueeze(0).to(device)
    metadata = torch.tensor([[
        age / 100.0,
        1.0 if sex == "M" else 0.0,
        1.0 if view == "AP" else 0.0,
    ]], dtype=torch.float32, device=device)

    with torch.no_grad():
        probability = torch.sigmoid(model(x, metadata)).item()

    label = "Pneumonia" if probability >= cfg["training"]["threshold"] else "No pneumonia"

    st.subheader("Model prediction")
    st.metric("Pneumonia probability", f"{probability:.1%}")
    st.write("Prediction:", label)

    if st.button("Generate Grad-CAM"):
        target_layer = model.image_encoder.features.denseblock4
        gradcam = LayerGradCam(model, target_layer)
        attr = gradcam.attribute(x, target=0, additional_forward_args=(metadata,))
        attr = LayerAttribution.interpolate(attr, x.shape[-2:])
        heatmap = attr[0].mean(dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        st.image(
            heatmap,
            caption="Grad-CAM attribution — regions contributing to the prediction",
            clamp=True,
        )

    st.divider()
    st.subheader("Human-in-the-Loop review")

    decision = st.radio(
        "Reviewer assessment",
        ["Accept model prediction", "Reject model prediction"],
    )
    corrected = st.selectbox(
        "Corrected label",
        ["No pneumonia", "Pneumonia"],
    )
    comment = st.text_area("Reviewer comment")

    if st.button("Submit feedback"):
        corrected_label = 1 if corrected == "Pneumonia" else 0
        save_feedback(
            image_path=uploaded.name,
            model_probability=probability,
            model_label=label,
            reviewer_decision=decision,
            corrected_label=corrected_label,
            comment=comment,
        )
        st.success("Feedback recorded for the HITL research queue.")
