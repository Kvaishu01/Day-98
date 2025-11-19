import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt

st.title("🎭 Mask R-CNN – Image Segmentation ")

# Load pretrained Mask R-CNN
@st.cache_resource
def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model

model = load_model()

# Transform
transform = T.Compose([T.ToTensor()])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Convert to numpy
    img_np = np.array(image)

    # Plot masks + boxes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)

    for i in range(len(prediction["scores"])):
        score = prediction["scores"][i].item()
        if score < 0.5:
            continue

        mask = prediction["masks"][i, 0].cpu().numpy()
        mask = mask > 0.5

        # Overlay mask
        img_np[mask] = [255, 0, 0]  # Red mask

        # Bounding box
        x1, y1, x2, y2 = prediction["boxes"][i].cpu().numpy()
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor="yellow", linewidth=2))

    ax.imshow(img_np)
    ax.axis("off")
    st.pyplot(fig)
