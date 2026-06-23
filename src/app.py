import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO(
    "runs/detect/runs/lisa_yolov8/weights/best.pt"
)

st.title("Traffic Sign Detection using YOLOv8")

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded)
    img = np.array(image)

    results = model(img)

    st.image(
        results[0].plot(),
        caption="Prediction"
    )