from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training on: {device}")

model = YOLO("yolov8n.pt")

model.train(
    data="datasets/lisa_yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=device,
    project="runs",
    name="lisa_yolov8"
)
