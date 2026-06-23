from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/lisa_yolov8/weights/best.pt"
)

results = model(
    source="datasets/lisa_yolov8/test/images",
    save=True,
    conf=0.25
)

print("Predictions saved.")