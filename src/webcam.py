from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/lisa_yolov8/weights/best.pt"
)

from ultralytics import YOLO
import cv2

# Load model
model = YOLO(
    "runs/detect/runs/lisa_yolov8/weights/best.pt"
)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Traffic Sign Detection", annotated_frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()