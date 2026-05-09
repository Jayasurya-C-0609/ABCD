from ultralytics import YOLO
import cv2

# Load YOLO model once
model = YOLO("models/best.pt")


def detect_qrs(frame, conf=0.5):

    results = model.predict(
        source=frame,
        conf=conf,
        device=0,
        verbose=False
    )

    detections = []

    boxes = results[0].boxes

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(box.conf[0])
        })

    return detections