import cv2
from ultralytics import YOLO

# Load YOLO model once
model = YOLO("models/best.pt")


def detect_qrs(frame, conf=0.5):
    original_h, original_w = frame.shape[:2]
    detect_width = 416
    scale = detect_width / original_w
    detect_height = int(original_h * scale)
    detect_frame = cv2.resize(frame, (detect_width, detect_height))

    results = model.predict(
        source=detect_frame,
        conf=conf,
        imgsz=416,
        device=0,
        verbose=False
    )

    detections = []

    boxes = results[0].boxes

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(box.conf[0])
        })

    return detections
