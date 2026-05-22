"""Red-zone YOLO detection helpers."""

from pathlib import Path

import cv2
from ultralytics import YOLO

from config import RED_MODEL_PATH, REDZONE_YOLO_CONF


_model = None
_model_available = None


def load_redzone_detector():
    """Load the optional red-zone detector once."""
    global _model, _model_available

    if _model_available is False:
        return None
    if _model is not None:
        return _model
    if not Path(RED_MODEL_PATH).exists():
        print(
            f"WARNING: REDdet model missing: {RED_MODEL_PATH}. "
            "Continuing with coordinate-only red-zone avoidance."
        )
        _model_available = False
        return None

    print(f"RED model loaded path: {RED_MODEL_PATH}")
    _model = YOLO(RED_MODEL_PATH)
    _model_available = True
    return _model


def detect_redzone_yolo(frame, conf=REDZONE_YOLO_CONF, debug_frame=None):
    """Return the strongest red-zone YOLO result for one frame."""
    model = load_redzone_detector()
    if model is None:
        return False, None, 0.0

    try:
        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=416,
            device=0,
            verbose=False,
        )
    except Exception as exc:
        print(f"WARNING: red-zone YOLO inference failed: {exc}")
        return False, None, 0.0

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False, None, 0.0

    best_box = max(boxes, key=lambda box: float(box.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    confidence = float(best_box.conf[0])
    bbox = (x1, y1, x2, y2)

    if debug_frame is not None:
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            debug_frame,
            f"RED {confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return confidence >= conf, bbox, confidence
