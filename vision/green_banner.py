"""Green banner YOLO detection for corridor entry."""

from pathlib import Path

from ultralytics import YOLO

from config import GREEN_BANNER_CONF, GREEN_BANNER_MODEL_PATH


_model = None
_model_available = None


def load_green_banner_detector():
    """Load the green banner model once and return it when available."""
    global _model, _model_available

    if _model_available is False:
        return None
    if _model is not None:
        return _model
    if not Path(GREEN_BANNER_MODEL_PATH).exists():
        print(f"WARNING: green banner model missing: {GREEN_BANNER_MODEL_PATH}")
        _model_available = False
        return None

    _model = YOLO(GREEN_BANNER_MODEL_PATH)
    _model_available = True
    print("Green banner model loaded")
    return _model


def detect_green_banner(frame, conf=GREEN_BANNER_CONF):
    """Return the strongest green banner detection for one frame."""
    model = load_green_banner_detector()
    if model is None:
        return None

    try:
        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=416,
            device=0,
            verbose=False,
        )
    except Exception as exc:
        print(f"WARNING: green banner YOLO inference failed: {exc}")
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_box = max(boxes, key=lambda box: float(box.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    confidence = float(best_box.conf[0])
    return {
        "bbox": (x1, y1, x2, y2),
        "confidence": confidence,
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
    }
