"""QR image-space alignment helpers."""

import math
import time

from config import PIXEL_ALIGN_SPEED, SOLVEPNP_MAX_SPEED


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def select_best_qr_detection(detections):
    """Pick the highest-confidence QR detection."""
    return max(detections, key=lambda detection: detection.get("confidence", 0.0))


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_center_distance(first_bbox, second_bbox):
    first_center = bbox_center(first_bbox)
    second_center = bbox_center(second_bbox)
    return math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )


def select_locked_qr_detection(detections, locked_bbox):
    """Track the detection closest to the target QR's previous bbox."""
    if not detections:
        return None
    if locked_bbox is None:
        return select_best_qr_detection(detections)
    return min(
        detections,
        key=lambda detection: bbox_center_distance(detection["bbox"], locked_bbox),
    )


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def select_reacquired_qr_detection(detections, locked_bbox):
    """Choose a reacquired QR, preferring the previous target location."""
    if not detections:
        return None
    if locked_bbox is not None:
        return select_locked_qr_detection(detections, locked_bbox)
    return max(
        detections,
        key=lambda detection: (
            detection.get("confidence", 0.0),
            bbox_area(detection["bbox"]),
        ),
    )


def bbox_near_frame_edge(bbox, frame, margin_ratio=0.08):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    edge_x = max(20, int(frame_width * margin_ratio))
    edge_y = max(20, int(frame_height * margin_ratio))
    return (
        x1 <= edge_x
        or y1 <= edge_y
        or x2 >= frame_width - edge_x
        or y2 >= frame_height - edge_y
    )


def recenter_locked_bbox(controller, bbox, frame):
    """Move a near-edge QR gently back toward the camera center."""
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = bbox_center(bbox)
    error_x = center_x - (frame_width / 2.0)
    error_y = center_y - (frame_height / 2.0)
    vx = clamp(
        -PIXEL_ALIGN_SPEED * (error_y / max(frame_height * 0.5, 1.0)),
        -SOLVEPNP_MAX_SPEED,
        SOLVEPNP_MAX_SPEED,
    )
    vy = clamp(
        PIXEL_ALIGN_SPEED * (error_x / max(frame_width * 0.5, 1.0)),
        -SOLVEPNP_MAX_SPEED,
        SOLVEPNP_MAX_SPEED,
    )
    print(f"solvePnP edge recenter error_x={error_x:.1f}, error_y={error_y:.1f}")
    print(f"solvePnP speed command vx={vx:.3f}, vy={vy:.3f}")
    controller.send_body_velocity(vx, vy, 0.0)
    time.sleep(0.2)
    controller.stop()
