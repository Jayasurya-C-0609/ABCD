"""QR corner preparation for solvePnP alignment."""

import cv2
import numpy as np

from config import QR_REAL_SIZE_M


_qr_detector = cv2.QRCodeDetector()


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def bbox_corners(bbox):
    x1, y1, x2, y2 = bbox
    return np.ascontiguousarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float64,
    )


def padded_bbox_crop(frame, bbox, pad=20):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    padded_bbox = (
        int(clamp(x1 - pad, 0, frame_width - 1)),
        int(clamp(y1 - pad, 0, frame_height - 1)),
        int(clamp(x2 + pad, 0, frame_width - 1)),
        int(clamp(y2 + pad, 0, frame_height - 1)),
    )
    px1, py1, px2, py2 = padded_bbox
    return frame[py1:py2, px1:px2], padded_bbox


def locked_bbox_qr_corners(frame, bbox):
    """Detect QR corners in a padded locked bbox, with bbox fallback."""
    crop, padded_bbox = padded_bbox_crop(frame, bbox)
    if crop.size == 0:
        return None, False, False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    thresholded = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    crop_offset = np.asarray(padded_bbox[:2], dtype=np.float64)

    for candidate in (crop, gray, equalized, thresholded):
        found, points = _qr_detector.detect(candidate)
        if found and points is not None and len(points) > 0:
            corners = np.ascontiguousarray(points.reshape(-1, 2)[:4], dtype=np.float64)
            return corners + crop_offset, True, False

    return bbox_corners(bbox), False, True


def qr_object_points():
    """Return square QR object coordinates in meters."""
    half_size = QR_REAL_SIZE_M / 2.0
    return np.ascontiguousarray(
        [
            [-half_size, -half_size, 0.0],
            [half_size, -half_size, 0.0],
            [half_size, half_size, 0.0],
            [-half_size, half_size, 0.0],
        ],
        dtype=np.float64,
    )
