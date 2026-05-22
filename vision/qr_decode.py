import cv2

# Load WeChat QR decoder
wechat_detector = cv2.wechat_qrcode.WeChatQRCode(
    "models/detect_2021nov.prototxt",
    "models/detect_2021nov.caffemodel",
    "models/sr_2021nov.prototxt",
    "models/sr_2021nov.caffemodel"
)

opencv_detector = cv2.QRCodeDetector()


def _clean_text(text):
    if isinstance(text, (list, tuple)):
        for item in text:
            cleaned = _clean_text(item)
            if cleaned:
                return cleaned
        return None

    if text is None:
        return None

    text = str(text).strip()
    return text if text else None


def _try_wechat(image):
    res, _ = wechat_detector.detectAndDecode(image)
    return _clean_text(res)


def _try_opencv(image):
    text, _, _ = opencv_detector.detectAndDecode(image)
    return _clean_text(text)


def decode_qr_crop(frame, bbox):

    x1, y1, x2, y2 = bbox

    # Dynamic padding
    box_size = max(x2 - x1, y2 - y1)
    pad = int(box_size * 0.45)

    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)

    x2_pad = min(frame.shape[1], x2 + pad)
    y2_pad = min(frame.shape[0], y2 + pad)

    # Crop QR
    qr_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

    if qr_crop.size == 0:
        return None, None

    scale = 4 if max(qr_crop.shape[:2]) < 250 else 2
    qr_crop = cv2.resize(qr_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(qr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)

    candidates = [
        qr_crop,
        gray,
        binary,
        sharpened,
    ]

    for candidate in candidates:
        decoded_text = _try_wechat(candidate)
        if decoded_text:
            return decoded_text, candidate

    for candidate in candidates:
        decoded_text = _try_opencv(candidate)
        if decoded_text:
            return decoded_text, candidate

    return None, binary
