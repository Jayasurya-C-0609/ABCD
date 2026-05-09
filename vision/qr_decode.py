import cv2

# Load WeChat QR decoder
wechat_detector = cv2.wechat_qrcode.WeChatQRCode(
    "models/detect_2021nov.prototxt",
    "models/detect_2021nov.caffemodel",
    "models/sr_2021nov.prototxt",
    "models/sr_2021nov.caffemodel"
)


def decode_qr_crop(frame, bbox):

    x1, y1, x2, y2 = bbox

    # Dynamic padding
    pad = int((x2 - x1) * 0.15)

    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)

    x2_pad = min(frame.shape[1], x2 + pad)
    y2_pad = min(frame.shape[0], y2 + pad)

    # Crop QR
    qr_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

    if qr_crop.size == 0:
        return None, None

    # Resize
    qr_crop = cv2.resize(
        qr_crop,
        None,
        fx=3,
        fy=3
    )

    # Convert grayscale
    gray = cv2.cvtColor(
        qr_crop,
        cv2.COLOR_BGR2GRAY
    )

    # Decode
    res, points = wechat_detector.detectAndDecode(gray)

    if len(res) > 0:
        return res[0], gray

    return None, gray