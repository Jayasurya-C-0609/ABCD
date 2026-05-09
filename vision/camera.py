import cv2


def _configure_capture(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def open_camera(source=0, droidcam_url=""):
    sources = []

    if droidcam_url:
        sources.append(("DroidCam URL", droidcam_url, None))

    if isinstance(source, int):
        sources.extend([
            ("MSMF", source, cv2.CAP_MSMF),
            ("Default", source, cv2.CAP_ANY),
        ])
    else:
        sources.append(("Video source", source, None))

    for name, value, backend in sources:
        print(f"Trying camera using {name}: {value}")
        cap = cv2.VideoCapture(value) if backend is None else cv2.VideoCapture(value, backend)
        _configure_capture(cap)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera opened using {name}: {value}")
                return cap

        cap.release()

    raise RuntimeError(
        "Camera not opened. For DroidCam, try setting DROIDCAM_URL in config.py "
        "to something like 'http://PHONE_IP:4747/video', or change CAMERA_SOURCE "
        "to 0, 1, or 2."
    )
