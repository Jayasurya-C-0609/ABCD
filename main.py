import time

import cv2

from config import (
    AUTO_PATH_END_WP_SEQ,
    AUTO_PATH_START_WP_SEQ,
    CAMERA_SOURCE,
    CONNECTION_STRING,
    CORRIDOR_ENTRANCE_WP_SEQ,
    DROIDCAM_URL,
    EXIT_CORRIDOR_WP_SEQ,
    POSITION_TOLERANCE_M,
    START_QR_WP_SEQ,
    SURFACE_ENTRANCE_WP_SEQ,
    TAKEOFF_ALT_M,
)
from mavlink.connection import MavlinkConnection
from mavlink.guided_control import GuidedController
from vision.camera import open_camera


CENTER_THRESHOLD_PX = 40
ALIGN_SPEED_MPS = 0.25
DECODE_ATTEMPTS = 5
ALIGN_TIMEOUT_S = 5.0
WAYPOINT_TIMEOUT_S = 60
START_QR_TIMEOUT_S = 45
DETECTION_INTERVAL_S = 0.8
WRONG_QR_SUPPRESS_S = 12.0
WINDOW_NAME = "SkyScan QR Mission"
_detect_qrs = None
_decode_qr_crop = None


def load_yolo_detector():
    global _detect_qrs

    if _detect_qrs is None:
        print("Loading YOLO QR detector...")
        from vision.qr_detect import detect_qrs as loaded_detect_qrs

        _detect_qrs = loaded_detect_qrs
        print("YOLO QR detector ready")


def detect_qrs(frame):
    load_yolo_detector()
    return _detect_qrs(frame)


def load_qr_decoder():
    global _decode_qr_crop

    if _decode_qr_crop is None:
        print("Loading WeChat QR decoder...")
        from vision.qr_decode import decode_qr_crop as loaded_decode_qr_crop

        _decode_qr_crop = loaded_decode_qr_crop
        print("WeChat QR decoder ready")


def decode_qr_crop(frame, bbox):
    load_qr_decoder()
    return _decode_qr_crop(frame, bbox)


def warm_up_vision_models():
    load_yolo_detector()
    load_qr_decoder()


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def qr_alignment_velocity(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    qr_cx = (x1 + x2) / 2
    qr_cy = (y1 + y2) / 2
    img_cx = w / 2
    img_cy = h / 2

    err_x = qr_cx - img_cx
    err_y = qr_cy - img_cy

    vx = 0.0
    vy = 0.0

    if err_x > CENTER_THRESHOLD_PX:
        vy = ALIGN_SPEED_MPS
    elif err_x < -CENTER_THRESHOLD_PX:
        vy = -ALIGN_SPEED_MPS

    if err_y < -CENTER_THRESHOLD_PX:
        vx = ALIGN_SPEED_MPS
    elif err_y > CENTER_THRESHOLD_PX:
        vx = -ALIGN_SPEED_MPS

    centered = abs(err_x) < CENTER_THRESHOLD_PX and abs(err_y) < CENTER_THRESHOLD_PX
    return vx, vy, centered, err_x, err_y


def read_camera_frame(cap):
    ret, frame = cap.read()

    if not ret or frame is None:
        print("No camera frame")
        return None

    return frame


def show_frame(frame, status):
    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF


def draw_detection(frame, detection, label="QR"):
    x1, y1, x2, y2 = detection["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def align_and_decode_qr(controller, cap, reason):
    print(f"{reason}: QR detected, aligning with center")
    last_velocity_time = 0
    centered_since = None
    best_bbox = None
    align_started_at = time.time()
    decode_frame = None

    while True:
        frame = read_camera_frame(cap)
        if frame is None:
            return None

        detections = detect_qrs(frame)

        if not detections:
            controller.stop()
            centered_since = None
            key = show_frame(frame, f"{reason}: waiting for QR")
            if key == ord("q"):
                return None
            continue

        detection = max(detections, key=lambda det: det["confidence"])
        best_bbox = detection["bbox"]
        decode_frame = frame.copy()
        draw_detection(frame, detection, "ALIGN")

        vx, vy, centered, err_x, err_y = qr_alignment_velocity(frame, best_bbox)
        cv2.putText(
            frame,
            f"err_x={err_x:.0f}, err_y={err_y:.0f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        if centered:
            controller.stop()
            if centered_since is None:
                centered_since = time.time()
            if time.time() - centered_since > 0.7:
                break
        else:
            centered_since = None
            if time.time() - last_velocity_time > 0.15:
                controller.send_body_velocity(vx, vy, 0)
                last_velocity_time = time.time()

        if time.time() - align_started_at > ALIGN_TIMEOUT_S and best_bbox is not None:
            controller.stop()
            print(f"{reason}: alignment timeout, trying decode from visible QR")
            break

        key = show_frame(frame, f"{reason}: aligning QR")
        if key == ord("q"):
            controller.stop()
            return None

    controller.stop()
    print(f"{reason}: decoding QR")

    for attempt in range(1, DECODE_ATTEMPTS + 1):
        frame = decode_frame if attempt == 1 and decode_frame is not None else read_camera_frame(cap)
        if frame is None:
            return None

        fresh_detections = detect_qrs(frame)
        if fresh_detections:
            best_bbox = max(fresh_detections, key=lambda det: det["confidence"])["bbox"]

        decoded_text, processed = decode_qr_crop(frame, best_bbox)

        if processed is not None:
            cv2.imshow("Processed QR", processed)

        if decoded_text:
            print(f"{reason}: decoded QR = {decoded_text}")
            show_frame(frame, f"{reason}: decoded {decoded_text}")
            return decoded_text.strip()

        print(f"{reason}: decode attempt {attempt} failed")
        show_frame(frame, f"{reason}: decode attempt {attempt}")
        time.sleep(0.3)

    print(f"{reason}: QR detected but value not decoded")
    return None


def goto_waypoint(controller, cap, waypoint, label, watch_for_qr=False, target_value=None):
    print(
        f"Moving to {label}: "
        f"Lat={waypoint['lat']:.7f}, Lon={waypoint['lon']:.7f}, Alt={waypoint['alt']:.1f}"
    )

    started_at = time.time()
    last_detection_time = 0
    suppress_qr_until = 0
    detections = []

    while time.time() - started_at < WAYPOINT_TIMEOUT_S:
        controller.goto_global_location(
            waypoint["lat"],
            waypoint["lon"],
            waypoint["alt"],
        )

        frame = read_camera_frame(cap)
        if frame is None:
            return "camera_failed", None

        qr_scan_enabled = watch_for_qr and time.time() >= suppress_qr_until

        if qr_scan_enabled and time.time() - last_detection_time >= DETECTION_INTERVAL_S:
            detections = detect_qrs(frame)
            last_detection_time = time.time()
        elif not qr_scan_enabled:
            detections = []

        if watch_for_qr:
            for detection in detections:
                draw_detection(frame, detection, "QR")

        if watch_for_qr and detections:
            controller.stop()
            time.sleep(0.6)
            decoded_text = align_and_decode_qr(controller, cap, label)

            if decoded_text is None:
                print(f"{label}: could not decode, resuming path")
            elif decoded_text == target_value:
                print("TARGET FOUND")
                return "target_found", decoded_text
            else:
                suppress_qr_until = time.time() + WRONG_QR_SUPPRESS_S
                print(
                    f"{label}: wrong QR {decoded_text}, "
                    f"resuming auto path and ignoring QR for {WRONG_QR_SUPPRESS_S:.0f}s"
                )

            detections = []

        distance = controller.distance_to_global_location(
            waypoint["lat"],
            waypoint["lon"],
            waypoint["alt"],
        )

        if distance is not None and distance <= POSITION_TOLERANCE_M:
            print(f"Reached {label}")
            return "reached", None

        key = show_frame(frame, f"Moving: {label}")
        if key == ord("q"):
            controller.stop()
            return "quit", None

        time.sleep(0.1)

    print(f"{label}: waypoint timeout, continuing")
    return "timeout", None


def scan_start_qr_at_waypoint_2(controller, cap, start_qr_waypoint):
    status, _ = goto_waypoint(controller, cap, start_qr_waypoint, "WP2 Start QR")
    if status in ("quit", "camera_failed"):
        return None

    print("At WP2: scanning start QR")
    started_at = time.time()
    last_detection_time = 0
    detections = []

    while time.time() - started_at < START_QR_TIMEOUT_S:
        frame = read_camera_frame(cap)
        if frame is None:
            return None

        if time.time() - last_detection_time >= DETECTION_INTERVAL_S:
            detections = detect_qrs(frame)
            last_detection_time = time.time()

        for detection in detections:
            draw_detection(frame, detection, "START QR")

        if detections:
            controller.stop()
            time.sleep(0.6)
            decoded_text = align_and_decode_qr(controller, cap, "START QR")
            if decoded_text:
                print(f"Stored target value from start QR: {decoded_text}")
                return decoded_text

            print("Start QR was detected but not decoded, trying again")

        key = show_frame(frame, "WP2: scan start QR")
        if key == ord("q"):
            controller.stop()
            return None

    print("Start QR scan timeout")
    return None


def build_surface_waypoints(mission_items):
    return [
        mission_items[seq]
        for seq in range(AUTO_PATH_START_WP_SEQ, AUTO_PATH_END_WP_SEQ + 1)
        if seq in mission_items
    ]


def run_surface_search(controller, cap, target_value, mission_items):
    print("Entering 40x30 m surface auto path")

    for waypoint in build_surface_waypoints(mission_items):
        status, decoded_text = goto_waypoint(
            controller,
            cap,
            waypoint,
            f"Surface WP{waypoint['seq']}",
            watch_for_qr=True,
            target_value=target_value,
        )

        if status == "target_found":
            return decoded_text

        if status in ("quit", "camera_failed"):
            return None

    print("Surface search completed, target was not found")
    return None


def main():
    cap = None
    controller = None

    try:
        print("Starting MAVLink mission startup")
        master = MavlinkConnection(CONNECTION_STRING).connect()
        controller = GuidedController(master)
        mission_items = controller.download_mission_items()

        required_wps = [
            START_QR_WP_SEQ,
            CORRIDOR_ENTRANCE_WP_SEQ,
            SURFACE_ENTRANCE_WP_SEQ,
            EXIT_CORRIDOR_WP_SEQ,
        ]
        missing_wps = [seq for seq in required_wps if seq not in mission_items]
        if missing_wps:
            raise RuntimeError(f"Mission is missing waypoint(s): {missing_wps}")

        controller.set_mode("GUIDED")
        controller.arm()
        controller.takeoff(TAKEOFF_ALT_M)

        print("Opening camera after takeoff")
        cap = open_camera(CAMERA_SOURCE, DROIDCAM_URL)
        print("Warming up QR models before flight scanning")
        warm_up_vision_models()

        target_value = scan_start_qr_at_waypoint_2(
            controller,
            cap,
            mission_items[START_QR_WP_SEQ],
        )
        if not target_value:
            print("Mission stopped: start QR target value was not decoded")
            controller.stop()
            return

        status, _ = goto_waypoint(
            controller,
            cap,
            mission_items[CORRIDOR_ENTRANCE_WP_SEQ],
            "WP3 Corridor Entrance",
        )
        if status in ("quit", "camera_failed"):
            return

        status, _ = goto_waypoint(
            controller,
            cap,
            mission_items[SURFACE_ENTRANCE_WP_SEQ],
            "WP4 Surface Entrance",
        )
        if status in ("quit", "camera_failed"):
            return

        found_value = run_surface_search(controller, cap, target_value, mission_items)

        if found_value:
            print(f"TARGET FOUND: {found_value}")
            print("Moving directly to exit corridor entrance")
            goto_waypoint(
                controller,
                cap,
                mission_items[EXIT_CORRIDOR_WP_SEQ],
                "Exit Corridor Entrance",
            )
        else:
            print("TARGET NOT FOUND")

        controller.stop()

    except Exception as exc:
        print(f"Mission error: {exc}")
        if controller is not None:
            controller.stop()
            controller.land()

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
