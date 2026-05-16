import time

import cv2

from config import (
    AUTO_PATH_END_WP_SEQ,
    AUTO_PATH_START_WP_SEQ,
    CAMERA_SOURCE,
    CONNECTION_STRING,
    CORRIDOR_ENTRANCE_WP_SEQ,
    CRUISE_SPEED_MPS,
    DROIDCAM_URL,
    EXIT_CORRIDOR_WP_SEQ,
    EXIT_CORRIDOR_ALT_M,
    PAYLOAD_DESCENT_ALT_M,
    PAYLOAD_HOVER_TIME_S,
    POSITION_TOLERANCE_M,
    RETURN_ALT_M,
    START_QR_WP_SEQ,
    SURFACE_ENTRANCE_WP_SEQ,
    TAKEOFF_ALT_M,
)
from mavlink.connection import MavlinkConnection
from mavlink.guided_control import GuidedController
from vision.camera import open_camera


DECODE_ATTEMPTS = 5
WAYPOINT_TIMEOUT_S = 60
START_QR_TIMEOUT_S = 45
DETECTION_INTERVAL_S = 0.8
WRONG_QR_SUPPRESS_S = 12.0
LOOP_SLEEP_S = 0.02
ALTITUDE_TOLERANCE_M = 0.3
POSITION_RETRY_TIMEOUT_S = 3.0
WINDOW_NAME = "SkyScan QR Mission"
_detect_qrs = None
_decode_qr_crop = None
_last_frame_time = None
_display_fps = 0.0


def set_mission_state(state_name):
    print(f"\n=== STATE: {state_name} ===")


def waypoint_with_altitude(waypoint, altitude_m):
    updated = dict(waypoint)
    updated["alt"] = altitude_m
    return updated


def target_text_matches(decoded_text, target_value):
    if decoded_text is None or target_value is None:
        return False

    return decoded_text.strip().casefold() == target_value.strip().casefold()


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


def read_camera_frame(cap):
    ret, frame = cap.read()

    if not ret or frame is None:
        print("No camera frame")
        return None

    return frame


def show_frame(frame, status):
    global _last_frame_time, _display_fps

    now = time.time()
    if _last_frame_time is not None:
        instant_fps = 1.0 / max(now - _last_frame_time, 0.001)
        _display_fps = (_display_fps * 0.85) + (instant_fps * 0.15)
    _last_frame_time = now

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {_display_fps:.1f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
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
    print(f"{reason}: QR detected, stopping and decoding without center alignment")
    controller.stop()

    for attempt in range(1, DECODE_ATTEMPTS + 1):
        frame = read_camera_frame(cap)
        if frame is None:
            return None

        detections = detect_qrs(frame)
        if not detections:
            print(f"{reason}: no QR visible on decode attempt {attempt}")
            key = show_frame(frame, f"{reason}: waiting for QR")
            if key == ord("q"):
                return None
            time.sleep(0.2)
            continue

        print(f"{reason}: decode attempt {attempt}, QR detections = {len(detections)}")

        for index, detection in enumerate(detections, start=1):
            draw_detection(frame, detection, f"QR {index}")
            decoded_text, processed = decode_qr_crop(frame, detection["bbox"])

            if processed is not None:
                cv2.imshow("Processed QR", processed)

            print(f"{reason}: QR {index} decoded = {decoded_text}")
            if decoded_text:
                show_frame(frame, f"{reason}: decoded {decoded_text}")
                return decoded_text.strip()

        show_frame(frame, f"{reason}: decode attempt {attempt}")
        time.sleep(0.3)

    print(f"{reason}: QR detected but value not decoded")
    return None


def hover_with_camera(controller, cap, hover_time_s, label, hold_waypoint=None):
    print(f"{label}: hovering for {hover_time_s:.1f}s")
    end_time = time.time() + hover_time_s

    while time.time() < end_time:
        remaining_s = max(0.0, end_time - time.time())
        print(f"{label}: hover timer {remaining_s:.1f}s remaining")
        if hold_waypoint is None:
            controller.stop()
        else:
            command_position_hold(controller, hold_waypoint)

        frame = read_camera_frame(cap)
        if frame is not None:
            key = show_frame(frame, label)
            if key == ord("q"):
                return "quit"
        time.sleep(0.2)

    return "done"


def get_latest_global_position(controller, timeout_s=0.2):
    msg = controller.master.recv_match(
        type="GLOBAL_POSITION_INT",
        blocking=True,
        timeout=timeout_s,
    )

    if msg is None:
        return None

    return {
        "type": "global",
        "source": "latest GLOBAL_POSITION_INT",
        "lat": msg.lat / 1e7,
        "lon": msg.lon / 1e7,
        "alt": msg.relative_alt / 1000.0,
    }


def get_safe_current_position(controller):
    end_time = time.time() + POSITION_RETRY_TIMEOUT_S

    while time.time() < end_time:
        position = controller.get_global_position(timeout_s=0.4)
        if position is not None:
            safe_position = {
                "type": "global",
                "source": "controller.get_global_position",
                "lat": position["lat"],
                "lon": position["lon"],
                "alt": position["relative_alt"],
            }
            print(f"Current position source used: {safe_position['source']}")
            return safe_position

        position = get_latest_global_position(controller, timeout_s=0.2)
        if position is not None:
            print(f"Current position source used: {position['source']}")
            return position

        local_position = controller.get_local_position(timeout_s=0.2)
        if local_position is not None:
            north, east, down = local_position
            safe_position = {
                "type": "local",
                "source": "LOCAL_POSITION_NED",
                "north": north,
                "east": east,
                "alt": -down,
            }
            print(f"Current position source used: {safe_position['source']}")
            return safe_position

        print("Current position unavailable, holding and retrying...")
        controller.stop()
        time.sleep(0.2)

    print("Current position unavailable after retry")
    return None


def command_position_hold(controller, position):
    if position["type"] == "global":
        controller.goto_global_location(position["lat"], position["lon"], position["alt"])
    else:
        controller.goto_local_position(position["north"], position["east"], position["alt"])


def get_current_altitude(controller):
    position = controller.get_global_position(timeout_s=0.05)
    if position is not None:
        return position["relative_alt"], "GLOBAL_POSITION_INT"

    local_position = controller.get_local_position(timeout_s=0.05)
    if local_position is not None:
        return -local_position[2], "LOCAL_POSITION_NED"

    return None, None


def change_altitude_at_current_xy(controller, cap, target_alt_m, label):
    current_position = get_safe_current_position(controller)
    if current_position is None:
        return "position_failed", None

    target_position = dict(current_position)
    target_position["alt"] = target_alt_m

    started_at = time.time()
    while time.time() - started_at < WAYPOINT_TIMEOUT_S:
        command_position_hold(controller, target_position)
        current_alt, altitude_source = get_current_altitude(controller)

        if current_alt is not None:
            print(
                f"{label}: current_alt={current_alt:.2f}m, "
                f"target_alt={target_alt_m:.2f}m, source={altitude_source}"
            )
            if abs(current_alt - target_alt_m) <= ALTITUDE_TOLERANCE_M:
                print(f"{label}: reached target altitude {target_alt_m:.2f}m")
                return "reached", target_position
        else:
            print(f"{label}: altitude unavailable while changing altitude")

        frame = read_camera_frame(cap)
        if frame is not None:
            key = show_frame(frame, label)
            if key == ord("q"):
                return "quit", target_position

        time.sleep(0.1)

    print(f"{label}: altitude change timeout")
    return "timeout", target_position


def goto_waypoint(
    controller,
    cap,
    waypoint,
    label,
    watch_for_qr=False,
    target_value=None,
    arrival_tolerance_m=None,
):
    print(
        f"Moving to {label}: "
        f"Lat={waypoint['lat']:.7f}, Lon={waypoint['lon']:.7f}, Alt={waypoint['alt']:.1f}"
    )

    started_at = time.time()
    last_detection_time = 0
    suppress_qr_until = 0
    detections = []
    arrival_tolerance_m = arrival_tolerance_m or POSITION_TOLERANCE_M

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
            elif target_text_matches(decoded_text, target_value):
                controller.stop()
                print(f"Correct QR detected: {decoded_text}")
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

        if distance is not None and distance <= arrival_tolerance_m:
            print(f"Reached {label}")
            return "reached", None

        key = show_frame(frame, f"Moving: {label}")
        if key == ord("q"):
            controller.stop()
            return "quit", None

        time.sleep(LOOP_SLEEP_S)

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


def run_post_target_sequence(controller, cap, mission_items):
    set_mission_state("TARGET_QR_FOUND")
    controller.stop()
    print("Correct QR detected, stopping lawn mower/search motion")

    set_mission_state("DESCEND_TO_5M")
    print("Descending to 5m")
    status, payload_point = change_altitude_at_current_xy(
        controller,
        cap,
        PAYLOAD_DESCENT_ALT_M,
        "Descend to 5m at target QR",
    )
    if status != "reached":
        print(
            "DESCEND_TO_5M: position/altitude hold failed, "
            "continuing directly to exit corridor entrance"
        )
        payload_point = None

    if payload_point is not None:
        set_mission_state("HOVER_AFTER_DECODE")
        status = hover_with_camera(
            controller,
            cap,
            PAYLOAD_HOVER_TIME_S,
            "Hover after decode at 5m",
            hold_waypoint=payload_point,
        )
        if status == "quit":
            return status

    set_mission_state("GOTO_EXIT_WP16")
    status, _ = goto_waypoint(
        controller,
        cap,
        waypoint_with_altitude(mission_items[EXIT_CORRIDOR_WP_SEQ], PAYLOAD_DESCENT_ALT_M),
        "Exit Corridor WP16 at 5m",
        watch_for_qr=False,
    )
    if status in ("quit", "camera_failed"):
        return status
    print("Reached waypoint 16")

    set_mission_state("GOTO_WP3")
    status, _ = goto_waypoint(
        controller,
        cap,
        waypoint_with_altitude(mission_items[CORRIDOR_ENTRANCE_WP_SEQ], RETURN_ALT_M),
        "Corridor WP3 at 10m",
        watch_for_qr=False,
    )
    if status in ("quit", "camera_failed"):
        return status
    print("Reached waypoint 3")

    set_mission_state("DESCEND_FOR_CORRIDOR")
    print("Corridor altitude active: descending to 2-3m")
    status, corridor_low_point = change_altitude_at_current_xy(
        controller,
        cap,
        EXIT_CORRIDOR_ALT_M,
        "Corridor WP3 descent to 3m",
    )
    if status != "reached":
        return status

    set_mission_state("MOVE_WP3_TO_WP4")
    print("Corridor altitude active: moving WP3 to WP4 at 2-3m")
    status, _ = goto_waypoint(
        controller,
        cap,
        waypoint_with_altitude(mission_items[SURFACE_ENTRANCE_WP_SEQ], EXIT_CORRIDOR_ALT_M),
        "Surface Entrance WP4 at 3m",
        watch_for_qr=False,
    )
    if status in ("quit", "camera_failed"):
        return status
    print("Reached waypoint 4")

    set_mission_state("ASCEND_AFTER_WP4")
    print("Ascending back to 10m")
    status, _ = goto_waypoint(
        controller,
        cap,
        waypoint_with_altitude(mission_items[SURFACE_ENTRANCE_WP_SEQ], RETURN_ALT_M),
        "Surface Entrance WP4 at 10m",
        watch_for_qr=False,
        arrival_tolerance_m=0.8,
    )
    if status in ("quit", "camera_failed"):
        return status

    remaining_return_waypoints = [
        mission_items[seq]
        for seq in sorted(mission_items)
        if seq > EXIT_CORRIDOR_WP_SEQ
    ]

    if remaining_return_waypoints:
        set_mission_state("RETURN_PATH_AFTER_WP4")

        for waypoint in remaining_return_waypoints:
            status, _ = goto_waypoint(
                controller,
                cap,
                waypoint_with_altitude(waypoint, RETURN_ALT_M),
                f"Return WP{waypoint['seq']} at 10m",
                watch_for_qr=False,
            )
            if status in ("quit", "camera_failed"):
                return status

    print("Post-target mission sequence complete")
    return "complete"


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
        controller.set_cruise_speed(CRUISE_SPEED_MPS)
        controller.arm()
        controller.takeoff(TAKEOFF_ALT_M)
        controller.set_cruise_speed(CRUISE_SPEED_MPS)

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
            post_status = run_post_target_sequence(controller, cap, mission_items)

            if post_status != "complete":
                print(f"Post-target mission ended with status: {post_status}")
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
