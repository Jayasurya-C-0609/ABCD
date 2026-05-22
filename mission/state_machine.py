import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config import (
    AUTO_PATH_END_WP_SEQ,
    AUTO_PATH_START_WP_SEQ,
    CAMERA_BODY_MAPPING,
    CAMERA_POSITION_BODY,
    BYPASS_FORWARD_SPEED,
    BYPASS_SIDE_SPEED,
    BYPASS_TURN_HOVER,
    ALLOW_CORRIDOR_WITHOUT_BANNER,
    ALTITUDE_TOLERANCE_M,
    CAMERA_SOURCE,
    CONNECTION_STRING,
    CORRIDOR_ALTITUDE_M,
    CORRIDOR_ENTRANCE_WP_SEQ,
    CORRIDOR_SPEED,
    CRUISE_SPEED_MPS,
    DETECTION_INTERVAL_S,
    DROIDCAM_URL,
    ENABLE_SOLVEPNP_ALIGNMENT,
    EXIT_CORRIDOR_END_WP_SEQ,
    EXIT_CORRIDOR_WP_SEQ,
    GREEN_ALIGN_SPEED,
    GREEN_ALIGN_TIMEOUT,
    GREEN_ALIGN_TOLERANCE_X,
    GREEN_ALIGN_TOLERANCE_Y,
    GREEN_DETECT_FPS,
    GREEN_SEARCH_TIMEOUT,
    GREEN_YAW_SPEED,
    LOOP_SLEEP_S,
    PAYLOAD_DESCENT_ALT_M,
    PAYLOAD_HOVER_TIME_S,
    PAYLOAD_POSITION_BODY,
    PIXEL_ALIGN_SPEED,
    PIXEL_ALIGN_TIMEOUT,
    PIXEL_ALIGN_TOL_X,
    PIXEL_ALIGN_TOL_Y,
    POSITION_MOVING_THRESHOLD_M,
    POSITION_RETRY_TIMEOUT_S,
    POSITION_STUCK_TIMEOUT_S,
    QR_DETECT_FPS,
    QR_MODEL_PATH,
    QR_REACQUIRE_CONF,
    QR_CAMERA_MATRIX,
    QR_DIST_COEFFS,
    R_BODY_CAMERA,
    REDZONE_POLYGON_LATLON,
    REDZONE_BYPASS_MARGIN_M,
    REDZONE_SIDE_STEP_SPEED_MPS,
    REDZONE_SIDE_STEP_TIME_S,
    REDZONE_VISUAL_CENTER_MARGIN_RATIO,
    REDZONE_VISUAL_MIN_AREA_RATIO,
    RETURN_ALT_M,
    START_QR_ALT_M,
    START_QR_MAX_RETRIES,
    START_QR_SCAN_TIMEOUT,
    START_QR_WP_SEQ,
    SOLVEPNP_ALIGN_TOLERANCE_M,
    SOLVEPNP_KP,
    SOLVEPNP_LOST_TIMEOUT,
    SOLVEPNP_MAX_SPEED,
    SOLVEPNP_REQUIRED_SUCCESS_FRAMES,
    SURFACE_ALTITUDE_M,
    SURFACE_SPEED_MPS,
    SURFACE_ENTRANCE_WP_SEQ,
    TAKEOFF_ALT_M,
    TARGET_FOUND_HOVER_SECONDS,
    WAYPOINT_ACCEPT_RADIUS,
    WAYPOINT_COMMAND_INTERVAL_S,
    WAYPOINT_DEBUG_INTERVAL_S,
    WAYPOINT_TIMEOUT,
    WP2_SCAN_NUDGE_SPEED_MPS,
    WP2_SCAN_NUDGE_TIME_S,
    WRONG_QR_SUPPRESS_S,
)
from mavlink.connection import MavlinkConnection
from mavlink.guided_control import GuidedController
from navigation.qr_alignment import (
    bbox_center,
    bbox_near_frame_edge,
    recenter_locked_bbox,
    select_best_qr_detection,
    select_locked_qr_detection,
    select_reacquired_qr_detection,
)
from navigation.corridor_nav import waypoint_with_altitude
from navigation.solvepnp_alignment import locked_bbox_qr_corners, qr_object_points
from navigation.waypoint_nav import (
    distance_from_position_to_waypoint,
    ground_distance_between_latlon,
    latlon_to_xy,
    position_changed_enough as position_changed_by_threshold,
    xy_to_latlon,
)
from vision.camera import open_camera
from vision.green_banner import detect_green_banner, load_green_banner_detector
from vision.red_zone import detect_redzone_yolo, load_redzone_detector


WAYPOINT_TIMEOUT_S = WAYPOINT_TIMEOUT
WINDOW_NAME = "SkyScan QR Mission"
_detect_qrs = None
_decode_qr_crop = None
_camera_position_body = np.asarray(CAMERA_POSITION_BODY, dtype=np.float64)
_payload_position_body = np.asarray(PAYLOAD_POSITION_BODY, dtype=np.float64)
_camera_body_rotations = {
    "DOWNWARD_NORMAL": [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "DOWNWARD_SWAP": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "DOWNWARD_INVERT_X": [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "DOWNWARD_INVERT_Y": [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "DOWNWARD_SWAP_AND_INVERT": [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
}
_r_body_camera = np.asarray(
    _camera_body_rotations.get(CAMERA_BODY_MAPPING, R_BODY_CAMERA),
    dtype=np.float64,
)
_qr_camera_matrix = np.asarray(QR_CAMERA_MATRIX, dtype=np.float64)
_qr_dist_coeffs = np.asarray(QR_DIST_COEFFS, dtype=np.float64)
_last_frame_time = None
_display_fps = 0.0
_home_lat = None
_home_lon = None
_last_valid_position = None
_redzone_geometry_logged = False
_status_log_times = {}
_target_qr_lock = {
    "target_locked": False,
    "locked_target_text": None,
    "locked_target_bbox": None,
}


def set_mission_state(state_name):
    print(f"\n=== STATE: {state_name} ===")


def log_status(key, message, interval_s=1.0):
    """Print repeated loop status at a readable rate."""
    now = time.time()
    if now - _status_log_times.get(key, 0.0) >= interval_s:
        print(message)
        _status_log_times[key] = now


def target_text_matches(decoded_text, target_value):
    if decoded_text is None or target_value is None:
        return False

    return decoded_text.strip().casefold() == target_value.strip().casefold()


def lock_target_qr(decoded_text, bbox):
    _target_qr_lock["target_locked"] = True
    _target_qr_lock["locked_target_text"] = decoded_text
    _target_qr_lock["locked_target_bbox"] = bbox
    print("target_locked True")
    print(f"locked_target_text={decoded_text}")
    print(f"locked_target_bbox={bbox}")


def initialize_home_origin(controller, mission_items):
    global _home_lat, _home_lon

    home_item = mission_items.get(0)
    if home_item and abs(home_item.get("lat", 0.0)) > 0.000001:
        _home_lat = home_item["lat"]
        _home_lon = home_item["lon"]
    else:
        position = controller.get_global_position(timeout_s=2.0)
        if position is not None:
            _home_lat = position["lat"]
            _home_lon = position["lon"]
        else:
            first_item = next(iter(mission_items.values()))
            _home_lat = first_item["lat"]
            _home_lon = first_item["lon"]

    print(f"home_lat={_home_lat:.7f}, home_lon={_home_lon:.7f}")


def cache_position(position):
    global _last_valid_position

    if position is not None:
        _last_valid_position = dict(position)

    return position


def load_yolo_detector():
    global _detect_qrs

    if _detect_qrs is None:
        if not Path(QR_MODEL_PATH).exists():
            raise FileNotFoundError(f"Required QR model missing: {QR_MODEL_PATH}")

        print(f"QR model loaded path: {QR_MODEL_PATH}")
        print("Loading YOLO QR detector...")
        from vision.qr_detect import detect_qrs as loaded_detect_qrs

        _detect_qrs = loaded_detect_qrs
        print("YOLO QR detector ready")


def detect_qrs(frame, conf=0.5):
    load_yolo_detector()
    return _detect_qrs(frame, conf=conf)


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
    load_redzone_detector()
    load_green_banner_detector()


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def read_camera_frame(cap):
    ret, frame = cap.read()

    if not ret or frame is None:
        log_status("camera_frame_missing", "No camera frame")
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


def draw_green_banner(frame, detection):
    x1, y1, x2, y2 = detection["bbox"]
    cx, cy = detection["center"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    cv2.putText(
        frame,
        f"GREEN {detection['confidence']:.2f}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 0),
        2,
    )


def align_to_green_banner(controller, cap, corridor_name, current_waypoint=None):
    detection_interval_s = 1.0 / GREEN_DETECT_FPS
    search_started_at = time.time()
    alignment_started_at = None
    last_detection_at = 0.0
    last_warning_at = 0.0
    detection = None
    last_banner_detection = None
    corridor_title = corridor_name.capitalize()

    print(f"Aligning with {corridor_name} corridor")
    while True:
        now = time.time()
        if last_banner_detection is None and now - search_started_at >= GREEN_SEARCH_TIMEOUT:
            controller.stop()
            print("green banner detected False")
            print("WARNING: green banner not detected before search timeout")
            if ALLOW_CORRIDOR_WITHOUT_BANNER:
                print("ALLOW_CORRIDOR_WITHOUT_BANNER=True, moving to corridor")
                return "search_timeout_allowed"
            return "search_timeout_blocked"

        if (
            alignment_started_at is not None
            and now - alignment_started_at >= GREEN_ALIGN_TIMEOUT
        ):
            controller.stop()
            print("Green banner detected but center alignment failed, continuing to corridor")
            print("alignment timeout fallback")
            print("alignment success False")
            return "alignment_timeout_fallback"

        frame = read_camera_frame(cap)
        if frame is None:
            if time.time() - last_warning_at >= 1.0:
                log_status("green_frame_missing", "Green banner frame unavailable, holding/searching")
                last_warning_at = time.time()
            controller.send_body_velocity_yaw_rate(0.0, 0.0, 0.0, GREEN_YAW_SPEED * 0.5)
            time.sleep(LOOP_SLEEP_S)
            continue

        now = time.time()
        if now - last_detection_at >= detection_interval_s:
            detection = detect_green_banner(frame)
            last_detection_at = now
            if detection is not None:
                last_banner_detection = detection
                if alignment_started_at is None:
                    alignment_started_at = now
                    print(f"{corridor_title} green banner detected")
                    print("green banner detected True")
                    print(f"confidence: {detection['confidence']:.3f}")
                    print(f"bbox: {detection['bbox']}")

        current_alt = None
        current_position = get_current_position(controller)
        if current_position is not None:
            current_alt = current_position.get("relative_alt")

        active_detection = detection or last_banner_detection
        if active_detection is None:
            if now - last_warning_at >= 1.0:
                log_status(
                    "green_search",
                    "Green banner not detected, holding/searching "
                    f"current waypoint={current_waypoint} altitude={current_alt}",
                )
                last_warning_at = now
            controller.send_body_velocity_yaw_rate(0.0, 0.0, 0.0, GREEN_YAW_SPEED)
            key = show_frame(frame, f"{corridor_title} green banner search")
            if key == ord("q"):
                controller.stop()
                print("green banner alignment failure: operator quit")
                return "quit"
            time.sleep(LOOP_SLEEP_S)
            continue

        draw_green_banner(frame, active_detection)
        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)
        banner_center = active_detection["center"]
        error_x = banner_center[0] - frame_center[0]
        error_y = banner_center[1] - frame_center[1]
        log_status(
            "green_align",
            "Green banner align "
            f"frame_center={frame_center} banner_center={banner_center} "
            f"error_x={error_x} error_y={error_y}",
        )

        if (
            abs(error_x) < GREEN_ALIGN_TOLERANCE_X
            and abs(error_y) < GREEN_ALIGN_TOLERANCE_Y
        ):
            controller.stop()
            print("commanded vx=0.000, vy=0.000, yaw=0.000")
            print("alignment success True")
            print(f"{corridor_title} corridor alignment complete")
            return "aligned"

        vx = clamp(
            -GREEN_ALIGN_SPEED * (error_y / max(frame_height * 0.5, 1.0)),
            -GREEN_ALIGN_SPEED,
            GREEN_ALIGN_SPEED,
        )
        vy = clamp(
            GREEN_ALIGN_SPEED * (error_x / max(frame_width * 0.5, 1.0)),
            -GREEN_ALIGN_SPEED,
            GREEN_ALIGN_SPEED,
        )
        yaw_rate = clamp(
            GREEN_YAW_SPEED * (error_x / max(frame_width * 0.5, 1.0)),
            -GREEN_YAW_SPEED,
            GREEN_YAW_SPEED,
        )
        log_status(
            "green_align_command",
            f"Green banner command vx={vx:.3f} vy={vy:.3f} yaw={yaw_rate:.3f} "
            f"current waypoint={current_waypoint} altitude={current_alt}",
        )
        controller.send_body_velocity_yaw_rate(vx, vy, 0.0, yaw_rate)

        key = show_frame(frame, f"{corridor_title} green banner align")
        if key == ord("q"):
            controller.stop()
            print("green banner alignment failure: operator quit")
            return "quit"

        time.sleep(LOOP_SLEEP_S)


def green_banner_result_allows_corridor(result):
    return result in (
        "aligned",
        "alignment_timeout_fallback",
        "search_timeout_allowed",
    )


def search_exit_green_banner_until_seen(controller, cap):
    detection_interval_s = 1.0 / GREEN_DETECT_FPS
    last_detection_at = 0.0
    last_status_at = 0.0
    detection = None

    print("Searching exit green banner")
    while True:
        frame = read_camera_frame(cap)
        if frame is None:
            print("Exit green banner search frame unavailable")
            time.sleep(LOOP_SLEEP_S)
            continue

        now = time.time()
        if now - last_detection_at >= detection_interval_s:
            detection = detect_green_banner(frame)
            last_detection_at = now

        if detection is not None:
            draw_green_banner(frame, detection)
            show_frame(frame, "Exit green banner detected")
            controller.stop()
            print("Exit green banner detected")
            print("Exit green banner detected, proceeding to corridor")
            print(f"exit banner confidence: {detection['confidence']:.3f}")
            print(f"exit banner bbox: {detection['bbox']}")
            return True

        if now - last_status_at >= WAYPOINT_DEBUG_INTERVAL_S:
            current_position = get_current_position(controller)
            current_alt = current_position["alt"] if current_position is not None else None
            print("current state: SEARCH_EXIT_GREEN_BANNER")
            print("exit_banner_seen=False")
            print(f"current altitude: {current_alt}")
            last_status_at = now

        controller.send_body_velocity_yaw_rate(0.0, 0.0, 0.0, GREEN_YAW_SPEED)
        key = show_frame(frame, "Search exit green banner WP27")
        if key == ord("q"):
            controller.stop()
            return False

        time.sleep(LOOP_SLEEP_S)


def move_exit_corridor_wp27_to_wp28(controller, cap, wp28, exit_banner_seen):
    exit_wp28_command_sent = False
    wp28_target = waypoint_with_altitude(wp28, CORRIDOR_ALTITUDE_M)
    last_status_at = 0.0
    last_frame_failed_at = 0.0

    set_mission_state("MOVE_EXIT_CORRIDOR_WP27_TO_WP28")
    controller.set_mode("GUIDED")
    controller.set_cruise_speed(CORRIDOR_SPEED)
    print("Moving WP27 to WP28")

    while True:
        if not exit_wp28_command_sent:
            controller.goto_global_location(
                wp28_target["lat"],
                wp28_target["lon"],
                CORRIDOR_ALTITUDE_M,
            )
            exit_wp28_command_sent = True
            print("WP28 goto command sent")

        current_position = get_current_position(controller)
        distance_to_wp28 = None
        current_alt = None
        if current_position is not None:
            distance_to_wp28 = distance_from_position_to_waypoint(
                current_position,
                wp28_target,
            )
            current_alt = current_position["alt"]

        now = time.time()
        if now - last_status_at >= WAYPOINT_DEBUG_INTERVAL_S:
            distance_text = (
                f"{distance_to_wp28:.2f}m"
                if distance_to_wp28 is not None
                else "None"
            )
            print("current state: MOVE_EXIT_CORRIDOR_WP27_TO_WP28")
            print(f"exit_banner_seen={exit_banner_seen}")
            print(f"exit_wp28_command_sent={exit_wp28_command_sent}")
            print("target waypoint = 28")
            print(f"distance_to_wp28={distance_text}")
            print(f"current altitude: {current_alt}")
            print(f"commanded altitude = {CORRIDOR_ALTITUDE_M:.1f}")
            last_status_at = now

        if distance_to_wp28 is not None and distance_to_wp28 <= WAYPOINT_ACCEPT_RADIUS:
            controller.stop()
            print("Reached WP28, switching RTL")
            print("Reached WP28 exit corridor end")
            return "reached"

        frame = read_camera_frame(cap)
        if frame is not None:
            key = show_frame(frame, "Moving WP27 to WP28")
            if key == ord("q"):
                controller.stop()
                return "quit"
        elif now - last_frame_failed_at >= WAYPOINT_DEBUG_INTERVAL_S:
            print("WP27 to WP28 camera frame unavailable; continuing waypoint movement")
            last_frame_failed_at = now

        time.sleep(LOOP_SLEEP_S)


def draw_detection(frame, detection, label="QR"):
    detection_type = detection.get("type", "qr")
    x1, y1, x2, y2 = detection["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{label} {detection_type}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def waypoint_to_xy(waypoint):
    return latlon_to_xy(waypoint["lat"], waypoint["lon"], _home_lat, _home_lon)


def position_changed_enough(previous_position, current_position):
    return position_changed_by_threshold(
        previous_position,
        current_position,
        POSITION_MOVING_THRESHOLD_M,
    )


def get_controller_mode(controller):
    if hasattr(controller, "get_mode"):
        return controller.get_mode()
    return getattr(controller.master, "flightmode", "UNKNOWN")


def get_controller_armed_status(controller):
    if hasattr(controller, "is_armed"):
        return controller.is_armed()
    try:
        return bool(controller.master.motors_armed())
    except Exception:
        return False


def print_waypoint_debug(controller, label, waypoint, position, distance):
    current_alt = position["alt"] if position is not None else None
    distance_text = f"{distance:.2f}m" if distance is not None else "None"
    altitude_text = f"{current_alt:.2f}m" if current_alt is not None else "None"
    print(
        f"STATUS {label}: "
        f"target_wp={waypoint.get('seq', 'runtime')} "
        f"distance={distance_text} "
        f"alt={altitude_text} "
        f"target_alt={waypoint['alt']:.1f}m "
        f"mode={get_controller_mode(controller)} "
        f"armed={get_controller_armed_status(controller)} "
        f"target_lat={waypoint['lat']:.7f}, "
        f"target_lon={waypoint['lon']:.7f}"
    )


def redzone_polygon_to_xy():
    return [
        latlon_to_xy(lat, lon, _home_lat, _home_lon)
        for lat, lon in REDZONE_POLYGON_LATLON
    ]


def make_redzone_geometry(margin=REDZONE_BYPASS_MARGIN_M):
    global _redzone_geometry_logged

    polygon_xy = redzone_polygon_to_xy()
    x_values = [point[0] for point in polygon_xy]
    y_values = [point[1] for point in polygon_xy]
    actual_box = {
        "xmin": min(x_values),
        "xmax": max(x_values),
        "ymin": min(y_values),
        "ymax": max(y_values),
    }
    expanded_box = {
        "xmin": actual_box["xmin"] - margin,
        "xmax": actual_box["xmax"] + margin,
        "ymin": actual_box["ymin"] - margin,
        "ymax": actual_box["ymax"] + margin,
    }
    polygon_edges = [
        (polygon_xy[index], polygon_xy[(index + 1) % len(polygon_xy)])
        for index in range(len(polygon_xy))
    ]

    if not _redzone_geometry_logged:
        print(f"redzone_polygon_xy: {polygon_xy}")
        print(f"redzone_bounding_box_xy: {actual_box}")
        print(f"redzone_expanded_bounding_box_xy: {expanded_box}")
        print(f"redzone_bounding_box_width: {expanded_box['xmax'] - expanded_box['xmin']:.2f} m")
        print(f"redzone_bounding_box_height: {expanded_box['ymax'] - expanded_box['ymin']:.2f} m")
        print(f"redzone_actual_polygon_width: {actual_box['xmax'] - actual_box['xmin']:.2f} m")
        print(f"redzone_actual_polygon_height: {actual_box['ymax'] - actual_box['ymin']:.2f} m")
        _redzone_geometry_logged = True

    return {
        "polygon_xy": polygon_xy,
        "polygon_edges": polygon_edges,
        "actual_box": actual_box,
        "expanded_box": expanded_box,
    }


def make_redzone_box(margin=REDZONE_BYPASS_MARGIN_M):
    return make_redzone_geometry(margin)["expanded_box"]


def point_inside_box(point, box):
    x, y = point
    return box["xmin"] <= x <= box["xmax"] and box["ymin"] <= y <= box["ymax"]


def point_inside_polygon(point, polygon):
    x, y = point
    inside = False
    count = len(polygon)

    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]

        cross = ((y - y1) * (x2 - x1)) - ((x - x1) * (y2 - y1))
        if (
            abs(cross) < 1e-9
            and min(x1, x2) <= x <= max(x1, x2)
            and min(y1, y2) <= y <= max(y1, y2)
        ):
            return True

        crosses_y = (y1 > y) != (y2 > y)
        if crosses_y:
            denominator = y2 - y1
            if abs(denominator) < 1e-12:
                continue
            intersect_x = ((x2 - x1) * (y - y1) / denominator) + x1
            if x < intersect_x:
                inside = not inside

    return inside


def orientation(a, b, c):
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def point_on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and point_on_segment(p1, q1, p2):
        return True
    if o2 == 0 and point_on_segment(p1, q2, p2):
        return True
    if o3 == 0 and point_on_segment(q1, p1, q2):
        return True
    if o4 == 0 and point_on_segment(q1, p2, q2):
        return True

    return False


def segment_crosses_box(start_xy, end_xy, box):
    if point_inside_box(start_xy, box) or point_inside_box(end_xy, box):
        return True

    x1, y1 = start_xy
    x2, y2 = end_xy
    segment_xmin = min(x1, x2)
    segment_xmax = max(x1, x2)
    segment_ymin = min(y1, y2)
    segment_ymax = max(y1, y2)

    if (
        segment_xmax < box["xmin"]
        or segment_xmin > box["xmax"]
        or segment_ymax < box["ymin"]
        or segment_ymin > box["ymax"]
    ):
        return False

    if abs(x1 - x2) < 1e-6:
        vertical_overlap = segment_ymax >= box["ymin"] and segment_ymin <= box["ymax"]
        if box["xmin"] <= x1 <= box["xmax"] and vertical_overlap:
            return True

    box_edges = [
        ((box["xmin"], box["ymin"]), (box["xmax"], box["ymin"])),
        ((box["xmax"], box["ymin"]), (box["xmax"], box["ymax"])),
        ((box["xmax"], box["ymax"]), (box["xmin"], box["ymax"])),
        ((box["xmin"], box["ymax"]), (box["xmin"], box["ymin"])),
    ]

    for edge_start, edge_end in box_edges:
        if segments_intersect(start_xy, end_xy, edge_start, edge_end):
            return True

    return False


def segment_intersects_polygon(start_xy, end_xy, polygon_edges):
    for edge_start, edge_end in polygon_edges:
        if segments_intersect(start_xy, end_xy, edge_start, edge_end):
            return True

    return False


def redzone_segment_decision(start_xy, end_xy, target_xy=None, block_current_near=False):
    geometry = make_redzone_geometry(REDZONE_BYPASS_MARGIN_M)
    target_xy = target_xy if target_xy is not None else end_xy
    crosses_box = segment_crosses_box(start_xy, end_xy, geometry["expanded_box"])
    intersects_polygon = False
    if crosses_box:
        intersects_polygon = segment_intersects_polygon(
            start_xy,
            end_xy,
            geometry["polygon_edges"],
        )
    waypoint_inside_polygon = point_inside_polygon(target_xy, geometry["polygon_xy"])
    current_inside_polygon = point_inside_polygon(start_xy, geometry["polygon_xy"])
    current_near_polygon = point_inside_box(start_xy, geometry["expanded_box"])
    final_redzone_blocked = (
        intersects_polygon
        or waypoint_inside_polygon
        or current_inside_polygon
        or (block_current_near and current_near_polygon)
    )

    return {
        "geometry": geometry,
        "crosses_box": crosses_box,
        "intersects_polygon": intersects_polygon,
        "waypoint_inside_polygon": waypoint_inside_polygon,
        "current_inside_polygon": current_inside_polygon,
        "current_near_polygon": current_near_polygon,
        "final_redzone_blocked": final_redzone_blocked,
    }


def print_redzone_decision(decision):
    print(
        "REDZONE "
        f"crosses_box={decision['crosses_box']} "
        f"intersects_polygon={decision['intersects_polygon']} "
        f"waypoint_inside_polygon={decision['waypoint_inside_polygon']} "
        f"current_inside_polygon={decision['current_inside_polygon']} "
        f"current_near_polygon={decision['current_near_polygon']} "
        f"blocked={decision['final_redzone_blocked']}"
    )


def route_length(points):
    total = 0.0
    for first, second in zip(points, points[1:]):
        total += ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5
    return total


def route_crosses_box(points, box):
    for point in points[1:-1]:
        if point_inside_box(point, box):
            return True

    for first, second in zip(points, points[1:]):
        if segment_crosses_box(first, second, box):
            return True

    return False


def generate_box_bypass(start_xy, end_xy, box):
    clearance_m = 0.25
    dx = abs(end_xy[0] - start_xy[0])
    dy = abs(end_xy[1] - start_xy[1])

    if dy >= dx:
        original_x = start_xy[0]
        direction = "DOWN" if end_xy[1] < start_xy[1] else "UP"

        left_x = box["xmin"] - clearance_m
        right_x = box["xmax"] + clearance_m
        left_distance = abs(original_x - left_x)
        right_distance = abs(right_x - original_x)

        if left_distance <= right_distance:
            chosen_side = "LEFT"
            side_x = left_x
        else:
            chosen_side = "RIGHT"
            side_x = right_x

        if direction == "DOWN":
            before_y = box["ymax"] + clearance_m
            after_y = box["ymin"] - clearance_m
        else:
            before_y = box["ymin"] - clearance_m
            after_y = box["ymax"] + clearance_m

        before_redzone_point = (original_x, before_y)
        side_point_1 = (side_x, before_y)
        side_point_2 = (side_x, after_y)
        return_point = (original_x, after_y)
    else:
        original_y = start_xy[1]
        direction = "LEFT" if end_xy[0] < start_xy[0] else "RIGHT"

        below_y = box["ymin"] - clearance_m
        above_y = box["ymax"] + clearance_m
        below_distance = abs(original_y - below_y)
        above_distance = abs(above_y - original_y)

        if below_distance <= above_distance:
            chosen_side = "LOWER"
            side_y = below_y
        else:
            chosen_side = "UPPER"
            side_y = above_y

        if direction == "RIGHT":
            before_x = box["xmin"] - clearance_m
            after_x = box["xmax"] + clearance_m
        else:
            before_x = box["xmax"] + clearance_m
            after_x = box["xmin"] - clearance_m

        before_redzone_point = (before_x, original_y)
        side_point_1 = (before_x, side_y)
        side_point_2 = (after_x, side_y)
        return_point = (after_x, original_y)

    bypass_points = [
        before_redzone_point,
        side_point_1,
        side_point_2,
        return_point,
    ]

    full_route = [start_xy] + bypass_points + [end_xy]
    safe_route = bypass_points[:]
    if route_crosses_box(safe_route, box):
        print("Bypass generation failed: close local bypass still crosses expanded red-zone box")
        return None, None

    print(
        f"BYPASS direction={direction} side={chosen_side} "
        f"distance={route_length(full_route):.2f}m points={bypass_points}"
    )
    return chosen_side, bypass_points


def bypass_points_are_safe(bypass_points_xy, box):
    for point in bypass_points_xy:
        if point_inside_box(point, box):
            print(f"Generated bypass point is inside expanded red-zone box, rejecting: {point}")
            return False
    return True


def bypass_xy_to_latlon_waypoints(bypass_points_xy, altitude):
    bypass_points_latlon = []
    for x, y in bypass_points_xy:
        lat, lon = xy_to_latlon(x, y, _home_lat, _home_lon)
        bypass_points_latlon.append({"lat": lat, "lon": lon, "alt": altitude})

    print(f"generated bypass points in lat/lon: {bypass_points_latlon}")
    return bypass_points_latlon


def send_world_velocity(controller, vx_east, vy_north, vz=0.0):
    controller.master.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        controller.master.target_system,
        controller.master.target_component,
        1,
        0b0000111111000111,
        0, 0, 0,
        vy_north, vx_east, vz,
        0, 0, 0,
        0, 0,
    )


def move_world_delta_slow(controller, cap, dx, dy, label):
    distance = math.sqrt((dx ** 2) + (dy ** 2))
    if distance < 0.05:
        return "complete"

    if abs(dx) >= abs(dy):
        speed = BYPASS_FORWARD_SPEED
    else:
        speed = BYPASS_SIDE_SPEED

    duration = distance / max(speed, 0.05)
    vx_east = (dx / distance) * speed
    vy_north = (dy / distance) * speed
    print(f"{label}: distance={distance:.2f}m speed={speed:.2f}m/s")

    end_time = time.time() + duration
    while time.time() < end_time:
        send_world_velocity(controller, vx_east, vy_north, 0.0)
        frame = read_camera_frame(cap)
        if frame is not None:
            key = show_frame(frame, label)
            if key == ord("q"):
                controller.stop()
                return "quit"
        time.sleep(0.1)

    controller.stop()
    time.sleep(BYPASS_TURN_HOVER)
    return "complete"


def follow_bypass_velocity_path(controller, cap, bypass_points_xy):
    current_position = get_current_position(controller)
    if current_position is None:
        print("Bypass failed: current position unavailable")
        controller.stop()
        return "position_failed"

    current_xy = latlon_to_xy(
        current_position["lat"],
        current_position["lon"],
        _home_lat,
        _home_lon,
    )

    print("bypass started")
    for index, target_xy in enumerate(bypass_points_xy, start=1):
        dx = target_xy[0] - current_xy[0]
        dy = target_xy[1] - current_xy[1]
        status = move_world_delta_slow(
            controller,
            cap,
            dx,
            dy,
            f"Bypass segment {index}",
        )
        if status != "complete":
            return status
        current_xy = target_xy

    print("returning to original yellow path")
    print("bypass completed")
    return "complete"


def follow_bypass_waypoints(controller, cap, bypass_points_latlon, altitude):
    for index, bypass_point in enumerate(bypass_points_latlon, start=1):
        waypoint = waypoint_with_altitude(bypass_point, altitude)
        status, _ = goto_waypoint_until_reached(
            controller,
            cap,
            waypoint,
            f"Runtime red-zone bypass WP{index}",
            watch_for_qr=False,
            avoid_redzone_visual=True,
            avoid_redzone_box=False,
        )
        if status in ("quit", "camera_failed", "redzone_blocked"):
            return status

        print(f"bypass waypoint reached: {index}")

    return "complete"


def redzone_visual_is_close(frame, bbox):
    if bbox is None:
        return False

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, w * h)
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2
    near_center = (
        abs(bbox_cx - (w / 2)) < w * REDZONE_VISUAL_CENTER_MARGIN_RATIO
        and abs(bbox_cy - (h / 2)) < h * REDZONE_VISUAL_CENTER_MARGIN_RATIO
    )
    return area_ratio >= REDZONE_VISUAL_MIN_AREA_RATIO and near_center


def perform_redzone_visual_backup_avoid(controller, cap):
    set_mission_state("REDZONE_VISUAL_BACKUP_AVOID")
    print("YOLO visual backup avoidance triggered")
    controller.stop()
    time.sleep(0.5)

    end_time = time.time() + REDZONE_SIDE_STEP_TIME_S
    while time.time() < end_time:
        controller.send_body_velocity(0.0, -REDZONE_SIDE_STEP_SPEED_MPS, 0.0)
        frame = read_camera_frame(cap)
        if frame is not None:
            show_frame(frame, "RED visual backup side-step")
        time.sleep(0.1)

    controller.stop()


def pixel_align_and_decode_qr(controller, cap, reason, target_value):
    set_mission_state("QR_PIXEL_DETECT_AND_DECODE")
    print(f"{reason}: QR detected, starting rough pixel alignment")
    started_at = time.time()
    decode_attempt = 0
    pixel_success_logged = False

    while time.time() - started_at < PIXEL_ALIGN_TIMEOUT:
        frame = read_camera_frame(cap)
        if frame is None:
            return None

        detections = detect_qrs(frame)
        log_status("pixel_qr_count", f"QR detections count: {len(detections)}")
        if not detections:
            key = show_frame(frame, f"{reason}: pixel QR search")
            if key == ord("q"):
                return None
            time.sleep(LOOP_SLEEP_S)
            continue

        detection = select_best_qr_detection(detections)
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame.shape[:2]
        qr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        frame_center = (frame_width // 2, frame_height // 2)
        error_x = qr_center[0] - frame_center[0]
        error_y = qr_center[1] - frame_center[1]
        log_status(
            "pixel_qr_status",
            f"selected QR bbox={bbox} pixel error_x={error_x} error_y={error_y}",
        )
        draw_detection(frame, detection, "QR pixel")

        decode_attempt += 1
        decoded_text, processed = decode_qr_crop(frame, bbox)
        if processed is not None:
            cv2.imshow("Processed QR", processed)
        if decoded_text:
            print(f"decoded QR text: {decoded_text}")
            decoded_text = decoded_text.strip()
            target_matched = target_text_matches(decoded_text, target_value)
            print(f"target matched {target_matched}")
            controller.stop()
            show_frame(frame, f"{reason}: decoded {decoded_text}")
            if target_matched:
                lock_target_qr(decoded_text, bbox)
                set_mission_state("TARGET_QR_FOUND_HOVER")
                print("Correct target QR found")
                if ENABLE_SOLVEPNP_ALIGNMENT:
                    print("Target QR decoded, solvePnP alignment enabled")
                else:
                    print("Skipping solvePnP alignment")
            else:
                print("QR decoded but not target, resuming path")
                set_mission_state("NORMAL_LAWN_MOWER")
            return decoded_text

        if (
            abs(error_x) <= PIXEL_ALIGN_TOL_X
            and abs(error_y) <= PIXEL_ALIGN_TOL_Y
        ):
            if not pixel_success_logged:
                print("pixel alignment success")
                pixel_success_logged = True
            controller.stop()
        else:
            vx = clamp(
                -PIXEL_ALIGN_SPEED * (error_y / max(frame_height * 0.5, 1.0)),
                -PIXEL_ALIGN_SPEED,
                PIXEL_ALIGN_SPEED,
            )
            vy = clamp(
                PIXEL_ALIGN_SPEED * (error_x / max(frame_width * 0.5, 1.0)),
                -PIXEL_ALIGN_SPEED,
                PIXEL_ALIGN_SPEED,
            )
            log_status("pixel_qr_command", f"pixel commanded vx={vx:.3f}, vy={vy:.3f}")
            controller.send_body_velocity(vx, vy, 0.0)

        key = show_frame(frame, f"{reason}: QR pixel decode {decode_attempt}")
        if key == ord("q"):
            controller.stop()
            return None
        time.sleep(LOOP_SLEEP_S)

    controller.stop()
    print("pixel alignment timeout")
    set_mission_state("NORMAL_LAWN_MOWER")
    return None


def solvepnp_target_alignment(controller, cap, target_value):
    set_mission_state("SOLVEPNP_TARGET_ALIGNMENT")
    if not _target_qr_lock["target_locked"]:
        print("target_locked False")
        print("solvePnP target alignment cannot start without a locked target")
        controller.stop()
        return "target_not_locked"

    object_points = qr_object_points()
    consecutive_success_count = 0
    solvepnp_success_frames = 0
    error_window = deque(maxlen=5)
    last_valid_error = None
    was_aligned_within_fallback = False
    alignment_started_at = time.time()
    last_seen_at = time.time()
    last_missing_corners_log_at = 0.0
    reacquire_attempt = 0
    solvepnp_state = "SOLVEPNP_TARGET_ALIGNMENT"
    movement_active = True
    print(f"solvePnP camera body mapping: {CAMERA_BODY_MAPPING}")
    print(f"target_locked {_target_qr_lock['target_locked']}")
    print(f"locked_target_text={_target_qr_lock['locked_target_text']}")

    def stop_motion_once():
        nonlocal movement_active
        if movement_active:
            controller.stop()
            movement_active = False

    def finish_lost_target(reason):
        stop_motion_once()
        if was_aligned_within_fallback:
            set_mission_state("TARGET_ALIGNMENT_COMPLETE")
            print(
                "solvePnP timeout after prior <=0.15m alignment; "
                "accepting last valid alignment"
            )
            print(f"last_valid_error={last_valid_error}")
            return "reached"
        print(f"solvePnP failed after target loss: {reason}")
        print("Holding after solvePnP_failed")
        print(f"last_valid_error={last_valid_error}")
        return "solvepnp_failed"

    while True:
        if time.time() - alignment_started_at >= SOLVEPNP_LOST_TIMEOUT:
            return finish_lost_target("alignment timeout")

        frame = read_camera_frame(cap)
        if frame is None:
            solvepnp_success_frames = 0
            stop_motion_once()
            lost_timer = time.time() - last_seen_at
            log_status("solvepnp_frame_loss", f"solvePnP frame missing lost_timer={lost_timer:.1f}")
            if lost_timer >= SOLVEPNP_LOST_TIMEOUT:
                return finish_lost_target("frame unavailable too long")
            time.sleep(LOOP_SLEEP_S)
            continue

        detections = detect_qrs(frame)
        selected_detection = select_locked_qr_detection(
            detections,
            _target_qr_lock["locked_target_bbox"],
        )
        selected_bbox = None if selected_detection is None else selected_detection["bbox"]
        log_status(
            "solvepnp_target",
            f"solvePnP target_locked={_target_qr_lock['target_locked']} "
            f"locked_target_text={_target_qr_lock['locked_target_text']} "
            f"selected bbox={selected_bbox}",
        )

        if selected_bbox is None:
            solvepnp_success_frames = 0
            stop_motion_once()
            if solvepnp_state != "SOLVEPNP_REACQUIRE_TARGET":
                solvepnp_state = "SOLVEPNP_REACQUIRE_TARGET"
                set_mission_state(solvepnp_state)
            reacquire_attempt += 1
            full_frame_detections = detect_qrs(frame, conf=QR_REACQUIRE_CONF)
            full_frame_qr_count = len(full_frame_detections)
            selected_detection = select_reacquired_qr_detection(
                full_frame_detections,
                _target_qr_lock["locked_target_bbox"],
            )
            selected_bbox = (
                None if selected_detection is None else selected_detection["bbox"]
            )
            lost_timer = time.time() - last_seen_at
            log_status(
                "solvepnp_reacquire",
                f"solvePnP reacquire attempt={reacquire_attempt} "
                f"full_frame_qr_count={full_frame_qr_count} selected bbox={selected_bbox} "
                f"lost_timer={lost_timer:.1f} last_valid_error={last_valid_error}",
            )
            if selected_bbox is not None:
                _target_qr_lock["locked_target_bbox"] = selected_bbox
                last_seen_at = time.time()
                solvepnp_state = "SOLVEPNP_TARGET_ALIGNMENT"
                set_mission_state(solvepnp_state)
                print("solvePnP locked QR reacquired")
                continue

            key = show_frame(frame, "solvePnP tracking locked QR")
            if key == ord("q"):
                return "quit"
            if lost_timer >= SOLVEPNP_LOST_TIMEOUT:
                return finish_lost_target("QR bbox unavailable")
            log_status("solvepnp_reacquire_motion", "solvePnP reacquire search motion vx=-0.080")
            controller.send_body_velocity(-0.08, 0.0, 0.0)
            movement_active = True
            time.sleep(0.2)
            stop_motion_once()
            time.sleep(0.1)
            continue

        if solvepnp_state == "SOLVEPNP_REACQUIRE_TARGET":
            solvepnp_state = "SOLVEPNP_TARGET_ALIGNMENT"
            set_mission_state(solvepnp_state)
            print("solvePnP locked QR reacquired")

        _target_qr_lock["locked_target_bbox"] = selected_bbox
        last_seen_at = time.time()
        bbox_center_xy = bbox_center(selected_bbox)
        near_edge = bbox_near_frame_edge(selected_bbox, frame)
        log_status(
            "solvepnp_bbox",
            f"solvePnP bbox center={bbox_center_xy} bbox near edge={near_edge}",
        )
        if near_edge:
            stop_motion_once()
            recenter_locked_bbox(controller, selected_bbox, frame)
            movement_active = False
            key = show_frame(frame, "solvePnP recenter QR edge bbox")
            if key == ord("q"):
                return "quit"
            continue

        corners, corners_found, using_bbox_fallback = locked_bbox_qr_corners(
            frame,
            selected_bbox,
        )
        if corners_found:
            log_status("solvepnp_corners", "corners found True")
        elif time.time() - last_missing_corners_log_at >= 1.0:
            print("corners found False")
            last_missing_corners_log_at = time.time()

        if corners is None:
            solvepnp_success_frames = 0
            stop_motion_once()
            lost_timer = time.time() - last_seen_at
            log_status(
                "solvepnp_corner_loss",
                f"solvePnP corners unavailable lost_timer={lost_timer:.1f} "
                f"last_valid_error={last_valid_error}",
            )
            key = show_frame(frame, "solvePnP waiting for QR corners")
            if key == ord("q"):
                return "quit"
            if lost_timer >= SOLVEPNP_LOST_TIMEOUT:
                return finish_lost_target("QR corners unavailable")
            time.sleep(0.1)
            continue

        if using_bbox_fallback:
            log_status("solvepnp_bbox_fallback", "Using YOLO bbox corners for solvePnP testing")
        last_seen_at = time.time()
        lost_timer = 0.0
        solvepnp_success, _, tvec = cv2.solvePnP(
            object_points,
            corners,
            _qr_camera_matrix,
            _qr_dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        log_status("solvepnp_pose", f"solvePnP success {solvepnp_success}")
        if not solvepnp_success:
            solvepnp_success_frames = 0
            stop_motion_once()
            log_status(
                "solvepnp_pose_loss",
                f"solvePnP pose unavailable lost_timer={lost_timer:.1f} "
                f"last_valid_error={last_valid_error}",
            )
            time.sleep(0.1)
            continue

        solvepnp_success_frames += 1
        qr_position_camera = tvec.reshape(3)
        qr_position_body = (_r_body_camera @ qr_position_camera) + _camera_position_body
        horizontal_error_x = qr_position_body[0] - _payload_position_body[0]
        horizontal_error_y = qr_position_body[1] - _payload_position_body[1]
        error_window.append((horizontal_error_x, horizontal_error_y))
        error_x_m = sum(error[0] for error in error_window) / len(error_window)
        error_y_m = sum(error[1] for error in error_window) / len(error_window)
        last_valid_error = (error_x_m, error_y_m)
        if abs(error_x_m) <= 0.15 and abs(error_y_m) <= 0.15:
            was_aligned_within_fallback = True
        vx = clamp(
            SOLVEPNP_KP * error_x_m,
            -SOLVEPNP_MAX_SPEED,
            SOLVEPNP_MAX_SPEED,
        )
        vy = clamp(
            SOLVEPNP_KP * error_y_m,
            -SOLVEPNP_MAX_SPEED,
            SOLVEPNP_MAX_SPEED,
        )
        current_alt, _ = get_current_altitude(controller)
        corner_center = corners.mean(axis=0)
        frame_height, frame_width = frame.shape[:2]
        visual_error_x = corner_center[0] - (frame_width / 2.0)
        visual_error_y = corner_center[1] - (frame_height / 2.0)
        visually_near_center = (
            abs(visual_error_x) <= PIXEL_ALIGN_TOL_X
            and abs(visual_error_y) <= PIXEL_ALIGN_TOL_Y
        )
        log_status(
            "solvepnp_motion",
            f"solvePnP error_x_m={error_x_m:.3f} error_y_m={error_y_m:.3f} "
            f"depth_z={qr_position_body[2]:.3f} speed vx={vx:.3f} vy={vy:.3f}",
        )
        if abs(error_x_m) > 1.0 and visually_near_center:
            print("solvePnP mapping likely wrong")

        if solvepnp_success_frames < SOLVEPNP_REQUIRED_SUCCESS_FRAMES:
            stop_motion_once()
            print(
                "solvePnP waiting for required success frames "
                f"({solvepnp_success_frames}/{SOLVEPNP_REQUIRED_SUCCESS_FRAMES})"
            )
            log_status(
                "solvepnp_stabilize",
                f"solvePnP stabilizing lost_timer={lost_timer:.1f} "
                f"success_frames={solvepnp_success_frames}",
            )
            key = show_frame(frame, "solvePnP stabilizing pose")
            if key == ord("q"):
                return "quit"
            time.sleep(0.1)
            continue

        altitude_ready = current_alt is not None and 4.8 <= current_alt <= 5.2
        errors_ready = (
            abs(error_x_m) < SOLVEPNP_ALIGN_TOLERANCE_M
            and abs(error_y_m) < SOLVEPNP_ALIGN_TOLERANCE_M
        )
        if errors_ready and altitude_ready:
            consecutive_success_count += 1
        else:
            consecutive_success_count = 0

        alignment_complete = consecutive_success_count >= SOLVEPNP_REQUIRED_SUCCESS_FRAMES
        log_status(
            "solvepnp_status",
            f"solvePnP lost_timer={lost_timer:.1f} "
            f"success_count={consecutive_success_count} "
            f"alignment_complete={alignment_complete}",
        )
        if alignment_complete:
            stop_motion_once()
            set_mission_state("TARGET_ALIGNMENT_COMPLETE")
            print("solvePnP target alignment complete")
            print("alignment complete")
            return "reached"

        for _ in range(2):
            controller.send_body_velocity(vx, vy, 0.0)
            movement_active = True
            time.sleep(0.1)
        stop_motion_once()
        time.sleep(0.1)

        key = show_frame(frame, "solvePnP target alignment")
        if key == ord("q"):
            controller.stop()
            return "quit"


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


def local_position_to_global(north, east, down):
    if _home_lat is None or _home_lon is None:
        return {
            "type": "local",
            "source": "LOCAL_POSITION_NED",
            "north": north,
            "east": east,
            "alt": -down,
        }

    lat, lon = xy_to_latlon(east, north, _home_lat, _home_lon)
    return {
        "type": "global",
        "source": "LOCAL_POSITION_NED converted using home",
        "lat": lat,
        "lon": lon,
        "alt": -down,
    }


def get_current_position(controller, print_source=False):
    global_position = controller.get_global_position(timeout_s=0.0)
    if global_position is not None:
        position = cache_position({
            "type": "global",
            "source": "GLOBAL_POSITION_INT",
            "lat": global_position["lat"],
            "lon": global_position["lon"],
            "alt": global_position["relative_alt"],
        })
        if print_source:
            print("Position source: GLOBAL_POSITION_INT")
        return position

    local_position = controller.get_local_position(timeout_s=0.0)
    if local_position is not None:
        north, east, down = local_position
        position = cache_position(local_position_to_global(north, east, down))
        position["source"] = "LOCAL_POSITION_NED"
        if print_source:
            print("Position source: LOCAL_POSITION_NED")
        return position

    if _last_valid_position is not None:
        if print_source:
            print("Position source: last_valid_position")
        return dict(_last_valid_position)

    return None


def get_safe_position(controller):
    end_time = time.time() + POSITION_RETRY_TIMEOUT_S

    while time.time() < end_time:
        position = get_current_position(controller, print_source=True)
        if position is not None:
            return position

        print("Current position unavailable, waiting for telemetry cache...")
        time.sleep(0.2)

    print("Current position unavailable after retry")
    return None


def command_position_hold(controller, position):
    position_type = position.get("type", "global" if "lat" in position and "lon" in position else "local")
    if position_type == "global":
        controller.goto_global_location(position["lat"], position["lon"], position["alt"])
    else:
        controller.goto_local_position(position["north"], position["east"], position["alt"])


def get_current_altitude(controller):
    position = controller.get_global_position(timeout_s=0.05)
    if position is not None:
        cache_position({
            "type": "global",
            "source": "GLOBAL_POSITION_INT",
            "lat": position["lat"],
            "lon": position["lon"],
            "alt": position["relative_alt"],
        })
        return position["relative_alt"], "GLOBAL_POSITION_INT"

    local_position = controller.get_local_position(timeout_s=0.05)
    if local_position is not None:
        return -local_position[2], "LOCAL_POSITION_NED"

    return None, None


def change_altitude_at_current_xy(controller, cap, target_alt_m, label):
    current_position = get_safe_position(controller)
    if current_position is None:
        return "position_failed", None

    target_position = dict(current_position)
    target_position["alt"] = target_alt_m

    started_at = time.time()
    last_status_time = 0
    while time.time() - started_at < WAYPOINT_TIMEOUT_S:
        command_position_hold(controller, target_position)
        current_alt, altitude_source = get_current_altitude(controller)
        if current_alt is None and _last_valid_position is not None:
            current_alt = _last_valid_position["alt"]
            altitude_source = "last_valid_position cache"

        if current_alt is not None:
            if time.time() - last_status_time >= WAYPOINT_DEBUG_INTERVAL_S:
                print(
                    f"STATUS {label}: altitude={current_alt:.2f}m "
                    f"target_alt={target_alt_m:.2f}m source={altitude_source}"
                )
                last_status_time = time.time()
            if abs(current_alt - target_alt_m) <= ALTITUDE_TOLERANCE_M:
                print(f"{label}: reached target altitude {target_alt_m:.2f}m")
                return "reached", target_position
        else:
            if time.time() - last_status_time >= WAYPOINT_DEBUG_INTERVAL_S:
                print(f"STATUS {label}: altitude unavailable")
                last_status_time = time.time()

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
    avoid_redzone_visual=False,
    avoid_redzone_box=True,
):
    print(
        f"Moving to {label}: "
        f"Lat={waypoint['lat']:.7f}, Lon={waypoint['lon']:.7f}, Alt={waypoint['alt']:.1f}"
    )

    started_at = time.time()
    last_detection_time = 0
    last_debug_time = 0
    last_goto_sent_time = 0
    last_position_change_time = time.time()
    last_motion_position = None
    stuck_warning_printed = False
    suppress_qr_until = 0
    detections = []
    arrival_tolerance_m = arrival_tolerance_m or WAYPOINT_ACCEPT_RADIUS

    if avoid_redzone_box and len(REDZONE_POLYGON_LATLON) >= 3:
        destination_xy = waypoint_to_xy(waypoint)
        geometry = make_redzone_geometry(REDZONE_BYPASS_MARGIN_M)
        waypoint_inside_polygon = point_inside_polygon(destination_xy, geometry["polygon_xy"])
        if waypoint_inside_polygon:
            print("REDZONE blocked=True reason=target_inside_polygon")
            print(f"Destination rejected inside actual red-zone polygon: {destination_xy}")
            controller.stop()
            return "redzone_blocked", None

    while time.time() - started_at < WAYPOINT_TIMEOUT_S:
        now = time.time()
        current_position = get_current_position(controller)

        if last_goto_sent_time == 0 or now - last_goto_sent_time >= WAYPOINT_COMMAND_INTERVAL_S:
            if avoid_redzone_box and current_position is not None and len(REDZONE_POLYGON_LATLON) >= 3:
                current_xy = latlon_to_xy(
                    current_position["lat"],
                    current_position["lon"],
                    _home_lat,
                    _home_lon,
                )
                target_xy = waypoint_to_xy(waypoint)
                decision = redzone_segment_decision(current_xy, target_xy)
                if decision["final_redzone_blocked"]:
                    print_redzone_decision(decision)
                    print(f"REDZONE blocked=True current_wp={label}")
                    controller.stop()
                    return "redzone_blocked", None

            controller.goto_global_location(
                waypoint["lat"],
                waypoint["lon"],
                waypoint["alt"],
            )
            last_goto_sent_time = now

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
            decoded_text = pixel_align_and_decode_qr(
                controller,
                cap,
                label,
                target_value,
            )

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

        if avoid_redzone_visual:
            red_detected, red_bbox, red_confidence = detect_redzone_yolo(frame, debug_frame=frame)
            if red_detected and redzone_visual_is_close(frame, red_bbox):
                perform_redzone_visual_backup_avoid(controller, cap)
                set_mission_state("RESUME_LAWN_MOWER")
                print(f"Resuming path after visual red-zone backup, confidence={red_confidence:.2f}")

        if current_position is None:
            distance = None
            started_at = time.time()
        else:
            distance = distance_from_position_to_waypoint(current_position, waypoint)

            if position_changed_enough(last_motion_position, current_position):
                last_motion_position = dict(current_position)
                last_position_change_time = time.time()
                stuck_warning_printed = False

        if time.time() - last_debug_time >= WAYPOINT_DEBUG_INTERVAL_S:
            print_waypoint_debug(controller, label, waypoint, current_position, distance)
            last_debug_time = time.time()

        if current_position is not None and time.time() - last_position_change_time >= POSITION_STUCK_TIMEOUT_S:
            if not stuck_warning_printed:
                print("Position not updating / drone not moving")
                stuck_warning_printed = True
            controller.set_mode("GUIDED")
            controller.set_cruise_speed(SURFACE_SPEED_MPS)
            waypoint["alt"] = SURFACE_ALTITUDE_M if waypoint.get("seq") in range(AUTO_PATH_START_WP_SEQ, AUTO_PATH_END_WP_SEQ + 1) else waypoint["alt"]
            controller.goto_global_location(
                waypoint["lat"],
                waypoint["lon"],
                waypoint["alt"],
            )
            last_goto_sent_time = time.time()
            last_position_change_time = time.time()

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


def goto_waypoint_until_reached(
    controller,
    cap,
    waypoint,
    label,
    watch_for_qr=False,
    target_value=None,
    arrival_tolerance_m=None,
    avoid_redzone_visual=False,
    avoid_redzone_box=True,
):
    while True:
        status, decoded_text = goto_waypoint(
            controller,
            cap,
            waypoint,
            label,
            watch_for_qr=watch_for_qr,
            target_value=target_value,
            arrival_tolerance_m=arrival_tolerance_m,
            avoid_redzone_visual=avoid_redzone_visual,
            avoid_redzone_box=avoid_redzone_box,
        )

        if status == "timeout":
            print(f"{label}: timeout occurred, resending same waypoint command")
            continue

        return status, decoded_text


def change_altitude_until_reached(controller, cap, target_alt_m, label):
    while True:
        status, target_position = change_altitude_at_current_xy(
            controller,
            cap,
            target_alt_m,
            label,
        )
        if status == "timeout":
            print(f"{label}: timeout occurred, retrying altitude change")
            continue
        return status, target_position


def wp2_scan_adjustment(controller, cap, retry_number):
    direction = 1.0 if retry_number % 2 == 1 else -1.0
    movement = "forward" if direction > 0 else "backward"
    print(f"WP2 QR retry {retry_number}: small {movement} camera adjustment")
    end_time = time.time() + WP2_SCAN_NUDGE_TIME_S

    while time.time() < end_time:
        controller.send_body_velocity(WP2_SCAN_NUDGE_SPEED_MPS * direction, 0.0, 0.0)
        frame = read_camera_frame(cap)
        if frame is not None:
            show_frame(frame, f"WP2 QR adjust {movement}")
        time.sleep(0.1)

    controller.stop()
    time.sleep(0.5)


def scan_start_qr_at_waypoint_2(controller, cap, start_qr_waypoint):
    start_qr_waypoint = waypoint_with_altitude(start_qr_waypoint, START_QR_ALT_M)
    status, _ = goto_waypoint_until_reached(
        controller,
        cap,
        start_qr_waypoint,
        "WP2 Start QR",
        arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
    )
    if status in ("quit", "camera_failed"):
        return None

    print("WP2 QR scan started")
    detection_interval_s = 1.0 / QR_DETECT_FPS

    for retry_number in range(1, START_QR_MAX_RETRIES + 1):
        print(f"WP2 QR scan retry number: {retry_number}/{START_QR_MAX_RETRIES}")
        scan_started_at = time.time()
        last_detection_time = 0
        detections = []

        while time.time() - scan_started_at < START_QR_SCAN_TIMEOUT:
            command_position_hold(controller, start_qr_waypoint)

            frame = read_camera_frame(cap)
            if frame is None:
                return None

            elapsed_s = time.time() - scan_started_at
            if time.time() - last_detection_time >= detection_interval_s:
                detections = detect_qrs(frame)
                last_detection_time = time.time()
                print(
                    f"WP2 QR scan elapsed time: {elapsed_s:.1f}s, "
                    f"QR detections count: {len(detections)}"
                )

            for detection in detections:
                detection_type = detection.get("type", "qr")
                print(f"WP2 raw detection object keys: {list(detection.keys())}")
                print(f"WP2 detection type: {detection_type}")
                print(f"WP2 QR bbox: {detection.get('bbox')}")
                draw_detection(frame, detection, "START QR")

            if detections:
                for detection in detections:
                    bbox = detection.get("bbox")
                    if bbox is None:
                        continue

                    decoded_text, processed = decode_qr_crop(frame, bbox)
                    if processed is not None:
                        cv2.imshow("Processed QR", processed)

                    print(f"WP2 QR decoded text: {decoded_text}")
                    if decoded_text:
                        target_location = decoded_text.strip()
                        print(f"WP2 target value saved: {target_location}")
                        return target_location

                print("WP2 QR detected but not decoded, continuing scan")
                detections = []

            key = show_frame(frame, "WP2: scan start QR")
            if key == ord("q"):
                controller.stop()
                return None

            time.sleep(LOOP_SLEEP_S)

        print(f"WP2 QR scan retry {retry_number} reached 60 seconds without decode")
        if retry_number < START_QR_MAX_RETRIES:
            wp2_scan_adjustment(controller, cap, retry_number)

    print("ERROR: WP2 compulsory start QR was not decoded after all retries")
    print("Holding position at WP2. Mission will not continue without target value.")
    while True:
        command_position_hold(controller, start_qr_waypoint)
        frame = read_camera_frame(cap)
        if frame is not None:
            key = show_frame(frame, "WP2 QR failed: holding")
            if key == ord("q"):
                controller.stop()
                return None
        time.sleep(0.2)

    return None


def build_surface_waypoints(mission_items):
    missing_surface_wps = [
        seq
        for seq in range(AUTO_PATH_START_WP_SEQ, AUTO_PATH_END_WP_SEQ + 1)
        if seq not in mission_items
    ]
    if missing_surface_wps:
        raise RuntimeError(f"Mission is missing surface waypoint(s): {missing_surface_wps}")

    waypoints = []
    for seq in range(AUTO_PATH_START_WP_SEQ, AUTO_PATH_END_WP_SEQ + 1):
        original_waypoint = mission_items[seq]
        commanded_waypoint = waypoint_with_altitude(original_waypoint, SURFACE_ALTITUDE_M)
        commanded_waypoint["original_alt"] = original_waypoint.get("alt")
        waypoints.append(commanded_waypoint)

    return waypoints


def run_surface_search(controller, cap, target_value, mission_items):
    set_mission_state("NORMAL_LAWN_MOWER")
    print("Entering 40x30 m surface auto path")
    controller.set_cruise_speed(SURFACE_SPEED_MPS)
    if len(REDZONE_POLYGON_LATLON) < 3:
        print("Red-zone coordinate polygon not configured; coordinate avoidance inactive")

    previous_waypoint = mission_items[SURFACE_ENTRANCE_WP_SEQ]

    for waypoint in build_surface_waypoints(mission_items):
        active_segment_start_wp = previous_waypoint
        active_segment_target_wp = waypoint
        resume_target_wp = waypoint
        current_index = previous_waypoint.get("seq", "runtime")
        next_index = waypoint.get("seq", "runtime")
        print(
            f"SEGMENT current_wp={current_index} target_wp={next_index} "
            f"alt={SURFACE_ALTITUDE_M:.1f}m"
        )
        print(
            f"active segment start/target: "
            f"{active_segment_start_wp.get('seq', 'runtime')} -> "
            f"{active_segment_target_wp.get('seq', 'runtime')}"
        )
        print(f"resume_target_wp: {resume_target_wp.get('seq', 'runtime')}")

        intersects_redzone = False
        box = None

        if len(REDZONE_POLYGON_LATLON) >= 3:
            live_position = get_current_position(controller)
            if live_position is not None:
                current_wp_xy = latlon_to_xy(
                    live_position["lat"],
                    live_position["lon"],
                    _home_lat,
                    _home_lon,
                )
            else:
                current_wp_xy = waypoint_to_xy(previous_waypoint)
            next_wp_xy = waypoint_to_xy(waypoint)
            decision = redzone_segment_decision(
                current_wp_xy,
                next_wp_xy,
                block_current_near=True,
            )
            box = decision["geometry"]["expanded_box"]
            intersects_redzone = decision["final_redzone_blocked"]
            print_redzone_decision(decision)

        if intersects_redzone:
            set_mission_state("REDZONE_COORDINATE_BYPASS")
            print("REDZONE_COORDINATE_BYPASS triggered")
            controller.stop()
            chosen_side, bypass_points_xy = generate_box_bypass(
                current_wp_xy,
                next_wp_xy,
                box,
            )

            if not bypass_points_xy:
                print("Bypass generation failed, holding position instead of crossing red zone")
                controller.stop()
                return None

            if not bypass_points_are_safe(bypass_points_xy, box):
                print("Bypass rejected, holding position instead of crossing red zone")
                controller.stop()
                return None

            bypass_status = follow_bypass_velocity_path(
                controller,
                cap,
                bypass_points_xy,
            )
            if bypass_status != "complete":
                print("Bypass failed, holding position")
                controller.stop()
                return None

            set_mission_state("RESUME_LAWN_MOWER")
            print("returning to original yellow path")

        print(f"Going to resume_target_wp: {resume_target_wp.get('seq', 'runtime')}")

        status, decoded_text = goto_waypoint_until_reached(
            controller,
            cap,
            resume_target_wp,
            f"Surface WP{resume_target_wp['seq']}",
            watch_for_qr=True,
            target_value=target_value,
            arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
            avoid_redzone_visual=True,
            avoid_redzone_box=True,
        )

        if status == "target_found":
            return decoded_text

        if status in ("quit", "camera_failed", "redzone_blocked"):
            return None

        print(f"reached resume target: {resume_target_wp.get('seq', 'runtime')}")
        previous_waypoint = resume_target_wp

    print("Surface search completed, target was not found")
    return None


def run_exit_corridor_sequence(controller, cap, mission_items):
    set_mission_state("MOVE_TO_EXIT_CORRIDOR_ENTRANCE")
    print("Moving to exit corridor entrance WP27")
    status, _ = goto_waypoint_until_reached(
        controller,
        cap,
        waypoint_with_altitude(mission_items[EXIT_CORRIDOR_WP_SEQ], SURFACE_ALTITUDE_M),
        "Exit Corridor WP27 at 10m",
        watch_for_qr=False,
        arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
    )
    if status in ("quit", "camera_failed"):
        return status
    print("Reached WP27")
    print("Reached exit corridor entrance WP27")

    set_mission_state("SEARCH_EXIT_GREEN_BANNER")
    exit_banner_seen = search_exit_green_banner_until_seen(controller, cap)
    if not exit_banner_seen:
        print("Exit green banner search ended before corridor move")
        return "green_banner_failed"

    print("Descending to corridor altitude")
    print("Moving from WP27 to WP28")
    print("Moving WP27 to WP28")
    status = move_exit_corridor_wp27_to_wp28(
        controller,
        cap,
        mission_items[EXIT_CORRIDOR_END_WP_SEQ],
        exit_banner_seen,
    )
    if status != "reached":
        return status

    set_mission_state("RTL_AFTER_EXIT")
    print("Switching to RTL")
    controller.set_mode("RTL")
    print("RTL command sent")

    print("Post-target mission sequence complete")
    return "rtl_started"


def run_post_target_sequence(controller, cap, mission_items, target_value):
    print("Correct QR detected, stopping lawn mower/search motion")

    if not ENABLE_SOLVEPNP_ALIGNMENT:
        set_mission_state("TARGET_QR_FOUND_HOVER")
        print(f"Hovering for {TARGET_FOUND_HOVER_SECONDS} seconds")
        print("Skipping solvePnP alignment")
        target_hold_point = get_safe_position(controller)
        status = hover_with_camera(
            controller,
            cap,
            TARGET_FOUND_HOVER_SECONDS,
            "Target QR found hover",
            hold_waypoint=target_hold_point,
        )
        if status == "quit":
            return status
        return run_exit_corridor_sequence(controller, cap, mission_items)

    set_mission_state("TARGET_QR_CONFIRMED")
    set_mission_state("DESCEND_FOR_PAYLOAD")
    print("Descending to 5m")
    status, payload_point = change_altitude_until_reached(
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
        solvepnp_status = solvepnp_target_alignment(controller, cap, target_value)
        if solvepnp_status != "reached":
            print(f"solvePnP target alignment ended with status: {solvepnp_status}")
            return solvepnp_status
        aligned_payload_point = get_current_position(controller)
        if aligned_payload_point is not None:
            payload_point = aligned_payload_point

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

    set_mission_state("ASCEND_AFTER_HOVER")
    print("ascending to 10 m")
    status, _ = change_altitude_until_reached(
        controller,
        cap,
        RETURN_ALT_M,
        "Ascend after 5m hover",
    )
    if status != "reached":
        print("ASCEND_AFTER_HOVER: position/altitude hold failed, continuing to WP27 at 10m")

    return run_exit_corridor_sequence(controller, cap, mission_items)


def main():
    cap = None
    controller = None
    rtl_started = False

    try:
        print("Starting MAVLink mission startup")
        master = MavlinkConnection(CONNECTION_STRING).connect()
        controller = GuidedController(master)
        mission_items = controller.download_mission_items()
        controller.start_telemetry_cache()
        time.sleep(0.5)
        initialize_home_origin(controller, mission_items)

        required_wps = [
            START_QR_WP_SEQ,
            CORRIDOR_ENTRANCE_WP_SEQ,
            SURFACE_ENTRANCE_WP_SEQ,
            EXIT_CORRIDOR_WP_SEQ,
            EXIT_CORRIDOR_END_WP_SEQ,
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

        status, _ = goto_waypoint_until_reached(
            controller,
            cap,
            waypoint_with_altitude(mission_items[CORRIDOR_ENTRANCE_WP_SEQ], START_QR_ALT_M),
            "WP3 Entrance Green Banner Approach",
            arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
        )
        if status in ("quit", "camera_failed"):
            return

        set_mission_state("SEARCH_ENTRANCE_GREEN_BANNER")
        print("Searching entrance green banner")
        set_mission_state("ALIGN_ENTRANCE_GREEN_BANNER")
        green_result = align_to_green_banner(
            controller,
            cap,
            "entrance",
            current_waypoint=f"WP{CORRIDOR_ENTRANCE_WP_SEQ}",
        )
        if not green_banner_result_allows_corridor(green_result):
            print("Entrance green banner alignment failed, holding before corridor")
            controller.stop()
            return

        set_mission_state("ENTER_ENTRANCE_CORRIDOR")
        print("Descending to corridor altitude")
        status, _ = change_altitude_until_reached(
            controller,
            cap,
            CORRIDOR_ALTITUDE_M,
            "WP3 descend to entrance corridor altitude",
        )
        if status in ("quit", "position_failed"):
            return

        print("Moving through entrance corridor WP3 to WP4")
        controller.set_cruise_speed(CORRIDOR_SPEED)
        print("moving to corridor")
        print(f"current corridor waypoint: WP{CORRIDOR_ENTRANCE_WP_SEQ}")
        status, _ = goto_waypoint_until_reached(
            controller,
            cap,
            waypoint_with_altitude(mission_items[CORRIDOR_ENTRANCE_WP_SEQ], CORRIDOR_ALTITUDE_M),
            "WP3 Corridor Entrance",
            arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
        )
        if status in ("quit", "camera_failed"):
            return

        print(f"current corridor waypoint: WP{SURFACE_ENTRANCE_WP_SEQ}")
        status, _ = goto_waypoint_until_reached(
            controller,
            cap,
            waypoint_with_altitude(mission_items[SURFACE_ENTRANCE_WP_SEQ], CORRIDOR_ALTITUDE_M),
            "WP4 Surface Entrance",
            arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
        )
        if status in ("quit", "camera_failed"):
            return

        controller.set_cruise_speed(CRUISE_SPEED_MPS)
        set_mission_state("ASCEND_AFTER_ENTRANCE_CORRIDOR")
        print("Entrance corridor completed, ascending to 10m")
        status, _ = change_altitude_until_reached(
            controller,
            cap,
            SURFACE_ALTITUDE_M,
            "Ascend after WP4 to 10m",
        )
        if status in ("quit", "position_failed"):
            return

        found_value = run_surface_search(controller, cap, target_value, mission_items)

        if found_value:
            print(f"TARGET FOUND: {found_value}")
            post_status = run_post_target_sequence(controller, cap, mission_items, target_value)
            rtl_started = post_status == "rtl_started"

            if post_status not in ("complete", "rtl_started"):
                print(f"Post-target mission ended with status: {post_status}")
        else:
            print("TARGET NOT FOUND")

        if not rtl_started:
            controller.stop()

    except Exception as exc:
        print(f"Mission error: {exc}")
        if controller is not None:
            controller.stop()
            controller.land()

    finally:
        if controller is not None:
            controller.stop_telemetry_cache()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


