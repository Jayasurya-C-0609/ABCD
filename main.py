import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    AUTO_PATH_END_WP_SEQ,
    AUTO_PATH_START_WP_SEQ,
    BYPASS_FORWARD_SPEED,
    BYPASS_SIDE_SPEED,
    BYPASS_TURN_HOVER,
    CAMERA_SOURCE,
    CONNECTION_STRING,
    CORRIDOR_ENTRANCE_WP_SEQ,
    CRUISE_SPEED_MPS,
    DROIDCAM_URL,
    EXIT_CORRIDOR_END_WP_SEQ,
    EXIT_CORRIDOR_WP_SEQ,
    EXIT_CORRIDOR_ALT_M,
    PAYLOAD_DESCENT_ALT_M,
    PAYLOAD_HOVER_TIME_S,
    POSITION_TOLERANCE_M,
    QR_MODEL_PATH,
    RED_MODEL_PATH,
    REDZONE_POLYGON_LATLON,
    REDZONE_BYPASS_MARGIN_M,
    REDZONE_SIDE_STEP_SPEED_MPS,
    REDZONE_SIDE_STEP_TIME_S,
    REDZONE_VISUAL_CENTER_MARGIN_RATIO,
    REDZONE_VISUAL_MIN_AREA_RATIO,
    REDZONE_YOLO_CONF,
    RETURN_ALT_M,
    START_QR_ALT_M,
    START_QR_WP_SEQ,
    SURFACE_ALTITUDE_M,
    SURFACE_SPEED_MPS,
    SURFACE_ENTRANCE_WP_SEQ,
    TAKEOFF_ALT_M,
)
from mavlink.connection import MavlinkConnection
from mavlink.guided_control import GuidedController
from vision.camera import open_camera


DECODE_ATTEMPTS = 5
WAYPOINT_TIMEOUT = 25
WAYPOINT_TIMEOUT_S = WAYPOINT_TIMEOUT
WAYPOINT_ACCEPT_RADIUS = 1.0
START_QR_TIMEOUT_S = 45
START_QR_SCAN_TIMEOUT = 60
START_QR_MAX_RETRIES = 3
QR_DETECT_FPS = 5
WP2_SCAN_NUDGE_SPEED_MPS = 0.2
WP2_SCAN_NUDGE_TIME_S = 1.0
DETECTION_INTERVAL_S = 0.8
WRONG_QR_SUPPRESS_S = 12.0
LOOP_SLEEP_S = 0.02
WAYPOINT_COMMAND_INTERVAL_S = 1.0
WAYPOINT_DEBUG_INTERVAL_S = 1.0
POSITION_STUCK_TIMEOUT_S = 3.0
POSITION_MOVING_THRESHOLD_M = 0.15
ALTITUDE_TOLERANCE_M = 0.3
POSITION_RETRY_TIMEOUT_S = 3.0
WINDOW_NAME = "SkyScan QR Mission"
_detect_qrs = None
_decode_qr_crop = None
_red_model = None
_red_model_available = None
_last_frame_time = None
_display_fps = 0.0
_home_lat = None
_home_lon = None
_last_valid_position = None
_redzone_geometry_logged = False


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
    load_redzone_detector()


def load_redzone_detector():
    global _red_model, _red_model_available

    if _red_model_available is False:
        return None

    if _red_model is None:
        if not Path(RED_MODEL_PATH).exists():
            print(
                f"WARNING: REDdet model missing: {RED_MODEL_PATH}. "
                "Continuing with coordinate-only red-zone avoidance."
            )
            _red_model_available = False
            return None

        print(f"RED model loaded path: {RED_MODEL_PATH}")
        _red_model = YOLO(RED_MODEL_PATH)
        _red_model_available = True

    return _red_model


def detect_redzone_yolo(frame, conf=REDZONE_YOLO_CONF, debug_frame=None):
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

    print(f"red-zone YOLO confidence = {confidence:.2f}")
    print(f"red-zone bbox = {bbox}")

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


def latlon_to_xy(lat, lon, home_lat, home_lon):
    lat_scale_m = 111320.0
    lon_scale_m = 111320.0 * max(0.01, abs(math.cos(math.radians(home_lat))))
    x = (lon - home_lon) * lon_scale_m
    y = (lat - home_lat) * lat_scale_m
    return x, y


def xy_to_latlon(x, y, home_lat, home_lon):
    lat_scale_m = 111320.0
    lon_scale_m = 111320.0 * max(0.01, abs(math.cos(math.radians(home_lat))))
    lat = home_lat + (y / lat_scale_m)
    lon = home_lon + (x / lon_scale_m)
    return lat, lon


def waypoint_to_xy(waypoint):
    return latlon_to_xy(waypoint["lat"], waypoint["lon"], _home_lat, _home_lon)


def ground_distance_between_latlon(first_lat, first_lon, second_lat, second_lon):
    earth_radius_m = 6371000.0
    lat1 = math.radians(first_lat)
    lat2 = math.radians(second_lat)
    dlat = math.radians(second_lat - first_lat)
    dlon = math.radians(second_lon - first_lon)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return earth_radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_from_position_to_waypoint(position, waypoint):
    ground_distance = ground_distance_between_latlon(
        position["lat"],
        position["lon"],
        waypoint["lat"],
        waypoint["lon"],
    )
    altitude_error = waypoint["alt"] - position["alt"]
    return math.sqrt((ground_distance ** 2) + (altitude_error ** 2))


def position_changed_enough(previous_position, current_position):
    if previous_position is None or current_position is None:
        return True

    ground_delta = ground_distance_between_latlon(
        previous_position["lat"],
        previous_position["lon"],
        current_position["lat"],
        current_position["lon"],
    )
    alt_delta = abs(current_position["alt"] - previous_position["alt"])
    return math.sqrt((ground_delta ** 2) + (alt_delta ** 2)) >= POSITION_MOVING_THRESHOLD_M


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
    position = controller.get_global_position(timeout_s=0.0)
    if position is None:
        return None

    return cache_position({
        "type": "global",
        "source": "GLOBAL_POSITION_INT",
        "lat": position["lat"],
        "lon": position["lon"],
        "alt": position["relative_alt"],
    })


def get_latest_gps_raw_position(controller, timeout_s=0.2):
    msg = controller.get_cached_message("GPS_RAW_INT")
    if msg is None or msg["lat"] == 0 or msg["lon"] == 0:
        return None

    altitude = _last_valid_position["alt"] if _last_valid_position is not None else 0.0
    return cache_position({
        "type": "global",
        "source": "GPS_RAW_INT",
        "lat": msg["lat"],
        "lon": msg["lon"],
        "alt": altitude,
    })


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


def get_safe_current_position(controller):
    return get_safe_position(controller)


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
    current_position = get_safe_current_position(controller)
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


def run_post_target_sequence(controller, cap, mission_items):
    set_mission_state("TARGET_QR_FOUND")
    controller.stop()
    print("correct QR found")
    print("Correct QR detected, stopping lawn mower/search motion")

    set_mission_state("DESCEND_TO_5M")
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
        print("ASCEND_AFTER_HOVER: position/altitude hold failed, continuing to WP16 at 10m")

    set_mission_state("GOTO_EXIT_WP16")
    print("going to WP16 first")
    status, _ = goto_waypoint_until_reached(
        controller,
        cap,
        waypoint_with_altitude(mission_items[EXIT_CORRIDOR_WP_SEQ], RETURN_ALT_M),
        "Exit Corridor WP16 at 10m",
        watch_for_qr=False,
        arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
    )
    if status in ("quit", "camera_failed"):
        return status
    print("reached WP16")

    set_mission_state("DESCEND_AT_EXIT_WP16")
    print("Exit corridor altitude active: descending to 2-3m at waypoint 16")
    status, corridor_low_point = change_altitude_until_reached(
        controller,
        cap,
        EXIT_CORRIDOR_ALT_M,
        "Exit Corridor WP16 descent to 3m",
    )
    if status != "reached":
        return status

    set_mission_state("MOVE_EXIT_WP16_TO_WP17")
    print("moving WP16 to WP17")
    print("Exit corridor altitude active: moving WP16 to WP17 at 2-3m")
    status, _ = goto_waypoint_until_reached(
        controller,
        cap,
        waypoint_with_altitude(mission_items[EXIT_CORRIDOR_END_WP_SEQ], EXIT_CORRIDOR_ALT_M),
        "Exit Corridor WP17 at 3m",
        watch_for_qr=False,
        arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
    )
    if status in ("quit", "camera_failed"):
        return status
    print("Reached waypoint 17")

    set_mission_state("ASCEND_AFTER_EXIT_WP17")
    print("ascending after exit corridor")
    status, _ = change_altitude_until_reached(
        controller,
        cap,
        RETURN_ALT_M,
        "Ascend after WP17",
    )
    if status != "reached":
        return status

    set_mission_state("RETURN_TO_HOME")
    print("Waypoint 17 reached, switching to RTL")
    controller.set_mode("RTL")

    print("Post-target mission sequence complete")
    return "rtl_started"


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
            waypoint_with_altitude(mission_items[CORRIDOR_ENTRANCE_WP_SEQ], EXIT_CORRIDOR_ALT_M),
            "WP3 Corridor Entrance",
            arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
        )
        if status in ("quit", "camera_failed"):
            return

        status, _ = goto_waypoint_until_reached(
            controller,
            cap,
            waypoint_with_altitude(mission_items[SURFACE_ENTRANCE_WP_SEQ], EXIT_CORRIDOR_ALT_M),
            "WP4 Surface Entrance",
            arrival_tolerance_m=WAYPOINT_ACCEPT_RADIUS,
        )
        if status in ("quit", "camera_failed"):
            return

        set_mission_state("ASCEND_TO_SURFACE_ALTITUDE")
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
            post_status = run_post_target_sequence(controller, cap, mission_items)
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


if __name__ == "__main__":
    main()
