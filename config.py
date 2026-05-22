"""Configuration for the Mission Planner / ArduPilot Mission 2 QR flow."""

# MAVLink connection for Mission Planner SITL.
# Common values:
#   Mission Planner SITL UDP: "udp:127.0.0.1:14550"
#   ArduPilot SITL TCP:       "tcp:127.0.0.1:5760"
CONNECTION_STRING = "tcp:127.0.0.1:5763"

# Detection model paths
QR_MODEL_PATH = "models/QRdet.pt"
RED_MODEL_PATH = "models/REDdet.pt"
GREEN_BANNER_MODEL_PATH = "models/Green banner.pt"

# Camera settings
# For DroidCam virtual webcam, try CAMERA_SOURCE = 0, 1, or 2.
# For DroidCam WiFi/IP mode, set DROIDCAM_URL like "http://192.168.1.10:4747/video".
CAMERA_SOURCE = 0
DROIDCAM_URL = ""

# Flight settings
TAKEOFF_ALT_M = 5.0
START_QR_ALT_M = 5.0
SURFACE_ALTITUDE_M = 10.0
CRUISE_SPEED_MPS = 0.8
SURFACE_SPEED_MPS = 1.0
PAYLOAD_DESCENT_ALT_M = 5.0
PAYLOAD_HOVER_TIME_S = 7.0
CORRIDOR_ALTITUDE_M = 3.0
CORRIDOR_SPEED = 0.4
RETURN_ALT_M = 10.0
REDZONE_YOLO_CONF = 0.5
GREEN_BANNER_CONF = 0.5
GREEN_ALIGN_TOLERANCE_X = 60
GREEN_ALIGN_TOLERANCE_Y = 80
GREEN_ALIGN_TIMEOUT = 5
GREEN_ALIGN_SPEED = 0.25
GREEN_YAW_SPEED = 0.15
GREEN_DETECT_FPS = 5
GREEN_SEARCH_TIMEOUT = 15
ALLOW_CORRIDOR_WITHOUT_BANNER = True
PIXEL_ALIGN_TOL_X = 100
PIXEL_ALIGN_TOL_Y = 100
PIXEL_ALIGN_TIMEOUT = 3.0
PIXEL_ALIGN_SPEED = 0.20
QR_REAL_SIZE_M = 1.0
SOLVEPNP_KP = 0.35
SOLVEPNP_MAX_SPEED = 0.25
SOLVEPNP_ALIGN_TOLERANCE_M = 0.10

# Replace these values with measured Mission 2 camera/payload calibration.
CAMERA_POSITION_BODY = [0.0, 0.0, 0.0]
PAYLOAD_POSITION_BODY = [0.0, 0.0, 0.0]

# OpenCV camera frame: x right, y down, z forward.
# Drone BODY_NED frame: x forward, y right, z down.
R_BODY_CAMERA = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]

# Replace with calibrated intrinsics and distortion for the mission camera.
QR_CAMERA_MATRIX = [
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
]
QR_DIST_COEFFS = [0.0, 0.0, 0.0, 0.0, 0.0]
REDZONE_VISUAL_MIN_AREA_RATIO = 0.08
REDZONE_VISUAL_CENTER_MARGIN_RATIO = 0.35
SAFETY_MARGIN = 0.5
REDZONE_BYPASS_MARGIN_M = SAFETY_MARGIN
REDZONE_SIDE_STEP_SPEED_MPS = 0.4
REDZONE_SIDE_STEP_TIME_S = 3.0
BYPASS_SIDE_SPEED = 0.3
BYPASS_FORWARD_SPEED = 0.4
BYPASS_TURN_HOVER = 1.0

# Mission Planner waypoint sequence numbers.
# These must match the green waypoint numbers shown in Mission Planner.
START_QR_WP_SEQ = 2
CORRIDOR_ENTRANCE_WP_SEQ = 3
SURFACE_ENTRANCE_WP_SEQ = 4
AUTO_PATH_START_WP_SEQ = 5
AUTO_PATH_END_WP_SEQ = 27
EXIT_CORRIDOR_WP_SEQ = 27
EXIT_CORRIDOR_END_WP_SEQ = 28

# Fixed red-zone polygon for coordinate-first avoidance.
REDZONE_POLYGON_LATLON = [
    (-35.3637285289132, 149.165268838406),
    (-35.3637247010921, 149.165355339646),
    (-35.3637646197890, 149.165360033512),
    (-35.3637695412708, 149.165273532271),
]

