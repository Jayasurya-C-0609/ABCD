"""
Configuration for Mission Planner / ArduPilot SITL QR simulation.

This project simulates QR detection using drone position instead of a real camera.
When the simulated drone comes near a configured QR point, the offboard code
acts as if YOLO detected the QR and WeChat decoded it.
"""

# MAVLink connection for Mission Planner SITL.
# Common values:
#   Mission Planner SITL UDP: "udp:127.0.0.1:14550"
#   ArduPilot SITL TCP:       "tcp:127.0.0.1:5760"
CONNECTION_STRING = "tcp:127.0.0.1:5763"

# Camera settings
# For DroidCam virtual webcam, try CAMERA_SOURCE = 0, 1, or 2.
# For DroidCam WiFi/IP mode, set DROIDCAM_URL like "http://192.168.1.10:4747/video".
CAMERA_SOURCE = 0
DROIDCAM_URL = ""

# Flight settings
TAKEOFF_ALT_M = 5.0
START_QR_ALT_M = 5.0
SEARCH_ALT_M = 10.0
CRUISE_SPEED_MPS = 0.8
POSITION_TOLERANCE_M = 3.0
PAYLOAD_DESCENT_ALT_M = 5.0
PAYLOAD_HOVER_TIME_S = 7.0
EXIT_CORRIDOR_ALT_M = 3.0
RETURN_ALT_M = 10.0

# Mission Planner waypoint sequence numbers.
# These must match the green waypoint numbers shown in Mission Planner.
START_QR_WP_SEQ = 2
CORRIDOR_ENTRANCE_WP_SEQ = 3
SURFACE_ENTRANCE_WP_SEQ = 4
AUTO_PATH_START_WP_SEQ = 5
AUTO_PATH_END_WP_SEQ = 16
EXIT_CORRIDOR_WP_SEQ = 16

# 40 x 30 m delivery surface in local coordinates.
# x axis = 40 m side, y axis = 30 m side.
AREA_LENGTH_M = 40.0
AREA_WIDTH_M = 30.0
LANE_SPACING_M = 5.0

# Simulated QR behavior
QR_DETECTION_RADIUS_M = 2.2      # Drone "detects" QR when closer than this
QR_ALIGN_TIME_S = 2.0            # Simulated hover/align time
QR_ALREADY_SEEN_RADIUS_M = 1.5   # Prevent repeated detection of same QR

# Start QR gives this target ID.
# In real mission, this comes from the QR near start point.
TARGET_ID = "DROP_A"

# Simulated QR locations inside the 40x30m surface.
# In real camera flight, YOLO+WeChat replaces this.
SIMULATED_QR_POINTS = [
    {"name": "WRONG_QR_1", "x": 8.0,  "y": 6.0,  "data": "DROP_B"},
    {"name": "WRONG_QR_2", "x": 21.0, "y": 16.0, "data": "DROP_C"},
    {"name": "TARGET_QR",  "x": 33.0, "y": 24.0, "data": "DROP_A"},
]

# Mission Planner/SITL home is used as origin. The code converts local meters to GPS.
# Use NED-like convention:
#   local x = East meters
#   local y = North meters
# If your rectangle direction is different, change these signs or rotate coordinates later.
LOCAL_X_IS_EAST = True
LOCAL_Y_IS_NORTH = True
