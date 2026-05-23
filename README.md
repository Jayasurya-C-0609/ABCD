# Drone Offboard QR Mission

Mission 2 SkyScan control code for Mission Planner / ArduPilot SITL. The drone flies a QR-guided search mission, avoids the red-zone fence, uses green banner detection for corridor entry, and exits through WP27 to WP28 before RTL.

## Current Mission Flow

1. Connect to Mission Planner SITL using MAVLink.
2. Download uploaded Mission Planner waypoints.
3. Arm and take off in GUIDED mode.
4. Move to WP2 and scan the compulsory start QR.
5. Save the decoded QR text as the target location.
6. Move toward WP3 and search for the entrance green banner.
7. Align roughly with the green banner.
8. Descend to corridor altitude and fly WP3 to WP4.
9. Ascend to surface altitude and start the lawn mower search.
10. Follow surface waypoints sequentially.
11. Avoid the red-zone using coordinate polygon checks and YOLO visual backup.
12. Detect and decode QR codes during the lawn mower search.
13. If the decoded QR matches the target:
    - stop search movement
    - hover for 10 seconds
    - skip solvePnP alignment by default
    - move to WP27
    - detect the exit green banner
    - descend to corridor altitude
    - fly WP27 to WP28
    - switch to RTL

## Important Waypoints

| Waypoint | Meaning |
| --- | --- |
| WP2 | Start QR scan point |
| WP3 | Entrance corridor front / entrance side |
| WP4 | Entrance corridor back / delivery-zone side |
| WP5 onward | Surface lawn mower search |
| WP27 | Exit corridor entrance |
| WP28 | Exit corridor exit |

## Project Structure

```text
main.py                         Mission entry point
config.py                       Mission constants and tuning values
mission/state_machine.py         Main Mission 2 state machine
mavlink/connection.py            MAVLink connection helper
mavlink/guided_control.py        GUIDED mode movement and telemetry cache
vision/camera.py                 Camera opening helper
vision/qr_detect.py              QR YOLO detection
vision/qr_decode.py              WeChat QR decoding
vision/green_banner.py           Green banner YOLO detection
vision/red_zone.py               Red-zone YOLO detection
navigation/waypoint_nav.py       Distance and lat/lon helpers
navigation/qr_alignment.py       QR pixel tracking helpers
navigation/solvepnp_alignment.py solvePnP QR corner helpers
navigation/corridor_nav.py       Corridor waypoint helpers
```

## Requirements

Install the Python dependencies:

```bash
pip install pymavlink opencv-python opencv-contrib-python ultralytics numpy
```

The WeChat QR decoder needs the model files in `models/`:

```text
detect_2021nov.prototxt
detect_2021nov.caffemodel
sr_2021nov.prototxt
sr_2021nov.caffemodel
```

YOLO model files:

```text
models/QRdet.pt
models/REDdet.pt
models/Green banner.pt
```

Make sure the green banner model filename matches `GREEN_BANNER_MODEL_PATH` in `config.py`.

## Mission Planner Setup

1. Open Mission Planner.
2. Start ArduPilot SITL.
3. Select Copter / Quad.
4. Upload the Mission 2 waypoints to the vehicle.
5. Keep Mission Planner connected.
6. Confirm the connection string in `config.py`.

Common connection strings:

```python
CONNECTION_STRING = "tcp:127.0.0.1:5763"
CONNECTION_STRING = "udp:127.0.0.1:14550"
CONNECTION_STRING = "tcp:127.0.0.1:5760"
```

## Running The Mission

From the project root:

```bash
python main.py
```

The mission expects:

- Mission waypoints already uploaded to Mission Planner.
- Camera available through `CAMERA_SOURCE` or `DROIDCAM_URL`.
- QR, red-zone, and green-banner model files present.
- Vehicle in a state where GUIDED mode, arming, and takeoff are allowed.

## Key Config Values

Edit [config.py](config.py) for mission tuning.

```python
TAKEOFF_ALT_M = 5.0
START_QR_ALT_M = 5.0
SURFACE_ALTITUDE_M = 10.0
CORRIDOR_ALTITUDE_M = 3.0
CORRIDOR_SPEED = 0.4
WAYPOINT_ACCEPT_RADIUS = 1.0
```

Target QR behavior:

```python
ENABLE_SOLVEPNP_ALIGNMENT = False
TARGET_FOUND_HOVER_SECONDS = 10
```

When `ENABLE_SOLVEPNP_ALIGNMENT = False`, the drone skips payload solvePnP alignment after the correct target QR is decoded. It stops, hovers for 10 seconds, and goes to the exit corridor.

Green banner detection:

```python
GREEN_BANNER_CONF = 0.5
GREEN_ALIGN_TIMEOUT = 5
GREEN_ALIGN_TOLERANCE_X = 60
GREEN_ALIGN_TOLERANCE_Y = 80
ALLOW_CORRIDOR_WITHOUT_BANNER = True
```

Red-zone safety:

```python
SAFETY_MARGIN = 0.5
REDZONE_BYPASS_MARGIN_M = SAFETY_MARGIN
```

## Green Banner Alignment

When a green banner is detected, the code calculates:

```text
error_x = banner_center_x - frame_center_x
error_y = banner_center_y - frame_center_y
```

The drone sends slow body-frame velocity and yaw corrections to roughly center the banner. If perfect alignment is not reached before timeout, the mission can continue to the corridor when `ALLOW_CORRIDOR_WITHOUT_BANNER = True`.

## Red-Zone Avoidance

The red-zone fence polygon is configured in `config.py`:

```python
REDZONE_POLYGON_LATLON = [
    (-35.3637285289132, 149.165268838406),
    (-35.3637247010921, 149.165355339646),
    (-35.3637646197890, 149.165360033512),
    (-35.3637695412708, 149.165273532271),
]
```

The mission converts the polygon to local XY meters, checks segment intersection against the actual polygon, and only uses the expanded bounding box as a quick pre-check.

## Syntax Check

PowerShell may not expand `mission/*.py` correctly for `py_compile`, so use the explicit file list:

```powershell
python -m py_compile main.py config.py mission\__init__.py mission\state_machine.py navigation\__init__.py navigation\corridor_nav.py navigation\qr_alignment.py navigation\solvepnp_alignment.py navigation\waypoint_nav.py vision\__init__.py vision\camera.py vision\green_banner.py vision\qr_decode.py vision\qr_detect.py vision\red_zone.py mavlink\__init__.py mavlink\connection.py mavlink\guided_control.py
```

## Troubleshooting

If the camera does not open:

- Try changing `CAMERA_SOURCE` in `config.py`.
- If using DroidCam/IP camera, set `DROIDCAM_URL`.

If a model is not found:

- Check the exact filename in `models/`.
- Check the matching path in `config.py`.

If the drone does not move to WP27 after target QR:

- Confirm `ENABLE_SOLVEPNP_ALIGNMENT = False`.
- Confirm WP27 and WP28 exist in the uploaded Mission Planner mission.
- Check that `EXIT_CORRIDOR_WP_SEQ = 27` and `EXIT_CORRIDOR_END_WP_SEQ = 28`.

If QR is decoded but mission does not continue:

- Check that the decoded text exactly matches the WP2 target QR text.
- Matching is case-insensitive, but extra unexpected characters can still cause mismatch.
