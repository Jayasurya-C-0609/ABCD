# Drone Offboard QR Simulation with Mission Planner SITL

This is the first code version: **QR detection logic simulation**.
It does not use a real camera. It simulates QR detection by checking whether the SITL drone is near pre-defined QR points.

## What this tests

- MAVLink connection to Mission Planner SITL
- GUIDED mode takeoff
- 40 m x 30 m lawn-mower path
- Simulated QR detection
- Decode all detected QRs
- Compare decoded value with target ID
- If wrong QR: return to resume point and continue path
- If target QR: go to target QR position and hover

## Install

```bash
pip install pymavlink
```

Optional later for real vision:

```bash
pip install opencv-python opencv-contrib-python ultralytics numpy
```

## Mission Planner steps

1. Open Mission Planner.
2. Go to **Simulation**.
3. Select **Copter** and **Quad**.
4. Start SITL simulation.
5. Wait for map vehicle to appear.
6. Keep Mission Planner connected.
7. In VS Code terminal, run:

```bash
cd drone_offboard_qr_sim
python main.py
```

## Change QR locations

Edit `config.py`:

```python
SIMULATED_QR_POINTS = [
    {"name": "WRONG_QR_1", "x": 8.0,  "y": 6.0,  "data": "DROP_B"},
    {"name": "TARGET_QR",  "x": 33.0, "y": 24.0, "data": "DROP_A"},
]
```

The target is:

```python
TARGET_ID = "DROP_A"
```

## Important

Mission Planner cannot place real QR images into the simulated camera view. This code uses position-based fake QR detection for SITL. Later, replace `SimulatedQRDetector` with `YOLOQRDetector` and replace `SimulatedQRDecoder` with `WeChatQRDecoder` for real camera/video testing.
