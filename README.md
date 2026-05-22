# Drone Offboard QR Mission with Mission Planner SITL

This project runs the Mission 2 QR search flow with Mission Planner SITL and a
camera-backed vision pipeline.

## What this tests

- MAVLink connection to Mission Planner SITL
- GUIDED mode takeoff
- Start QR detection and decoding at WP2
- Entrance green banner detection before WP3 to WP4 corridor movement
- Mission waypoint lawn-mower surface search
- Red-zone coordinate and YOLO avoidance
- Target QR payload hover flow
- Exit green banner detection before WP27 to WP28 corridor movement
- RTL after the exit corridor

## Install

```bash
pip install pymavlink opencv-python opencv-contrib-python ultralytics numpy
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

## Important

Upload the mission waypoints before running `main.py`, and keep the QR and
green banner models in the `models` directory.
