import math


# ============================================================
# HOME LOCATION
# Copy from Mission Planner home location
# ============================================================
HOME_LAT = -35.3633516
HOME_LON = 149.1652413


# ============================================================
# EXACT RED-ZONE FENCE POINTS FROM MISSION PLANNER
# ============================================================
REDZONE_POLYGON_LATLON = [
    (-35.3637285289132, 149.165268838406),
    (-35.3637247010921, 149.165355339646),
    (-35.3637646197890, 149.165360033512),
    (-35.3637695412708, 149.165273532271),
]


# ============================================================
# ALTITUDES
# ============================================================
START_QR_ALT = 5.0
CORRIDOR_ALT = 3.0
SURFACE_ALT = 10.0


# ============================================================
# SEARCH AREA IN LOCAL XY METERS
# x = left/right from home
# y = forward/back from home
#
# Adjust these values based on your Mission Planner field.
# ============================================================
AREA_X_MIN = -5.0
AREA_X_MAX = 18.0
AREA_Y_MIN = -55.0
AREA_Y_MAX = -15.0

SWEEP_SPACING = 5.0
SAFETY_MARGIN = 1.0


# ============================================================
# LAT/LON <-> LOCAL XY CONVERSION
# ============================================================
def latlon_to_xy(lat, lon):
    R = 6378137.0
    x = math.radians(lon - HOME_LON) * R * math.cos(math.radians(HOME_LAT))
    y = math.radians(lat - HOME_LAT) * R
    return x, y


def xy_to_latlon(x, y):
    R = 6378137.0
    lat = HOME_LAT + math.degrees(y / R)
    lon = HOME_LON + math.degrees(
        x / (R * math.cos(math.radians(HOME_LAT)))
    )
    return lat, lon


# ============================================================
# RED-ZONE BOUNDING BOX
# ============================================================
def make_redzone_box():
    pts_xy = [latlon_to_xy(lat, lon) for lat, lon in REDZONE_POLYGON_LATLON]

    xs = [p[0] for p in pts_xy]
    ys = [p[1] for p in pts_xy]

    box = {
        "xmin": min(xs) - SAFETY_MARGIN,
        "xmax": max(xs) + SAFETY_MARGIN,
        "ymin": min(ys) - SAFETY_MARGIN,
        "ymax": max(ys) + SAFETY_MARGIN,
    }

    width = box["xmax"] - box["xmin"]
    height = box["ymax"] - box["ymin"]

    print("Red-zone polygon XY:")
    for p in pts_xy:
        print(" ", p)

    print("Expanded red-zone box:", box)
    print(f"Expanded box width  = {width:.2f} m")
    print(f"Expanded box height = {height:.2f} m")

    return box


def segment_crosses_box(x, y1, y2, box):
    """
    Since lawn mower lines are mostly vertical,
    check whether this vertical sweep line crosses the red-zone box.
    """

    if not (box["xmin"] <= x <= box["xmax"]):
        return False

    seg_y_min = min(y1, y2)
    seg_y_max = max(y1, y2)

    if seg_y_max < box["ymin"]:
        return False

    if seg_y_min > box["ymax"]:
        return False

    return True


def remove_duplicate_points(path_xy, min_dist=0.5):
    clean = []

    for p in path_xy:
        if not clean:
            clean.append(p)
            continue

        last = clean[-1]
        d = math.hypot(p[0] - last[0], p[1] - last[1])

        if d >= min_dist:
            clean.append(p)

    return clean


# ============================================================
# TRUE ZIGZAG PATH WITH RED-ZONE BYPASS
# ============================================================
def generate_zigzag_with_redzone():
    box = make_redzone_box()
    path_xy = []

    x_values = []
    x = AREA_X_MAX

    while x >= AREA_X_MIN:
        x_values.append(x)
        x -= SWEEP_SPACING

    direction_down = True

    for x in x_values:
        if direction_down:
            y_start = AREA_Y_MAX
            y_end = AREA_Y_MIN
        else:
            y_start = AREA_Y_MIN
            y_end = AREA_Y_MAX

        # ----------------------------------------------------
        # CASE 1: This sweep line crosses red zone
        # ----------------------------------------------------
        if segment_crosses_box(x, y_start, y_end, box):
            left_side_x = box["xmin"] - 1.0
            right_side_x = box["xmax"] + 1.0

            # choose nearest side
            if abs(x - left_side_x) < abs(x - right_side_x):
                bypass_x = left_side_x
                side_name = "LEFT"
            else:
                bypass_x = right_side_x
                side_name = "RIGHT"

            if direction_down:
                before_y = box["ymax"] + 1.0
                after_y = box["ymin"] - 1.0
            else:
                before_y = box["ymin"] - 1.0
                after_y = box["ymax"] + 1.0

            print(f"Sweep x={x:.2f} crosses red zone. Bypass side: {side_name}")

            # Zigzag with local bypass:
            # move on original line -> side -> straight -> return -> continue
            path_xy.append((x, y_start))
            path_xy.append((x, before_y))
            path_xy.append((bypass_x, before_y))
            path_xy.append((bypass_x, after_y))
            path_xy.append((x, after_y))
            path_xy.append((x, y_end))

        # ----------------------------------------------------
        # CASE 2: Normal sweep line
        # ----------------------------------------------------
        else:
            path_xy.append((x, y_start))
            path_xy.append((x, y_end))

        direction_down = not direction_down

    path_xy = remove_duplicate_points(path_xy)
    return path_xy


# ============================================================
# WRITE MISSION PLANNER WAYPOINT FILE
# ============================================================
def write_waypoints_file(filename, path_xy):
    with open(filename, "w") as f:
        f.write("QGC WPL 110\n")

        seq = 0

        # ----------------------------------------------------
        # Home / dummy row
        # ----------------------------------------------------
        f.write(
            f"{seq}\t1\t0\t16\t0\t0\t0\t0\t"
            f"{HOME_LAT:.10f}\t{HOME_LON:.10f}\t0\t1\n"
        )
        seq += 1

        # ----------------------------------------------------
        # WP1 Takeoff
        # ----------------------------------------------------
        f.write(
            f"{seq}\t0\t3\t22\t0\t0\t0\t0\t"
            f"{HOME_LAT:.10f}\t{HOME_LON:.10f}\t{START_QR_ALT:.1f}\t1\n"
        )
        seq += 1

        # ----------------------------------------------------
        # WP2 Start QR point
        # ----------------------------------------------------
        wp2_lat, wp2_lon = xy_to_latlon(0.0, -4.0)
        f.write(
            f"{seq}\t0\t3\t16\t0\t0\t0\t0\t"
            f"{wp2_lat:.10f}\t{wp2_lon:.10f}\t{START_QR_ALT:.1f}\t1\n"
        )
        seq += 1

        # ----------------------------------------------------
        # WP3 Entrance corridor front
        # ----------------------------------------------------
        wp3_lat, wp3_lon = xy_to_latlon(14.0, -12.0)
        f.write(
            f"{seq}\t0\t3\t16\t0\t0\t0\t0\t"
            f"{wp3_lat:.10f}\t{wp3_lon:.10f}\t{CORRIDOR_ALT:.1f}\t1\n"
        )
        seq += 1

        # ----------------------------------------------------
        # WP4 Entrance corridor back
        # ----------------------------------------------------
        wp4_lat, wp4_lon = xy_to_latlon(14.0, -18.0)
        f.write(
            f"{seq}\t0\t3\t16\t0\t0\t0\t0\t"
            f"{wp4_lat:.10f}\t{wp4_lon:.10f}\t{CORRIDOR_ALT:.1f}\t1\n"
        )
        seq += 1

        # ----------------------------------------------------
        # WP5 onward: Zigzag surface path
        # ----------------------------------------------------
        for x, y in path_xy:
            lat, lon = xy_to_latlon(x, y)
            f.write(
                f"{seq}\t0\t3\t16\t0\t0\t0\t0\t"
                f"{lat:.10f}\t{lon:.10f}\t{SURFACE_ALT:.1f}\t1\n"
            )
            seq += 1

        # ----------------------------------------------------
        # Final waypoint: return near home at 10 m
        # ----------------------------------------------------
        f.write(
            f"{seq}\t0\t3\t16\t0\t0\t0\t0\t"
            f"{HOME_LAT:.10f}\t{HOME_LON:.10f}\t{SURFACE_ALT:.1f}\t1\n"
        )
        seq += 1

    print("\nMission file created successfully.")
    print(f"File name: {filename}")
    print(f"Total mission rows: {seq}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    path = generate_zigzag_with_redzone()
    write_waypoints_file("mission_zigzag_redzone_safe.waypoints", path)