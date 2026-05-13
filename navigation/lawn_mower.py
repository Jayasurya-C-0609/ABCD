def generate_lawnmower_path(length_m: float, width_m: float, lane_spacing_m: float):
    """
    Generate a simple boustrophedon/lawn-mower path inside a rectangle.
    Coordinates are local meters: x from 0..length, y from 0..width.
    """
    waypoints = []
    y = 0.0
    lane_index = 0

    while y <= width_m:
        if lane_index % 2 == 0:
            waypoints.append((0.0, y))
            waypoints.append((length_m, y))
        else:
            waypoints.append((length_m, y))
            waypoints.append((0.0, y))

        y += lane_spacing_m
        lane_index += 1

    return waypoints
