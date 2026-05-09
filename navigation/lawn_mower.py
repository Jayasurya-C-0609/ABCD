def generate_lawnmower_path(length_m: float, width_m: float, lane_spacing_m: float):
    """
    Generate a simple boustrophedon/lawn-mower path inside a rectangle.
    Coordinates are local meters: x from 0..length, y from 0..width.
    """
    def generate_lawnmower_path(length=40, width=30, lane_spacing=5):
        waypoints = []
        y = 0

        while y <= width:
            waypoints.append((0, y))       # start of lane
            waypoints.append((length, y))  # end of lane
            y += lane_spacing

        return waypoints
