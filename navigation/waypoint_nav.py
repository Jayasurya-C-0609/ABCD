"""Waypoint and local-plane navigation math."""

import math


def latlon_to_xy(lat, lon, home_lat, home_lon):
    """Convert latitude/longitude into local east/north meters."""
    lat_scale_m = 111320.0
    lon_scale_m = 111320.0 * max(0.01, abs(math.cos(math.radians(home_lat))))
    x = (lon - home_lon) * lon_scale_m
    y = (lat - home_lat) * lat_scale_m
    return x, y


def xy_to_latlon(x, y, home_lat, home_lon):
    """Convert local east/north meters into latitude/longitude."""
    lat_scale_m = 111320.0
    lon_scale_m = 111320.0 * max(0.01, abs(math.cos(math.radians(home_lat))))
    lat = home_lat + (y / lat_scale_m)
    lon = home_lon + (x / lon_scale_m)
    return lat, lon


def ground_distance_between_latlon(first_lat, first_lon, second_lat, second_lon):
    """Return haversine ground distance between two global points."""
    earth_radius_m = 6371000.0
    first_lat_rad = math.radians(first_lat)
    second_lat_rad = math.radians(second_lat)
    d_lat = math.radians(second_lat - first_lat)
    d_lon = math.radians(second_lon - first_lon)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(first_lat_rad)
        * math.cos(second_lat_rad)
        * math.sin(d_lon / 2.0) ** 2
    )
    return earth_radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def distance_from_position_to_waypoint(position, waypoint):
    """Measure global horizontal distance when both inputs are valid."""
    ground_distance = ground_distance_between_latlon(
        position["lat"],
        position["lon"],
        waypoint["lat"],
        waypoint["lon"],
    )
    altitude_error = waypoint["alt"] - position["alt"]
    return math.sqrt((ground_distance ** 2) + (altitude_error ** 2))


def position_changed_enough(previous_position, current_position, threshold_m):
    """Report when cached global position changed enough to count as motion."""
    if previous_position is None or current_position is None:
        return True
    ground_delta = ground_distance_between_latlon(
        previous_position["lat"],
        previous_position["lon"],
        current_position["lat"],
        current_position["lon"],
    )
    alt_delta = abs(current_position["alt"] - previous_position["alt"])
    return math.sqrt((ground_delta ** 2) + (alt_delta ** 2)) >= threshold_m
