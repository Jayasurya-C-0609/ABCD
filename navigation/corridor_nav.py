"""Small corridor waypoint helpers."""


def waypoint_with_altitude(waypoint, altitude_m):
    """Clone a mission waypoint with a commanded altitude."""
    updated = dict(waypoint)
    updated["alt"] = altitude_m
    return updated
