import time
import math
from pymavlink import mavutil


class GuidedController:
    def __init__(self, master):
        self.master = master

    def set_mode(self, mode_name: str):
        mode_mapping = self.master.mode_mapping()
        mode_id = mode_mapping[mode_name]

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        print(f"Mode command sent: {mode_name}")
        time.sleep(1)

    def arm(self):
        print("Arming...")
        self.master.arducopter_arm()
        self.master.motors_armed_wait()
        print("Armed")

    def takeoff(self, altitude_m):
        print(f"Taking off to {altitude_m} m...")

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,
            0, 0,
            altitude_m
        )

        print("Takeoff command sent")
        time.sleep(8)

    def download_mission_items(self, timeout_s=2.0):
        print("Downloading mission waypoints from vehicle...")
        self.master.mav.mission_request_list_send(
            self.master.target_system,
            self.master.target_component
        )

        count_msg = self.master.recv_match(
            type="MISSION_COUNT",
            blocking=True,
            timeout=timeout_s
        )

        if count_msg is None:
            raise RuntimeError("No mission received from vehicle. Upload/write the mission to the drone first.")

        mission_items = {}

        for seq in range(count_msg.count):
            self.master.mav.mission_request_int_send(
                self.master.target_system,
                self.master.target_component,
                seq
            )

            msg = self.master.recv_match(
                type=["MISSION_ITEM_INT", "MISSION_ITEM"],
                blocking=True,
                timeout=timeout_s
            )

            if msg is None:
                raise RuntimeError(f"Mission waypoint {seq} was not received from vehicle.")

            if msg.get_type() == "MISSION_ITEM_INT":
                lat = msg.x / 1e7
                lon = msg.y / 1e7
            else:
                lat = msg.x
                lon = msg.y

            mission_items[msg.seq] = {
                "seq": msg.seq,
                "lat": lat,
                "lon": lon,
                "alt": msg.z,
                "command": msg.command,
                "frame": msg.frame,
            }

        print(f"Downloaded {len(mission_items)} mission waypoints")
        return mission_items

    def get_global_position(self, timeout_s=1.0):
        msg = self.master.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=timeout_s
        )

        if msg is None:
            return None

        return {
            "lat": msg.lat / 1e7,
            "lon": msg.lon / 1e7,
            "relative_alt": msg.relative_alt / 1000.0,
        }

    def distance_to_global_location(self, lat, lon, alt_m):
        position = self.get_global_position(timeout_s=0.2)

        if position is None:
            return None

        earth_radius_m = 6371000.0
        lat1 = math.radians(position["lat"])
        lat2 = math.radians(lat)
        dlat = math.radians(lat - position["lat"])
        dlon = math.radians(lon - position["lon"])

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        ground_distance = earth_radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        altitude_error = alt_m - position["relative_alt"]

        return math.sqrt(ground_distance ** 2 + altitude_error ** 2)

    def goto_global_location(self, lat, lon, alt_m):
        self.master.mav.set_position_target_global_int_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000,
            int(lat * 1e7),
            int(lon * 1e7),
            alt_m,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

    def get_local_position(self, timeout_s=1.0):
        msg = self.master.recv_match(
            type="LOCAL_POSITION_NED",
            blocking=True,
            timeout=timeout_s
        )

        if msg is None:
            return None

        return msg.x, msg.y, msg.z

    def distance_to_local_position(self, north_m, east_m, alt_m):
        position = self.get_local_position(timeout_s=0.2)

        if position is None:
            return None

        current_north, current_east, current_down = position
        target_down = -alt_m

        return math.sqrt(
            (north_m - current_north) ** 2
            + (east_m - current_east) ** 2
            + (target_down - current_down) ** 2
        )

    def goto_local_position(self, north_m, east_m, alt_m):
        """
        LOCAL_NED:
        x/north + = north
        y/east  + = east
        z/down  + = down, so altitude above home is negative z.
        """
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,
            north_m, east_m, -alt_m,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )

    def send_body_velocity(self, vx, vy, vz=0.0):
        """
        BODY_NED:
        vx + = forward
        vx - = backward
        vy + = right
        vy - = left
        vz + = down
        vz - = up
        """

        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    def stop(self):
        self.send_body_velocity(0, 0, 0)

    def hold(self, duration_s):
        end_time = time.time() + duration_s

        while time.time() < end_time:
            self.stop()
            time.sleep(0.2)

    def land(self):
        print("Landing...")
        self.set_mode("LAND")
