import time
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

    def land(self):
        print("Landing...")
        self.set_mode("LAND")