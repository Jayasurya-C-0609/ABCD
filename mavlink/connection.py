from pymavlink import mavutil

class MavlinkConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.master = None

    def connect(self):
        print(f"Connecting to MAVLink: {self.connection_string}")
        self.master = mavutil.mavlink_connection(self.connection_string)
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print(f"Connected: system={self.master.target_system}, component={self.master.target_component}")
        return self.master
