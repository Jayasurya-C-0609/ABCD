from mavlink.connection import MavlinkConnection
from config import CONNECTION_STRING

master = MavlinkConnection(CONNECTION_STRING).connect()
print("MAVLink heartbeat connected successfully")