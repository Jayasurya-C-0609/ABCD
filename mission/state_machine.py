from enum import Enum, auto

class MissionState(Enum):
    CONNECT = auto()
    TAKEOFF = auto()
    SCAN_LAWNMOWER = auto()
    QR_ALIGN = auto()
    QR_DECODE = auto()
    TARGET_FOUND = auto()
    RESUME_PATH = auto()
    LAND = auto()
