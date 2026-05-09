import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CAMERA_SOURCE, DROIDCAM_URL
from vision.camera import open_camera

cap = open_camera(CAMERA_SOURCE, DROIDCAM_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        break

    cv2.imshow("Camera Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
