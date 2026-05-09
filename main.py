import cv2
import time

from vision.qr_detect import detect_qrs
from vision.qr_decode import decode_qr_crop

from mavlink.connection import MavlinkConnection
from mavlink.guided_control import GuidedController
from config import CONNECTION_STRING


TARGET_ID = "DROP_A"

CAMERA_INDEX = 0
SEARCH_ALTITUDE = 10.0

CENTER_THRESHOLD = 40
MOVE_SPEED = 0.25


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def qr_alignment_velocity(frame, bbox):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    qr_cx = (x1 + x2) / 2
    qr_cy = (y1 + y2) / 2

    img_cx = w / 2
    img_cy = h / 2

    err_x = qr_cx - img_cx
    err_y = qr_cy - img_cy

    vx = 0.0
    vy = 0.0

    # QR right side -> move right
    if err_x > CENTER_THRESHOLD:
        vy = MOVE_SPEED
    elif err_x < -CENTER_THRESHOLD:
        vy = -MOVE_SPEED

    # QR top side -> move forward
    if err_y < -CENTER_THRESHOLD:
        vx = MOVE_SPEED
    elif err_y > CENTER_THRESHOLD:
        vx = -MOVE_SPEED

    centered = abs(err_x) < CENTER_THRESHOLD and abs(err_y) < CENTER_THRESHOLD

    return vx, vy, centered, err_x, err_y


def main():
    master = MavlinkConnection(CONNECTION_STRING).connect()
    controller = GuidedController(master)

    controller.set_mode("GUIDED")
    controller.arm()
    controller.takeoff(SEARCH_ALTITUDE)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera not opened")
        return

    prev_time = time.time()
    last_velocity_time = 0

    print("Camera QR control started")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera frame")
            break

        detections = detect_qrs(frame)

        h, w = frame.shape[:2]
        cv2.circle(frame, (w // 2, h // 2), 6, (0, 0, 255), -1)

        if len(detections) == 0:
            controller.stop()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            vx, vy, centered, err_x, err_y = qr_alignment_velocity(
                frame,
                (x1, y1, x2, y2)
            )

            cv2.putText(
                frame,
                f"err_x={err_x:.0f}, err_y={err_y:.0f}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # Send velocity slowly, not too fast
            if time.time() - last_velocity_time > 0.15:
                controller.send_body_velocity(vx, vy, 0)
                last_velocity_time = time.time()

            if centered:
                controller.stop()
                print("QR CENTERED - decoding")

                decoded_text, processed = decode_qr_crop(
                    frame,
                    (x1, y1, x2, y2)
                )

                if processed is not None:
                    cv2.imshow("Processed QR", processed)

                if decoded_text:
                    print("Decoded QR:", decoded_text)

                    cv2.putText(
                        frame,
                        decoded_text[:50],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    if decoded_text == TARGET_ID:
                        print("TARGET QR FOUND")
                        controller.stop()

                    else:
                        print("WRONG QR - continue searching")

            break  # handle only first QR for now

        current_time = time.time()
        fps = 1 / max(current_time - prev_time, 0.001)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow("YOLO + WeChat + Mission Planner", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            controller.stop()
            break

        if key == ord("l"):
            controller.land()
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()