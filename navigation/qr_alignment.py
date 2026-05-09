import time

def simulated_qr_align(controller, align_time_s: float):
    """
    In real drone:
      QR bbox center error -> body-frame velocity command -> QR center alignment.
    In SITL-only QR simulation:
      just hover for a short time to represent stop + align + stable decode.
    """
    print(f"Simulated QR alignment: hovering for {align_time_s:.1f}s")
    controller.hold(align_time_s)


def bbox_center_error(bbox, frame_width, frame_height):
    """
    Real-camera helper for later.
    Returns normalized error [-1..1] from image center.
    """
    x1, y1, x2, y2 = bbox
    qr_cx = (x1 + x2) / 2.0
    qr_cy = (y1 + y2) / 2.0
    img_cx = frame_width / 2.0
    img_cy = frame_height / 2.0
    err_x = (qr_cx - img_cx) / img_cx
    err_y = (qr_cy - img_cy) / img_cy
    return err_x, err_y
