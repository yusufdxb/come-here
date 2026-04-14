"""Quick test: grab one camera frame from GO2 via videohub and run face detection.

Usage (on Jetson, source ROS2 first):
    python3 -u test_face_detector_camera.py
"""

import datetime
import random
import sys
import time

import cv2
import numpy as np
import rclpy
from unitree_api.msg import Request, Response

from come_here_perception.mediapipe_face_detector import MediapipeFaceDetector

VIDEOHUB_API_ID = 1001
OUTPUT_PATH = "/home/unitree/come-here/face_test_output.jpg"


def make_req(api_id):
    msg = Request()
    msg.header.identity.api_id = api_id
    msg.header.identity.id = int(datetime.datetime.now().timestamp() * 1000 % 2147483648) + random.randint(0, 999)
    msg.header.lease.id = 0
    msg.header.policy.priority = 0
    msg.header.policy.noreply = False
    msg.parameter = ''
    msg.binary = []
    return msg


def main():
    print("Initializing MediaPipe face detector...", flush=True)
    detector = MediapipeFaceDetector(min_detection_confidence=0.5)
    detector.setup()
    print("Detector ready.", flush=True)

    rclpy.init()
    node = rclpy.create_node('face_detector_test')
    req_pub = node.create_publisher(Request, '/api/videohub/request', 10)

    frame = None

    def on_response(msg):
        nonlocal frame
        if msg.header.identity.api_id != VIDEOHUB_API_ID:
            return
        if msg.header.status.code != 0 or len(msg.binary) < 100:
            return
        jpeg_data = bytes(msg.binary)
        if jpeg_data[:2] != b'\xff\xd8':
            return
        img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frame = img

    node.create_subscription(Response, '/api/videohub/response', on_response, 10)
    time.sleep(0.5)

    print("Requesting camera frames...", flush=True)
    for _ in range(10):
        req_pub.publish(make_req(VIDEOHUB_API_ID))
        rclpy.spin_once(node, timeout_sec=0.5)
        if frame is not None:
            break

    if frame is None:
        print("ERROR: No camera frame received. Is the GO2 robot on?", flush=True)
        detector.teardown()
        rclpy.shutdown()
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Got frame: {w}x{h}", flush=True)

    t0 = time.time()
    result = detector.detect(frame)
    elapsed = time.time() - t0
    print(f"Inference time: {elapsed * 1000:.1f} ms", flush=True)
    print(
        f"face_present={result.face_present} count={result.face_count} "
        f"conf={result.max_confidence:.2f} "
        f"center=({result.center_x_norm:.2f},{result.center_y_norm:.2f})",
        flush=True,
    )

    # Annotate: draw a cross at the reported centroid if a face was seen
    if result.face_present:
        cx = int(result.center_x_norm * w)
        cy = int(result.center_y_norm * h)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
        cv2.putText(
            frame,
            f"face {result.max_confidence:.2f}",
            (cx + 15, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"Saved annotated frame to {OUTPUT_PATH}", flush=True)

    detector.teardown()
    node.destroy_node()
    rclpy.shutdown()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
