"""Quick test: grab one camera frame from GO2 via videohub and run YOLO.

Usage (on Jetson, source ROS2 first):
    python3 -u test_yolo_camera.py
"""

import sys
import time
import json
import datetime
import random

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from unitree_api.msg import Request, Response
from ultralytics import YOLO

VIDEOHUB_API_ID = 1001
MODEL_PATH = "/home/unitree/come-here/models/yolo11n.pt"


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
    print("Loading YOLO...", flush=True)
    model = YOLO(MODEL_PATH)
    print("YOLO loaded.", flush=True)

    rclpy.init()
    node = rclpy.create_node('yolo_test')
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

    # Request a few frames
    print("Requesting camera frames...", flush=True)
    for i in range(10):
        req_pub.publish(make_req(VIDEOHUB_API_ID))
        rclpy.spin_once(node, timeout_sec=0.5)
        if frame is not None:
            break

    if frame is None:
        print("ERROR: No camera frame received. Is the GO2 robot on?", flush=True)
        rclpy.shutdown()
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Got frame: {w}x{h}", flush=True)

    # Run YOLO
    print("Running YOLO inference...", flush=True)
    t0 = time.time()
    results = model(frame, conf=0.45, verbose=False, classes=[0])[0]
    elapsed = time.time() - t0
    print(f"Inference time: {elapsed:.2f}s", flush=True)

    persons = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        persons.append((x1, y1, x2, y2, conf))
        print(f"  Person: [{x1},{y1}]-[{x2},{y2}] conf={conf:.2f}", flush=True)

    if not persons:
        print("No persons detected.", flush=True)
    else:
        print(f"Detected {len(persons)} person(s).", flush=True)

    # Save annotated image
    for x1, y1, x2, y2, conf in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("/home/unitree/come-here/yolo_test_output.jpg", frame)
    print("Saved annotated frame to ~/come-here/yolo_test_output.jpg", flush=True)

    node.destroy_node()
    rclpy.shutdown()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
