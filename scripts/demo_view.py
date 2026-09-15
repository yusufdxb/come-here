#!/usr/bin/env python3
"""Operator view for the come-here class demo. Runs ON the Jetson; subscribe-only.

The Jetson's DDS is bound to the robot link, and a raw 1080p frame is 6 MB, so
the laptop never subscribes to ROS. This process draws the overlay here and
serves small JPEGs over plain HTTP (the ODIN lab_viewer pattern). The launch
starts it under `nice -n 19` so it always yields to perception and control.
It publishes nothing and commands nothing: safe to start or kill any time.

    python3 scripts/demo_view.py --port 8088        # started by the launch (view:=true)
    ./scripts/demo_view.sh                          # on the laptop: opens the window

Drawn: every person box (grey inside the gate, red outside), the selected
caller (green, confidence and bearing), the gate band, the image center,
and a header with STATE, DOA, turn, target, distance and frame rates.
"""

import argparse
import json
import math
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String

CAMERA_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
HFOV_DEG = 90.0   # same constant as yolo_person_detector.CAMERA_HFOV_DEG

PAGE = b"""<!doctype html><title>GO2 come-here</title>
<body style="margin:0;background:#000"><img src="/camera.mjpg" style="width:100%"></body>"""


class Rate:
    def __init__(self):
        self._last = None
        self.hz = 0.0

    def tick(self):
        now = time.monotonic()
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.hz = inst if self.hz == 0.0 else 0.8 * self.hz + 0.2 * inst
        self._last = now

    def value(self):
        return 0.0 if self._last is None or time.monotonic() - self._last > 3.0 else self.hz


class Slot:
    """Latest JPEG + counter: a stream waits for a NEW frame, so no queue builds up."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg, self._seq = None, 0

    def put(self, jpeg):
        with self._cond:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cond.notify_all()

    def wait_newer(self, seen, timeout=2.0):
        with self._cond:
            if self._seq == seen:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


def bearing_to_x(bearing_rad, width):
    f = (width / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    b = max(-1.5, min(1.5, bearing_rad))
    return int(round(width / 2.0 - f * math.tan(b)))


class DemoView(Node):
    def __init__(self, width, quality, fps):
        super().__init__('come_here_demo_view')
        self._width, self._quality, self._gap = width, quality, 1.0 / max(1.0, fps)
        self._last_encode = 0.0
        self.slot = Slot()
        self.cam_rate, self.det_rate = Rate(), Rate()
        self._boxes, self._boxes_s = None, 0.0
        self._status, self._status_s = {}, 0.0
        self.create_subscription(Image, '/camera/image_raw', self._on_image, CAMERA_QOS)
        self.create_subscription(Float64MultiArray, '/come_here/person_boxes', self._on_boxes, 10)
        self.create_subscription(String, '/come_here/status', self._on_status, 10)
        self.get_logger().info('demo view: subscribe-only, publishing nothing')

    def _on_boxes(self, msg):
        self._boxes, self._boxes_s = list(msg.data), time.monotonic()
        self.det_rate.tick()

    def _on_status(self, msg):
        try:
            self._status, self._status_s = json.loads(msg.data), time.monotonic()
        except ValueError:
            pass

    def _on_image(self, msg):
        self.cam_rate.tick()
        now = time.monotonic()
        if now - self._last_encode < self._gap:
            return
        self._last_encode = now
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            frame = buf.reshape(msg.height, msg.width, 3)
            if msg.encoding.lower() == 'rgb8':
                frame = frame[:, :, ::-1]
        except ValueError as exc:
            self.get_logger().warn(f'frame dropped: {exc}', throttle_duration_sec=5.0)
            return
        scale = self._width / float(frame.shape[1])
        img = cv2.resize(frame, (self._width, max(1, int(frame.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
        self.draw(img, now)
        ok, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if ok:
            self.slot.put(jpg.tobytes())

    def draw(self, img, now):
        h, w = img.shape[:2]
        st = self._status if now - self._status_s < 1.5 else {}
        # Gate band: where the caller may be selected.
        gc, gh = st.get('gate_center_deg'), st.get('gate_half_deg')
        if gc is not None and gh is not None:
            overlay = img.copy()
            if gh > 0:
                xa = bearing_to_x(math.radians(gc + gh), w)
                xb = bearing_to_x(math.radians(gc - gh), w)
                cv2.rectangle(overlay, (max(0, xa), 0), (min(w - 1, xb), h), (60, 200, 60), -1)
                cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
                cv2.line(img, (bearing_to_x(math.radians(gc), w), 40),
                         (bearing_to_x(math.radians(gc), w), h), (60, 220, 60), 1)
            else:
                cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 200), -1)
                cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
                cv2.putText(img, 'GATE CLOSED (turning)', (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
        cv2.line(img, (w // 2, 40), (w // 2, h), (200, 200, 200), 1)
        # Boxes.
        b = self._boxes
        stale = b is None or now - self._boxes_s > 1.5
        if not stale and len(b) >= 6 and b[0] > 0 and b[1] > 0:
            sx, sy = w / b[0], h / b[1]
            n, sel = int(b[2]), int(b[3])
            for i in range(n):
                x1, y1, x2, y2, conf, bearing, in_gate = b[6 + 7 * i: 13 + 7 * i]
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                if i == sel:
                    color, thick = (60, 255, 60), 3
                    label = f'TARGET {conf:.2f} {math.degrees(bearing):+.0f}deg'
                elif in_gate:
                    color, thick, label = (190, 190, 190), 1, f'{conf:.2f}'
                else:
                    color, thick, label = (60, 60, 230), 1, f'out {math.degrees(bearing):+.0f}'
                cv2.rectangle(img, p1, p2, color, thick)
                cv2.putText(img, label, (p1[0] + 2, max(52, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if i == sel else 1, cv2.LINE_AA)
        elif stale:
            cv2.putText(img, 'detections STALE', (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (60, 60, 235), 2, cv2.LINE_AA)
        face = st.get('face') or {}
        if face.get('present') and face.get('center_x') is not None:
            fx = int(face['center_x'] * w)
            cv2.drawMarker(img, (fx, 60), (255, 200, 60), cv2.MARKER_TRIANGLE_DOWN, 18, 2)
        # Header.
        cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)

        def fmt(v, spec, unit=''):
            return '-' if v is None else f'{v:{spec}}{unit}'
        line1 = (f'STATE {st.get("phase", "no status")}'
                 + ('  ESTOP' if st.get('estopped') else '')
                 + f'   DOA {fmt(st.get("doa_deg"), "+.0f", "deg")} conf {fmt(st.get("doa_conf"), ".2f")}'
                 + f'   turn {fmt(st.get("turn_target_deg"), "+.0f")}/{fmt(st.get("turn_turned_deg"), "+.0f")}')
        line2 = (f'TARGET conf {fmt(st.get("target_conf"), ".2f")} '
                 f'bearing {fmt(st.get("target_bearing_deg"), "+.0f", "deg")} '
                 f'dist {fmt(st.get("target_distance_m"), ".2f", "m")} ({st.get("target_distance_source") or "-"}) '
                 f'bbox {fmt(st.get("target_bbox_h_frac"), ".2f")}   '
                 f'cam {self.cam_rate.value():.1f}Hz det {self.det_rate.value():.1f}Hz')
        cv2.putText(img, line1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, line2, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

    def status(self):
        return dict(self._status, camera_hz=round(self.cam_rate.value(), 1),
                    detections_hz=round(self.det_rate.value(), 1))


def make_handler(view):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *args):
            pass

        def _bytes(self, body, ctype):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self._bytes(PAGE, 'text/html')
            elif self.path.startswith('/status.json'):
                self._bytes(json.dumps(view.status()).encode(), 'application/json')
            elif self.path.startswith('/camera.mjpg'):
                self.send_response(200)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                seen = -1
                try:
                    while True:
                        jpeg, seq = view.slot.wait_newer(seen)
                        if seq == seen or not jpeg:
                            continue
                        seen = seq
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode())
                        self.wfile.write(jpeg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8088)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--quality', type=int, default=60)
    ap.add_argument('--fps', type=float, default=8.0)
    args = ap.parse_args()
    # rclpy's own handlers would stop only the spin thread and leave the HTTP
    # loop (and the process) running; Ctrl+C and SIGTERM end both here.
    from rclpy.executors import ExternalShutdownException
    from rclpy.signals import SignalHandlerOptions

    def _interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    view = DemoView(args.width, args.quality, args.fps)

    def _spin():
        try:
            rclpy.spin(view)
        except (ExternalShutdownException, rclpy.executors.ShutdownException):
            pass

    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    server = ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(view))
    server.daemon_threads = True
    print(f'demo view on http://0.0.0.0:{args.port}/camera.mjpg', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        rclpy.try_shutdown()          # wakes the spin thread
        spinner.join(timeout=3.0)     # let it leave the executor before teardown
        view.destroy_node()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
