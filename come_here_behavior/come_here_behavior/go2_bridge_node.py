"""Bridge node: translate behavior commands into GO2 Sport + audiohub API calls.

Subscribes:
  /come_here/cmd_rotate  (std_msgs/Float64) - target yaw angle in rad (+ = left)
  /come_here/cmd_move    (std_msgs/Float64) - forward velocity in m/s, 0 = stop
  /come_here/cmd_sit     (std_msgs/Bool)    - True triggers StandDown
  /come_here/cmd_stand   (std_msgs/Bool)    - True triggers BalanceStand
  /come_here/cmd_say     (std_msgs/String)  - phrase to vocalize via audiohub WAV

Publishes:
  /api/sport/request     (unitree_api/msg/Request) - Sport API requests
  /api/audiohub/request  (unitree_api/msg/Request) - audiohub WAV-streaming requests

Parameters:
  cmd_z                 (float, default 2.0)   - yaw rate magnitude for Move
  deg_per_sec           (float, default 90.0)  - empirical rotation speed for duration calc
  sit_api_id            (int,   default 1005)  - Sport API ID for sit (StandDown)
  stand_api_id          (int,   default 1002)  - Sport API ID for stand (BalanceStand)
  move_api_id           (int,   default 1008)  - Sport API ID for Move(x, y, z)
  stop_move_api_id      (int,   default 1003)  - Sport API ID for StopMove
  move_control_rate_hz  (float, default 10.0)  - tick rate of Move control loop
  wav_dir               (str)                  - directory containing <phrase>.wav files
  wav_chunk_size_bytes  (int,   default 16384) - audiohub base64 chunk size
  wav_chunk_delay_s     (float, default 0.15)  - sleep between audiohub chunk publishes
"""

import base64
import datetime
import json
import os
import random
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String
from unitree_api.msg import Request


def make_req(api_id, params=None):
    """Build a unitree_api Request for Sport or audiohub, matching the demo helper."""
    msg = Request()
    msg.header.identity.api_id = api_id
    msg.header.identity.id = (
        int(datetime.datetime.now().timestamp() * 1000 % 2147483648)
        + random.randint(0, 999)
    )
    msg.header.lease.id = 0
    msg.header.policy.priority = 0
    msg.header.policy.noreply = False
    if params is not None:
        msg.parameter = json.dumps(params) if isinstance(params, dict) else str(params)
    else:
        msg.parameter = ''
    msg.binary = []
    return msg


class Go2BridgeNode(Node):
    def __init__(self):
        super().__init__('go2_bridge_node')

        # -- Parameters --
        self.declare_parameter('cmd_z', 2.0)
        self.declare_parameter('deg_per_sec', 90.0)
        self.declare_parameter('sit_api_id', 1005)
        self.declare_parameter('stand_api_id', 1002)
        self.declare_parameter('move_api_id', 1008)
        self.declare_parameter('stop_move_api_id', 1003)
        self.declare_parameter('move_control_rate_hz', 10.0)
        self.declare_parameter(
            'wav_dir', '/home/unitree/come-here/come_here_audio/scripts'
        )
        self.declare_parameter('wav_chunk_size_bytes', 16384)
        self.declare_parameter('wav_chunk_delay_s', 0.15)

        self._cmd_z: float = float(self.get_parameter('cmd_z').value)
        self._deg_per_sec: float = float(self.get_parameter('deg_per_sec').value)
        self._sit_api_id: int = int(self.get_parameter('sit_api_id').value)
        self._stand_api_id: int = int(self.get_parameter('stand_api_id').value)
        self._move_api_id: int = int(self.get_parameter('move_api_id').value)
        self._stop_move_api_id: int = int(self.get_parameter('stop_move_api_id').value)
        self._move_rate_hz: float = float(
            self.get_parameter('move_control_rate_hz').value
        )
        self._wav_dir: str = str(self.get_parameter('wav_dir').value)
        self._wav_chunk_size: int = int(self.get_parameter('wav_chunk_size_bytes').value)
        self._wav_chunk_delay_s: float = float(
            self.get_parameter('wav_chunk_delay_s').value
        )

        # -- Publishers --
        self._sport_pub = self.create_publisher(Request, '/api/sport/request', 10)
        self._audio_pub = self.create_publisher(Request, '/api/audiohub/request', 10)

        # -- Runtime state --
        # Forward-velocity state
        self._current_vx: float = 0.0
        self._last_was_zero: bool = True  # suppress the first StopMove at startup
        self._vx_lock = threading.Lock()

        # Rotation thread control: incrementing generation invalidates older rotations.
        self._rotate_generation: int = 0
        self._rotate_lock = threading.Lock()
        self._rotate_cancel = threading.Event()
        self._rotate_thread = None

        # Audio thread: drop new cmd_say if a previous playback is still streaming.
        self._audio_busy = threading.Event()

        # -- Subscribers --
        self.create_subscription(
            Float64, '/come_here/cmd_rotate', self._rotate_cb, 10
        )
        self.create_subscription(
            Float64, '/come_here/cmd_move', self._move_cb, 10
        )
        self.create_subscription(
            Bool, '/come_here/cmd_sit', self._sit_cb, 10
        )
        self.create_subscription(
            Bool, '/come_here/cmd_stand', self._stand_cb, 10
        )
        self.create_subscription(
            String, '/come_here/cmd_say', self._say_cb, 10
        )

        # Forward-velocity control timer
        self._move_timer = self.create_timer(
            1.0 / self._move_rate_hz, self._move_tick
        )

        self.get_logger().info(
            f'go2_bridge_node started (cmd_z={self._cmd_z}, '
            f'deg_per_sec={self._deg_per_sec}, wav_dir={self._wav_dir})'
        )

    # -- cmd_rotate --

    def _rotate_cb(self, msg: Float64) -> None:
        target_rad: float = float(msg.data)
        target_deg: float = abs(target_rad * 180.0 / 3.141592653589793)
        duration: float = max(0.3, min(target_deg / self._deg_per_sec, 4.0))
        sign: float = 1.0 if target_rad > 0 else -1.0

        # Cancel any in-flight rotation, bump generation, start a new thread.
        with self._rotate_lock:
            self._rotate_generation += 1
            my_gen: int = self._rotate_generation
            self._rotate_cancel.set()
            # Fresh event for the new rotation thread
            self._rotate_cancel = threading.Event()
            cancel_event = self._rotate_cancel

            thread = threading.Thread(
                target=self._rotate_worker,
                args=(my_gen, sign, duration, target_rad, cancel_event),
                daemon=True,
            )
            self._rotate_thread = thread
            thread.start()

        self.get_logger().info(
            f'cmd_rotate: {target_rad:+.2f} rad ({sign * target_deg:+.1f}°), '
            f'dur={duration:.2f}s, gen={my_gen}'
        )

    def _rotate_worker(
        self,
        generation: int,
        sign: float,
        duration: float,
        target_rad: float,
        cancel_event: threading.Event,
    ) -> None:
        """Publish Move at ~20 Hz for `duration`, then StopMove. Abort on cancel."""
        publish_dt: float = 0.05  # 20 Hz
        start: float = time.time()
        z_cmd: float = self._cmd_z * sign
        params = {'x': 0.0, 'y': 0.0, 'z': z_cmd}

        while (time.time() - start) < duration:
            if cancel_event.is_set():
                # A newer rotation preempted us — do NOT publish StopMove,
                # the newer worker is already driving the robot.
                self.get_logger().info(
                    f'rotate gen={generation} preempted at '
                    f'{time.time() - start:.2f}s'
                )
                return
            self._sport_pub.publish(make_req(self._move_api_id, params))
            time.sleep(publish_dt)

        # Only the still-current generation should emit StopMove.
        with self._rotate_lock:
            if generation != self._rotate_generation:
                return
        self._sport_pub.publish(make_req(self._stop_move_api_id))
        elapsed: float = time.time() - start
        self.get_logger().info(
            f'rotate gen={generation} done: {target_rad:+.2f} rad in {elapsed:.2f}s'
        )

    # -- cmd_move + timer --

    def _move_cb(self, msg: Float64) -> None:
        with self._vx_lock:
            self._current_vx = float(msg.data)

    def _move_tick(self) -> None:
        with self._vx_lock:
            vx: float = self._current_vx
        if abs(vx) > 1e-3:
            self._sport_pub.publish(
                make_req(self._move_api_id, {'x': vx, 'y': 0.0, 'z': 0.0})
            )
            self._last_was_zero = False
        else:
            # Publish StopMove exactly once per zero-transition.
            if not self._last_was_zero:
                self._sport_pub.publish(make_req(self._stop_move_api_id))
                self._last_was_zero = True

    # -- cmd_sit / cmd_stand --

    def _sit_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.get_logger().info(f'cmd_sit: api_id={self._sit_api_id}')
        self._sport_pub.publish(make_req(self._sit_api_id))

    def _stand_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.get_logger().info(f'cmd_stand: api_id={self._stand_api_id}')
        self._sport_pub.publish(make_req(self._stand_api_id))

    # -- cmd_say --

    def _say_cb(self, msg: String) -> None:
        phrase: str = msg.data or ''
        if not phrase:
            return
        filename: str = phrase.strip().lower().replace(' ', '_') + '.wav'
        wav_path: str = os.path.join(self._wav_dir, filename)

        if not os.path.isfile(wav_path):
            self.get_logger().warn(
                f'cmd_say: WAV not found for "{phrase}" at {wav_path}, dropping'
            )
            return

        if self._audio_busy.is_set():
            self.get_logger().warn(
                f'cmd_say: previous playback still in flight, dropping "{phrase}"'
            )
            return

        self._audio_busy.set()
        thread = threading.Thread(
            target=self._play_wav_worker, args=(wav_path, phrase), daemon=True
        )
        thread.start()

    def _play_wav_worker(self, wav_path: str, phrase: str) -> None:
        """Stream a WAV file to the GO2 speaker via audiohub (start/chunk/end)."""
        try:
            with open(wav_path, 'rb') as f:
                wav_data: bytes = f.read()
            b64: str = base64.b64encode(wav_data).decode('utf-8')
            chunk_size: int = self._wav_chunk_size
            chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]

            self.get_logger().info(
                f'cmd_say: streaming "{phrase}" ({len(wav_data)} bytes, '
                f'{len(chunks)} chunks)'
            )

            # Start session
            self._audio_pub.publish(make_req(4001))
            time.sleep(0.1)

            # Chunks
            for i, chunk in enumerate(chunks):
                payload = {
                    'current_block_index': i + 1,
                    'total_block_number': len(chunks),
                    'block_content': chunk,
                }
                self._audio_pub.publish(make_req(4003, payload))
                time.sleep(self._wav_chunk_delay_s)

            # Let last chunk play out, then end session
            time.sleep(1.5)
            self._audio_pub.publish(make_req(4002))
        except Exception as exc:
            self.get_logger().error(f'cmd_say: playback error for "{phrase}": {exc}')
        finally:
            self._audio_busy.clear()


def main(args=None):
    rclpy.init(args=args)
    node = Go2BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
