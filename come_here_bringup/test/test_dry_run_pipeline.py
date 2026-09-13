"""End-to-end dry run of the class demo with the real behavior_node and go2_bridge_node.

Both nodes run as separate processes with professor_demo.yaml. The bridge is in
dry run, so Sport requests go to /come_here/dry_run/sport_request, and the test
asserts on exactly what the robot would have been sent:

  * a centered caller: WALK Moves are forward-only, a StopMove follows
    arrival and no Move comes after it, and a trial record is written
  * a lost caller: StopMove within the debounce, no Move while lost
  * e-stop during an approach: StopMove, and no Move afterwards
  * nothing is ever published on the real /api/sport/request

The mcf check is switched off here (no robot to answer CheckMode). Needs the
built workspace and unitree_api; skipped otherwise. DDS is isolated with its
own domain id and ROS_LOCALHOST_ONLY=1.
"""

import json
import os
import pathlib
import signal
import subprocess
import time

import pytest

try:
    import rclpy
    from ament_index_python.packages import PackageNotFoundError, get_package_prefix
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Bool, Float64MultiArray, String
    from unitree_api.msg import Request
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason='needs ROS 2 and unitree_api')

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / 'come_here_bringup' / 'config' / 'professor_demo.yaml'
DOMAIN_ID = 40 + os.getpid() % 20
MOVE_API_ID = 1008
STOP_MOVE_API_ID = 1003


def _executable(name):
    try:
        prefix = get_package_prefix('come_here_behavior')
    except PackageNotFoundError:
        return None
    path = os.path.join(prefix, 'lib', 'come_here_behavior', name)
    return path if os.access(path, os.X_OK) else None


class Harness:
    def __init__(self):
        self.ctx = Context()
        rclpy.init(context=self.ctx, domain_id=DOMAIN_ID)
        self.node = rclpy.create_node('dry_run_harness', context=self.ctx)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        self.dry = []      # (time, api_id, parameter)
        self.real = []
        self.states = []
        self.trials = []
        n = self.node
        n.create_subscription(Request, '/come_here/dry_run/sport_request', self._on_dry, 100)
        n.create_subscription(Request, '/api/sport/request',
                              lambda m: self.real.append(m.header.identity.api_id), 100)
        n.create_subscription(String, '/come_here/state', self._on_state, 100)
        n.create_subscription(String, '/come_here/trial_summary',
                              lambda m: self.trials.append(json.loads(m.data)), 10)
        self.wake_pub = n.create_publisher(String, '/come_here/wake_phrase', 10)
        self.person_pub = n.create_publisher(Float64MultiArray, '/come_here/person_detection', 10)
        self.estop_pub = n.create_publisher(Bool, '/come_here/estop', 10)

    def _on_dry(self, msg):
        self.dry.append((time.monotonic(), msg.header.identity.api_id, msg.parameter))

    def _on_state(self, msg):
        if not self.states or self.states[-1] != msg.data:
            self.states.append(msg.data)

    def spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.01)

    def spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            self.executor.spin_once(timeout_sec=0.02)
        return predicate()

    def person(self, bearing=0.0, bbox=0.55, detected=True):
        msg = Float64MultiArray()
        if detected:
            msg.data = [bearing, 3.0, 0.9, 1.0, bbox, 2.0, 0.1]
        else:
            msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        self.person_pub.publish(msg)

    def wake(self):
        self.wake_pub.publish(String(data='come here'))

    def moves(self, since=0.0):
        return [(t, json.loads(p)) for t, a, p in self.dry if a == MOVE_API_ID and t >= since]

    def stops(self, since=0.0):
        return [t for t, a, _p in self.dry if a == STOP_MOVE_API_ID and t >= since]

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.try_shutdown(context=self.ctx)


@pytest.fixture
def stack(tmp_path, monkeypatch):
    behavior, bridge = _executable('behavior_node'), _executable('go2_bridge_node')
    if behavior is None or bridge is None:
        pytest.skip('come_here_behavior not installed; build the workspace first')
    monkeypatch.setenv('ROS_DOMAIN_ID', str(DOMAIN_ID))
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    params = ['--ros-args', '--params-file', str(CONFIG)]
    procs = [
        subprocess.Popen([behavior] + params + ['-p', f'trial_log_dir:={tmp_path}'],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
        subprocess.Popen([bridge] + params + ['-p', 'dry_run:=true',
                                              '-p', "require_motion_mode:=''",
                                              '-p', f'wav_dir:={tmp_path}'],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
    ]
    harness = Harness()
    try:
        ready = harness.spin_until(
            lambda: (harness.node.count_subscribers('/come_here/person_detection') >= 1
                     and harness.node.count_publishers('/come_here/dry_run/sport_request') >= 1
                     and bool(harness.states)),
            20.0,
        )
        assert ready, 'behavior_node and go2_bridge_node did not come up'
        harness.spin(2.0)  # let behavior_node and the bridge discover each other
        yield harness, tmp_path
    finally:
        harness.close()
        for proc in procs:
            proc.send_signal(signal.SIGINT)
        for proc in procs:
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()


def _approach(h, bearing=0.02, bbox_start=0.55, arrive=True, timeout=12.0):
    """Stream detections at 5 Hz; the bbox grows only while the robot walks.

    Returns once ARRIVED (arrive=True) or once this call has seen a Move
    (arrive=False). Only Moves sent after the call started count.
    """
    started = time.monotonic()
    states_before = len(h.states)
    bbox = bbox_start
    deadline = started + timeout
    while time.monotonic() < deadline:
        walking = bool(h.moves(since=started))
        if arrive and walking:
            bbox = min(0.8, bbox + 0.03)
        h.person(bearing, bbox)
        h.spin(0.2)
        if arrive and 'ARRIVED' in h.states[states_before:]:
            return True
        if not arrive and walking:
            return True
    return False


def test_centered_caller_lost_caller_and_estop(stack):
    h, log_dir = stack

    # 1. Centered caller: wake -> ACQUIRE_PERSON -> WALK -> ARRIVED -> IDLE.
    h.wake()
    assert _approach(h), f'never arrived; states={h.states}'
    arrived_at = time.monotonic()
    assert h.spin_until(lambda: h.states[-1] == 'IDLE', 5.0), h.states
    moves = h.moves()
    assert moves, 'no Move was sent'
    assert all(not (m['x'] != 0.0 and m['z'] != 0.0) for _t, m in moves), 'combined vx+yaw'
    assert all(m['x'] == 0.6 for _t, m in moves if m['x'] != 0.0)
    last_move = moves[-1][0]
    assert any(t > last_move for t in h.stops()), 'no StopMove after the last Move'
    assert h.moves(since=arrived_at + 0.3) == [], 'Move after arrival'
    assert h.spin_until(lambda: h.trials, 3.0)
    assert h.trials[0]['stop_reason'] == 'arrived_bbox'
    assert h.trials[0]['success'] is True
    assert h.trials[0]['combined_commands'] == 0
    assert len((log_dir / 'trials.jsonl').read_text().splitlines()) == 1

    # 2. Lost caller mid-walk: StopMove within the 0.3 s debounce (+ tick and DDS).
    h.wake()
    assert _approach(h, bbox_start=0.5, arrive=False), f'no walk on trial 2; states={h.states}'
    first_miss = time.monotonic()
    while time.monotonic() - first_miss < 2.0:
        h.person(detected=False)
        h.spin(0.2)
    stops = h.stops(since=first_miss)
    assert stops and stops[0] - first_miss <= 1.0, f'stop latency {stops[:1]}'
    assert h.moves(since=stops[0] + 0.1) == [], 'Move while the caller was lost'
    assert h.states[-1] == 'ACQUIRE_PERSON'

    # 3. Caller reacquired, walking again, then the operator e-stop.
    assert _approach(h, bbox_start=0.5, arrive=False, timeout=6.0), 'no walk after reacquire'
    h.spin(0.5)
    estop_at = time.monotonic()
    h.estop_pub.publish(Bool(data=True))
    assert h.spin_until(lambda: h.stops(since=estop_at), 2.0), 'no StopMove after e-stop'
    blocked_from = time.monotonic()
    while time.monotonic() - blocked_from < 1.5:
        h.person(bbox=0.5)
        h.spin(0.2)
    assert h.moves(since=blocked_from) == [], 'Move after e-stop'
    assert h.spin_until(lambda: len(h.trials) >= 2, 3.0)
    assert h.trials[1]['stop_reason'] == 'estop'
    assert h.trials[1]['lost_events'] == 1

    assert h.real == [], 'dry run published on the real /api/sport/request'
