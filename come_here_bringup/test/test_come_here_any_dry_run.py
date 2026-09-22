"""Multi-process dry run of Come Here ANY with the real ANY behavior and bridge nodes.

Both nodes run as separate processes with come_here_any.yaml, the bridge in dry
run: requests go to /come_here/dry_run/{sport,obstacles_avoid}_request and the
native enable replies are SIMULATED. The test asserts on exactly what the robot
would have been sent. It proves routing, ownership and sequencing through the
native backend. It proves NOTHING about how the GO2 avoids obstacles.

  * startup: the native backend enables (FreeAvoid 2048 / obstacles_avoid sequence)
  * centered caller: Moves only through the selected backend, single-axis,
    StopMove after arrival, verified final facing, Sit, trial record with the
    native fields and mode 'any'
  * e-stop: stop, and obstacles_avoid API control released
  * shutdown (SIGINT): the release sequence goes out
  * nothing ever reaches the real /api/sport/request or /api/obstacles_avoid/request

Needs the built workspace and unitree_api; skipped otherwise.
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
    from nav_msgs.msg import Odometry
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Bool, Float64MultiArray, String
    from unitree_api.msg import Request
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason='needs ROS 2 and unitree_api')

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / 'come_here_bringup' / 'config' / 'come_here_any.yaml'
DOMAIN_ID = 60 + os.getpid() % 20


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
        n = self.node = rclpy.create_node('any_dry_run_harness', context=self.ctx)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(n)
        self.sport, self.oa, self.real = [], [], []
        self.states, self.trials, self.status = [], [], []
        n.create_subscription(Request, '/come_here/dry_run/sport_request',
                              lambda m: self.sport.append(self._rec(m)), 200)
        n.create_subscription(Request, '/come_here/dry_run/obstacles_avoid_request',
                              lambda m: self.oa.append(self._rec(m)), 200)
        for topic in ('/api/sport/request', '/api/obstacles_avoid/request'):
            n.create_subscription(Request, topic,
                                  lambda m: self.real.append(m.header.identity.api_id), 100)
        n.create_subscription(String, '/come_here/state', self._on_state, 100)
        n.create_subscription(String, '/come_here/trial_summary',
                              lambda m: self.trials.append(json.loads(m.data)), 10)
        n.create_subscription(String, '/come_here/bridge_status',
                              lambda m: self.status.append(json.loads(m.data)), 10)
        self.wake_pub = n.create_publisher(String, '/come_here/wake_phrase', 10)
        self.person_pub = n.create_publisher(Float64MultiArray, '/come_here/person_detection', 10)
        self.estop_pub = n.create_publisher(Bool, '/come_here/estop', 10)
        self.odom_pub = n.create_publisher(Odometry, '/utlidar/robot_odom', 10)
        n.create_timer(0.05, self._odom)

    @staticmethod
    def _rec(m):
        return (time.monotonic(), m.header.identity.api_id,
                json.loads(m.parameter) if m.parameter else None)

    def _odom(self):
        msg = Odometry()
        msg.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(msg)

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
        msg.data = ([bearing, 3.0, 0.9, 1.0, bbox, 2.0, 0.1] if detected
                    else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
        self.person_pub.publish(msg)

    def moves(self, since=0.0):
        """Every motion request on the selected backend's dry-run topic."""
        return ([(t, p) for t, a, p in self.sport if a == 1008 and t >= since]
                + [(t, p) for t, a, p in self.oa if a == 1003 and t >= since])

    def nonzero_moves(self, since=0.0):
        return [(t, p) for t, p in self.moves(since)
                if any(abs(p.get(k, 0.0)) > 0 for k in ('x', 'y', 'z', 'yaw'))]

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.try_shutdown(context=self.ctx)


@pytest.fixture(params=['sport_freeavoid', 'obstacles_avoid'])
def stack(request, tmp_path, monkeypatch):
    behavior = _executable('come_here_any_behavior_node')
    bridge = _executable('native_avoid_bridge_node')
    if behavior is None or bridge is None:
        pytest.skip('come_here_behavior (ANY executables) not installed; build the workspace')
    monkeypatch.setenv('ROS_DOMAIN_ID', str(DOMAIN_ID))
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    params = ['--ros-args', '--params-file', str(CONFIG)]
    procs = [
        subprocess.Popen([behavior] + params + [
            '-p', f'trial_log_dir:={tmp_path}', '-p', 'skip_turn_to_sound:=true',
            '-p', 'face_timeout_s:=0.2', '-p', 'sit_settle_s:=0.5'],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
        subprocess.Popen([bridge] + params + [
            '-p', 'dry_run:=true', '-p', "require_motion_mode:=''",
            '-p', f'wav_dir:={tmp_path}', '-p', f'native_avoid_backend:={request.param}'],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
    ]
    h = Harness()
    try:
        ready = h.spin_until(
            lambda: (h.node.count_subscribers('/come_here/person_detection') >= 1
                     and h.node.count_publishers('/come_here/dry_run/sport_request') >= 1
                     and bool(h.states) and bool(h.status)
                     and h.status[-1].get('native_avoid_enabled') is True),
            25.0)
        assert ready, f'ANY nodes did not come up; status={h.status[-1:]}'
        h.spin(1.5)
        yield h, tmp_path, request.param, procs
    finally:
        h.close()
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in procs:
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()


def _approach(h, bearing=0.02, timeout=15.0):
    """5 Hz detections; the box grows only while the robot is commanded forward."""
    started = time.monotonic()
    bbox = 0.55
    while time.monotonic() < started + timeout:
        if h.nonzero_moves(since=started):
            bbox = min(0.9, bbox + 0.03)
        h.person(bearing, bbox)
        h.spin(0.2)
        if any(t > started for t, a, _p in h.sport if a == 1009):
            return True
    return False


def test_any_dry_run_walks_through_the_native_backend_and_sits_facing(stack):
    h, log_dir, backend, procs = stack
    status = h.status[-1]
    assert status['mode'] == 'any' and status['native_avoid_backend'] == backend
    assert status['native_avoid_enable_result'] == 'dry_run_simulated'
    if backend == 'sport_freeavoid':
        assert (2048, {'data': True}) in [(a, p) for _t, a, p in h.sport]
    else:
        oa = [(a, p) for _t, a, p in h.oa]
        assert (1001, {'enable': True}) in oa
        assert (1004, {'is_remote_commands_from_api': True}) in oa

    h.wake_pub.publish(String(data='come here'))
    assert _approach(h), f'never sat; states={h.states}'
    moves = h.nonzero_moves()
    assert moves, 'no motion request was sent'
    if backend == 'sport_freeavoid':
        assert all(p == {'x': 0.6, 'y': 0.0, 'z': 0.0} for _t, p in moves)
        assert not [p for _t, a, p in h.oa if a == 1003]
    else:
        assert all(p == {'x': 0.6, 'y': 0.0, 'yaw': 0.0, 'mode': 0} for _t, p in moves)
        assert not [p for _t, a, p in h.sport if a == 1008]
    last_move = moves[-1][0]
    assert any(t > last_move for t, a, _p in h.sport if a == 1003), 'no StopMove after motion'
    assert h.spin_until(lambda: h.trials, 6.0), f'no trial record; states={h.states}'
    trial = h.trials[0]
    assert trial['mode'] == 'any' and trial['success'] is True
    assert trial['final_align_verified'] is True
    assert trial['stop_reason'] == 'arrived_bbox'
    assert trial['native_avoid_backend'] == backend
    assert trial['native_avoid_simulated'] is True
    assert trial['native_avoid_enabled_at_wake'] is True
    assert trial['combined_commands'] == 0 and trial['lateral_commands'] == 0
    log = log_dir / 'trials.jsonl'
    # The record is published before it is appended to the file.
    assert h.spin_until(lambda: log.exists() and log.read_text().strip(), 3.0)
    assert len(log.read_text().splitlines()) == 1
    assert h.real == [], 'dry run published on a real robot request topic'


def test_any_dry_run_estop_and_shutdown_release(stack):
    h, _log_dir, backend, procs = stack
    h.wake_pub.publish(String(data='come here'))
    started = time.monotonic()
    while time.monotonic() - started < 8.0 and not h.nonzero_moves(since=started):
        h.person(0.02, 0.55)
        h.spin(0.2)
    assert h.nonzero_moves(since=started), f'no motion; states={h.states}'
    estop_at = time.monotonic()
    h.estop_pub.publish(Bool(data=True))
    assert h.spin_until(lambda: any(t > estop_at and a == 1003 for t, a, _p in h.sport), 3.0)
    blocked = time.monotonic()
    while time.monotonic() - blocked < 1.5:
        h.person(0.02, 0.55)
        h.spin(0.2)
    assert h.nonzero_moves(since=blocked) == [], 'motion after the e-stop'
    if backend == 'obstacles_avoid':
        assert any(t > estop_at and a == 1004 and p == {'is_remote_commands_from_api': False}
                   for t, a, p in h.oa), 'API control not released on e-stop'

    shutdown_at = time.monotonic()
    procs[1].send_signal(signal.SIGINT)
    assert h.spin_until(lambda: procs[1].poll() is not None, 10.0)
    h.spin(0.5)
    if backend == 'sport_freeavoid':
        assert any(t > shutdown_at and a == 2048 and p == {'data': False}
                   for t, a, p in h.sport), 'FreeAvoid not disabled at shutdown'
    else:
        assert any(t > shutdown_at and a == 1001 and p == {'enable': False}
                   for t, a, p in h.oa), 'obstacles_avoid switch not restored at shutdown'
    assert any(t > shutdown_at and a == 1003 for t, a, _p in h.sport), 'no final StopMove'
    assert h.real == []
