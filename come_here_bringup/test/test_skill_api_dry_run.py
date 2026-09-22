"""Skill interface v2 end to end: real behavior_node (enable_skill_api:=true) and
go2_bridge_node in dry run, with professor_demo.yaml, as separate processes.

Asserts on the Sport requests the robot would have been sent:
  * localize_caller -> orient_to_caller -> acquire_caller -> approach_caller -> sit:
    one yaw-only rotation, forward-only Moves, StopMove after arrival, NO Sit before the
    approach result, then exactly one Sit, only after the sit request
  * cancel mid-walk: StopMove within 0.5 s, no Move after it, no e-stop latched
  * stale handoffs, reused request ids and out-of-order requests: rejected, no motion
  * a wake phrase is ignored with the skill interface on
  * nothing is ever published on the real /api/sport/request
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_dry_run_pipeline import (  # noqa: E402
    _AVAILABLE, CONFIG, DOMAIN_ID, Harness, _executable,
)

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason='needs ROS 2 and unitree_api')
SIT_API_ID = 1009

if _AVAILABLE:
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray, String


class SkillHarness(Harness):
    def __init__(self):
        super().__init__()
        self.results = []
        self.request_pub = self.node.create_publisher(String, '/come_here/skill_request', 10)
        self.direction_pub = self.node.create_publisher(
            Float64MultiArray, '/come_here/audio_direction', 10)
        self.node.create_subscription(
            String, '/come_here/skill_result', lambda m: self.results.append(json.loads(m.data)),
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def request(self, rid, skill, goal='g1', seq=None, **args):
        self.seq = getattr(self, 'seq', 0) + 1 if seq is None else seq
        self.request_pub.publish(String(data=json.dumps(
            {'v': 2, 'goal_id': goal, 'request_id': rid, 'seq': self.seq, 'skill': skill,
             'args': args})))
        return rid

    def result(self, rid, timeout=5.0):
        self.spin_until(lambda: any(r['request_id'] == rid for r in self.results), timeout)
        found = [r for r in self.results if r['request_id'] == rid]
        assert len(found) == 1, f'{rid}: {found}'
        return found[0]


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
        subprocess.Popen([behavior] + params + ['-p', f'trial_log_dir:={tmp_path}',
                                                '-p', 'enable_skill_api:=true',
                                                '-p', 'walk_budget_arrives:=false',
                                                '-p', 'max_walk_distance_m:=3.0'],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
        subprocess.Popen([bridge] + params + ['-p', 'dry_run:=true',
                                              '-p', "require_motion_mode:=''",
                                              '-p', f'wav_dir:={tmp_path}'],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
    ]
    h = SkillHarness()
    try:
        ready = h.spin_until(
            lambda: (h.node.count_subscribers('/come_here/skill_request') >= 1
                     and h.node.count_publishers('/come_here/skill_result') >= 1
                     and h.node.count_publishers('/come_here/dry_run/sport_request') >= 1
                     and bool(h.states)),
            20.0)
        assert ready, 'behavior_node (skill API on) and go2_bridge_node did not come up'
        h.spin(2.0)
        yield h
    finally:
        h.close()
        for proc in procs:
            proc.send_signal(signal.SIGINT)
        for proc in procs:
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()


def _stream(h, seconds, bearing=0.02, bbox=0.55, grow=False, until=None):
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        if grow and h.moves(since=started):
            bbox = min(0.9, bbox + 0.03)
        h.person(bearing, bbox)
        h.spin(0.2)
        if until is not None and until():
            return True
    return until() if until is not None else True


def _sits(h, since=0.0):
    return [t for t, a, _p in h.dry if a == SIT_API_ID and t >= since]


def test_decomposed_sequence_cancel_and_rejections_over_dds(stack):
    h = stack
    wake_pub = h.node.create_publisher(String, '/come_here/wake_phrase', 10)
    t0 = time.monotonic()

    # 0. A wake phrase does nothing with the skill interface on.
    wake_pub.publish(String(data='come here'))
    h.spin(1.0)
    assert h.states[-1] == 'IDLE' and h.moves(since=t0) == []

    # 1. localize: a fresh confident bearing, no motion.
    h.direction_pub.publish(Float64MultiArray(data=[1.0, 0.9]))
    h.spin(0.2)
    h.request('L1', 'localize_caller')
    r = h.result('L1')
    assert r['status'] == 'succeeded' and r['data']['bearing_rad'] == pytest.approx(1.0)
    assert h.moves(since=t0) == []

    # 2. orient: one yaw-only rotation by the localized bearing, then stopped.
    turn_at = time.monotonic()
    h.request('O1', 'orient_to_caller', localization='L1')
    r = h.result('O1', timeout=15.0)
    assert r['status'] == 'succeeded' and r['data']['turn_rad'] == pytest.approx(1.0), r
    turn_moves = h.moves(since=turn_at)
    assert turn_moves and all(m['x'] == 0.0 and m['z'] > 0.0 for _t, m in turn_moves)
    assert any(t > turn_moves[-1][0] for t in h.stops(since=turn_at))

    # 3. acquire: the caller in view, no motion.
    acq_at = time.monotonic()
    h.request('A1', 'acquire_caller')
    assert _stream(h, 5.0, bbox=0.55, until=lambda: any(
        x['request_id'] == 'A1' for x in h.results))
    r = h.result('A1')
    assert r['status'] == 'succeeded' and r['data']['confidence'] > 0.5, r
    assert h.moves(since=acq_at) == []

    # 4. approach: forward-only, arrives, StopMove, NO Sit.
    app_at = time.monotonic()
    h.request('P1', 'approach_caller', acquisition='A1')
    assert _stream(h, 15.0, grow=True,
                   until=lambda: any(x['request_id'] == 'P1' for x in h.results)), h.states
    r = h.result('P1')
    arrived_at = time.monotonic()
    assert r['status'] == 'succeeded' and r['reason'] == 'arrived_bbox', r
    assert r['data']['posture'] == 'standing' and r['run_id']
    moves = h.moves(since=app_at)
    assert moves and all(m['z'] == 0.0 and m['x'] >= 0.0 for _t, m in moves)
    assert any(t > moves[-1][0] for t in h.stops(since=app_at)), 'no StopMove after the last Move'
    assert _sits(h) == [], 'approach_caller sent a Sit'
    h.spin(2.0)                                         # standing still, waiting
    assert _sits(h) == [] and h.moves(since=arrived_at) == []

    # 5. sit: exactly one Sit, after the sit request, never before.
    sit_at = time.monotonic()
    h.request('S1', 'sit', arrival='P1')
    r = h.result('S1', timeout=10.0)
    assert r['status'] == 'succeeded' and r['reason'] == 'sit_commanded', r
    assert h.spin_until(lambda: _sits(h), 3.0)
    assert len(_sits(h)) == 1 and _sits(h)[0] > sit_at
    assert h.moves(since=sit_at) == []

    # 6. seated: motion requests refused; operator reset stands the robot up.
    h.request('L2', 'localize_caller')
    assert h.result('L2')['reason'] == 'not_idle'
    reset_pub = h.node.create_publisher(__import__('std_msgs.msg', fromlist=['Bool']).Bool,
                                        '/come_here/reset', 10)
    h.spin(0.3)
    reset_pub.publish(__import__('std_msgs.msg', fromlist=['Bool']).Bool(data=True))
    assert h.spin_until(lambda: h.states[-1] == 'IDLE', 5.0), h.states[-3:]

    # 7. cancel mid-walk (new goal): StopMove promptly, no Move after, no e-stop.
    h.request('A2', 'acquire_caller', goal='g2', seq=1)
    assert _stream(h, 5.0, bbox=0.4, until=lambda: any(
        x['request_id'] == 'A2' for x in h.results))
    assert h.result('A2')['status'] == 'succeeded'
    h.request('P2', 'approach_caller', goal='g2', seq=2, acquisition='A2')
    assert _stream(h, 8.0, bbox=0.4, until=lambda: bool(h.moves(since=time.monotonic() - 0.3)))
    cancel_at = time.monotonic()
    h.request('C2', 'cancel', goal='g2', seq=3, request_id='P2')
    assert h.spin_until(lambda: h.stops(since=cancel_at), 2.0), 'no StopMove after cancel'
    assert h.stops(since=cancel_at)[0] - cancel_at <= 0.5
    _stream(h, 1.5, bbox=0.4)
    assert h.moves(since=cancel_at + 0.3) == [], 'Move after cancel'
    assert h.result('P2')['status'] == 'cancelled'
    assert h.result('C2')['status'] == 'succeeded'
    assert h.spin_until(lambda: h.states[-1] == 'IDLE', 1.0), h.states[-3:]

    # 8. stale / reused / out-of-order requests: refused without motion.
    quiet_from = time.monotonic()
    h.request('X1', 'sit', goal='g2', seq=4, arrival='P1')         # goal g1's arrival
    h.request('P2', 'approach_caller', goal='g2', seq=5, acquisition='A2')   # reused id
    h.request('X3', 'localize_caller', goal='g2', seq=2)           # out of order
    h.request('X4', 'localize_caller', goal='g1', seq=99)          # retired goal
    h.spin(1.0)
    assert h.result('X1')['reason'] == 'no_arrival'
    assert [x['reason'] for x in h.results
            if x['request_id'] == 'P2' and x['status'] == 'rejected'] == ['duplicate_request_id']
    assert h.result('X3')['reason'] == 'out_of_order'
    assert h.result('X4')['reason'] == 'stale_goal'
    assert h.moves(since=quiet_from) == [] and _sits(h, since=quiet_from) == []

    assert len(_sits(h)) == 1
    assert h.real == [], 'dry run published on the real /api/sport/request'
