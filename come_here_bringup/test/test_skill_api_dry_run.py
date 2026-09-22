"""Skill interface end to end: real behavior_node (enable_skill_api:=true) and
go2_bridge_node in dry run, with professor_demo.yaml, as separate processes.

Asserts on the Sport requests the robot would have been sent:
  * approach_person{stop}: forward-only Moves, StopMove after arrival, no Sit, result
    succeeded with the request's goal_id
  * cancel mid-walk: StopMove within 0.5 s, no Move after it, no e-stop latched, and the
    next request is accepted
  * turn_to_voice: one rotation (yaw-only Moves), stopped, result reported
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

    def request(self, gid, skill, **args):
        self.request_pub.publish(String(data=json.dumps(
            {'v': 1, 'goal_id': gid, 'skill': skill, 'args': args})))

    def result(self, goal_id, timeout=5.0):
        self.spin_until(lambda: any(r['goal_id'] == goal_id for r in self.results), timeout)
        found = [r for r in self.results if r['goal_id'] == goal_id]
        assert len(found) == 1, f'{goal_id}: {found}'
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


def test_approach_cancel_and_turn_over_dds(stack):
    h = stack

    # 1. approach_person{stop}: arrive, stop, never sit.
    t0 = time.monotonic()
    h.request('a1', 'approach_person', arrival='stop')
    assert _stream(h, 15.0, grow=True,
                   until=lambda: any(r['goal_id'] == 'a1' for r in h.results)), h.states
    r = h.result('a1')
    assert r['status'] == 'succeeded' and r['reason'] == 'arrived_bbox', r
    assert r['run_id']
    moves = h.moves(since=t0)
    assert moves and all(m['z'] == 0.0 and m['x'] >= 0.0 for _t, m in moves)
    assert any(t > moves[-1][0] for t in h.stops(since=t0)), 'no StopMove after the last Move'
    assert all(a != SIT_API_ID for _t, a, _p in h.dry), 'Sit sent for a stop arrival'
    assert h.spin_until(lambda: h.states[-1] == 'IDLE', 5.0)

    # 2. cancel mid-walk: StopMove promptly, no Move after, no e-stop.
    h.request('a2', 'approach_person', arrival='stop')
    assert _stream(h, 8.0, bbox=0.5, until=lambda: bool(h.moves(since=time.monotonic() - 0.3)))
    cancel_at = time.monotonic()
    h.request('c2', 'cancel', goal_id='a2')
    assert h.spin_until(lambda: h.stops(since=cancel_at), 2.0), 'no StopMove after cancel'
    assert h.stops(since=cancel_at)[0] - cancel_at <= 0.5
    _stream(h, 1.5, bbox=0.5)
    assert h.moves(since=cancel_at + 0.3) == [], 'Move after cancel'
    assert h.result('a2')['status'] == 'cancelled'
    assert h.result('c2')['status'] == 'succeeded'
    assert h.spin_until(lambda: h.states[-1] == 'IDLE', 1.0), h.states[-3:]
    h.request('a3', 'approach_person', arrival='stop')
    assert h.spin_until(lambda: h.states[-1] == 'ACQUIRE_PERSON', 3.0), 'next request refused'
    h.request('c3', 'cancel', goal_id='a3')
    assert h.result('a3')['status'] == 'cancelled'

    # 3. a stale cancel and a reused goal_id are refused without motion.
    quiet_from = time.monotonic()
    h.request('c4', 'cancel', goal_id='a2')
    h.request('a1', 'approach_person', arrival='stop')
    h.spin(1.0)
    assert h.result('c4')['reason'] == 'not_active'
    assert [x['reason'] for x in h.results
            if x['goal_id'] == 'a1' and x['status'] == 'rejected'] == ['duplicate_goal_id']
    assert h.moves(since=quiet_from) == []

    # 4. turn_to_voice: one yaw-only rotation, then stopped and reported.
    turn_at = time.monotonic()
    h.direction_pub.publish(Float64MultiArray(data=[1.0, 0.9]))
    h.spin(0.2)
    h.request('t1', 'turn_to_voice')
    r = h.result('t1', timeout=15.0)
    assert r['data']['turn_rad'] == pytest.approx(1.0, abs=1e-3), r
    turn_moves = h.moves(since=turn_at)
    assert turn_moves and all(m['x'] == 0.0 and m['z'] > 0.0 for _t, m in turn_moves)
    assert any(t > turn_moves[-1][0] for t in h.stops(since=turn_at))
    assert h.spin_until(lambda: h.states[-1] == 'IDLE', 1.0), h.states[-3:]

    assert h.real == [], 'dry run published on the real /api/sport/request'
