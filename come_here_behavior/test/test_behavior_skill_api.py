"""Node-level tests for the skill interface of BehaviorNode (need a ROS 2 runtime).

Flag off (the default and the deployed configuration): no skill topic exists and the
trial record's config snapshot is unchanged. Flag on: results carry the request's goal_id
and the trial's run_id, are latched for late subscribers, and bad input never moves.
"""

import dataclasses
import json
import time

import pytest
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from come_here_behavior.behavior_node import BehaviorNode
from come_here_behavior.come_here_fsm import FsmConfig, State

MOTION_PUBS = ('_velocity_pub', '_rotate_pub', '_sit_pub', '_stand_pub')
FAKED = MOTION_PUBS + ('_say_pub', '_face_req_pub', '_state_pub', '_trial_pub')


class FakePub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture(scope='module', autouse=True)
def _rclpy_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_node(fake_skill_pub=True, **params):
    params.setdefault('trial_log_enabled', False)
    node = BehaviorNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    for name in FAKED:
        setattr(node, name, FakePub())
    if fake_skill_pub and node._skill_pub is not None:
        node._skill_pub = FakePub()
    node._now = lambda: 1000.0
    return node


def _request(node, body):
    m = String()
    m.data = body if isinstance(body, str) else json.dumps(body)
    node._skill_request_cb(m)


def _topics(node):
    subs = {s.topic_name for s in node.subscriptions}
    pubs = {p.topic_name for p in node.publishers}
    return subs, pubs


def test_flag_off_creates_no_skill_topics():
    node = _make_node()
    try:
        subs, pubs = _topics(node)
        assert '/come_here/skill_request' not in subs
        assert '/come_here/skill_result' not in pubs
        assert node._skill_pub is None
        assert node.get_parameter('enable_skill_api').value is False
    finally:
        node.destroy_node()


def test_flag_off_config_snapshot_is_the_fsm_config_only():
    node = _make_node()
    try:
        assert set(node._config_snapshot) == {f.name for f in dataclasses.fields(FsmConfig)}
        assert 'enable_skill_api' not in node._config_snapshot
    finally:
        node.destroy_node()


def test_flag_on_creates_exactly_the_two_skill_topics():
    off, on = _make_node(), _make_node(fake_skill_pub=False, enable_skill_api=True)
    try:
        subs_off, pubs_off = _topics(off)
        subs_on, pubs_on = _topics(on)
        assert subs_on - subs_off == {'/come_here/skill_request'}
        assert pubs_on - pubs_off == {'/come_here/skill_result'}
    finally:
        off.destroy_node()
        on.destroy_node()


@pytest.mark.parametrize('body', ['{not json', '[]', 'null', '"x"',
                                  json.dumps({'v': 1, 'goal_id': 'g', 'skill': 'fly'})])
def test_malformed_requests_are_answered_and_never_move(body):
    node = _make_node(enable_skill_api=True)
    try:
        _request(node, body)
        results = [json.loads(m.data) for m in node._skill_pub.msgs]
        assert [r['status'] for r in results] == ['rejected']
        assert results[0]['run_id'] is None
        for name in MOTION_PUBS:
            assert getattr(node, name).msgs == [], name
        assert node._fsm.state == State.IDLE and not node._fsm.trial_active
    finally:
        node.destroy_node()


def test_accepted_request_result_carries_goal_id_and_the_trial_run_id():
    node = _make_node(enable_skill_api=True)
    try:
        _request(node, {'v': 1, 'goal_id': 'a1', 'skill': 'approach_person',
                        'args': {'arrival': 'stop'}})
        assert node._fsm.state == State.ACQUIRE_PERSON
        assert [list(m.data) for m in node._velocity_pub.msgs] == [[0.0, 0.0]]
        run_id = node._trial_meta['run_id']
        _request(node, {'v': 1, 'goal_id': 'c1', 'skill': 'cancel',
                        'args': {'goal_id': 'a1'}})
        results = {r['goal_id']: r for r in
                   (json.loads(m.data) for m in node._skill_pub.msgs)}
        assert results['a1']['status'] == 'cancelled' and results['a1']['run_id'] == run_id
        assert results['c1']['status'] == 'succeeded'
        record = json.loads(node._trial_pub.msgs[-1].data)
        assert record['run_id'] == run_id and record['goal_id'] == 'a1'
        assert record['wake'] == {'source': 'skill_api'}
        assert list(node._velocity_pub.msgs[-1].data) == [0.0, 0.0]
    finally:
        node.destroy_node()


def test_result_is_latched_for_a_late_subscriber():
    node = _make_node(fake_skill_pub=False, enable_skill_api=True)
    late = Node('late_skill_subscriber')
    got = []
    try:
        _request(node, {'v': 1, 'goal_id': 'r1', 'skill': 'cancel',
                        'args': {'goal_id': 'none'}})     # answered: rejected
        late.create_subscription(
            String, '/come_here/skill_result', lambda m: got.append(json.loads(m.data)),
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        deadline = time.monotonic() + 5.0
        while not got and time.monotonic() < deadline:
            rclpy.spin_once(late, timeout_sec=0.1)
        assert got and got[0]['goal_id'] == 'r1' and got[0]['reason'] == 'not_active'
    finally:
        late.destroy_node()
        node.destroy_node()
