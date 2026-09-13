"""Node-level safety tests for Go2BridgeNode (need the unitree_api package).

The command rules are unit-tested without ROS in test_motion_gate.py. These
tests check the wiring: what the bridge actually publishes on the Sport API.

  C2        /come_here/estop latches a StopMove and blocks later Move publishes
  C4        cmd_velocity is validated and clamped before it becomes a Move
  watchdog  a Move stream stops when cmd_velocity stops arriving
  dry_run   Sport requests go to the dry-run topic

C3 (StopMove on SIGINT / SIGTERM) is covered at process level in
test_shutdown_stop.py; destroy_node() is also checked here.

Skipped on machines without the Unitree SDK message package, because the
bridge imports ``unitree_api`` at module scope.
"""

import json
import math
import time

import pytest

try:
    import rclpy
    from rclpy.parameter import Parameter
    from std_msgs.msg import Bool, Float64, Float64MultiArray
    from come_here_behavior.go2_bridge_node import (
        DRY_RUN_SPORT_TOPIC,
        SPORT_TOPIC,
        Go2BridgeNode,
    )
    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False

# A module-level pytest.importorskip would mark the shared `test` package
# skipped and break collection of the sibling test files.
pytestmark = pytest.mark.skipif(
    not _BRIDGE_AVAILABLE,
    reason='unitree_api (Unitree SDK) not installed; GO2 bridge cannot be imported',
)

MOVE_API_ID = 1008
STOP_MOVE_API_ID = 1003


class FakePub:
    """Records published messages instead of sending them over DDS."""

    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(scope='module', autouse=True)
def _rclpy_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_node(**params):
    node = Go2BridgeNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in params.items()]
    )
    # Swap the real publisher out before anything can be published on DDS.
    node._sport_pub = FakePub()
    node._now = FakeClock()
    return node


@pytest.fixture
def node():
    n = _make_node()
    yield n
    n.destroy_node()


def _vel(*values):
    m = Float64MultiArray()
    m.data = [float(v) for v in values]
    return m


def _bool(value):
    m = Bool()
    m.data = value
    return m


def _api_ids(node, since=0):
    return [r.header.identity.api_id for r in node._sport_pub.msgs[since:]]


def _moves(node, since=0):
    return [
        json.loads(r.parameter)
        for r in node._sport_pub.msgs[since:]
        if r.header.identity.api_id == MOVE_API_ID
    ]


def _mark(node):
    return len(node._sport_pub.msgs)


# -- C4: validation and clamping --

def test_forward_command_becomes_single_axis_moves(node):
    node._velocity_cb(_vel(0.6, 0.0))
    node._velocity_tick()
    moves = _moves(node)
    assert len(moves) == 2
    assert all(m == {'x': 0.6, 'y': 0.0, 'z': 0.0} for m in moves)


def test_over_limit_forward_is_clamped(node):
    node._velocity_cb(_vel(1.5, 0.0))  # default max_vx 1.0, reject above 2.0
    assert _moves(node)[-1]['x'] == 1.0


def test_absurd_command_never_moves(node):
    node._velocity_cb(_vel(50.0, 0.0))
    node._velocity_tick()
    assert _moves(node) == []


def test_non_finite_command_while_moving_stops(node):
    node._velocity_cb(_vel(0.6, 0.0))
    mark = _mark(node)
    node._velocity_cb(_vel(math.nan, 0.0))
    node._velocity_tick()
    assert _api_ids(node, mark) == [STOP_MOVE_API_ID]


def test_malformed_command_while_moving_stops(node):
    node._velocity_cb(_vel(0.6, 0.0))
    mark = _mark(node)
    node._velocity_cb(_vel(0.6))
    node._velocity_tick()
    assert _api_ids(node, mark) == [STOP_MOVE_API_ID]


def test_combined_forward_and_yaw_is_rejected(node):
    node._velocity_cb(_vel(0.6, 0.6))
    node._velocity_tick()
    assert _moves(node) == []


# -- watchdog --

def test_watchdog_stops_when_commands_stop_arriving(node):
    node._velocity_cb(_vel(0.6, 0.0))
    node._now.t += 0.3
    node._velocity_tick()
    assert _api_ids(node)[-1] == MOVE_API_ID
    node._now.t += 0.3  # 0.6 s since the last command, timeout is 0.5 s
    node._velocity_tick()
    assert _api_ids(node)[-1] == STOP_MOVE_API_ID
    mark = _mark(node)
    node._now.t += 0.05
    node._velocity_tick()
    assert _api_ids(node, mark) == []


# -- C2: latched e-stop --

def test_estop_publishes_stopmove_and_blocks_motion(node):
    node._velocity_cb(_vel(0.6, 0.0))
    node._estop_cb(_bool(True))
    assert _api_ids(node)[-1] == STOP_MOVE_API_ID
    mark = _mark(node)
    for _ in range(5):
        node._velocity_cb(_vel(0.6, 0.0))
        node._velocity_tick()
    assert _api_ids(node, mark) == []


def test_estop_release_does_not_resume_motion_without_zero_command(node):
    node._velocity_cb(_vel(0.6, 0.0))
    node._estop_cb(_bool(True))
    node._estop_cb(_bool(False))
    mark = _mark(node)
    node._velocity_cb(_vel(0.6, 0.0))
    node._velocity_tick()
    assert MOVE_API_ID not in _api_ids(node, mark)
    node._velocity_cb(_vel(0.0, 0.0))
    node._velocity_cb(_vel(0.6, 0.0))
    assert _api_ids(node)[-1] == MOVE_API_ID


def test_every_estop_message_sends_stopmove(node):
    node._estop_cb(_bool(True))
    node._estop_cb(_bool(True))
    assert _api_ids(node).count(STOP_MOVE_API_ID) == 2


def test_deferred_sit_is_not_sent_while_estopped(node):
    node._estop_cb(_bool(True))
    mark = _mark(node)
    node._sit_cb(_bool(True))
    time.sleep(0.7)  # a deferred Sit would fire 0.5 s after cmd_sit
    assert _api_ids(node, mark) == []


def test_rotate_command_ignored_when_disabled():
    n = _make_node(enable_rotate_command=False)
    try:
        msg = Float64()
        msg.data = 0.5
        n._rotate_cb(msg)
        time.sleep(0.2)
        assert _moves(n) == []
    finally:
        n.destroy_node()


# -- dry run and configuration --

def test_dry_run_uses_dry_run_topic():
    n = Go2BridgeNode(parameter_overrides=[Parameter('dry_run', value=True)])
    try:
        assert n._sport_pub.topic_name == DRY_RUN_SPORT_TOPIC
    finally:
        n._sport_pub = FakePub()
        n.destroy_node()


def test_default_uses_sport_topic():
    n = Go2BridgeNode()
    try:
        assert n._sport_pub.topic_name == SPORT_TOPIC
    finally:
        n._sport_pub = FakePub()
        n.destroy_node()


def test_invalid_limits_refuse_to_start():
    with pytest.raises(ValueError):
        Go2BridgeNode(parameter_overrides=[Parameter('max_vx', value=-1.0)])


# -- C3: StopMove from destroy_node --

def test_destroy_node_publishes_stopmove():
    n = _make_node()
    pub = n._sport_pub
    n.destroy_node()
    assert pub.msgs[-1].header.identity.api_id == STOP_MOVE_API_ID


# -- motion mode: read-only CheckMode, motion refused unless mcf --

def _mode_response(name, code=0, api_id=1001):
    from unitree_api.msg import Response
    r = Response()
    r.header.identity.api_id = api_id
    r.header.status.code = code
    r.data = json.dumps({'form': '0', 'name': name})
    return r


@pytest.fixture
def mode_node():
    n = _make_node(require_motion_mode='mcf')
    n._mode_pub = FakePub()
    yield n
    n.destroy_node()


def test_motion_refused_until_mcf_is_verified(mode_node):
    mode_node._velocity_cb(_vel(0.6, 0.0))
    mode_node._velocity_tick()
    assert _moves(mode_node) == []
    mode_node._mode_response_cb(_mode_response('mcf'))
    mode_node._velocity_cb(_vel(0.0, 0.0))  # a new trial always starts with a zero
    mode_node._velocity_cb(_vel(0.6, 0.0))
    assert _moves(mode_node) == [{'x': 0.6, 'y': 0.0, 'z': 0.0}]


def test_wrong_mode_refuses_motion(mode_node):
    mode_node._mode_response_cb(_mode_response('ai'))
    mode_node._velocity_cb(_vel(0.0, 0.0))
    mode_node._velocity_cb(_vel(0.6, 0.0))
    assert _moves(mode_node) == []


def test_failed_check_mode_status_does_not_enable_motion(mode_node):
    mode_node._mode_response_cb(_mode_response('mcf', code=7002))
    mode_node._velocity_cb(_vel(0.0, 0.0))
    mode_node._velocity_cb(_vel(0.6, 0.0))
    assert _moves(mode_node) == []


def test_mode_leaving_mcf_stops_the_robot(mode_node):
    mode_node._mode_response_cb(_mode_response('mcf'))
    mode_node._velocity_cb(_vel(0.0, 0.0))
    mode_node._velocity_cb(_vel(0.6, 0.0))
    mode_node._mode_response_cb(_mode_response('ai'))
    assert _api_ids(mode_node)[-1] == STOP_MOVE_API_ID
    mark = _mark(mode_node)
    mode_node._velocity_cb(_vel(0.6, 0.0))
    mode_node._velocity_tick()
    assert MOVE_API_ID not in _api_ids(mode_node, mark)


def test_check_mode_never_reaches_the_sport_topic(mode_node):
    for _ in range(3):
        mode_node._mode_check_tick()
        mode_node._now.t += 1.1
    assert [r.header.identity.api_id for r in mode_node._mode_pub.msgs] == [1001, 1001, 1001]
    mode_node._mode_response_cb(_mode_response('mcf'))
    mode_node._velocity_cb(_vel(0.0, 0.0))
    mode_node._velocity_cb(_vel(0.6, 0.0))
    mode_node._estop_cb(_bool(True))
    assert 1001 not in _api_ids(mode_node)  # 1001 on /api/sport/request is Damp


def test_sport_api_1001_refuses_to_start():
    with pytest.raises(ValueError):
        Go2BridgeNode(parameter_overrides=[Parameter('stop_move_api_id', value=1001)])


def test_normal_mode_requirement_refuses_to_start():
    with pytest.raises(ValueError):
        Go2BridgeNode(parameter_overrides=[Parameter('require_motion_mode', value='normal')])


def test_bridge_status_reports_the_estop_latch(node):
    node._status_pub = FakePub()
    node._estop_cb(_bool(True))
    status = json.loads(node._status_pub.msgs[-1].data)
    assert status['estopped'] is True
    assert status['dry_run'] is False
