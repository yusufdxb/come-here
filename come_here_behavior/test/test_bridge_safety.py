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
    from std_msgs.msg import Bool, Float64, Float64MultiArray, String
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
LIE_API_ID = 1005
STAND_UP_API_ID = 1004
BALANCE_API_ID = 1002


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


def test_estop_engage_edge_sends_one_stopmove_and_repeats_are_rate_limited(node):
    # Was one StopMove per True; now engage edge + bounded reassert (1 s apart).
    node._estop_cb(_bool(True))
    node._estop_cb(_bool(True))
    assert _api_ids(node).count(STOP_MOVE_API_ID) == 1
    node._now.t += 1.0
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


def test_unreadable_check_mode_reply_neither_enables_nor_stops(mode_node):
    mode_node._mode_response_cb(_mode_response('mcf'))
    mode_node._velocity_cb(_vel(0.0, 0.0))
    mode_node._velocity_cb(_vel(0.6, 0.0))
    garbled = _mode_response('mcf')
    garbled.data = 'not json'
    mark = _mark(mode_node)
    mode_node._mode_response_cb(garbled)
    mode_node._velocity_cb(_vel(0.6, 0.0))
    assert _api_ids(mode_node, mark) == [MOVE_API_ID]


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


# -- closed-loop turn on odometry --

def _odom(yaw):
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.pose.pose.orientation.z = math.sin(yaw / 2.0)
    m.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return m


def test_rotate_turns_until_odometry_yaw_reaches_the_target():
    import threading
    n = _make_node(dry_run=True, cmd_z=1.0, max_yaw_rate=1.0, rotate_deadband_rad=0.05)
    n._now = time.monotonic          # the worker needs real time for odometry ages
    try:
        n._odom_cb(_odom(0.0))
        state = {'yaw': 0.0, 'stop': False}

        def robot():
            # A fake GO2: integrates the last commanded yaw rate at 1 rad/s scale.
            last = time.monotonic()
            while not state['stop']:
                time.sleep(0.01)
                now = time.monotonic()
                moves = _moves(n)
                if moves and not n._last_was_zero:
                    state['yaw'] += moves[-1]['z'] * (now - last)
                last = now
                n._odom_cb(_odom(state['yaw']))

        t = threading.Thread(target=robot, daemon=True)
        t.start()
        msg = Float64()
        msg.data = math.radians(60)
        n._rotate_cb(msg)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not n._last_was_zero:
            time.sleep(0.02)
        state['stop'] = True
        t.join(1.0)
        assert n._last_was_zero, 'turn never issued StopMove'
        assert 0.95 < math.degrees(state['yaw']) / 60.0 < 1.15
        assert all(m['x'] == 0.0 and m['z'] > 0 for m in _moves(n))
        assert any(r.header.identity.api_id == STOP_MOVE_API_ID
                   for r in n._sport_pub.msgs)  # StopMove closes the turn
    finally:
        n.destroy_node()


def test_rotate_stops_when_odometry_goes_stale():
    n = _make_node(dry_run=True, cmd_z=1.0, max_yaw_rate=1.0, odom_max_age_s=0.2)
    n._now = time.monotonic
    try:
        n._odom_cb(_odom(0.0))
        msg = Float64()
        msg.data = math.radians(90)
        n._rotate_cb(msg)                 # odometry never updates again
        time.sleep(0.6)
        assert n._last_was_zero
        moves = _moves(n)
        assert 1 <= len(moves) <= 8       # about 0.2 s of Moves, then StopMove
    finally:
        n.destroy_node()


def test_rotate_without_odometry_falls_back_to_the_timed_turn():
    n = _make_node(dry_run=True, cmd_z=1.0, max_yaw_rate=1.0, deg_per_sec=90.0)
    n._now = time.monotonic
    try:
        msg = Float64()
        msg.data = math.radians(45)         # 0.5 s at 90 deg/s
        n._rotate_cb(msg)
        time.sleep(0.9)
        assert n._last_was_zero
        assert 6 <= len(_moves(n)) <= 12
    finally:
        n.destroy_node()


class _Remote:
    def __init__(self, lx=0.0, ly=0.0, rx=0.0, ry=0.0, keys=0):
        self.lx, self.ly, self.rx, self.ry, self.keys = lx, ly, rx, ry, keys


def test_remote_buttons_alone_do_not_trip_the_override(node):
    node._remote_cb(_Remote(keys=0x0200))
    node._remote_cb(_Remote(lx=0.1, ry=-0.15))
    assert not node._gate.estopped


def test_remote_stick_latches_the_estop_and_stops(node):
    node._remote_cb(_Remote(ly=0.6))
    assert node._gate.estopped
    node._remote_cb(_Remote(lx=float('nan')))      # already latched: no error, still latched
    assert node._gate.estopped


def test_phrase_filenames_match_the_sound_files():
    from come_here_behavior.go2_bridge_node import phrase_filename
    assert phrase_filename('I am coming') == 'i_am_coming.wav'
    assert phrase_filename("Got you. I'm on my way.") == 'got_you_i_m_on_my_way.wav'
    assert phrase_filename(' Found you. ') == 'found_you.wav'


# -- cmd_posture: the settle sequence lie / stand_up / balance --

POSTURE_IDS = {'lie': LIE_API_ID, 'stand_up': STAND_UP_API_ID, 'balance': BALANCE_API_ID}


def _str(value):
    m = String()
    m.data = value
    return m


def _grant_authority(n, epoch=1):
    n._authority.on_grant(json.dumps({
        'v': 1, 'owner': 'come_here', 'epoch': epoch, 'guardian': 'g1',
        'state': 'RUN', 'mode': 'mcf'}), n._now())
    n._authority_poll()


@pytest.mark.parametrize('value', sorted(POSTURE_IDS))
def test_posture_each_value_sends_one_request(node, value):
    node._posture_cb(_str(value))
    assert _api_ids(node) == [POSTURE_IDS[value]]


@pytest.mark.parametrize('value', ['', 'Lie', 'lie ', 'sit', 'damp', '1001', 'stand', 'lie_down'])
def test_posture_unknown_value_is_ignored(node, value):
    node._posture_cb(_str(value))
    assert _api_ids(node) == []


def test_posture_ids_come_from_parameters():
    n = _make_node(lie_api_id=1006, stand_up_api_id=1010, balance_api_id=1007)
    try:
        for value, api_id in (('lie', 1006), ('stand_up', 1010), ('balance', 1007)):
            n._posture_cb(_str(value))
            assert _api_ids(n)[-1] == api_id
    finally:
        n.destroy_node()


@pytest.mark.parametrize('param', ['lie_api_id', 'stand_up_api_id', 'balance_api_id'])
def test_posture_api_id_1001_is_refused_at_startup(param):
    with pytest.raises(ValueError, match='Damp'):
        Go2BridgeNode(parameter_overrides=[Parameter(param, value=1001)])


def test_posture_lie_is_sent_under_estop_and_leaves_it_latched(node):
    node._estop_cb(_bool(True))
    mark = _mark(node)
    node._posture_cb(_str('lie'))
    assert _api_ids(node, mark) == [LIE_API_ID]
    assert node._gate.estopped
    assert _api_ids(node).count(STOP_MOVE_API_ID) == 1  # only the engage StopMove


@pytest.mark.parametrize('value', ['stand_up', 'balance'])
def test_posture_loading_commands_are_refused_under_estop(node, value):
    node._estop_cb(_bool(True))
    mark = _mark(node)
    node._posture_cb(_str(value))
    assert _api_ids(node, mark) == []


@pytest.mark.parametrize('value', ['stand_up', 'balance'])
def test_posture_loading_commands_are_refused_while_inhibited(value):
    n = _make_node(require_motion_mode='mcf')  # inhibited until CheckMode reports mcf
    try:
        n._posture_cb(_str(value))
        assert _api_ids(n) == []
    finally:
        n.destroy_node()


def test_posture_lie_needs_no_authority_and_does_not_change_it():
    n = _make_node(motion_authority_topic='/auth', grant_timeout_s=0.3)
    try:
        n._posture_cb(_str('lie'))
        assert _api_ids(n) == [LIE_API_ID]
        assert n._authority_dropped == 0 and not n._authority.owned(n._now())
    finally:
        n.destroy_node()


@pytest.mark.parametrize('value', ['stand_up', 'balance'])
def test_posture_loading_commands_need_authority_like_cmd_stand(value):
    n = _make_node(motion_authority_topic='/auth', grant_timeout_s=0.3)
    try:
        n._posture_cb(_str(value))
        assert _api_ids(n) == [] and n._authority_dropped == 1
        _grant_authority(n)
        n._posture_cb(_str(value))
        assert _api_ids(n) == [POSTURE_IDS[value]]
    finally:
        n.destroy_node()


def test_posture_lie_while_active_sends_stopmove_and_retry_succeeds(node):
    node._velocity_cb(_vel(0.6, 0.0))
    assert node._gate.active
    mark = _mark(node)
    node._posture_cb(_str('lie'))
    assert _api_ids(node, mark) == [STOP_MOVE_API_ID]
    assert not node._gate.active
    mark = _mark(node)
    node._posture_cb(_str('lie'))
    assert _api_ids(node, mark) == [LIE_API_ID]


@pytest.mark.parametrize('value', ['stand_up', 'balance'])
def test_posture_loading_commands_are_refused_while_active(node, value):
    node._velocity_cb(_vel(0.6, 0.0))
    mark = _mark(node)
    node._posture_cb(_str(value))
    assert _api_ids(node, mark) == []


def test_posture_lie_during_a_rotation_stops_it_and_is_refused():
    n = _make_node(cmd_z=1.0, max_yaw_rate=1.0, deg_per_sec=90.0, rotate_closed_loop=False)
    n._now = time.monotonic
    try:
        msg = Float64()
        msg.data = math.radians(90)         # 1 s timed turn
        n._rotate_cb(msg)
        assert n._rotating()
        mark = _mark(n)
        n._posture_cb(_str('lie'))
        assert LIE_API_ID not in _api_ids(n, mark)
        assert _api_ids(n, mark)[-1] == STOP_MOVE_API_ID
        time.sleep(0.2)                     # the preempted worker exits
        assert not n._rotating()
        mark = _mark(n)
        n._posture_cb(_str('lie'))
        assert _api_ids(n, mark) == [LIE_API_ID]
    finally:
        n.destroy_node()


@pytest.mark.parametrize('value', sorted(POSTURE_IDS))
def test_posture_ignored_when_posture_commands_disabled(value):
    n = _make_node(enable_posture_commands=False)
    try:
        n._posture_cb(_str(value))
        assert _api_ids(n) == []
    finally:
        n.destroy_node()


def test_posture_dry_run_publishes_only_on_the_dry_run_topic():
    n = Go2BridgeNode(parameter_overrides=[Parameter('dry_run', value=True)])
    try:
        assert n._sport_pub.topic_name == DRY_RUN_SPORT_TOPIC
        real_pub, n._sport_pub = n._sport_pub, FakePub()
        n._posture_cb(_str('lie'))
        assert _api_ids(n) == [LIE_API_ID]  # goes through the dry-run publisher
        n._sport_pub = real_pub
    finally:
        n._sport_pub = FakePub()
        n.destroy_node()


# -- authority revoke: StopMove only if motion was active --
# StopMove on a sitting GO2 stands it back up (hardware, 2026-10-13), so a
# revoke on an idle robot must send nothing. E-stop is unchanged.

def _authority_node(**params):
    n = _make_node(motion_authority_topic='/auth', grant_timeout_s=0.3, **params)
    _grant_authority(n)
    assert n._authority.owned(n._now())
    return n


def _revoke(n):
    n._now.t += 1.0  # past grant_timeout_s with no refreshed grant
    n._authority_poll()
    assert not n._authority.owned(n._now())


def test_revoke_after_sit_sends_no_stopmove():
    n = _authority_node(sit_api_id=1009)  # the deployed value (behavior_params.yaml)
    try:
        n._sit_cb(_bool(True))
        time.sleep(0.7)  # the deferred Sit fires 0.5 s after cmd_sit
        assert _api_ids(n) == [STOP_MOVE_API_ID, 1009]
        assert n._last_posture == 'sit'
        mark = _mark(n)
        _revoke(n)
        assert _api_ids(n, mark) == []
    finally:
        n.destroy_node()


def test_revoke_while_idle_sends_no_stopmove():
    n = _authority_node()
    try:
        _revoke(n)
        assert _api_ids(n) == []
    finally:
        n.destroy_node()


def test_revoke_during_walking_sends_stopmove():
    n = _authority_node()
    try:
        n._velocity_cb(_vel(0.6, 0.0))
        assert _api_ids(n)[-1] == MOVE_API_ID
        mark = _mark(n)
        _revoke(n)
        assert _api_ids(n, mark) == [STOP_MOVE_API_ID]
    finally:
        n.destroy_node()


def test_revoke_after_walking_stopped_by_watchdog_sends_no_second_stopmove():
    n = _authority_node(cmd_velocity_timeout_s=0.5)
    try:
        n._velocity_cb(_vel(0.6, 0.0))
        n._now.t += 0.2
        n._authority.on_grant(json.dumps({
            'v': 1, 'owner': 'come_here', 'epoch': 1, 'guardian': 'g1',
            'state': 'RUN', 'mode': 'mcf'}), n._now())
        n._now.t += 0.3
        n._velocity_tick()  # command is 0.5+ s old: watchdog stops
        n._now.t += 0.3
        n._velocity_tick()
        assert _api_ids(n)[-1] == STOP_MOVE_API_ID
        mark = _mark(n)
        _revoke(n)
        assert _api_ids(n, mark) == []
    finally:
        n.destroy_node()


def test_revoke_during_rotate_sends_stopmove():
    n = _make_node(motion_authority_topic='/auth', grant_timeout_s=0.3, cmd_z=1.0,
                   max_yaw_rate=1.0, deg_per_sec=90.0, rotate_closed_loop=False)
    n._now = time.monotonic  # the rotate worker needs a real clock
    try:
        _grant_authority(n)
        msg = Float64()
        msg.data = math.radians(180)         # 2 s timed turn
        n._rotate_cb(msg)
        assert n._rotating()
        time.sleep(0.15)
        assert MOVE_API_ID in _api_ids(n)
        time.sleep(0.3)                      # grant (0.3 s) expires mid-rotation
        assert n._rotating()
        mark = _mark(n)
        n._authority_poll()
        assert _api_ids(n, mark)[0] == STOP_MOVE_API_ID
        time.sleep(0.2)                      # the preempted worker exits
        assert not n._rotating()
    finally:
        n.destroy_node()


def test_estop_while_sitting_still_sends_stopmove():
    # Unchanged on purpose: the e-stop path sends StopMove even when idle.
    # Whether that stands a sitting GO2 is an open hardware question.
    n = _authority_node()
    try:
        n._sit_cb(_bool(True))
        time.sleep(0.7)
        mark = _mark(n)
        n._estop_cb(_bool(True))
        assert _api_ids(n, mark) == [STOP_MOVE_API_ID]
    finally:
        n.destroy_node()


def test_estop_during_walking_still_sends_stopmove_and_authority_revoke_after_is_quiet():
    n = _authority_node()
    try:
        n._velocity_cb(_vel(0.6, 0.0))
        mark = _mark(n)
        n._estop_cb(_bool(True))
        assert _api_ids(n, mark) == [STOP_MOVE_API_ID]
        mark = _mark(n)
        _revoke(n)  # StopMove already sent by the e-stop; robot is idle now
        assert _api_ids(n, mark) == []
    finally:
        n.destroy_node()
