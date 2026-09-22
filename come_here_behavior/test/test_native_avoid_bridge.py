"""Node-level tests for NativeAvoidBridgeNode (Come Here ANY; need unitree_api).

What the ANY bridge actually publishes, with fake publishers and a fake clock:

  H  backend ownership: every Move and StopMove, including the rotate worker's,
     goes through the one native backend; nothing moves before it is ENABLED
  I  the legacy and ANY command formats are mutually rejected (2 vs 3 elements)
  D/E enable success (live, replies answered) and failure (error / timeout)
  F/G stop and suspend sequences; K shutdown releases API control and avoidance
  J  an exception in spin still runs the stop / release (run_node finally)
  live motion refused unless native_live_motion_cleared; odometry stale -> stop

Skipped without the Unitree SDK message package, like test_bridge_safety.py.
"""

import json
import math

import pytest

try:
    import rclpy
    from rclpy.parameter import Parameter
    from std_msgs.msg import Bool, Float64, Float64MultiArray
    from unitree_api.msg import Response
    from come_here_behavior.native_avoid_bridge_node import NativeAvoidBridgeNode
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason='unitree_api (Unitree SDK) not installed')


class FakePub:
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


def make(**params):
    params.setdefault('dry_run', True)
    node = NativeAvoidBridgeNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    node._sport_pub = FakePub()
    node._oa_pub = FakePub()
    node._now = FakeClock()
    fresh_odom(node)
    return node


def fresh_odom(node, yaw=0.0):
    with node._odom_lock:
        node._odom_yaw = yaw
        node._odom_stamp_s = node._now()


def vel(*values):
    m = Float64MultiArray()
    m.data = [float(v) for v in values]
    return m


def flag(value):
    m = Bool()
    m.data = value
    return m


def calls(node, which='both'):
    """(service, api_id, params) of everything published, in order."""
    out = []
    pubs = {'sport': node._sport_pub, 'oa': node._oa_pub}
    merged = []
    for name, pub in pubs.items():
        if which in ('both', name):
            merged += [(m.header.identity.id, name, m) for m in pub.msgs]
    merged.sort(key=lambda x: x[0])
    for _, name, m in merged:
        params = json.loads(m.parameter) if m.parameter else None
        out.append((name, m.header.identity.api_id, params))
    return out


def api_ids(node, which='both'):
    return [c[1] for c in calls(node, which)]


def enable_dry(node):
    node._native_tick()
    assert node._backend.enabled, node._backend.status()


def answer(node, service_pub, code=0, data='{}'):
    """Answer the last request on ``service_pub`` as the robot would."""
    req = service_pub.msgs[-1]
    r = Response()
    r.header.identity.id = req.header.identity.id
    r.header.identity.api_id = req.header.identity.api_id
    r.header.status.code = code
    r.data = data
    return r


# -- dry run: routing through the backend -----------------------------------

def test_dry_run_enables_with_simulated_replies_and_says_so():
    n = make()
    n._native_tick()
    st = n._backend.status()
    assert st.state == 'enabled' and st.enable_result == 'dry_run_simulated'
    assert api_ids(n) == [1, 2048]                 # version query + FreeAvoid(true)
    assert calls(n)[1][2] == {'data': True}
    n.destroy_node()


def test_no_motion_is_published_before_the_backend_is_enabled():
    n = make()
    n._velocity_cb(vel(0.6, 0.0, 0.0))
    n._velocity_tick()
    assert 1008 not in api_ids(n)
    enable_dry(n)
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.6, 0.0, 0.0))
    assert calls(n)[-1] == ('sport', 1008, {'x': 0.6, 'y': 0.0, 'z': 0.0})
    n.destroy_node()


def test_obstacles_avoid_backend_moves_on_the_obstacles_avoid_service_only():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)
    mark = len(n._sport_pub.msgs)
    n._velocity_cb(vel(0.5, 0.0, 0.0))
    n._velocity_tick()
    assert [m.header.identity.api_id for m in n._sport_pub.msgs[mark:]] == []
    oa_moves = [c for c in calls(n, 'oa') if c[1] == 1003]
    assert oa_moves[-1][2] == {'x': 0.5, 'y': 0.0, 'yaw': 0.0, 'mode': 0}
    assert all(m.header.policy.noreply for m in n._oa_pub.msgs
               if m.header.identity.api_id == 1003)
    n.destroy_node()


def test_rotate_worker_turns_through_the_native_backend():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)
    n._publish_move(0.0, 0.8)                      # what the rotate worker calls
    assert calls(n, 'oa')[-1] == ('oa', 1003, {'x': 0.0, 'y': 0.0, 'yaw': 0.8, 'mode': 0})
    n.destroy_node()


def test_rotate_is_refused_while_native_is_not_enabled():
    n = make()
    msg = Float64()
    msg.data = 0.5
    n._rotate_cb(msg)
    assert 1008 not in api_ids(n)
    n.destroy_node()


# -- I: formats are mutually exclusive, single axis by default ---------------

def test_legacy_two_element_command_is_malformed_here():
    n = make()
    enable_dry(n)
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.6, 0.0))
    assert 1008 not in api_ids(n)
    n.destroy_node()


def test_combined_and_lateral_rejected_by_default():
    n = make()
    enable_dry(n)
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.6, 0.0, 0.5))
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.0, 0.2, 0.0))
    assert 1008 not in api_ids(n)
    n.destroy_node()


def test_lateral_allowed_only_with_its_flag():
    n = make(native_allow_lateral=True)
    enable_dry(n)
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.5, 0.2, 0.0))
    assert calls(n)[-1] == ('sport', 1008, {'x': 0.5, 'y': 0.2, 'z': 0.0})
    n.destroy_node()


# -- live mode: clearance, enable success / failure ---------------------------

def test_live_without_clearance_never_enables_and_never_moves():
    n = make(dry_run=False)
    for _ in range(5):
        n._native_tick()
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.6, 0.0, 0.0))
    n._velocity_tick()
    assert 2048 not in api_ids(n) and 1008 not in api_ids(n)
    assert 'native_live_motion_not_cleared' in n._gate.inhibit_reason
    n.destroy_node()


def test_live_cleared_enable_waits_for_real_replies():
    n = make(dry_run=False, native_live_motion_cleared=True)
    n._native_tick()
    assert api_ids(n) == [1]
    n._response_cb('sport', answer(n, n._sport_pub, data='"1.0.0.1"'))
    assert api_ids(n) == [1, 2048]
    assert not n._backend.enabled
    n._velocity_cb(vel(0.6, 0.0, 0.0))
    assert 1008 not in api_ids(n)                  # still enabling: refused
    n._response_cb('sport', answer(n, n._sport_pub))
    assert n._backend.enabled
    st = n._backend.status()
    assert st.enable_result == 'ok' and st.api_version_match is True
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.6, 0.0, 0.0))
    assert calls(n)[-1][1] == 1008
    n.destroy_node()


def test_live_enable_error_latches_failed_and_blocks_motion():
    n = make(dry_run=False, native_live_motion_cleared=True)
    n._native_tick()
    n._response_cb('sport', answer(n, n._sport_pub, data='"1.0.0.1"'))
    n._response_cb('sport', answer(n, n._sport_pub, code=3203))
    n._native_tick()
    assert n._backend.failed and 'native_avoid_failed' in n._gate.inhibit_reason
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.6, 0.0, 0.0))
    assert 1008 not in api_ids(n)
    n._native_tick()
    assert api_ids(n).count(2048) == 1             # no retry until restart
    n.destroy_node()


def test_live_enable_timeout_fails():
    n = make(dry_run=False, native_live_motion_cleared=True, native_response_timeout_s=1.0)
    n._native_tick()
    n._now.t += 1.5
    n._native_tick()                               # version query unanswered: non-fatal
    assert api_ids(n) == [1, 2048] and not n._backend.failed
    n._now.t += 1.5
    n._native_tick()                               # FreeAvoid unanswered: fatal
    assert n._backend.failed and 1008 not in api_ids(n)
    n.destroy_node()


def test_enable_waits_for_the_mcf_check():
    n = make(require_motion_mode='mcf')
    n._mode_pub = FakePub()
    n._native_tick()
    assert not n._backend.enabled and 2048 not in api_ids(n)
    r = Response()
    r.header.identity.api_id = 1001
    r.header.status.code = 0
    r.data = json.dumps({'form': '0', 'name': 'mcf'})
    n._mode_response_cb(r)
    n._native_tick()
    assert n._backend.enabled
    n.destroy_node()


# -- stop paths ---------------------------------------------------------------

def test_watchdog_stop_goes_through_the_backend():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.5, 0.0, 0.0))
    fresh_odom(n)
    n._now.t += 0.6
    fresh_odom(n)
    n._velocity_tick()
    tail = calls(n)[-2:]
    assert tail[0] == ('oa', 1003, {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'mode': 0})
    assert tail[1][:2] == ('sport', 1003)
    n.destroy_node()


def test_estop_stops_and_releases_obstacles_avoid_api_control():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)
    assert n._backend.api_control_taken
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.5, 0.0, 0.0))
    n._estop_cb(flag(True))
    labels = [c for c in calls(n)][-5:]
    assert ('oa', 1004, {'is_remote_commands_from_api': False}) in labels
    assert not n._backend.api_control_taken and n._backend.state == 'disabled'
    n._native_tick()
    assert n._backend.state == 'disabled'          # no re-enable while e-stopped
    mark = len(n._oa_pub.msgs)
    n._velocity_cb(vel(0.5, 0.0, 0.0))
    assert [m for m in n._oa_pub.msgs[mark:] if m.header.identity.api_id == 1003
            and json.loads(m.parameter)['x'] != 0.0] == []
    n.destroy_node()


def test_remote_stick_override_also_releases_api_control():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)

    class Remote:
        lx, ly, rx, ry = 0.0, 0.7, 0.0, 0.0
    n._remote_cb(Remote())
    assert n._gate.estopped and not n._backend.api_control_taken
    n.destroy_node()


def test_stale_odometry_stops_a_moving_robot():
    n = make()
    enable_dry(n)
    n._velocity_cb(vel(0.0, 0.0, 0.0))
    n._velocity_cb(vel(0.6, 0.0, 0.0))
    n._now.t += 0.2
    n._velocity_cb(vel(0.6, 0.0, 0.0))             # odometry now 0.2 s old: fine
    n._now.t += 0.5
    n._velocity_tick()                              # 0.7 s old: stop
    assert calls(n)[-1][1] == 1003
    n.destroy_node()


def test_sit_suspends_api_control_and_holds_until_stand():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)
    n._sit_cb(flag(True))
    assert not n._backend.api_control_taken and n._posture_hold
    n._native_tick()
    assert n._backend.state == 'disabled'
    n._stand_cb(flag(True))
    n._native_tick()
    assert n._backend.enabled
    n.destroy_node()


def test_shutdown_releases_everything_obstacles_avoid():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)
    oa, sport = n._oa_pub, n._sport_pub
    mark_oa, mark_sp = len(oa.msgs), len(sport.msgs)
    n.destroy_node()
    oa_tail = [(m.header.identity.api_id, json.loads(m.parameter) if m.parameter else None)
               for m in oa.msgs[mark_oa:]]
    assert (1004, {'is_remote_commands_from_api': False}) in oa_tail
    assert (1001, {'enable': False}) in oa_tail     # switch restored to its initial state
    assert sport.msgs[-1].header.identity.api_id == 1003   # final StopMove


def test_shutdown_disables_freeavoid_and_stops():
    n = make()
    enable_dry(n)
    sport = n._sport_pub
    mark = len(sport.msgs)
    n.destroy_node()
    tail = [(m.header.identity.api_id, m.parameter) for m in sport.msgs[mark:]]
    assert (2048, json.dumps({'data': False})) in tail
    assert tail[-1][0] == 1003


def test_exception_in_spin_still_stops_and_releases(monkeypatch):
    from come_here_behavior import node_runner
    created = {}

    def factory():
        created['node'] = make(native_avoid_backend='obstacles_avoid')
        created['node']._native_tick()
        return created['node']

    def boom(node):
        raise RuntimeError('simulated callback crash')
    monkeypatch.setattr(node_runner.rclpy, 'init', lambda **kw: None)
    monkeypatch.setattr(node_runner.rclpy, 'spin', boom)
    monkeypatch.setattr(node_runner.rclpy, 'try_shutdown', lambda: None)
    import signal
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with pytest.raises(RuntimeError):
            node_runner.run_node(factory)
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    n = created['node']
    oa_ids = [m.header.identity.api_id for m in n._oa_pub.msgs]
    assert 1004 in oa_ids[oa_ids.index(1004) + 1:] or oa_ids.count(1004) >= 2  # taken, released
    assert n._sport_pub.msgs[-1].header.identity.api_id == 1003


def test_status_reports_the_native_state():
    n = make(native_avoid_backend='obstacles_avoid')
    enable_dry(n)
    pub = FakePub()
    n._status_pub = pub
    n._publish_status()
    st = json.loads(pub.msgs[-1].data)
    assert st['mode'] == 'any' and st['native_avoid_enabled'] is True
    assert st['native_avoid_enable_result'] == 'dry_run_simulated'
    assert st['api_control_taken'] is True and st['native_live_motion_cleared'] is False
    n.destroy_node()


def test_invalid_backend_refuses_to_start():
    with pytest.raises(ValueError):
        make(native_avoid_backend='custom_planner')
