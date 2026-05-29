"""Tests for the sport_to_twist_sim_bridge utility.

The bridge translates GO2 Sport API requests (unitree_api/msg/Request on
/api/sport/request) into geometry_msgs/Twist on /robot0/cmd_vel so come-here
can drive the go2_omniverse Isaac Sim in a closed loop.

Requires the Unitree SDK (``unitree_api``). On a machine without it the whole
module is skipped, because the bridge imports ``unitree_api`` at module scope
and cannot be loaded there. Run these on the Jetson or in any colcon workspace
that also builds unitree_ros2. The skip mirrors test_bridge_safety.py.

The bridge lives in ``scripts/`` (it is a sim/test utility, not a packaged
console_script), so it is loaded by file path via importlib rather than a
normal package import.

Covers the translation contract:
  Move (api_id 1008)      -> Twist with x/y/z from the parameter JSON
  StopMove (api_id 1003)  -> all-zero Twist (the sim halt path)
  unparseable Move JSON   -> nothing published, no crash
  other api_ids           -> ignored (no Twist published)
"""

import importlib.util
import json
import pathlib

import pytest

try:
    import rclpy
    from geometry_msgs.msg import Twist  # noqa: F401  (used via published msgs)
    from unitree_api.msg import Request
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

# The bridge imports unitree_api at module scope, so it only loads on a machine
# with the Unitree SDK. Skip the whole module cleanly otherwise. A module-level
# pytest.importorskip would instead mark the shared `test` package skipped and
# break collection of the sibling test files.
pytestmark = pytest.mark.skipif(
    not _SDK_AVAILABLE,
    reason='unitree_api (Unitree SDK) not installed; sim bridge cannot be imported',
)


def _load_bridge_module():
    """Load the sim bridge by file path (it lives in scripts/, not the package)."""
    script = (
        pathlib.Path(__file__).resolve().parent.parent
        / 'scripts'
        / 'sport_to_twist_sim_bridge.py'
    )
    spec = importlib.util.spec_from_file_location('sport_to_twist_sim_bridge', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePub:
    """Records published messages instead of sending them over DDS."""

    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture(scope='module', autouse=True)
def _rclpy_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope='module')
def bridge_module():
    return _load_bridge_module()


@pytest.fixture
def node(bridge_module):
    n = bridge_module.SportToTwistSimBridge()
    n._pub = FakePub()
    yield n
    n.destroy_node()


def _request(api_id, parameter=''):
    r = Request()
    r.header.identity.api_id = api_id
    r.parameter = parameter
    return r


# -- Move (api_id 1008) --

def test_move_translates_xyz_into_twist(node, bridge_module):
    node._on_request(
        _request(bridge_module.MOVE_API_ID, json.dumps({'x': 0.5, 'y': -0.2, 'z': 1.2}))
    )
    assert len(node._pub.msgs) == 1
    twist = node._pub.msgs[-1]
    assert twist.linear.x == pytest.approx(0.5)
    assert twist.linear.y == pytest.approx(-0.2)
    assert twist.angular.z == pytest.approx(1.2)


def test_move_missing_keys_default_to_zero(node, bridge_module):
    node._on_request(_request(bridge_module.MOVE_API_ID, json.dumps({'x': 0.3})))
    twist = node._pub.msgs[-1]
    assert twist.linear.x == pytest.approx(0.3)
    assert twist.linear.y == pytest.approx(0.0)
    assert twist.angular.z == pytest.approx(0.0)


def test_move_with_empty_parameter_publishes_zero_twist(node, bridge_module):
    node._on_request(_request(bridge_module.MOVE_API_ID, ''))
    assert len(node._pub.msgs) == 1
    twist = node._pub.msgs[-1]
    assert twist.linear.x == pytest.approx(0.0)
    assert twist.angular.z == pytest.approx(0.0)


def test_move_with_unparseable_json_publishes_nothing(node, bridge_module):
    node._on_request(_request(bridge_module.MOVE_API_ID, 'not-json{'))
    assert node._pub.msgs == []


# -- StopMove (api_id 1003): the sim halt path --

def test_stopmove_publishes_zero_twist(node, bridge_module):
    node._on_request(_request(bridge_module.STOP_MOVE_API_ID))
    assert len(node._pub.msgs) == 1
    twist = node._pub.msgs[-1]
    assert twist.linear.x == 0.0
    assert twist.linear.y == 0.0
    assert twist.angular.z == 0.0


# -- Other api_ids are ignored --

def test_unrelated_api_id_publishes_nothing(node):
    node._on_request(_request(1006))  # RecoveryStand
    assert node._pub.msgs == []
