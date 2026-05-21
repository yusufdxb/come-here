"""Safety tests for Go2BridgeNode — skeptic-review Priority 0 fixes.

Requires the Unitree SDK (``unitree_api``). On a machine without it (e.g. a
dev laptop) the whole module is skipped, because the GO2 bridge imports
``unitree_api`` at module scope and cannot be loaded there. Run these on the
Jetson or in any colcon workspace that also builds unitree_ros2.

Covers:
  C2 — /come_here/estop must halt the robot and block subsequent Move publishes
  C3 — destroy_node() must emit a StopMove on shutdown
  C4 — cmd_velocity must be clamped; a non-finite component collapses to a stop
"""

import pytest

try:
    import rclpy
    from std_msgs.msg import Bool, Float64MultiArray
    from come_here_behavior.go2_bridge_node import Go2BridgeNode
    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False

# The GO2 bridge imports unitree_api at module scope, so it only loads on a
# machine with the Unitree SDK. Skip the whole module cleanly otherwise. A
# module-level pytest.importorskip would instead mark the shared `test`
# package skipped and break collection of the sibling test files.
pytestmark = pytest.mark.skipif(
    not _BRIDGE_AVAILABLE,
    reason='unitree_api (Unitree SDK) not installed; GO2 bridge cannot be imported',
)


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


@pytest.fixture
def node():
    n = Go2BridgeNode()
    n._sport_pub = FakePub()
    yield n
    n.destroy_node()


def _bool(value):
    m = Bool()
    m.data = value
    return m


def _vel(vx, yaw):
    m = Float64MultiArray()
    m.data = [vx, yaw]
    return m


def _has_stopmove(node, fake_pub):
    return any(
        r.header.identity.api_id == node._stop_move_api_id
        for r in fake_pub.msgs
    )


# -- C4: cmd_velocity clamping --

def test_velocity_cb_clamps_excess_forward_speed(node):
    node._velocity_cb(_vel(50.0, 0.0))
    assert node._vel_vx == node._max_vx


def test_velocity_cb_clamps_excess_yaw(node):
    node._velocity_cb(_vel(0.0, -9.0))
    assert node._vel_yaw == -node._max_yaw_rate


def test_velocity_cb_passes_normal_setpoint_unchanged(node):
    node._velocity_cb(_vel(0.6, 0.6))
    assert node._vel_vx == 0.6
    assert node._vel_yaw == 0.6


def test_velocity_cb_nan_collapses_to_stop(node):
    node._velocity_cb(_vel(0.6, 0.6))
    assert node._vel_active is True
    node._velocity_cb(_vel(float('nan'), 0.6))
    assert node._vel_active is False


# -- C2: emergency stop --

def test_estop_engaged_publishes_stopmove_and_disarms(node):
    node._velocity_cb(_vel(0.6, 0.6))
    assert node._vel_active is True
    node._estop_cb(_bool(True))
    assert node._estopped is True
    assert node._vel_active is False
    assert _has_stopmove(node, node._sport_pub)


def test_estop_blocks_subsequent_velocity(node):
    node._estop_cb(_bool(True))
    node._velocity_cb(_vel(0.6, 0.6))
    assert node._vel_active is False


def test_estop_released_re_enables_motion(node):
    node._estop_cb(_bool(True))
    node._estop_cb(_bool(False))
    assert node._estopped is False
    node._velocity_cb(_vel(0.6, 0.6))
    assert node._vel_active is True


# -- C3: StopMove on shutdown --

def test_destroy_node_publishes_stopmove():
    n = Go2BridgeNode()
    n._sport_pub = FakePub()
    sport = n._sport_pub
    n.destroy_node()
    assert _has_stopmove(n, sport)
