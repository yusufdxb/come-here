"""Process-level shutdown test (skeptic review C3).

The node-level tests call ``destroy_node()`` while the ROS context is still
alive, so they cannot catch the real failure: rclpy's default signal handlers
shut the context down before ``main()`` reaches ``destroy_node()``, and the
final stop was never delivered.

These tests start the real installed executables, send SIGINT / SIGTERM, and
assert that the stop command is observed on the wire by an independent
subscriber. DDS is isolated with a dedicated domain id and
``ROS_LOCALHOST_ONLY=1``. The bridge is started with ``dry_run:=true``.

Skipped when the workspace has not been built (no installed executables). The
bridge cases also need the Unitree SDK message package ``unitree_api``.
"""

import os
import signal
import subprocess
import time

import pytest

try:
    import rclpy
    from ament_index_python.packages import PackageNotFoundError, get_package_prefix
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Float64MultiArray
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False

try:
    from unitree_api.msg import Request
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _ROS_AVAILABLE, reason='ROS 2 not sourced')

DOMAIN_ID = 60 + os.getpid() % 30
STOP_MOVE_API_ID = 1003
SPORT_TOPICS = ('/api/sport/request', '/come_here/dry_run/sport_request')
SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]  # SIGHUP: dropped SSH session


def _executable(package: str, name: str):
    try:
        prefix = get_package_prefix(package)
    except PackageNotFoundError:
        return None
    path = os.path.join(prefix, 'lib', package, name)
    return path if os.access(path, os.X_OK) else None


def _spin_until(executor, predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        executor.spin_once(timeout_sec=0.05)
    return predicate()


@pytest.fixture
def listener(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', str(DOMAIN_ID))
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('PYTHONUNBUFFERED', '1')
    ctx = Context()
    rclpy.init(context=ctx, domain_id=DOMAIN_ID)
    node = rclpy.create_node('c3_shutdown_listener', context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)
    yield node, executor
    executor.shutdown()
    node.destroy_node()
    rclpy.try_shutdown(context=ctx)


def _signal_process(cmd, node, executor, topics, sig):
    """Start cmd, wait for it to advertise topics, deliver sig, return output."""
    proc = subprocess.Popen(
        cmd, env=dict(os.environ), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    try:
        advertised = _spin_until(
            executor,
            lambda: any(node.count_publishers(t) > 0 for t in topics),
            20.0,
        )
        assert advertised, 'node under test never advertised its command topic'
        # Let the remote writer match our reader before the signal.
        _spin_until(executor, lambda: False, 1.5)
        proc.send_signal(sig)
        output, _ = proc.communicate(timeout=10.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    return output


@pytest.mark.parametrize('sig', SIGNALS, ids=['sigint', 'sigterm', 'sighup'])
def test_behavior_node_publishes_stop_on_signal(listener, sig):
    exe = _executable('come_here_behavior', 'behavior_node')
    if exe is None:
        pytest.skip('behavior_node not installed; build the workspace first')
    node, executor = listener
    received = []
    node.create_subscription(
        Float64MultiArray, '/come_here/cmd_velocity',
        lambda m: received.append(list(m.data)), 10,
    )
    output = _signal_process(
        [exe], node, executor, ['/come_here/cmd_velocity'], sig,
    )
    got_stop = _spin_until(executor, lambda: [0.0, 0.0] in received, 3.0)
    assert got_stop, (
        f'no zero cmd_velocity observed after {sig.name}; '
        f'received={received}; node output tail:\n{output[-2000:]}'
    )


@pytest.mark.skipif(not _SDK_AVAILABLE, reason='unitree_api not installed')
@pytest.mark.parametrize('sig', SIGNALS, ids=['sigint', 'sigterm', 'sighup'])
def test_bridge_publishes_stopmove_on_signal(listener, sig):
    exe = _executable('come_here_behavior', 'go2_bridge_node')
    if exe is None:
        pytest.skip('go2_bridge_node not installed; build the workspace first')
    node, executor = listener
    api_ids = {topic: [] for topic in SPORT_TOPICS}
    for topic in SPORT_TOPICS:
        node.create_subscription(
            Request, topic,
            lambda m, t=topic: api_ids[t].append(m.header.identity.api_id), 10,
        )
    output = _signal_process(
        [exe, '--ros-args', '-p', 'dry_run:=true'],
        node, executor, SPORT_TOPICS, sig,
    )
    dry_run_ids = api_ids['/come_here/dry_run/sport_request']
    got_stop = _spin_until(
        executor, lambda: STOP_MOVE_API_ID in dry_run_ids, 3.0,
    )
    assert got_stop, (
        f'no StopMove observed after {sig.name}; api_ids={api_ids}; '
        f'node output tail:\n{output[-2000:]}'
    )
    assert api_ids['/api/sport/request'] == [], 'dry run leaked onto the real Sport topic'
