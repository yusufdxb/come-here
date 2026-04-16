"""Smoke tests for come_here_bringup/launch/come_here.launch.py.

These tests do not start any nodes. They only assemble the LaunchDescription
and inspect its structure, which catches regressions where:

  * the launch file fails to import,
  * the ``use_mock`` argument is no longer declared, or
  * ``go2_bridge_node`` stops being gated out of mock mode.

Each test runs offline and does not require a ROS 2 runtime beyond the
``launch`` / ``launch_ros`` Python packages that ship with ROS 2.
"""

import importlib.util
import pathlib

import pytest

from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch_ros.actions import Node


LAUNCH_FILE = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'launch' / 'come_here.launch.py'
)


def _load_launch_module():
    spec = importlib.util.spec_from_file_location(
        'come_here_launch_under_test', LAUNCH_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def launch_description():
    mod = _load_launch_module()
    return mod.generate_launch_description()


def test_use_mock_argument_declared(launch_description):
    args = [
        e for e in launch_description.entities
        if isinstance(e, DeclareLaunchArgument) and e.name == 'use_mock'
    ]
    assert len(args) == 1, 'use_mock launch argument must be declared'
    assert args[0].default_value is not None


def test_launch_description_has_five_nodes(launch_description):
    nodes = [e for e in launch_description.entities if isinstance(e, Node)]
    assert len(nodes) == 5, (
        f'Expected 5 Node actions (audio, perception, face_detector, '
        f'behavior, go2_bridge), got {len(nodes)}'
    )


def test_go2_bridge_is_gated_by_use_mock(launch_description):
    """go2_bridge_node must be the only node carrying UnlessCondition.

    This enforces that mock-mode launches do not attempt to import
    unitree_api.
    """
    nodes = [e for e in launch_description.entities if isinstance(e, Node)]
    conditional = [n for n in nodes if isinstance(n.condition, UnlessCondition)]
    assert len(conditional) == 1, (
        'Exactly one node (go2_bridge_node) must be gated by UnlessCondition'
    )
