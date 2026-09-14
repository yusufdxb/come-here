"""The class demo launch and config: structure, scope, and no silently ignored parameters.

A misspelled key in a ROS 2 params file is ignored without an error, so a typo
in a safety limit would quietly fall back to the node default. Every key in
professor_demo.yaml must be a parameter its node actually declares.
"""

import importlib.util
import pathlib
import re

import pytest
import yaml
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

REPO = pathlib.Path(__file__).resolve().parents[2]
LAUNCH_FILE = REPO / 'come_here_bringup' / 'launch' / 'professor_demo.launch.py'
CONFIG_FILE = REPO / 'come_here_bringup' / 'config' / 'professor_demo.yaml'

NODE_SOURCES = {
    'audio_node': ['come_here_audio/come_here_audio/audio_node.py'],
    'perception_node': ['come_here_perception/come_here_perception/perception_node.py'],
    'behavior_node': ['come_here_behavior/come_here_behavior/behavior_node.py'],
    'go2_bridge_node': ['come_here_behavior/come_here_behavior/go2_bridge_node.py'],
}


@pytest.fixture(scope='module')
def config():
    return yaml.safe_load(CONFIG_FILE.read_text())


@pytest.fixture(scope='module')
def launch_description():
    spec = importlib.util.spec_from_file_location('professor_demo_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _declared(node_name):
    names = set()
    for rel in NODE_SOURCES[node_name]:
        source = (REPO / rel).read_text()
        names |= set(re.findall(r"(?:declare_parameter|\bp)\(\s*'([a-z0-9_]+)'", source))
    if node_name == 'behavior_node':
        fsm = (REPO / 'come_here_behavior/come_here_behavior/come_here_fsm.py').read_text()
        body = fsm.split('class FsmConfig', 1)[1].split('def validate', 1)[0]
        names |= set(re.findall(r'^    ([a-z0-9_]+): [A-Za-z]+ = ', body, flags=re.M))
    return names


def _params(config, node):
    return config[node]['ros__parameters']


@pytest.mark.parametrize('node', sorted(NODE_SOURCES))
def test_every_config_key_is_declared_by_its_node(config, node):
    unknown = set(_params(config, node)) - _declared(node)
    assert unknown == set(), f'{node} would silently ignore: {sorted(unknown)}'


def test_launch_starts_only_the_demo_nodes(launch_description):
    nodes = [e for e in launch_description.entities if isinstance(e, Node)]
    names = sorted(n.node_executable for n in nodes)
    assert names == ['audio_node', 'behavior_node', 'go2_bridge_node', 'perception_node']
    processes = [e for e in launch_description.entities if isinstance(e, ExecuteProcess)
                 and not isinstance(e, Node)]
    assert len(processes) == 1  # the camera publisher


def test_dry_run_is_the_default(launch_description):
    args = {e.name: e for e in launch_description.entities if isinstance(e, DeclareLaunchArgument)}
    assert args['dry_run'].default_value[0].text == 'true'


def test_launch_and_config_agree_on_walk_budget(launch_description, config):
    args = {e.name: e for e in launch_description.entities if isinstance(e, DeclareLaunchArgument)}
    launch_default = float(args['max_walk_distance_m'].default_value[0].text)
    assert launch_default == _params(config, 'behavior_node')['max_walk_distance_m']


def test_demo_scope_is_minimal(config):
    behavior = _params(config, 'behavior_node')
    bridge = _params(config, 'go2_bridge_node')
    assert behavior['arrival_mode'] == 'stop'
    assert behavior['speak_text'] == ''
    assert bridge['enable_posture_commands'] is False
    assert bridge['allow_combined_motion'] is False
    assert bridge['require_motion_mode'] == 'mcf'


def test_turn_to_sound_uses_software_doa_and_closed_loop_turns(config, launch_description):
    audio = _params(config, 'audio_node')
    behavior = _params(config, 'behavior_node')
    bridge = _params(config, 'go2_bridge_node')
    assert audio['doa_source'] == 'software'
    assert audio['enable_doa'] is False              # the stuck firmware register stays off
    assert audio['mic_channels'] == 6                # raw capsules needed for DOA
    assert audio['mic_beam_channel'] == 0            # Whisper keeps the DSP beam
    assert behavior['skip_turn_to_sound'] is False
    assert behavior['direction_max_age_s'] > 0
    assert bridge['enable_rotate_command'] is True
    assert bridge['rotate_closed_loop'] is True
    assert bridge['odom_topic'] == '/utlidar/robot_odom'
    assert bridge['cmd_z'] <= bridge['max_yaw_rate']
    args = {e.name: e for e in launch_description.entities if isinstance(e, DeclareLaunchArgument)}
    assert args['skip_turn_to_sound'].default_value[0].text == 'false'
    assert float(args['doa_offset_deg'].default_value[0].text) == audio['respeaker_frame_offset_deg']


def test_single_bearing_smoothing_layer(config):
    assert _params(config, 'perception_node')['bearing_ema_alpha'] == 1.0
    assert _params(config, 'behavior_node')['bearing_ema_alpha'] == 0.3


def test_hardware_tuned_motion_values_are_preserved(config):
    behavior = _params(config, 'behavior_node')
    assert behavior['approach_speed'] == 0.6
    assert behavior['approach_align_threshold_rad'] == 0.15
    assert (behavior['approach_ccw_yaw'], behavior['approach_cw_yaw']) == (0.6, 0.6)
    assert (behavior['approach_min_align_s'], behavior['approach_min_walk_s']) == (0.4, 1.5)
    assert behavior['bbox_stop_fraction'] == 0.75
    assert _params(config, 'go2_bridge_node')['republish_rate_hz'] == 20.0


def test_bridge_limits_cover_the_commanded_setpoints(config):
    behavior = _params(config, 'behavior_node')
    bridge = _params(config, 'go2_bridge_node')
    assert behavior['approach_speed'] <= bridge['max_vx'] < bridge['reject_vx_above']
    assert max(behavior['approach_ccw_yaw'], behavior['approach_cw_yaw']) <= bridge['max_yaw_rate']
    assert bridge['max_yaw_rate'] < bridge['reject_yaw_rate_above']


def test_behavior_config_passes_fsm_validation(config):
    fsm = pytest.importorskip('come_here_behavior.come_here_fsm')
    fields = {f for f in fsm.FsmConfig.__dataclass_fields__}
    values = {k: v for k, v in _params(config, 'behavior_node').items() if k in fields}
    fsm.ComeHereFsm(fsm.FsmConfig(**values))  # raises on an inconsistent config
