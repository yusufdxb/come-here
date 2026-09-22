"""Come Here ANY launch/config contract, and the legacy demo regression contract.

LEGACY (professor_demo.launch.py + professor_demo.yaml) must keep its exact
semantics: legacy executables, no native-avoidance anything, single-axis gate,
dry run by default. ANY (come_here_any.launch.py + come_here_any.yaml) must be
a separate, opt-in entry point with its own executables, every key declared by
its node, dry run by default, live motion refused unless explicitly cleared,
and the shared sensor nodes configured identically to the legacy demo.
"""

import importlib.util
import pathlib
import re

import pytest
import yaml
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

REPO = pathlib.Path(__file__).resolve().parents[2]
BRINGUP = REPO / 'come_here_bringup'
BEHAVIOR = REPO / 'come_here_behavior' / 'come_here_behavior'
LEGACY_LAUNCH = BRINGUP / 'launch' / 'professor_demo.launch.py'
LEGACY_CONFIG = BRINGUP / 'config' / 'professor_demo.yaml'
ANY_LAUNCH = BRINGUP / 'launch' / 'come_here_any.launch.py'
ANY_CONFIG = BRINGUP / 'config' / 'come_here_any.yaml'
NATIVE_WORDS = ('native', 'freeavoid', 'free_avoid', 'obstacles_avoid', '2048', 'any_')


def _launch(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def _nodes(ld):
    return {n.node_executable: n for n in ld.entities if isinstance(n, Node)}


def _args(ld):
    return {e.name: e.default_value[0].text for e in ld.entities
            if isinstance(e, DeclareLaunchArgument)}


def _declared(sources, dataclass_files=()):
    names = set()
    for rel in sources:
        names |= set(re.findall(r"(?:declare_parameter|\bp)\(\s*'([a-z0-9_]+)'",
                                (BEHAVIOR / rel).read_text()))
    for rel, cls in dataclass_files:
        body = (BEHAVIOR / rel).read_text().split(f'class {cls}', 1)[1].split('def validate', 1)[0]
        names |= set(re.findall(r'^    ([a-z0-9_]+): [A-Za-z]+ = ', body, flags=re.M))
    return names


@pytest.fixture(scope='module')
def legacy():
    return _launch(LEGACY_LAUNCH), yaml.safe_load(LEGACY_CONFIG.read_text())


@pytest.fixture(scope='module')
def anyl():
    return _launch(ANY_LAUNCH), yaml.safe_load(ANY_CONFIG.read_text())


# -- LEGACY regression contract ----------------------------------------------

def test_legacy_launch_runs_only_the_legacy_behavior_and_bridge(legacy):
    ld, _cfg = legacy
    nodes = _nodes(ld)
    assert 'behavior_node' in nodes and 'go2_bridge_node' in nodes
    assert 'native_avoid_bridge_node' not in nodes
    assert 'come_here_any_behavior_node' not in nodes


def test_legacy_launch_and_config_never_mention_native_avoidance():
    for path in (LEGACY_LAUNCH, LEGACY_CONFIG):
        text = path.read_text().lower()
        for word in NATIVE_WORDS:
            assert word not in text, f'{path.name} mentions {word!r}'


def test_legacy_safety_defaults_unchanged(legacy):
    ld, cfg = legacy
    assert _args(ld)['dry_run'] == 'true'
    bridge = cfg['go2_bridge_node']['ros__parameters']
    assert bridge['dry_run'] is True
    assert bridge['allow_combined_motion'] is False
    assert bridge['require_motion_mode'] == 'mcf'
    assert bridge['manual_override_estop'] is True
    assert bridge['cmd_velocity_timeout_s'] == 0.5
    beh = cfg['behavior_node']['ros__parameters']
    assert beh['walk_budget_arrives'] is True        # the Sep 14/15 demo value, unchanged
    assert beh['max_walk_distance_m'] == 1.5
    assert beh['align_by_rotate'] is True


def test_legacy_bridge_source_has_no_native_backend():
    src = (BEHAVIOR / 'go2_bridge_node.py').read_text().lower()
    for word in ('native', 'freeavoid', 'obstacles_avoid', '2048'):
        assert word not in src
    assert "declare_parameter('allow_combined_motion', false)" in src


def test_legacy_behavior_node_still_defaults_to_the_legacy_fsm():
    src = (BEHAVIOR / 'behavior_node.py').read_text()
    assert 'CONFIG_CLASS = FsmConfig' in src and 'FSM_CLASS = ComeHereFsm' in src
    assert 'native' not in src.lower()


# -- ANY entry point -----------------------------------------------------------

def test_any_launch_runs_the_any_executables_under_the_usual_names(anyl):
    ld, _cfg = anyl
    nodes = _nodes(ld)
    assert 'go2_bridge_node' not in nodes and 'behavior_node' not in nodes
    assert nodes['native_avoid_bridge_node']._Node__node_name == 'go2_bridge_node'
    assert nodes['come_here_any_behavior_node']._Node__node_name == 'behavior_node'
    assert sorted(nodes) == ['audio_node', 'come_here_any_behavior_node', 'face_detector_node',
                             'native_avoid_bridge_node', 'perception_node']


def test_any_is_dry_run_and_not_cleared_by_default(anyl):
    ld, cfg = anyl
    args = _args(ld)
    assert args['dry_run'] == 'true'
    assert args['native_live_motion_cleared'] == 'false'
    assert args['native_avoid_backend'] == 'sport_freeavoid'
    bridge = cfg['go2_bridge_node']['ros__parameters']
    assert bridge['dry_run'] is True and bridge['native_live_motion_cleared'] is False
    assert bridge['native_allow_lateral'] is False and bridge['native_allow_combined'] is False
    beh = cfg['behavior_node']['ros__parameters']
    assert beh['any_control_law'] == 'split'
    assert beh['any_allow_lateral'] is False and beh['any_allow_combined'] is False
    assert beh['walk_budget_arrives'] is False


def test_every_any_config_key_is_declared_by_its_node(anyl):
    _ld, cfg = anyl
    beh = _declared(['behavior_node.py', 'come_here_any_behavior_node.py'],
                    [('come_here_fsm.py', 'FsmConfig'),
                     ('come_here_any_controller.py', 'AnyFsmConfig')])
    bridge = _declared(['go2_bridge_node.py', 'native_avoid_bridge_node.py'])
    for node, declared in (('behavior_node', beh), ('go2_bridge_node', bridge)):
        unknown = set(cfg[node]['ros__parameters']) - declared
        assert unknown == set(), f'{node} would silently ignore: {sorted(unknown)}'


def test_any_sensor_nodes_match_the_legacy_demo(legacy, anyl):
    _l, lcfg = legacy
    _a, acfg = anyl
    for node in ('audio_node', 'perception_node', 'face_detector_node'):
        assert acfg[node] == lcfg[node], f'{node} differs from the legacy demo'


def test_any_caller_pipeline_values_match_the_legacy_demo(legacy, anyl):
    """Only the locomotion-related keys may differ; the caller pipeline is reused as is."""
    lbeh = legacy[1]['behavior_node']['ros__parameters']
    abeh = anyl[1]['behavior_node']['ros__parameters']
    allowed = {'max_walk_distance_m', 'walk_budget_arrives', 'approach_timeout_s',
               'final_align_timeout_s', 'max_align_turns'}
    diffs = {k for k in lbeh if k in abeh and lbeh[k] != abeh[k]} - allowed
    assert diffs == set(), f'caller pipeline drifted from the legacy demo: {sorted(diffs)}'
    assert set(lbeh) - set(abeh) == set()


def test_setup_installs_both_entry_points():
    setup = (BRINGUP / 'setup.py').read_text()
    assert 'launch/professor_demo.launch.py' in setup and 'launch/come_here_any.launch.py' in setup
    assert 'config/professor_demo.yaml' in setup and 'config/come_here_any.yaml' in setup
