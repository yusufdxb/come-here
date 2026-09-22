"""scripts/native_avoid_probe.py: argument refusals and displacement math (no robot)."""

import importlib.util
import math
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'native_avoid_probe.py'


@pytest.fixture(scope='module')
def probe():
    spec = importlib.util.spec_from_file_location('native_avoid_probe', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_motion_caps(probe):
    assert probe.check_motion_args(0.5, 0.0, 0.0, 1.0) == []
    assert probe.check_motion_args(0.7, 0.0, 0.0, 1.0)
    assert probe.check_motion_args(0.5, 0.4, 0.0, 1.0)
    assert probe.check_motion_args(0.5, 0.0, 1.5, 1.0)
    assert probe.check_motion_args(0.5, 0.0, 0.0, 2.5)
    assert probe.check_motion_args(-0.3, 0.0, 0.0, 1.0)
    assert probe.check_motion_args(math.nan, 0.0, 0.0, 1.0)
    assert probe.check_motion_args(0.5, 0.0, 0.0, 0.0)


def test_motion_needs_the_explicit_flag(probe):
    with pytest.raises(SystemExit, match='i-understand'):
        probe.main(['move', '--backend', 'oa', '--vx', '0.4', '--seconds', '1'])


def test_state_changes_need_supervised(probe):
    with pytest.raises(SystemExit):
        probe.main(['freeavoid', 'on'])


def test_over_cap_motion_refused_before_ros_starts(probe):
    with pytest.raises(SystemExit, match='exceeds'):
        probe.main(['move', '--backend', 'freeavoid', '--vx', '0.9', '--seconds', '1',
                    '--i-understand-this-moves-the-robot'])


def test_body_frame_displacement(probe):
    fwd, lat, dyaw = probe.body_frame_displacement((1.0, 1.0, math.pi / 2), (1.0, 2.0, math.pi))
    assert fwd == pytest.approx(1.0) and lat == pytest.approx(0.0, abs=1e-9)
    assert dyaw == pytest.approx(math.pi / 2)
    fwd, lat, _ = probe.body_frame_displacement((0.0, 0.0, 0.0), (0.5, 0.3, 0.0))
    assert fwd == pytest.approx(0.5) and lat == pytest.approx(0.3)   # +lateral = left


def test_speed_stats(probe):
    s = probe.speed_stats([(0.0, 0.3, 0.4), (0.5, 0.0, 0.0), (2.0, 1.0, 0.0)], 0.0, 1.0)
    assert s == {'max_speed': 0.5, 'min_speed': 0.0, 'mean_speed': 0.25, 'samples': 2}
    assert probe.speed_stats([], 0.0, 1.0) is None


def test_record_is_appended(probe, tmp_path):
    path = probe.append_record({'command': 'status'}, directory=str(tmp_path))
    probe.append_record({'command': 'stop'}, directory=str(tmp_path))
    assert len(pathlib.Path(path).read_text().splitlines()) == 2
