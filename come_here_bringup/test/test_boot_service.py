"""The boot service: what it must guarantee before it can run unattended.

The robot starts this with nobody logged in, so the rules that a human would
otherwise enforce (do not start twice, do not start half-ready, stop the robot
on shutdown, do not move unless motion was enabled) have to live in the unit
file and the wrapper.
"""
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
UNIT = REPO / 'systemd' / 'come-here.service.in'
WRAPPER = REPO / 'scripts' / 'come_here_boot.sh'
INSTALLER = REPO / 'scripts' / 'install_come_here_service.sh'


@pytest.fixture(scope='module')
def unit():
    return UNIT.read_text()


@pytest.fixture(scope='module')
def wrapper():
    return WRAPPER.read_text()


def _directive(text, key):
    match = re.search(rf'^{key}=(.+)$', text, re.MULTILINE)
    return None if match is None else match.group(1).strip()


@pytest.mark.parametrize('path', [UNIT, WRAPPER, INSTALLER])
def test_the_files_exist(path):
    assert path.is_file(), path


@pytest.mark.parametrize('path', [WRAPPER, INSTALLER])
def test_scripts_are_executable_and_valid_bash(path):
    assert path.stat().st_mode & 0o111, f'{path} is not executable'
    subprocess.run(['bash', '-n', str(path)], check=True)


def test_shutdown_sends_sigint_so_the_robot_is_told_to_stop(unit):
    # SIGTERM kills the launch before the C3 shutdown stop reaches DDS.
    assert _directive(unit, 'KillSignal') == 'SIGINT'
    assert int(_directive(unit, 'TimeoutStopSec')) >= 15


def test_a_boot_before_the_robot_is_powered_retries(unit):
    assert _directive(unit, 'Restart') == 'on-failure'
    assert int(_directive(unit, 'RestartSec')) >= 10


def test_the_service_runs_the_wrapper_as_a_normal_user_at_boot(unit):
    assert _directive(unit, 'ExecStart').endswith('scripts/come_here_boot.sh')
    assert _directive(unit, 'User') == '@USER@'
    assert _directive(unit, 'Environment') == 'HOME=@HOME@'
    assert _directive(unit, 'WantedBy') == 'multi-user.target'


def test_enabling_the_service_cannot_by_itself_command_motion(wrapper):
    # dry_run:=false only when the live flag file exists.
    assert 'DRY_RUN=true' in wrapper
    assert re.search(r'if \[ -f "\$LIVE_FLAG" \]', wrapper)
    assert 'dry_run:="$DRY_RUN"' in wrapper


def test_the_wrapper_refuses_to_start_a_second_stack(wrapper):
    assert 'already running' in wrapper
    # Matching the interpreter path keeps the wrapper's own command line out of
    # the count (a bare pattern would match itself and never start).
    assert '^/usr/bin/python3 .*lib/come_here_(audio|behavior|perception)/' in wrapper


def test_the_wrapper_requires_preflight_before_launching(wrapper):
    assert 'demo_preflight.sh' in wrapper
    assert wrapper.index('demo_preflight.sh') < wrapper.index('exec ros2 launch')


def test_the_wrapper_execs_the_launch_so_signals_reach_it(wrapper):
    assert 'exec ros2 launch come_here_bringup professor_demo.launch.py' in wrapper


def test_the_wrapper_sources_the_ros_env_outside_set_u(wrapper):
    # ROS setup files read unset variables; under set -u sourcing them exits.
    source_at = wrapper.index('source "$ROOT/scripts/demo_env.sh"')
    assert 'set +u' in wrapper[:source_at]
    assert 'ROS_DISTRO' in wrapper[source_at:]


def test_live_mode_waits_until_the_robot_is_standing(wrapper):
    # Lying is about 0.07 m and sitting about 0.25 m: a wake phrase must never
    # command a walk from either.
    guard = wrapper.index('/sportmodestate')
    assert guard < wrapper.index('exec ros2 launch')
    assert '0.28' in wrapper
    assert 'not standing' in wrapper


def test_the_installer_uses_the_account_that_owns_the_checkout():
    # Run under sudo, `id -un` is root: the service would run as root.
    text = INSTALLER.read_text()
    assert 'SUDO_USER' in text
    assert 'refusing to install a service that runs as root' in text
    assert 'getent passwd' in text


def test_the_live_flag_is_never_committed():
    assert '.come_here_live' in (REPO / '.gitignore').read_text()
