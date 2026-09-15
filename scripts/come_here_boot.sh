#!/usr/bin/env bash
# Boot wrapper for the come-here demo: the robot computer starts this from
# systemd (see systemd/come-here.service.in), so saying "come here" works with
# nobody logged in.
#
# It refuses to start unless the machine is actually ready, because a stack
# that comes up without the microphone, without robot odometry or while the
# robot is not in mcf is a stack that looks alive and cannot work. Preflight
# is the readiness check; on FAIL this exits non-zero and systemd retries.
#
# Dry run is the default. The service only commands motion when the live flag
# file exists, so enabling the service can never, on its own, make the robot
# walk.
#
#   touch ~/come-here-demo/.come_here_live     # motion enabled at boot
#   rm    ~/come-here-demo/.come_here_live     # dry run at boot
#
# exec keeps the launch as the service's main process, so systemd's SIGINT
# reaches it directly: that is the path that sends StopMove on shutdown.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE_FLAG="${COME_HERE_LIVE_FLAG:-$ROOT/.come_here_live}"
TRIAL_DIR="${COME_HERE_TRIAL_DIR:-$HOME/come_here_trials}"
mkdir -p "$TRIAL_DIR"

log() { printf '[come-here-boot] %s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

# ROS setup files read unset variables, so they cannot be sourced under set -u.
set +u
# shellcheck disable=SC1091
source "$ROOT/scripts/demo_env.sh" >/dev/null 2>&1
set -u
if [ -z "${ROS_DISTRO:-}" ]; then
  log "FAIL: $ROOT/scripts/demo_env.sh did not source; not starting"
  exit 1
fi

# Never two stacks: a second one fights the first for the microphone and the
# robot. Matching on the python interpreter path keeps this script's own
# command line out of the count.
running() {
  ps -eo args | grep -cE "^/usr/bin/python3 .*lib/come_here_(audio|behavior|perception)/" || true
}
if [ "$(running)" != "0" ]; then
  log "FAIL: come-here nodes are already running; not starting a second stack"
  exit 1
fi

if ! "$ROOT/scripts/demo_preflight.sh"; then
  log "NOT READY: preflight failed (robot off, mic missing, or DDS not up yet); systemd will retry"
  exit 1
fi

if [ -f "$LIVE_FLAG" ]; then
  DRY_RUN=false
  # Live mode only while the robot is already standing. Booting next to a
  # charger, lying down (body height about 0.07 m) or sitting (about 0.25 m),
  # a wake phrase would otherwise command a walk from the floor. Standing
  # measures about 0.32 m. systemd retries, so standing the robot up is all
  # the operator has to do.
  height="$("$ROOT/scripts/topic_probe.py" sport /sportmodestate 5 2>/dev/null || true)"
  if [ -z "$height" ] || ! awk -v h="$height" 'BEGIN { exit !(h + 0 >= 0.28) }'; then
    log "NOT READY: robot body height '${height:-unknown}' is not standing (>= 0.28 m); systemd will retry"
    exit 1
  fi
  log "LIVE: $LIVE_FLAG present and the robot is standing (body height ${height} m)"
else
  DRY_RUN=true
  log "DRY RUN: no $LIVE_FLAG, Sport API requests go to /come_here/dry_run/sport_request"
fi

log "starting professor_demo.launch.py dry_run:=$DRY_RUN (trial dir $TRIAL_DIR)"
exec ros2 launch come_here_bringup professor_demo.launch.py \
  dry_run:="$DRY_RUN" trial_log_dir:="$TRIAL_DIR"
