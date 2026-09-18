#!/usr/bin/env bash
# come-here class demo preflight. It NEVER moves the robot: it reads files,
# lists devices, listens to topics, and sends one read-only CheckMode query to
# /api/motion_switcher/request.
#
#   ./scripts/demo_preflight.sh          before launching (robot standing, nothing running)
#   ./scripts/demo_preflight.sh --live   while `professor_demo.launch.py` (dry run) is running
#
# Optional: EXPECTED_COMMIT=<sha> ./scripts/demo_preflight.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE=0
[ "${1:-}" = "--live" ] && LIVE=1
FAILS=()
WARNS=()
pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*"; FAILS+=("$*"); }
warn() { printf '  [WARN] %s\n' "$*"; WARNS+=("$*"); }
probe() { python3 "$ROOT/scripts/topic_probe.py" "$@" 2>/dev/null; }
json_field() { python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get(sys.argv[2]))" "$1" "$2" 2>/dev/null; }
at_least() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$1" "$2"; }

if [ -z "${ROS_DISTRO:-}" ]; then
  # ROS setup files read unset variables (AMENT_TRACE_SETUP_FILES): under
  # set -u sourcing them killed this script before it printed anything.
  set +u
  # shellcheck disable=SC1091
  source "$ROOT/scripts/demo_env.sh" >/dev/null 2>&1
  set -u
  [ -n "${ROS_DISTRO:-}" ] || { echo "FAIL: could not source $ROOT/scripts/demo_env.sh"; exit 1; }
fi

echo "== code"
commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true)"
if [ -n "$commit" ]; then
  pass "git commit $commit on $(git -C "$ROOT" branch --show-current 2>/dev/null)"
  if [ -n "${EXPECTED_COMMIT:-}" ]; then
    [ "$commit" = "${EXPECTED_COMMIT:0:${#commit}}" ] && pass "matches EXPECTED_COMMIT" \
      || fail "commit $commit is not EXPECTED_COMMIT $EXPECTED_COMMIT"
  fi
  [ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ] \
    || warn "uncommitted changes in $ROOT"
else
  warn "not a git checkout: trial logs will record commit 'unknown'"
fi

echo "== environment"
[ "${ROS_DISTRO:-}" = "humble" ] && pass "ROS 2 humble" || fail "ROS 2 humble not sourced: source scripts/demo_env.sh"
[ "${RMW_IMPLEMENTATION:-}" = "rmw_cyclonedds_cpp" ] && pass "RMW cyclonedds" \
  || fail "RMW_IMPLEMENTATION is '${RMW_IMPLEMENTATION:-unset}'"
if [ -n "${CYCLONEDDS_URI:-}" ]; then
  xml="${CYCLONEDDS_URI#file://}"
  if [ -f "$xml" ]; then
    iface="$(grep -o 'NetworkInterface [^>]*name="[^"]*"' "$xml" | head -1 | sed 's/.*name="\([^"]*\)".*/\1/')"
    [ -z "$iface" ] && iface="$(grep -o '<NetworkInterfaceAddress>[^<]*' "$xml" | head -1 | sed 's/.*>//')"
    if [ -n "$iface" ]; then
      state="$(cat "/sys/class/net/$iface/operstate" 2>/dev/null || echo missing)"
      [ "$state" = "up" ] && pass "DDS interface $iface is up" || fail "DDS interface '$iface' is $state"
    else
      warn "no interface named in $xml"
    fi
  else
    fail "CYCLONEDDS_URI points at a missing file: $xml"
  fi
else
  warn "CYCLONEDDS_URI is unset"
fi
if [ "$(date +%Y)" -ge 2026 ]; then pass "clock $(date -Iseconds)"; else warn "clock reads $(date): set it before recording trials"; fi

echo "== packages, models, camera script"
for pkg in come_here_audio come_here_perception come_here_behavior come_here_bringup unitree_api; do
  ros2 pkg prefix "$pkg" >/dev/null 2>&1 && pass "package $pkg" || fail "package $pkg not found"
done
launch_file="$(ros2 pkg prefix come_here_bringup 2>/dev/null)/share/come_here_bringup/launch/professor_demo.launch.py"
[ -f "$launch_file" ] && pass "professor_demo.launch.py installed" || fail "professor_demo.launch.py not installed: colcon build"
for module in faster_whisper sounddevice ultralytics cv2 usb yaml; do
  timeout 60 python3 -c "import $module" >/dev/null 2>&1 && pass "python module $module" || fail "python module $module missing"
done
config="$ROOT/come_here_bringup/config/professor_demo.yaml"
yolo="$(python3 -c "import yaml; print(yaml.safe_load(open('$config'))['perception_node']['ros__parameters']['model_path'])" 2>/dev/null)"
[ -n "$yolo" ] && [ -f "$yolo" ] && pass "YOLO model $yolo" || fail "YOLO model missing: ${yolo:-unreadable config}"
if [ -d "$ROOT/models/faster-whisper-base.en" ] || [ -d "$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-base.en" ]; then
  pass "faster-whisper base.en model cached"
else
  fail "faster-whisper base.en model not cached (no internet in the lab)"
fi
camera_script="${CAMERA_SCRIPT:-/home/unitree/go2_video_publisher.py}"
[ -f "$camera_script" ] && pass "camera script $camera_script" || fail "camera script missing: $camera_script"

echo "== microphone"
if lsusb 2>/dev/null | grep -qi '2886:0018'; then pass "ReSpeaker Mic Array on USB"; else fail "ReSpeaker Mic Array (2886:0018) not on USB"; fi
mic="$(timeout 30 python3 -c "from come_here_audio.mic_select import resolve; i,n,c,f=resolve('ReSpeaker', True); print(f'{f}|{i}:{n} ({c}ch)')" 2>&1 | tail -1)"
case "$mic" in
  True\|*) pass "capture device ${mic#True|}" ;;
  False\|*) fail "capture device is not the far-field array: ${mic#False|}" ;;
  *) fail "no ReSpeaker capture device: $mic" ;;
esac

echo "== DOA calibration and manual override"
CAL="${HOME}/come_here_trials/doa_calibration.json"
if [ -f "$CAL" ]; then
  pass "DOA calibration $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('offset %+.1f deg mirror %s, ahead std %s deg, measured %s' % (d['offset_deg'], d['mirror'], d.get('ahead_circ_std_deg'), d.get('measured_at')))" "$CAL" 2>&1)"
else
  fail "no DOA calibration at $CAL: run python3 scripts/calibrate_doa.py (the robot will not turn without it)"
fi
if python3 -c "from unitree_go.msg import WirelessController" 2>/dev/null; then
  pass "unitree_go WirelessController importable (remote stick override)"
else
  warn "unitree_go not importable: remote stick override disabled (e-stop console still works)"
fi

echo "== processes"
# Another stack on this Jetson holds the microphone (ALSA gives a capture
# device to one process) and may command the robot; it must be down first.
holders=""
for pid in $(fuser /dev/snd/pcm*c 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$'); do
  name="$(ps -o comm= -p "$pid" 2>/dev/null)"
  [ "$name" = "audio_node" ] && [ "$LIVE" = 1 ] && continue
  holders="$holders $name($pid)"
done
[ -z "$holders" ] && pass "no other process holds a microphone capture device" \
  || fail "microphone held by another stack, stop it first:$holders"
other_launch="$(pgrep -af 'ros2 launch' | grep -v -e pgrep -e come_here || true)"
[ -z "$other_launch" ] && pass "no other ros2 launch running" \
  || fail "another ros2 launch is running (it may own the Sport API): $(echo "$other_launch" | head -1 | cut -c1-90)"
running="$(pgrep -af 'lib/come_here_(audio|perception|behavior)/|go2_video_publisher' | grep -v -e pgrep -e demo_preflight || true)"
if [ "$LIVE" = 0 ]; then
  [ -z "$running" ] && pass "no come-here processes already running" || fail "already running (stop them first): $running"
else
  [ -n "$running" ] && pass "come-here processes running" || fail "no come-here processes running: start the dry-run launch first"
fi

echo "== robot (read-only)"
ros2 daemon stop >/dev/null 2>&1
odom_hz="$(probe rate /utlidar/robot_odom odom 3)"
at_least "${odom_hz:-0}" 10 && pass "GO2 DDS data flowing: /utlidar/robot_odom ${odom_hz} Hz" \
  || fail "no GO2 data on /utlidar/robot_odom (${odom_hz:-0} Hz): interface, domain, or robot off"
cloud_hz="$(probe rate /utlidar/cloud_base cloud 3)"
at_least "${cloud_hz:-0}" 5 && pass "LiDAR /utlidar/cloud_base ${cloud_hz} Hz" \
  || warn "LiDAR /utlidar/cloud_base ${cloud_hz:-0} Hz (the bbox stop still works without it)"
if timeout 15 ros2 run come_here_behavior check_motion_mode 2>/dev/null | grep -q 'MOTION MODE: PASS'; then
  pass "motion mode is mcf (read-only CheckMode)"
else
  fail "motion mode is not verified mcf: robot will refuse to move (do NOT SelectMode normal)"
fi

if [ "$LIVE" = 1 ]; then
  echo "== live dry-run stack"
  cam_hz="$(probe rate /camera/image_raw image 4)"
  at_least "${cam_hz:-0}" 3 && pass "camera /camera/image_raw ${cam_hz} Hz" \
    || fail "camera /camera/image_raw ${cam_hz:-0} Hz (need >= 3 Hz for a safe approach)"
  det_hz="$(probe rate /come_here/person_detection array 4)"
  at_least "${det_hz:-0}" 1 && pass "person detections ${det_hz} Hz" \
    || fail "person detections ${det_hz:-0} Hz (YOLO stalled or camera stale)"
  audio="$(probe json /come_here/audio_health 8)"
  if [ -n "$audio" ]; then
    age="$(json_field "$audio" capture_age_s)"
    calibrated="$(json_field "$audio" calibrated)"
    at_least 1.0 "${age:-99}" && pass "microphone capturing (age ${age}s, gate $(json_field "$audio" rms_gate), floor $(json_field "$audio" noise_floor))" \
      || fail "microphone capture stalled (age ${age}s)"
    [ "$calibrated" = "True" ] && pass "wake gate calibrated" || warn "wake gate not calibrated yet"
    case "$(json_field "$audio" mic)" in
      *6ch*) pass "audio_node captures 6 channels (raw capsules available for DOA)" ;;
      *) warn "audio_node mic is not 6-channel: software DOA is off, the demo is camera only" ;;
    esac
  else
    fail "no /come_here/audio_health from audio_node"
  fi
  bridge="$(probe json /come_here/bridge_status 5)"
  if [ -n "$bridge" ]; then
    [ "$(json_field "$bridge" motion_mode_verified)" = "True" ] && pass "bridge verified motion mode $(json_field "$bridge" motion_mode)" \
      || fail "bridge has NOT verified motion mode: $bridge"
    [ "$(json_field "$bridge" estopped)" = "False" ] && pass "e-stop not engaged" || warn "e-stop engaged"
    [ "$(json_field "$bridge" dry_run)" = "True" ] && pass "bridge in dry run" || warn "bridge is LIVE (dry_run false)"
    odom_age="$(json_field "$bridge" odom_age_s)"
    if [ -z "$odom_age" ] || [ "$odom_age" = "None" ]; then
      fail "bridge sees no /utlidar/robot_odom: turns would fall back to the timed guess"
    else
      at_least 0.5 "$odom_age" && pass "bridge odometry fresh (age ${odom_age}s): closed-loop turns available" \
        || fail "bridge odometry stale (age ${odom_age}s)"
    fi
  else
    fail "no /come_here/bridge_status from go2_bridge_node"
  fi
  subs="$(ros2 topic info /come_here/estop 2>/dev/null | sed -n 's/^Subscription count: //p')"
  [ "${subs:-0}" -ge 2 ] && pass "/come_here/estop has $subs subscribers (behavior + bridge)" \
    || fail "/come_here/estop has ${subs:-0} subscribers: a software e-stop would not reach both nodes"
fi

echo
if [ "${#FAILS[@]}" -gt 0 ]; then
  echo "DEMO PREFLIGHT: FAIL (${#FAILS[@]} blocker(s), ${#WARNS[@]} warning(s))"
  for item in "${FAILS[@]}"; do echo "  - $item"; done
  exit 1
fi
echo "DEMO PREFLIGHT: PASS (${#WARNS[@]} warning(s))"
exit 0
