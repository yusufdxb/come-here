#!/usr/bin/env bash
# Come Here ANY preflight. Runs the UNCHANGED legacy preflight (scripts/demo_preflight.sh)
# first, then the ANY-only checks. It never moves the robot: the robot checks are the
# read-only `native_avoid_probe.py status` queries (CheckMode, server api versions,
# obstacles_avoid SwitchGet, state snapshots).
#
#   ./scripts/come_here_any_preflight.sh [--backend sport_freeavoid|obstacles_avoid]
#   ./scripts/come_here_any_preflight.sh --live [--backend ...]   # while come_here_any.launch.py runs
#
# NO-GO when the native interface ANY needs is unavailable. ANY never falls back
# to legacy straight-line walking: run professor_demo.launch.py for that.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE_ARG=""
BACKEND="sport_freeavoid"
while [ $# -gt 0 ]; do
  case "$1" in
    --live) LIVE_ARG="--live" ;;
    --backend) BACKEND="$2"; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
  shift
done
case "$BACKEND" in sport_freeavoid|obstacles_avoid) ;; *) echo "bad --backend $BACKEND"; exit 2 ;; esac

FAILS=()
WARNS=()
pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*"; FAILS+=("$*"); }
warn() { printf '  [WARN] %s\n' "$*"; WARNS+=("$*"); }
probe() { python3 "$ROOT/scripts/topic_probe.py" "$@" 2>/dev/null; }
jq_py() { python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2], {}, {'d': d}))" "$1" "$2" 2>/dev/null; }

echo "=================== legacy preflight (unchanged) ==================="
"$ROOT/scripts/demo_preflight.sh" $LIVE_ARG
legacy_rc=$?
[ "$legacy_rc" = 0 ] || FAILS+=("legacy preflight failed (see above)")

echo
echo "=================== Come Here ANY ($BACKEND) ==================="
echo "== ANY entry point"
prefix="$(ros2 pkg prefix come_here_bringup 2>/dev/null)"
[ -f "$prefix/share/come_here_bringup/launch/come_here_any.launch.py" ] && pass "come_here_any.launch.py installed" \
  || fail "come_here_any.launch.py not installed: colcon build"
bprefix="$(ros2 pkg prefix come_here_behavior 2>/dev/null)"
for exe in come_here_any_behavior_node native_avoid_bridge_node; do
  [ -x "$bprefix/lib/come_here_behavior/$exe" ] && pass "executable $exe" || fail "executable $exe missing"
done
python3 -c "from unitree_go.msg import SportModeState, WirelessController" 2>/dev/null \
  && pass "unitree_go messages importable" || fail "unitree_go messages missing (probe and manual override need them)"

echo "== native avoidance API (read-only probe)"
if [ -n "$LIVE_ARG" ]; then
  echo "  (skipped: the running ANY bridge owns the native API; its status is checked below)"
else
  plog="$HOME/come_here_trials/native_avoid_probe.jsonl"
  lines() { if [ -f "$plog" ]; then wc -l < "$plog"; else echo 0; fi; }
  before="$(lines)"
  timeout 40 python3 "$ROOT/scripts/native_avoid_probe.py" --note preflight status >/dev/null 2>&1
  rec="$(tail -n 1 "$plog" 2>/dev/null)"
  after="$(lines)"
  if [ "$after" -le "$before" ] || [ -z "$rec" ]; then
    fail "native_avoid_probe.py status wrote no record"
  else
    [ "$(jq_py "$rec" "'mcf' in str(d['check_mode']['data'])")" = "True" ] && pass "CheckMode mcf" \
      || fail "CheckMode not mcf: $(jq_py "$rec" "d['check_mode']")"
    height="$(jq_py "$rec" "(d['snapshot']['sportmodestate'] or {}).get('body_height', 0)")"
    python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.28 else 1)" "${height:-0}" \
      && pass "robot standing (body_height $height)" || fail "robot not standing (body_height ${height:-none})"
    sv="$(jq_py "$rec" "d['sport_server_version']")"
    [ "$(jq_py "$rec" "d['sport_server_version']['code']")" = "0" ] && pass "sport server api version $sv" \
      || warn "sport server api version not answered: $sv"
    ov="$(jq_py "$rec" "d['obstacles_avoid_server_version']")"
    case "$ov" in *1.0.0.2*) pass "obstacles_avoid server api version matches the SDK (1.0.0.2)";;
      *) [ "$BACKEND" = obstacles_avoid ] && fail "obstacles_avoid server api version: $ov" \
           || warn "obstacles_avoid server api version: $ov" ;; esac
    sg="$(jq_py "$rec" "d['oa_switch_get']")"
    if [ "$(jq_py "$rec" "d['oa_switch_get']['code']")" = "0" ]; then
      pass "obstacles_avoid SwitchGet answered: $sg"
    elif [ "$BACKEND" = obstacles_avoid ]; then
      fail "obstacles_avoid service not answering (NO-GO for this backend): $sg"
    else
      warn "obstacles_avoid service not answering: $sg"
    fi
    pubs="$(jq_py "$rec" "d['response_publishers']")"
    [ "$(jq_py "$rec" "d['response_publishers']['sport'] >= 1")" = "True" ] && pass "sport service present ($pubs)" \
      || fail "no /api/sport/response publisher ($pubs)"
    if [ "$BACKEND" = sport_freeavoid ]; then
      warn "FreeAvoid (2048) has no getter: availability is proven only when the bridge's enable returns code 0"
    fi
  fi
fi

echo "== trial evidence and stop path"
[ -w "$HOME/come_here_trials" ] || mkdir -p "$HOME/come_here_trials" 2>/dev/null
[ -w "$HOME/come_here_trials" ] && pass "trial log dir writable" || fail "cannot write ~/come_here_trials"
ros2 pkg executables come_here_behavior 2>/dev/null | grep -q estop_console \
  && pass "estop_console available (ros2 run come_here_behavior estop_console)" || fail "estop_console missing"

if [ -n "$LIVE_ARG" ]; then
  echo "== running ANY bridge"
  bridge="$(probe json /come_here/bridge_status 5)"
  if [ -z "$bridge" ]; then
    fail "no /come_here/bridge_status"
  else
    [ "$(jq_py "$bridge" "d.get('mode')")" = "any" ] && pass "bridge is the ANY bridge" \
      || fail "bridge is not the ANY bridge (legacy launch running?)"
    [ "$(jq_py "$bridge" "d.get('native_avoid_backend')")" = "$BACKEND" ] && pass "backend $BACKEND" \
      || fail "bridge backend is $(jq_py "$bridge" "d.get('native_avoid_backend')"), expected $BACKEND"
    state="$(jq_py "$bridge" "d.get('native_avoid_state')")"
    dry="$(jq_py "$bridge" "d.get('dry_run')")"
    cleared="$(jq_py "$bridge" "d.get('native_live_motion_cleared')")"
    case "$state" in
      enabled) pass "native avoidance enabled ($(jq_py "$bridge" "d.get('native_avoid_enable_result')"))" ;;
      failed) fail "native avoidance FAILED: $(jq_py "$bridge" "d.get('native_avoid_failure')")" ;;
      *) [ "$dry" = "False" ] && [ "$cleared" != "True" ] \
           && warn "native avoidance $state: live motion not cleared (expected before Stage E)" \
           || fail "native avoidance $state" ;;
    esac
    [ "$dry" = "True" ] && pass "bridge in dry run (native replies SIMULATED)" \
      || { [ "$cleared" = "True" ] && warn "bridge LIVE and CLEARED for motion" \
           || warn "bridge LIVE, motion NOT cleared"; }
    [ "$(jq_py "$bridge" "d.get('estopped')")" = "False" ] && pass "e-stop not engaged" || warn "e-stop engaged"
    if [ "$BACKEND" = obstacles_avoid ]; then
      warn "obstacles_avoid holds API remote-command ownership while enabled: Stage B remote-check must have shown the stick override still arrives"
    fi
  fi
fi

echo
if [ "${#FAILS[@]}" -gt 0 ]; then
  echo "COME HERE ANY PREFLIGHT: NO-GO (${#FAILS[@]} blocker(s), ${#WARNS[@]} warning(s))"
  for item in "${FAILS[@]}"; do echo "  - $item"; done
  echo "Fallback: ros2 launch come_here_bringup professor_demo.launch.py dry_run:=false"
  exit 1
fi
echo "COME HERE ANY PREFLIGHT: GO for the configured stage (${#WARNS[@]} warning(s))"
exit 0
