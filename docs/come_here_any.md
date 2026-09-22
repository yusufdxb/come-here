# Come Here ANY (experimental)

Come Here ANY keeps the hardware-demonstrated caller pipeline of the class demo
(wake, voice direction, turn, DOA-gated acquisition, tracked caller, sit) and
swaps only the locomotion: the GO2's own onboard obstacle avoidance moves the
robot, while Come Here keeps deciding who the target is, where they are, when to
go, when to stop, and which way to face at the end.

It is an additive second backend. The legacy demo is unchanged and stays the
fallback:

| | Legacy (known good) | Come Here ANY (experimental) |
|---|---|---|
| Launch | `professor_demo.launch.py` | `come_here_any.launch.py` |
| Config | `professor_demo.yaml` | `come_here_any.yaml` |
| Behavior | `behavior_node` (`ComeHereFsm`) | `come_here_any_behavior_node` (`ComeHereAnyFsm`) |
| Bridge / motion owner | `go2_bridge_node`, Sport Move | `native_avoid_bridge_node`, `NativeAvoidBackend` |
| Command | `[vx, yaw_rate]`, single axis | `[vx, vy, yaw_rate]`, single axis unless Stage D clears more |
| Hardware status | full loop demonstrated (Sep 14/15/18) | **no ANY motion has run on the robot** |

## Status of the evidence

Precise claims only. "Software tested" never means "avoidance works".

| Claim | Status |
|---|---|
| FreeAvoid (Sport 2048) request accepted by the lab GO2 | observed 2026-09-21: `FreeAvoid(false)` and `FreeAvoid(true)` both returned code 0 in mcf, robot standing |
| obstacles_avoid service present and switchable | observed 2026-09-21: `SwitchGet` code 0 `{"enable":false}`, `SwitchSet(true)` code 0 then `SwitchGet` `enable:true`, `SwitchSet(false)` restored `enable:false` |
| Enabling obstacles_avoid alone produces no motion | observed 2026-09-21: 20 zero `Move` requests at 10 Hz with the switch on, measured body speed 0.000 m/s, yaw rate <= 0.025 rad/s |
| Enabling FreeAvoid alone produces no motion | **not established**: 0.29 m/s was measured within 3 s of `FreeAvoid(true)`; self-motion vs the robot being handled is unresolved (a StopMove backstop was sent) |
| Remote "Free Avoid" button = which API | **not established**. `/multiplestate obstaclesAvoidSwitch` went True when the operator enabled it from the remote, but did NOT follow SDK `FreeAvoid` calls. So the button may drive the obstacles_avoid switch, not Sport 2048. `native_avoid_probe.py watch` + `status` decide it (Stage A3) |
| FreeAvoid changes how ordinary Sport Move reacts to obstacles | **untested** |
| obstacles_avoid Move steers around, sidesteps, or only brakes | **untested** |
| Lateral `vy` / combined vx+yaw under native avoidance | **untested** |
| Remote stick still reported while `UseRemoteCommandFromApi(true)` | **untested** (the manual-override e-stop depends on it) |
| Server API versions vs SDK (sport 1.0.0.1, obstacles_avoid 1.0.0.2) | **not read yet** (`native_avoid_probe.py status`) |
| ANY routing, ownership, bounds, stop sequencing, verified facing | software tested: unit tests + a two-process dry run of the real nodes |
| Single-obstacle avoidance, multi-obstacle, occlusion, final facing on hardware | **not demonstrated** |

The September Come Here hardware evidence in the README belongs to the legacy
demo only.

## What the SDK actually offers

From `unitree_sdk2py` 1.0.1 (identical to upstream master for these files on
2026-09-21) and `unitree_ros2`:

* **`SportClient.FreeAvoid(flag)`**: Sport api **2048**, parameter
  `{"data": bool}`, a request with a reply. No getter exists. The official
  example calls `FreeAvoid(True)`, sleeps 2 s, calls `FreeAvoid(False)`.
* **`SportClient.SwitchAvoidMode()`**: Sport api **2058**, no parameters, no
  getter: a blind toggle. **Never sent by ANY**: a toggle whose state cannot be
  read back cannot be restored safely.
* **`ObstaclesAvoidClient`**: service `obstacles_avoid`
  (`/api/obstacles_avoid/request`, SDK api version 1.0.0.2): `SwitchSet` 1001
  `{"enable"}`, `SwitchGet` 1002, `Move` 1003 `{"x","y","yaw","mode":0}`
  (noreply), `UseRemoteCommandFromApi` 1004 `{"is_remote_commands_from_api"}`,
  `MoveToAbsolutePosition` / `MoveToIncrementPosition` (Move with mode 2 / 1,
  not used). The official example: loop `SwitchSet(True)` until `SwitchGet`,
  `UseRemoteCommandFromApi(True)`, `Move(0.5, 0, 0)` for 1 s, `Move(0,0,0)`,
  `UseRemoteCommandFromApi(False)`.
* Every service answers internal RPC api **1** with its server API version
  (`Client.GetServerApiVersion`).

The two are **not assumed equivalent**. ANY supports both as
`native_avoid_backend: sport_freeavoid` (default, the path to investigate
first) or `obstacles_avoid`; Stage C picks one on evidence. A custom LiDAR
planner is not part of this work.

## Architecture

### One motion owner

`native_avoid_bridge_node` subclasses the legacy `Go2BridgeNode`, so it keeps
the latched e-stop, the remote-stick manual override, the CheckMode refusal,
the watchdog, closed-loop rotate, deferred Sit and speech. It changes only who
receives motion:

* every Move and StopMove, including the rotate worker's, goes through one
  `NativeAvoidBackend` (sport_freeavoid: Sport Move 1008; obstacles_avoid:
  obstacles_avoid Move 1003);
* nothing moves unless the backend is ENABLED: enable is a sequence of
  requests that each need a code-0 reply; an error, a wrong read-back or no
  reply within 1.5 s latches FAILED until the node restarts (no retry, no
  fallback to legacy walking);
* live (`dry_run:=false`) also needs `native_live_motion_cleared:=true`,
  otherwise nothing is enabled and every command stops;
* the command formats exclude each other: the ANY gate rejects a 2-element
  legacy command, the legacy gate rejects a 3-element ANY command;
* the ANY launch starts only the ANY bridge; the legacy launch never
  instantiates the backend. Both use the node name `go2_bridge_node`, so
  starting both would be flagged by ROS as a duplicate node.

Legacy invariants stay legacy: `allow_combined_motion` is still false in
`professor_demo.yaml` and `MotionGate` is unchanged. The ANY gate has its own
`native_allow_lateral` / `native_allow_combined` flags, both false until
Stage D shows the native controller handles them.

### Stop is stronger than move

| Event | sport_freeavoid | obstacles_avoid |
|---|---|---|
| any stop (watchdog, zero, lost caller, odometry stale) | StopMove | Move(0,0,0), StopMove |
| e-stop, remote stick override, Sit, motion mode lost | StopMove | Move(0,0,0), StopMove, **UseRemoteCommandFromApi(false)** |
| shutdown (SIGINT/SIGTERM/SIGHUP, exception) | StopMove, FreeAvoid(false), StopMove | Move(0,0,0), StopMove, release API control, SwitchSet(initial), StopMove |

Stops never wait for replies. Any enable request that was published counts as
possibly applied: a lost or late reply still gets its release (API control) or
restore (switch, FreeAvoid) at e-stop and shutdown. The ordering is a design
choice; Stage B verifies it on the robot.

Known limit: after an e-stop release or a Stand, the bridge re-enables
avoidance on its own and (obstacles_avoid) takes API control again while idle.
Stage B `remote-check` must show the stick override still arrives before that
backend is used.

### Caller tracking (reused)

`ComeHereAnyFsm` subclasses `ComeHereFsm`: wake, voice direction, turn,
relisten, search turns, DOA-gated perception, the single bearing EMA, lost and
stale detection, reacquisition with N fresh frames and the e-stop are the
legacy code. Only pursuit, occlusion handling, bounds and the final facing
are new.

Pursuit is recomputed every tick from the latest caller geometry, never a
precomputed path. The default control law, `split`, is the legacy single-axis
law (yaw-only ALIGN, forward-only WALK, odometry align turns) with the walk
going through native avoidance. WALK re-aligns at once when the caller nears
the camera edge (`any_fov_stop_rad`, 34 deg) so avoidance cannot carry the
robot around a chair while losing the person. `combined` (vx + yaw) and
`vector` (`vx = g cos b`, `vy = g sin b` plus yaw that keeps the caller in view)
exist but refuse to load without their flags.

### Brief occlusion

On each fresh visual fix with a real range, the caller's position is stored in
the odometry frame (`caller_estimate.py`, explicit math, no TF). While the
camera is fresh but the caller is hidden, the robot keeps steering on the
predicted bearing, slowing from 0.6 toward the 0.5 m/s gait floor (the mcf
trot is only clean at 0.5 or more, so there is no creeping). With the defaults
the prediction lasts at most about 1.2 s (uncertainty grows 0.8 m/s from 0.25 m, refused past 1.2 m,
TTL 1.5 s). Then it stops and reacquires, with the perception gate centered
on the predicted bearing. Rules:

* the camera is authoritative; a stale camera stops at once (no prediction);
* a prediction never arrives: arrival needs a fresh frame;
* after the approach starts, a person who would put the caller more than
  1.2 m from the last fix (no growth with time) is rejected as someone else,
  so the robot stops and reports `caller_lost` instead of adopting a new
  target. The fix is remembered for `any_identity_memory_s` (15 s), which must
  outlast the 10 s reacquire window (validated at startup). With stale
  odometry nobody can be checked, so detections are rejected. A detection
  with no range at all cannot be placed and is not rejected (the perception
  node normally supplies a LiDAR or pinhole range).

### Arrival and final facing

Arrival needs a fresh visual observation (box height or distance). Commanded
walking distance is only an abort backstop in ANY (`walk_budget_arrives` must
be false), because a detour is longer than the straight line.

The robot then turns until the caller's image bearing (the person box center,
robot to caller) is within `final_align_rad` (9 deg). It stops and verifies
the bearing on at least 2 new frames over 0.5 s. Only a verified alignment
records the arrival and sits. A lost caller, a stale camera, 4 s without
verification or 3 drifting verify attempts: stop, no sit, `stop_reason:
final_align_failed`.

Nothing in the control law uses the caller's face, gaze, head or body
orientation. Tests drive a caller at +25 deg as "front-facing", "side-facing"
and "back-facing" (different detector confidence and face results) and require
identical robot commands. The legacy demo still has its original behavior
(final-align timeout, then sit); that is deliberately unchanged here.

### Bounds and stop reasons

`trial_timeout` (60 s), `approach_timeout` (30 s), `walk_budget` (commanded
6 m), `travel_budget` (5 m of odometry path), `displacement_budget` (4 m from
the start), `no_progress` (8 s without 0.3 m of range or +0.05 box height),
`odom_stale`, `odom_invalid` (a jump > 0.3 m in one sample), `caller_lost`
(reacquire timeout), `final_align_failed`, `estop`, `shutdown`. The bridge adds
`native_avoid_failed` (latched) and refuses motion when odometry is older than
0.5 s.

### Trial log

ANY records go to the same `trials.jsonl` with `mode: "any"` (legacy records
have no `mode` field), plus `control_law`, `travel_distance_m`,
`max_displacement_m`, `straight_line_progress_m`, `caller_prediction_used_s`,
`caller_reacquisitions`, `caller_loss_events`,
`caller_prediction_expired_events`, `identity_rejections`,
`no_progress_events`, `camera_stale_events`, `lateral_commands`,
`final_align_start_bearing_rad`, `final_align_end_bearing_rad`,
`final_align_verified`, `final_align_attempts`, `final_align_failure`, and the
bridge's `native_avoid_backend`, `native_avoid_enabled`,
`native_avoid_enabled_at_wake`, `native_avoid_enable_result`
(`ok` / `failed` / `dry_run_simulated`), `native_avoid_server_version`,
`native_avoid_api_version_match`, `api_control_taken`, `api_control_released`,
`native_avoid_simulated`. `success` needs a verified final facing.

## Dry run does not prove avoidance

`dry_run:=true` (the default) sends every request to
`/come_here/dry_run/sport_request` and `/come_here/dry_run/obstacles_avoid_request`
and **simulates** the enable replies (`native_avoid_enable_result:
dry_run_simulated`). It proves command routing, ownership, state transitions,
caller tracking, timeouts, request formation and stop sequencing. It says
nothing about how the GO2 avoids obstacles.

## Commands

Legacy demo (unchanged; the fallback):

```bash
source scripts/demo_env.sh
./scripts/demo_preflight.sh
ros2 launch come_here_bringup professor_demo.launch.py dry_run:=false
```

Come Here ANY:

```bash
source scripts/demo_env.sh
./scripts/come_here_any_preflight.sh --backend sport_freeavoid
ros2 launch come_here_bringup come_here_any.launch.py                      # dry run
ros2 launch come_here_bringup come_here_any.launch.py dry_run:=false \
    native_avoid_backend:=sport_freeavoid native_live_motion_cleared:=true  # only after Stage D
```

Stop / disable ANY immediately, from any second terminal:

```bash
ros2 run come_here_behavior estop_console        # type e (latched: stop + release API control)
python3 scripts/native_avoid_probe.py stop       # zero native Move, StopMove, release API control
```

Or push a remote stick (latches the e-stop, as in the legacy demo), or Ctrl+C
the launch (stop, release, restore). Then fall back to the legacy command above.

## Hardware validation ladder

Run each stage alone, with nothing else commanding the robot (no come-here,
FETCH or Nav2 launch), the robot standing in mcf, the operator holding the
remote, video recording. PASS criteria are written here before the test. A
FAIL stops the ladder.

**Stage A: no motion.**
A1 `native_avoid_probe.py status`. PASS: CheckMode mcf, both server versions
recorded, obstacles_avoid SwitchGet answered.
A2 `native_avoid_probe.py oa-switch on --supervised`, then `oa-switch off
--supervised`. PASS: read-back matches each time, measured speed < 0.1 m/s,
switch restored.
A3 `native_avoid_probe.py watch --seconds 60` while the operator toggles the
remote avoidance button on and off twice. PASS: the record shows which state
follows the button (`obstaclesAvoidSwitch` and/or `SwitchGet`, remote `keys`).
A4 robot on the ground, nobody touching it, `native_avoid_probe.py freeavoid on
--supervised --watch-s 5`, then `freeavoid off --supervised`. PASS: speed stays
< 0.1 m/s. This settles the unexplained 0.29 m/s. If the robot moves by itself,
`sport_freeavoid` is NO-GO.

**Stage B: stop path.**
B1 `native_avoid_probe.py remote-check --supervised`. PASS: stick events arrive
on `/wirelesscontroller` while API control is held. If none arrive,
`obstacles_avoid` with `native_use_remote_command_from_api: true` is a hardware
NO-GO (the stick e-stop would be blind).
B2 `native_avoid_probe.py move --backend oa --vx 0.3 --seconds 1.0
--i-understand-this-moves-the-robot`, and the same with `--backend freeavoid`.
Press Ctrl+C mid-move once. PASS: stops within 0.5 s, `speed_after_stop` < 0.05
m/s, API control released.
B3 the ANY launch in live mode with clearance, caller absent: engage
`estop_console`, then Ctrl+C. PASS: status shows `api_control_released`, the
shutdown release sequence in the bridge log, the robot stays still.

**Stage C: native avoidance without Come Here.** Open floor, one large box
about 1.5 m ahead, generous space on both sides.
`native_avoid_probe.py move --backend freeavoid --vx 0.5 --seconds 2.0 ...`,
repeat with FreeAvoid off (`--no-restore` omitted means it is restored), then
`--backend oa`. Record for each: stops, steers, sidesteps, rotates, or hits.
The record's `displacement.lateral_m`, `speed_while_commanded.min_speed` and
`range_obstacle_min` back the video. PASS for a backend: it clears the box
without contact on 3 of 3 runs, OR the result is clearly "only brakes" (then
report it: that backend cannot route around). The backend that routes around
becomes `native_avoid_backend`.

**Stage D: motion characterization** with the chosen backend (no box): forward
0.5; lateral 0.2 (only if C suggests vy works); yaw 0.6; forward + lateral;
forward + yaw. PASS per mode: the gait stays clean (no shaking or stumbling on
video) and measured displacement matches the command direction. Only modes
that pass may be enabled (`native_allow_lateral`, `native_allow_combined`,
`any_control_law`). After Stage D: `native_live_motion_cleared:=true` is
allowed.

**Stage E: ANY, static caller, one obstacle.** Caller visible at 3 m, box
between. PASS: sits about 0.8 m from the caller with `final_align_verified`
true, no contact, trial `success` true.

**Stage F: multiple obstacles** with generous spacing. PASS: the path adapts
continuously (video), same arrival criteria.

**Stage G: brief occlusion**: the caller is hidden by an obstacle for under 1 s.
PASS: `caller_prediction_used_s` > 0, no stop or a clean reacquire, arrival only
after the caller is seen again.

**Stage H: final orientation**: caller facing the robot, sideways, back turned.
PASS: `final_align_end_bearing_rad` within 9 deg in all three, no sit on any
`final_align_failed`.

## GO / NO-GO for the first ANY movement trial

NO-GO today. First ANY motion needs Stages A, B and C to pass and Stage D for
the modes in use. Until then run ANY only as `dry_run:=true`, and use
`professor_demo.launch.py` for demos.
