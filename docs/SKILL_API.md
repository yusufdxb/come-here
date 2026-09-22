# Stepwise skill interface v2 (optional, off by default)

`behavior_node` parameter `enable_skill_api` (default `false`). When false, no skill topic
exists, the wake phrase is subscribed as always, and the node behaves exactly as the
baseline (tag `baseline/pre-skill-api-f7192c9`): `test/test_baseline_golden.py` checks
every command the state machine emits in 13 deployed scenarios (4341 frames) against a
trace recorded from the baseline code, byte for byte. The parameter is not an `FsmConfig`
field, so the `config` snapshot in trial records does not change. The bridge, perception,
audio, configs and launch files are identical to the baseline.

With `enable_skill_api:=true` a supervisor owns the robot: `/come_here/wake_phrase` is not
subscribed (a wake would start the whole baseline sequence, sit included), and Come Here
runs one decomposed step per request. The interface gives task-level requests only: it
never carries a velocity, angle, speed or distance. The state machine stays the only
producer of `cmd_velocity`, `cmd_rotate` and `cmd_sit`; the bridge is unchanged.

## Topics

`/come_here/skill_request` (std_msgs/String, JSON, reliable, volatile, depth 10)

```json
{"v": 2, "goal_id": "<1..64 printable>", "request_id": "<1..64 printable, never reused>",
 "seq": 1, "skill": "...", "args": {}}
```

`/come_here/skill_result` (std_msgs/String, JSON, reliable, **transient_local**, depth 10)

```json
{"v": 2, "goal_id": "...", "request_id": "...", "skill": "...",
 "status": "succeeded|failed|cancelled|rejected", "reason": "...", "run_id": "...", "data": {}}
```

## Skills

A step that depends on an earlier one names that step's `request_id` (a handoff). A
handoff is used once, belongs to its goal, and is void after any motion (a bearing or a
track measured before a turn or a walk means something else afterwards) or an e-stop.

| skill | args | motion | runs | succeeds with |
|---|---|---|---|---|
| `localize_caller` | none | none | waits (listening_timeout_s, or relisten_timeout_s after a prompt) for a voice bearing measured after the robot last moved and after any prompt, within direction_max_age_s | `localized`: bearing_rad/deg, confidence, age_s. Else `no_direction`, or `low_confidence` (bearing and confidence reported, never turned on) |
| `orient_to_caller` | `localization` | one turn | one closed-loop turn by the NAMED localization's bearing (never a newer one), then settle | `turn_complete` (rotate reason reached/overshoot/timed) or `turn_not_needed`; residual kept as the next acquire's gate center |
| `acquire_caller` | none | none | ACQUIRE_PERSON (gated: the turn residual, else the bearing where a track was just lost, else ahead) until 2 fresh in-gate detections; no relisten, no search turns, no align | `acquired`: bearing, confidence, distance and source, bbox, age, gate. Else `acquire_timeout`. The gate is held on the track while the acquisition is valid (continuous track, <= 15 s, nothing moved) |
| `approach_caller` | `acquisition` | walk | the baseline approach (ACQUIRE, align turns, ALIGN/WALK, arrival rule incl. walk budget) with the arrival ALWAYS `stop`: zero velocity, StopMove, arrival_hold_s standing still | `arrived_bbox`, `arrived_distance` or `arrived_walk_budget` with `posture: standing` and `proximity_evidence` (false for the walk budget). A lost caller stops the robot and ends the step `track_lost` (no internal reacquisition); `approach_timeout` |
| `sit` | `arrival` | posture | settle (pre_sit_settle_s), Sit, sit_settle_s; no final align (no motion); then held seated until the operator reset | `sit_commanded`, `posture: unverified` (nothing here reads posture) |
| `ask_caller_again` | none | none | says relisten_speak_text; only an utterance after it can localize | `asked` |
| `cancel` | `request_id` | stop | the existing abort for the active request: zero velocity, no e-stop latched | the cancelled request `cancelled` (`data.cancel_request_id`), the cancel `succeeded` |

## Rejections (`status: rejected`, nothing moves, no trial starts)

`malformed`, `unknown_skill` (including the v1 names), `bad_args`, `duplicate_request_id`
(1024 remembered), `stale_goal` (a goal retired by a newer one), `out_of_order` (seq not
increasing within a goal), `not_idle` (a request or trial is active, or the robot sits),
`estopped`, `seated` (a skill sit happened and no operator reset since, even after an
e-stop), handoff problems: `no_localization`, `stale_localization`,
`moved_since_localization`, `no_acquisition`, `stale_acquisition`,
`moved_since_acquisition`, `caller_not_in_view`, `no_arrival`, `stale_arrival`,
`moved_since_arrival`; for cancel: `not_active`, `posture_phase` (sitting or changing
posture: the operator reset owns the stand-up).

## Invariants (each has a test)

- One active request; a second one is rejected and the first continues.
- Every request gets exactly one result carrying its goal_id and request_id.
- `approach_caller` never sits and ends stopped and standing; only `sit` sends Sit, only
  after a successful approach of the same goal, with nothing moved since.
- A new goal voids every handoff of the previous one and retires it.
- Cancel, timeouts and e-stop end a request through the existing stop paths (`_abort`,
  `on_estop`); watchdogs, the motion gate, the e-stop and the stick override are unchanged.
- The operator reset clears the skill-sit flag (it stands the robot from DONE, or, after
  an e-stop while seated, records that the operator stood it up).

## Tests

`come_here_behavior/test/test_fsm_skill_api.py` (68 tests: full sequence, each skill,
handoffs, correlation, rejections, a 60-episode random-request property test that no
velocity, turn or sit happens outside its skill), `test_behavior_skill_api.py` (node
wiring, flag off/on, wake not subscribed, latched results),
`come_here_bringup/test/test_skill_api_dry_run.py` (real behavior_node and bridge
processes in dry run: localize, orient, acquire, approach with no Sit, a separate sit
with exactly one Sit, cancel mid-walk, stale / reused / out-of-order requests; nothing on
`/api/sport/request`).

Not run on the robot.
