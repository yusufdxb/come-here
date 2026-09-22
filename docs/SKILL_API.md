# Stepwise skill interface (optional, off by default)

`behavior_node` parameter `enable_skill_api` (default `false`). When false, no skill topic
exists and the node behaves exactly as the baseline (tag `baseline/pre-skill-api-f7192c9`);
`test/test_baseline_golden.py` checks every command the state machine emits in 13 deployed
scenarios against a trace recorded from the baseline code. The parameter is not an
`FsmConfig` field, so the `config` snapshot in trial records does not change.

The interface gives a supervisor task-level requests only. It never carries a velocity,
angle, speed or distance: the turn angle is the state machine's own voice bearing and every
motion bound is the existing configuration. The state machine stays the only producer of
`cmd_velocity`, `cmd_rotate` and `cmd_sit`; the bridge is unchanged.

## Topics

`/come_here/skill_request` (std_msgs/String, JSON, reliable, volatile, depth 10)

```json
{"v": 1, "goal_id": "<1..64 printable chars, never reused>", "skill": "...", "args": {}}
```

| skill | args | runs |
|---|---|---|
| `turn_to_voice` | none | LISTENING then one voice turn, using the stored bearing (fresh and confident, else `no_direction`, never a guessed turn); ends stopped |
| `approach_person` | `arrival`: `stop` or `sit_and_identify` (required) | ACQUIRE, ALIGN/WALK, align turns, the chosen arrival; no relisten, no search turns |
| `cancel` | `goal_id`: the active request | the existing abort: zero velocity, trial closed, no e-stop latched |

`/come_here/skill_result` (std_msgs/String, JSON, reliable, **transient_local**, depth 10)

```json
{"v": 1, "goal_id": "...", "skill": "...", "status": "succeeded|failed|cancelled|rejected",
 "reason": "...", "run_id": "<trial run_id or null>", "data": {}}
```

Every request gets exactly one result with its own `goal_id`. A cancel produces two: the
cancelled request (`cancelled`, `data.cancel_goal_id`) and the cancel itself (`succeeded`).

Rejections (`status: rejected`, nothing moves, no trial starts): `malformed`,
`unknown_skill`, `bad_args`, `duplicate_goal_id`, `estopped`, `not_idle` (a trial or a
request is active, or the robot is seated), `not_active` (cancel for a goal that is not the
active one, including a stale goal), `posture_phase` (cancel while sitting or changing
posture; the operator reset owns the stand-up).

## Invariants

- One active request; a second one is rejected and the first continues.
- A result can only ever name the goal that produced it; a stale cancel cannot stop or
  advance a newer request.
- Cancel, timeouts and e-stop end a request through the existing stop paths (`_abort`,
  `on_estop`); watchdogs, the motion gate, the e-stop and the stick override are unchanged.
- A cancel after an arrival but before the requested sit reports `cancelled`
  (`data.stop_reason` keeps the arrival).
- `sit_and_identify` reports `data.posture: "unverified"`: nothing here reads posture.
- The turn residual is used as the next approach's gate center only when that approach
  immediately follows a successful turn request; an e-stop or any other trial clears it.

## Tests

`come_here_behavior/test/test_fsm_skill_api.py` (state machine, fake time),
`test_behavior_skill_api.py` (node wiring, flag off/on, latched results),
`come_here_bringup/test/test_skill_api_dry_run.py` (real behavior_node and bridge processes
in dry run: approach, cancel mid-walk, stale cancel, reused goal_id, voice turn; nothing on
`/api/sport/request`).

Not run on the robot.
