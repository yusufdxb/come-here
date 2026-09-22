"""Come Here ANY: the Unitree native obstacle-avoidance motion backend (pure Python).

No ROS and no Unitree SDK imports, so every rule here is unit-tested in CI.
``native_avoid_bridge_node`` turns the ``ApiCall`` objects returned here into
``unitree_api/Request`` messages and feeds the responses back.

Two native interfaces exist in the Unitree SDK (unitree_sdk2py 1.0.1, identical
to upstream master for these files on 2026-09-21):

``sport_freeavoid``
    Sport service api 2048 ``FreeAvoid({"data": bool})`` on /api/sport/request.
    Motion stays on Sport ``Move`` 1008 ``{"x", "y", "z"}``, the same request the
    legacy demo sends. No getter exists, so the prior state cannot be read back.
    Returned code 0 on the lab GO2 (mcf) on 2026-09-21. Whether it changes how
    ordinary Sport Move commands react to obstacles is NOT established.

``obstacles_avoid``
    The dedicated ``obstacles_avoid`` service on /api/obstacles_avoid/request
    (SDK api version 1.0.0.2): SwitchSet 1001 ``{"enable"}``, SwitchGet 1002,
    Move 1003 ``{"x", "y", "yaw", "mode": 0}`` (noreply), UseRemoteCommandFromApi
    1004 ``{"is_remote_commands_from_api"}``. The official example enables the
    switch, takes API remote-command ownership, then moves through this client.
    SwitchSet/SwitchGet returned code 0 on the lab GO2 on 2026-09-21; a zero
    Move with the switch on produced no motion. Moving through it is NOT tested.

Sport ``SwitchAvoidMode`` (2058, no parameters, a toggle with no getter) is
deliberately not used: a blind toggle cannot be restored to a known state.

Lifecycle (one backend instance per bridge, one motion owner at a time)::

    DISABLED --begin_enable--> ENABLING --every step code 0--> ENABLED
    ENABLING --error / timeout / wrong read-back--> FAILED (sticky until restart)
    ENABLED --suspend (e-stop)--> DISABLED (obstacles_avoid: API control released)

Stopping never waits for a reply: ``stop_calls`` and ``release_calls`` are sent
fire-and-forget, in order, because they run on e-stop and shutdown.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

SPORT = 'sport'
OBSTACLES_AVOID = 'obstacles_avoid'

REQUEST_TOPICS = {SPORT: '/api/sport/request', OBSTACLES_AVOID: '/api/obstacles_avoid/request'}
RESPONSE_TOPICS = {SPORT: '/api/sport/response',
                   OBSTACLES_AVOID: '/api/obstacles_avoid/response'}
DRY_RUN_REQUEST_TOPICS = {SPORT: '/come_here/dry_run/sport_request',
                          OBSTACLES_AVOID: '/come_here/dry_run/obstacles_avoid_request'}

# Internal RPC api id on every Unitree service: GetServerApiVersion (rpc/internal.py).
RPC_API_ID_API_VERSION = 1

SPORT_API_STOPMOVE = 1003
SPORT_API_MOVE = 1008
SPORT_API_FREEAVOID = 2048
SPORT_API_SWITCHAVOIDMODE = 2058   # documented only; never sent

OA_API_SWITCH_SET = 1001
OA_API_SWITCH_GET = 1002
OA_API_MOVE = 1003
OA_API_USE_REMOTE_COMMAND_FROM_API = 1004

SDK_API_VERSIONS = {SPORT: '1.0.0.1', OBSTACLES_AVOID: '1.0.0.2'}

BACKEND_FREEAVOID = 'sport_freeavoid'
BACKEND_OBSTACLES_AVOID = 'obstacles_avoid'
BACKENDS = (BACKEND_FREEAVOID, BACKEND_OBSTACLES_AVOID)

DISABLED = 'disabled'
ENABLING = 'enabling'
ENABLED = 'enabled'
FAILED = 'failed'


@dataclass(frozen=True)
class ApiCall:
    """One request to publish: service, api id, JSON parameters, reply expected."""

    service: str
    api_id: int
    params: Optional[dict] = None
    noreply: bool = False
    label: str = ''

    def parameter_json(self) -> str:
        return '' if self.params is None else json.dumps(self.params)


@dataclass
class _Step:
    call: ApiCall
    required: bool = True                   # False: a failure is recorded, not fatal
    expect: Optional[dict] = None           # response JSON fields that must match
    record: Optional[str] = None            # store the response under this key


@dataclass
class BackendStatus:
    backend: str
    state: str
    enable_result: Optional[str]
    server_version: Optional[str]
    api_version_match: Optional[bool]
    api_control_taken: bool
    api_control_released: bool
    initial_switch: Optional[bool]
    simulated: bool
    failure: Optional[str]
    log: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            'native_avoid_backend': self.backend,
            'native_avoid_state': self.state,
            'native_avoid_enabled': self.state == ENABLED,
            'native_avoid_enable_result': self.enable_result,
            'native_avoid_server_version': self.server_version,
            'native_avoid_api_version_match': self.api_version_match,
            'api_control_taken': self.api_control_taken,
            'api_control_released': self.api_control_released,
            'native_avoid_initial_switch': self.initial_switch,
            'native_avoid_simulated': self.simulated,
            'native_avoid_failure': self.failure,
        }


def move_call(backend: str, vx: float, vy: float, yaw_rate: float) -> ApiCall:
    """The single motion request for ``backend``; the legacy Move shape for freeavoid."""
    for v in (vx, vy, yaw_rate):
        if not math.isfinite(v):
            raise ValueError('non-finite motion command')
    if backend == BACKEND_FREEAVOID:
        return ApiCall(SPORT, SPORT_API_MOVE,
                       {'x': float(vx), 'y': float(vy), 'z': float(yaw_rate)}, False, 'move')
    if backend == BACKEND_OBSTACLES_AVOID:
        return ApiCall(OBSTACLES_AVOID, OA_API_MOVE,
                       {'x': float(vx), 'y': float(vy), 'yaw': float(yaw_rate), 'mode': 0},
                       True, 'oa_move')
    raise ValueError(f'unknown native avoid backend {backend!r}')


STOPMOVE = ApiCall(SPORT, SPORT_API_STOPMOVE, None, False, 'stop_move')


class NativeAvoidBackend:
    """Enable / stop / release sequencing for one native avoidance interface."""

    def __init__(self, backend: str, *, response_timeout_s: float = 1.5,
                 use_remote_command_from_api: bool = True,
                 restore_switch_on_release: bool = True,
                 disable_freeavoid_on_release: bool = True,
                 stop_with_sport_stopmove: bool = True,
                 simulate_responses: bool = False):
        if backend not in BACKENDS:
            raise ValueError(f'native_avoid_backend must be one of {BACKENDS}, got {backend!r}')
        if not (math.isfinite(response_timeout_s) and response_timeout_s > 0.0):
            raise ValueError('response_timeout_s must be > 0')
        self.backend = backend
        self._timeout_s = response_timeout_s
        self._use_remote = bool(use_remote_command_from_api)
        self._restore_switch = bool(restore_switch_on_release)
        self._disable_freeavoid = bool(disable_freeavoid_on_release)
        self._stop_with_stopmove = bool(stop_with_sport_stopmove)
        self._simulated = bool(simulate_responses)

        self._state = DISABLED
        self._steps: List[_Step] = []
        self._pending: Optional[_Step] = None
        self._pending_id: Optional[int] = None
        self._pending_since = 0.0
        self._next_id = 1
        self._enable_result: Optional[str] = None
        self._failure: Optional[str] = None
        self._server_version: Optional[str] = None
        self._initial_switch: Optional[bool] = None
        self._api_control_taken = False
        self._api_control_released = False
        self._freeavoid_set = False
        self._switch_set = False
        # Set when the request is PUBLISHED: a lost or late reply may still have
        # applied it, so release / restore must undo anything attempted.
        self._freeavoid_attempted = False
        self._switch_attempted = False
        self._api_control_attempted = False
        self._log: List[str] = []

    # -- read-only --

    @property
    def state(self) -> str:
        return self._state

    @property
    def enabled(self) -> bool:
        return self._state == ENABLED

    @property
    def failed(self) -> bool:
        return self._state == FAILED

    @property
    def api_control_taken(self) -> bool:
        return self._api_control_taken

    def status(self) -> BackendStatus:
        match = None
        if self._server_version is not None and not self._simulated:
            match = self._server_version == SDK_API_VERSIONS[self._service()]
        return BackendStatus(
            backend=self.backend, state=self._state, enable_result=self._enable_result,
            server_version=self._server_version, api_version_match=match,
            api_control_taken=self._api_control_taken,
            api_control_released=self._api_control_released,
            initial_switch=self._initial_switch, simulated=self._simulated,
            failure=self._failure, log=self.drain_log())

    def drain_log(self) -> List[str]:
        lines, self._log = self._log, []
        return lines

    # -- request ids (the bridge stamps each Request with the id returned here) --

    def new_request_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    # -- enable --

    def begin_enable(self, now: float) -> List[ApiCall]:
        """Start the enable sequence; returns the first call to publish (or none)."""
        if self._state != DISABLED:
            return []
        self._state = ENABLING
        self._enable_result = None
        self._failure = None
        service = self._service()
        self._steps = [_Step(ApiCall(service, RPC_API_ID_API_VERSION, {}, False, 'api_version'),
                             required=False, record='version')]
        if self.backend == BACKEND_FREEAVOID:
            self._steps.append(_Step(ApiCall(SPORT, SPORT_API_FREEAVOID, {'data': True},
                                             False, 'free_avoid_on')))
        else:
            self._steps += [
                _Step(ApiCall(OBSTACLES_AVOID, OA_API_SWITCH_GET, {}, False, 'switch_get_initial'),
                      record='initial_switch'),
                _Step(ApiCall(OBSTACLES_AVOID, OA_API_SWITCH_SET, {'enable': True}, False,
                              'switch_set_on')),
                _Step(ApiCall(OBSTACLES_AVOID, OA_API_SWITCH_GET, {}, False, 'switch_get_verify'),
                      expect={'enable': True}),
            ]
            if self._use_remote:
                self._steps.append(_Step(ApiCall(
                    OBSTACLES_AVOID, OA_API_USE_REMOTE_COMMAND_FROM_API,
                    {'is_remote_commands_from_api': True}, False, 'take_api_control')))
        self._log.append(f'native avoid ({self.backend}): enabling'
                         + (' [DRY RUN: responses simulated]' if self._simulated else ''))
        return self._advance(now)

    def on_published(self, request_id: int, now: float) -> None:
        """The bridge published the pending call with ``request_id``."""
        if self._pending is None or self._pending_id is not None:
            return
        self._pending_id = request_id
        call = self._pending.call
        if call.service == SPORT and call.api_id == SPORT_API_FREEAVOID:
            self._freeavoid_attempted = True
        elif call.service == OBSTACLES_AVOID and call.api_id == OA_API_SWITCH_SET:
            self._switch_attempted = True
        elif (call.service == OBSTACLES_AVOID
              and call.api_id == OA_API_USE_REMOTE_COMMAND_FROM_API):
            self._api_control_attempted = True

    def on_response(self, service: str, request_id: int, api_id: int, code: int,
                    data: str, now: float) -> List[ApiCall]:
        """Feed one Response; returns the next call to publish, if any."""
        step = self._pending
        if (step is None or self._pending_id is None or request_id != self._pending_id
                or service != step.call.service or api_id != step.call.api_id):
            return []
        return self._complete(step, code, data, now)

    def simulated_reply(self, now: float) -> List[ApiCall]:
        """Dry run: acknowledge the pending call as if the robot answered code 0."""
        step = self._pending
        if not self._simulated or step is None:
            return []
        data = '{}'
        if step.call.api_id == OA_API_SWITCH_GET and step.call.service == OBSTACLES_AVOID:
            data = json.dumps({'enable': self._switch_set})
        elif step.call.api_id == RPC_API_ID_API_VERSION:
            data = 'dry_run'
        return self._complete(step, 0, data, now)

    def tick(self, now: float) -> List[ApiCall]:
        """Time out a pending enable step (no reply = failure, fail closed).

        Returns the next call to publish when a non-fatal step timed out."""
        if (self._state == ENABLING and self._pending is not None
                and now - self._pending_since > self._timeout_s):
            step = self._pending
            if step.required:
                self._fail(f'no reply to {step.call.label} within {self._timeout_s:.1f}s')
                return []
            self._log.append(f'native avoid: {step.call.label} not answered (non-fatal)')
            self._pending = None
            self._pending_id = None
            return self._advance(now)
        return []

    def _complete(self, step: _Step, code: int, data: str, now: float) -> List[ApiCall]:
        self._pending = None
        self._pending_id = None
        label = step.call.label
        if code != 0:
            if step.required:
                self._fail(f'{label} returned code {code}')
                return []
            self._log.append(f'native avoid: {label} returned code {code} (non-fatal)')
            return self._advance(now)
        payload = None
        if step.record == 'version':
            self._server_version = _parse_version(data)
            self._log.append(f'native avoid: {self._service()} server api version '
                             f'{self._server_version!r}, SDK {SDK_API_VERSIONS[self._service()]}')
        else:
            payload = _parse_json(data)
        if step.record == 'initial_switch':
            enable = None if payload is None else payload.get('enable')
            if not isinstance(enable, bool):
                self._fail(f'{label}: unreadable reply {data!r}')
                return []
            if self._initial_switch is None and not self._switch_attempted:
                # Only the state before WE ever touched it is the one to restore.
                self._initial_switch = enable
        if step.expect:
            for key, value in step.expect.items():
                if payload is None or payload.get(key) != value:
                    self._fail(f'{label}: expected {key}={value}, got {data!r}')
                    return []
        if step.call.api_id == SPORT_API_FREEAVOID and step.call.service == SPORT:
            self._freeavoid_set = True
        if step.call.api_id == OA_API_SWITCH_SET and step.call.service == OBSTACLES_AVOID:
            self._switch_set = True
        if (step.call.api_id == OA_API_USE_REMOTE_COMMAND_FROM_API
                and step.call.service == OBSTACLES_AVOID):
            self._api_control_taken = True
            self._api_control_released = False
        return self._advance(now)

    def _advance(self, now: float) -> List[ApiCall]:
        if self._state != ENABLING:
            return []
        if not self._steps:
            self._state = ENABLED
            self._enable_result = 'dry_run_simulated' if self._simulated else 'ok'
            self._log.append(f'native avoid ({self.backend}): ENABLED'
                             + (' (simulated, nothing sent to the robot)' if self._simulated
                                else ''))
            return []
        self._pending = self._steps.pop(0)
        self._pending_id = None
        self._pending_since = now
        return [self._pending.call]

    def _fail(self, reason: str) -> None:
        self._state = FAILED
        self._failure = reason
        self._enable_result = 'failed'
        self._steps = []
        self._pending = None
        self._pending_id = None
        self._log.append(f'native avoid ({self.backend}) FAILED: {reason}')

    # -- motion, stop, release --

    def move(self, vx: float, vy: float, yaw_rate: float) -> List[ApiCall]:
        """The motion request, or nothing unless ENABLED (never move un-enabled)."""
        if self._state != ENABLED:
            return []
        return [move_call(self.backend, vx, vy, yaw_rate)]

    def stop_calls(self) -> List[ApiCall]:
        """Zero the native command, then the known-good Sport StopMove backstop."""
        calls = []
        if self.backend == BACKEND_OBSTACLES_AVOID:
            calls.append(move_call(self.backend, 0.0, 0.0, 0.0))
        if self.backend == BACKEND_FREEAVOID or self._stop_with_stopmove:
            calls.append(STOPMOVE)
        return calls

    def suspend(self) -> List[ApiCall]:
        """E-stop / manual override: stop and hand remote authority back.

        The backend returns to DISABLED (unless FAILED) and must be re-enabled
        before motion; obstacles_avoid API control is released immediately.
        """
        calls = self.stop_calls()
        if self._api_control_taken or self._api_control_attempted:
            calls.append(self._release_api_control())
        if self._state in (ENABLING, ENABLED):
            self._state = DISABLED
            self._pending = None
            self._pending_id = None
            self._steps = []
        return calls

    def release_calls(self) -> List[ApiCall]:
        """Shutdown: stop, give API control back, restore / disable avoidance."""
        calls = self.stop_calls()
        if self.backend == BACKEND_OBSTACLES_AVOID:
            if self._api_control_taken or self._api_control_attempted:
                calls.append(self._release_api_control())
            if (self._switch_set or self._switch_attempted) and self._restore_switch:
                # Unknown prior state (never read) restores to off, the safe default.
                initial = False if self._initial_switch is None else self._initial_switch
                calls.append(ApiCall(OBSTACLES_AVOID, OA_API_SWITCH_SET,
                                     {'enable': initial}, False, 'switch_restore'))
        elif (self._freeavoid_set or self._freeavoid_attempted) and self._disable_freeavoid:
            calls.append(ApiCall(SPORT, SPORT_API_FREEAVOID, {'data': False}, False,
                                 'free_avoid_off'))
        if self._state != FAILED:
            self._state = DISABLED
        self._pending = None
        self._pending_id = None
        self._steps = []
        return calls

    def _release_api_control(self) -> ApiCall:
        self._api_control_taken = False
        self._api_control_attempted = False
        self._api_control_released = True
        return ApiCall(OBSTACLES_AVOID, OA_API_USE_REMOTE_COMMAND_FROM_API,
                       {'is_remote_commands_from_api': False}, False, 'release_api_control')

    def _service(self) -> str:
        return SPORT if self.backend == BACKEND_FREEAVOID else OBSTACLES_AVOID


def _parse_json(data: str) -> Optional[dict]:
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_version(data: str) -> Optional[str]:
    """Server version reply: a bare string, a JSON string, or {"api_version": ...}."""
    if data is None:
        return None
    text = str(data).strip()
    payload = None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        pass
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ('api_version', 'version', 'data'):
            if isinstance(payload.get(key), str):
                return payload[key]
        return text
    return text or None


# -- 3-axis motion gate ------------------------------------------------------

@dataclass(frozen=True)
class NativeGateLimits:
    max_vx: float
    max_vy: float
    max_yaw_rate: float
    reject_vx_above: float
    reject_vy_above: float
    reject_yaw_rate_above: float
    command_timeout_s: float
    allow_combined: bool = False     # translation and yaw in the same command
    allow_lateral: bool = False      # any vy at all
    zero_epsilon: float = 1e-3

    def validate(self) -> None:
        for name in ('max_vx', 'max_vy', 'max_yaw_rate', 'reject_vx_above', 'reject_vy_above',
                     'reject_yaw_rate_above', 'command_timeout_s', 'zero_epsilon'):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name} must be finite and > 0, got {value}')
        if (self.reject_vx_above < self.max_vx or self.reject_vy_above < self.max_vy
                or self.reject_yaw_rate_above < self.max_yaw_rate):
            raise ValueError('reject_*_above must be >= the matching max_*')


MOVE = 'move'
STOP = 'stop'
NONE = 'none'


@dataclass(frozen=True)
class NativeGateDecision:
    action: str
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    reason: str = ''


# Inhibit reasons the legacy bridge sets and clears through inhibit(name) / inhibit(None).
MODE_INHIBITS = ('motion_mode_unverified', 'motion_mode')


class NativeMotionGate:
    """``[vx, vy, yaw_rate]`` gate for the native backend.

    Same fail-closed rules as the legacy ``MotionGate`` (validation, clamps,
    watchdog, latched e-stop, re-arm on zero), plus: a 2-element legacy command
    is malformed here (and a 3-element ANY command is malformed in the legacy
    gate), lateral and combined motion each need an explicit flag, and several
    inhibit reasons can be active at once. It exposes the attributes and
    methods ``Go2BridgeNode`` uses on its gate, so the ANY bridge can swap it in.
    """

    def __init__(self, limits: NativeGateLimits):
        limits.validate()
        self._limits = limits
        self._estopped = False
        self._inhibits: Dict[str, bool] = {}
        self._rearm_required = False
        self._active = False
        self._cmd = (0.0, 0.0, 0.0)
        self._last_command_s = 0.0

    @property
    def limits(self) -> NativeGateLimits:
        return self._limits

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def inhibited(self) -> bool:
        return bool(self._inhibits)

    @property
    def inhibit_reason(self) -> Optional[str]:
        return ','.join(sorted(self._inhibits)) or None

    @property
    def rearm_required(self) -> bool:
        return self._rearm_required

    @property
    def active(self) -> bool:
        return self._active

    def on_command(self, data: Sequence[float], now_s: float) -> NativeGateDecision:
        if self._estopped:
            return NativeGateDecision(NONE, reason='estopped')
        if self._inhibits:
            return self._disarm(f'inhibited:{self.inhibit_reason}')
        if len(data) != 3:
            return self._disarm('malformed')
        try:
            vx, vy, yaw = (float(v) for v in data)
        except (TypeError, ValueError):
            return self._disarm('malformed')
        if not all(math.isfinite(v) for v in (vx, vy, yaw)):
            return self._disarm('non_finite')
        lim = self._limits
        if (abs(vx) > lim.reject_vx_above or abs(vy) > lim.reject_vy_above
                or abs(yaw) > lim.reject_yaw_rate_above):
            return self._disarm('absurd')
        eps = lim.zero_epsilon
        moving_x, moving_y, moving_yaw = abs(vx) >= eps, abs(vy) >= eps, abs(yaw) >= eps
        if not (moving_x or moving_y or moving_yaw):
            self._rearm_required = False
            return self._disarm('zero')
        if self._rearm_required:
            return self._disarm('rearm_required')
        if moving_y and not lim.allow_lateral:
            return self._disarm('lateral')
        if (moving_x or moving_y) and moving_yaw and not lim.allow_combined:
            return self._disarm('combined')
        cvx = max(-lim.max_vx, min(vx, lim.max_vx)) if moving_x else 0.0
        cvy = max(-lim.max_vy, min(vy, lim.max_vy)) if moving_y else 0.0
        cyaw = max(-lim.max_yaw_rate, min(yaw, lim.max_yaw_rate)) if moving_yaw else 0.0
        self._active = True
        self._cmd = (cvx, cvy, cyaw)
        self._last_command_s = now_s
        reason = 'clamped' if (cvx, cvy, cyaw) != (vx, vy, yaw) else 'ok'
        return NativeGateDecision(MOVE, cvx, cvy, cyaw, reason)

    def on_tick(self, now_s: float) -> NativeGateDecision:
        if self._estopped or self._inhibits or not self._active:
            return NativeGateDecision(NONE)
        if now_s - self._last_command_s > self._limits.command_timeout_s:
            return self._disarm('watchdog')
        return NativeGateDecision(MOVE, *self._cmd, reason='republish')

    def engage_estop(self) -> NativeGateDecision:
        self._estopped = True
        self._rearm_required = True
        return self._disarm('estop')

    def release_estop(self) -> None:
        if self._estopped:
            self._estopped = False
            self._rearm_required = True

    def inhibit(self, reason: Optional[str]) -> NativeGateDecision:
        """Legacy bridge API: a mode reason, or None to clear the mode reasons only."""
        if reason is None:
            for key in MODE_INHIBITS:
                self._inhibits.pop(key, None)
            return NativeGateDecision(NONE)
        return self.set_inhibit(reason, True)

    def set_inhibit(self, reason: str, active: bool) -> NativeGateDecision:
        if not active:
            self._inhibits.pop(reason, None)
            return NativeGateDecision(NONE)
        self._inhibits[reason] = True
        self._rearm_required = True
        return self._disarm(f'inhibited:{reason}')

    def disarm(self, reason: str) -> NativeGateDecision:
        return self._disarm(reason)

    def _disarm(self, reason: str) -> NativeGateDecision:
        self._active = False
        self._cmd = (0.0, 0.0, 0.0)
        return NativeGateDecision(STOP, reason=reason)
