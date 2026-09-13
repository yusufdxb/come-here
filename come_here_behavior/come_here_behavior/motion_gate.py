"""Pure motion-command gate for the GO2 bridge.

No ROS and no Unitree SDK imports, so every rule here is unit-testable in CI.
The bridge feeds it each ``/come_here/cmd_velocity`` message plus a periodic
tick, and turns the returned decisions into Sport API Move / StopMove requests.

Rules, all failing closed:
  * malformed arrays, non-finite values and absurd magnitudes are rejected and
    collapse to a stop;
  * valid commands are clamped to the configured conservative limits;
  * combined forward + yaw commands are rejected unless explicitly allowed,
    because the stock ``mcf`` gait degrades on them (measured 2026-04-17);
  * an armed command that is not refreshed within ``command_timeout_s`` is
    dropped by the watchdog;
  * the operator e-stop latches: while engaged nothing moves, and after it is
    released a zero command is required before motion can re-arm.

Times are plain float seconds from a monotonic clock supplied by the caller.
"""

import math
from dataclasses import dataclass
from typing import Sequence

MOVE = 'move'
STOP = 'stop'
NONE = 'none'


@dataclass(frozen=True)
class GateDecision:
    """What the bridge should do: publish Move, publish StopMove, or nothing."""

    action: str
    vx: float = 0.0
    yaw_rate: float = 0.0
    reason: str = ''


@dataclass(frozen=True)
class GateLimits:
    max_vx: float
    max_yaw_rate: float
    reject_vx_above: float
    reject_yaw_rate_above: float
    command_timeout_s: float
    allow_combined: bool = False
    zero_epsilon: float = 1e-3

    def validate(self) -> None:
        positive = {
            'max_vx': self.max_vx,
            'max_yaw_rate': self.max_yaw_rate,
            'reject_vx_above': self.reject_vx_above,
            'reject_yaw_rate_above': self.reject_yaw_rate_above,
            'command_timeout_s': self.command_timeout_s,
            'zero_epsilon': self.zero_epsilon,
        }
        for name, value in positive.items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name} must be finite and > 0, got {value}')
        if self.reject_vx_above < self.max_vx:
            raise ValueError('reject_vx_above must be >= max_vx')
        if self.reject_yaw_rate_above < self.max_yaw_rate:
            raise ValueError('reject_yaw_rate_above must be >= max_yaw_rate')


class MotionGate:
    """Validates, clamps, times out and e-stop-latches velocity commands."""

    def __init__(self, limits: GateLimits):
        limits.validate()
        self._limits = limits
        self._estopped = False
        self._rearm_required = False
        self._active = False
        self._vx = 0.0
        self._yaw_rate = 0.0
        self._last_command_s = 0.0

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def rearm_required(self) -> bool:
        return self._rearm_required

    @property
    def active(self) -> bool:
        return self._active

    def on_command(self, data: Sequence[float], now_s: float) -> GateDecision:
        """Evaluate one ``[vx, yaw_rate]`` command."""
        if self._estopped:
            return GateDecision(NONE, reason='estopped')
        if len(data) != 2:
            return self._disarm('malformed')
        try:
            vx = float(data[0])
            yaw_rate = float(data[1])
        except (TypeError, ValueError):
            return self._disarm('malformed')
        if not (math.isfinite(vx) and math.isfinite(yaw_rate)):
            return self._disarm('non_finite')

        lim = self._limits
        if abs(vx) > lim.reject_vx_above or abs(yaw_rate) > lim.reject_yaw_rate_above:
            return self._disarm('absurd')

        moving_x = abs(vx) >= lim.zero_epsilon
        moving_yaw = abs(yaw_rate) >= lim.zero_epsilon
        if not moving_x and not moving_yaw:
            self._rearm_required = False
            return self._disarm('zero')
        if self._rearm_required:
            return self._disarm('rearm_required')
        if moving_x and moving_yaw and not lim.allow_combined:
            return self._disarm('combined')

        clamped_vx = max(-lim.max_vx, min(vx, lim.max_vx)) if moving_x else 0.0
        clamped_yaw = (
            max(-lim.max_yaw_rate, min(yaw_rate, lim.max_yaw_rate))
            if moving_yaw else 0.0
        )
        self._active = True
        self._vx = clamped_vx
        self._yaw_rate = clamped_yaw
        self._last_command_s = now_s
        reason = 'clamped' if (clamped_vx != vx or clamped_yaw != yaw_rate) else 'ok'
        return GateDecision(MOVE, clamped_vx, clamped_yaw, reason)

    def on_tick(self, now_s: float) -> GateDecision:
        """Periodic republish of the armed command, or a watchdog stop."""
        if self._estopped or not self._active:
            return GateDecision(NONE)
        if now_s - self._last_command_s > self._limits.command_timeout_s:
            return self._disarm('watchdog')
        return GateDecision(MOVE, self._vx, self._yaw_rate, 'republish')

    def engage_estop(self) -> GateDecision:
        self._estopped = True
        self._rearm_required = True
        return self._disarm('estop')

    def release_estop(self) -> None:
        if self._estopped:
            self._estopped = False
            self._rearm_required = True

    def disarm(self, reason: str) -> GateDecision:
        return self._disarm(reason)

    def _disarm(self, reason: str) -> GateDecision:
        self._active = False
        self._vx = 0.0
        self._yaw_rate = 0.0
        return GateDecision(STOP, reason=reason)
