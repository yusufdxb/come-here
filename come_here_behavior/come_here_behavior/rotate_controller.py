"""Closed-loop turn-in-place controller for TURN_TO_SOUND. No ROS imports.

The bridge used to rotate open loop: yaw rate times a calibrated duration.
On the mcf gait that is not repeatable (the rate takes time to latch, and a
clockwise turn does not mirror a counter-clockwise one), so the robot ended
up facing somewhere near the caller, not at the caller. This controller
turns until the robot's own odometry yaw has moved by the commanded angle.

Contract, per ``step(now, yaw, yaw_stamp)``:
  * closed loop when a yaw reading was available at construction: turn at
    ``yaw_rate`` in the chosen direction until the accumulated yaw change
    reaches ``target`` minus ``deadband``, or overshoots it;
  * stale odometry (older than ``odom_max_age_s``) ends the turn immediately,
    because a turn nobody can measure is a turn nobody can stop accurately;
  * without any yaw reading the turn falls back to the timed rotation, so the
    behaviour never blocks on a missing topic;
  * ``timeout_s`` bounds every turn.

Direction: the sign of ``target``, except that a target beyond
``prefer_ccw_beyond_rad`` turns counter-clockwise the long way round if
needed, because the mcf gait steps cleanly to the left and only twists to the
right at some rates (hardware, 2026-04-17). Yaw and rates follow ROS:
positive is left / counter-clockwise.
"""

import math
from dataclasses import dataclass
from typing import Optional

TWO_PI = 2.0 * math.pi


def wrap_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class RotateStep:
    yaw_rate: float   # command to publish now (0.0 when done)
    done: bool
    reason: str       # turning | reached | overshoot | timeout | odom_stale | timed


class RotateController:
    def __init__(
        self,
        target_rad: float,
        yaw_rate: float,
        start_s: float,
        start_yaw: Optional[float],
        deadband_rad: float = 0.12,
        timeout_s: float = 6.0,
        odom_max_age_s: float = 0.5,
        fallback_deg_per_sec: float = 57.0,
        prefer_ccw_beyond_rad: float = 2.6,
    ):
        if not (math.isfinite(target_rad) and abs(target_rad) <= math.pi):
            raise ValueError(f'target_rad out of range: {target_rad}')
        if not (math.isfinite(yaw_rate) and yaw_rate > 0.0):
            raise ValueError('yaw_rate must be > 0')
        if deadband_rad < 0.0 or timeout_s <= 0.0 or odom_max_age_s <= 0.0:
            raise ValueError('deadband_rad >= 0, timeout_s > 0, odom_max_age_s > 0')
        self._rate = float(yaw_rate)
        self._start_s = float(start_s)
        self._deadband = float(deadband_rad)
        self._timeout = float(timeout_s)
        self._odom_max_age = float(odom_max_age_s)

        if target_rad >= 0.0 or abs(target_rad) >= prefer_ccw_beyond_rad:
            self._sign = 1.0
            self._magnitude = target_rad if target_rad >= 0.0 else TWO_PI - abs(target_rad)
        else:
            self._sign = -1.0
            self._magnitude = abs(target_rad)

        self._closed_loop = start_yaw is not None
        self._prev_yaw = start_yaw
        self._turned = 0.0
        self._duration = max(0.3, min(math.degrees(self._magnitude)
                                      / max(1e-6, fallback_deg_per_sec), 4.0))
        self._done: Optional[RotateStep] = None

    @property
    def closed_loop(self) -> bool:
        return self._closed_loop

    @property
    def sign(self) -> float:
        return self._sign

    @property
    def magnitude_rad(self) -> float:
        """Angle to turn in the chosen direction (can exceed pi for the long way)."""
        return self._magnitude

    @property
    def turned_rad(self) -> float:
        """Signed yaw change measured so far (closed loop only)."""
        return self._turned

    @property
    def timed_duration_s(self) -> float:
        return self._duration

    def step(self, now: float, yaw: Optional[float] = None,
             yaw_stamp_s: Optional[float] = None) -> RotateStep:
        if self._done is not None:
            return self._done
        elapsed = now - self._start_s
        if elapsed > self._timeout:
            return self._finish('timeout')
        if not self._closed_loop:
            if elapsed >= self._duration:
                return self._finish('timed')
            return RotateStep(self._sign * self._rate, False, 'turning')

        if yaw is None or yaw_stamp_s is None or now - yaw_stamp_s > self._odom_max_age:
            return self._finish('odom_stale')
        self._turned += wrap_pi(yaw - self._prev_yaw)
        self._prev_yaw = yaw
        remaining = self._magnitude - self._sign * self._turned
        if remaining <= self._deadband:
            return self._finish('reached' if remaining >= -self._deadband else 'overshoot')
        return RotateStep(self._sign * self._rate, False, 'turning')

    def _finish(self, reason: str) -> RotateStep:
        self._done = RotateStep(0.0, True, reason)
        return self._done
