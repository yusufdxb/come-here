"""Short-lived caller position estimate in the odometry frame (pure Python, no TF).

The GO2 has no TF tree here, so the transform is written out and tested:

    caller_x = robot_x + range * cos(robot_yaw + bearing)
    caller_y = robot_y + range * sin(robot_yaw + bearing)

and back, for a later robot pose:

    range   = hypot(caller_x - robot_x, caller_y - robot_y)
    bearing = wrap(atan2(caller_y - robot_y, caller_x - robot_x) - robot_yaw)

Bearing is positive to the robot's left (the perception convention). Only a
visual observation with a real range (LiDAR or pinhole) updates the estimate.
A prediction is valid for ``ttl_s``; its 1-sigma position uncertainty grows
from ``base_sigma_m`` at ``sigma_growth_mps`` (the caller may walk) and the
prediction is refused beyond ``max_sigma_m``. A prediction never counts as
seeing the caller: callers use it to keep steering briefly, never to arrive.
"""

import math
from dataclasses import dataclass
from typing import Optional


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x, self.y, self.yaw))


@dataclass(frozen=True)
class CallerPrediction:
    bearing_rad: float
    range_m: float
    age_s: float
    sigma_m: float


def caller_position(pose: Pose2D, bearing_rad: float, range_m: float):
    heading = pose.yaw + bearing_rad
    return (pose.x + range_m * math.cos(heading), pose.y + range_m * math.sin(heading))


def relative_polar(pose: Pose2D, cx: float, cy: float):
    dx, dy = cx - pose.x, cy - pose.y
    return wrap_pi(math.atan2(dy, dx) - pose.yaw), math.hypot(dx, dy)


class CallerEstimate:
    def __init__(self, ttl_s: float = 1.5, base_sigma_m: float = 0.25,
                 sigma_growth_mps: float = 0.8, max_sigma_m: float = 1.2):
        for name, value in (('ttl_s', ttl_s), ('base_sigma_m', base_sigma_m),
                            ('max_sigma_m', max_sigma_m)):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f'{name} must be > 0')
        if not (math.isfinite(sigma_growth_mps) and sigma_growth_mps >= 0.0):
            raise ValueError('sigma_growth_mps must be >= 0')
        self.ttl_s = ttl_s
        self.base_sigma_m = base_sigma_m
        self.sigma_growth_mps = sigma_growth_mps
        self.max_sigma_m = max_sigma_m
        self.reset()

    def reset(self) -> None:
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._t: Optional[float] = None

    @property
    def has_fix(self) -> bool:
        return self._t is not None

    def age(self, now: float) -> Optional[float]:
        return None if self._t is None else now - self._t

    def position(self):
        return None if self._t is None else (self._cx, self._cy)

    def observe(self, pose: Pose2D, bearing_rad: float, range_m: float, now: float) -> bool:
        """Record a visual fix; False (ignored) without a usable range or pose."""
        if (not pose.finite() or not math.isfinite(bearing_rad) or not math.isfinite(range_m)
                or range_m <= 0.0):
            return False
        self._cx, self._cy = caller_position(pose, bearing_rad, range_m)
        self._t = now
        return True

    def sigma(self, now: float) -> Optional[float]:
        age = self.age(now)
        return None if age is None else self.base_sigma_m + self.sigma_growth_mps * max(0.0, age)

    def predict(self, pose: Pose2D, now: float) -> Optional[CallerPrediction]:
        """Bearing/range from ``pose`` to the last fix, or None (no fix, expired, uncertain)."""
        if self._t is None or not pose.finite():
            return None
        age = now - self._t
        if age < 0.0 or age > self.ttl_s:
            return None
        sigma = self.sigma(now)
        if sigma > self.max_sigma_m:
            return None
        bearing, rng = relative_polar(pose, self._cx, self._cy)
        if rng <= 1e-6:
            return None
        return CallerPrediction(bearing, rng, age, sigma)

    def consistent(self, pose: Pose2D, bearing_rad: float, range_m: float, now: float,
                   max_jump_m: float, memory_s: float, growth_mps: float = 0.0) -> bool:
        """Could this observation be the same caller? True when there is no recent fix
        to compare against (or no range to place it).

        The allowance does NOT grow with the prediction uncertainty by default: a
        person far from the last fix is treated as someone else, and the robot
        stops (caller lost) rather than adopting them."""
        if self._t is None or now - self._t > memory_s:
            return True
        if not pose.finite() or not math.isfinite(range_m) or range_m <= 0.0:
            return True
        ox, oy = caller_position(pose, bearing_rad, range_m)
        jump = math.hypot(ox - self._cx, oy - self._cy)
        allowed = max_jump_m + growth_mps * max(0.0, now - self._t)
        return jump <= allowed
