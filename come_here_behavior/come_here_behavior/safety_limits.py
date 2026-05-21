"""Pure numeric safety guards for come-here motion commands.

No ROS dependencies — importable and unit-testable on any machine, including
ones without the Unitree SDK. Both the behavior node and the GO2 bridge use
these helpers so that no externally-sourced number reaches the robot
unclamped or non-finite.
"""

import math


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to the closed interval [lo, hi]."""
    return max(lo, min(value, hi))


def clamp_velocity(
    vx: float,
    yaw_rate: float,
    max_vx: float,
    max_yaw_rate: float,
) -> tuple[float, float]:
    """Clamp a (vx, yaw_rate) command to safe magnitudes.

    A non-finite component (NaN/inf) makes the whole command untrustworthy,
    so the command collapses to a full stop (0.0, 0.0) rather than moving the
    robot on a partially-corrupt setpoint.

    Returns (vx, yaw_rate) as floats.
    """
    if not (math.isfinite(vx) and math.isfinite(yaw_rate)):
        return 0.0, 0.0
    return (
        clamp(vx, -max_vx, max_vx),
        clamp(yaw_rate, -max_yaw_rate, max_yaw_rate),
    )
