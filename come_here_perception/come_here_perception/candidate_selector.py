"""Pick the caller among several people: the one consistent with the voice direction.

Pure Python. The behavior node publishes a gate (center bearing, half width):
right after the turn the center is where the voice should now be in the camera;
while approaching it follows the tracked caller. A person outside the gate is
never selected, however large or close. Inside the gate the smallest angular
distance to the center wins, then higher confidence, then the taller box.
Nobody inside the gate means nobody is selected: the robot does not walk.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    bearing_rad: float
    distance_m: float
    confidence: float
    bbox_h_frac: float
    box_px: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def angular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def select(candidates: Sequence[Candidate], center_rad: float,
           half_width_rad: float) -> Tuple[Optional[int], List[bool]]:
    """(index of the selected candidate or None, in_gate flag per candidate)."""
    in_gate = [
        half_width_rad > 0.0
        and math.isfinite(c.bearing_rad)
        and angular_distance(c.bearing_rad, center_rad) <= half_width_rad
        for c in candidates
    ]
    ranked = sorted(
        (i for i, ok in enumerate(in_gate) if ok),
        key=lambda i: (round(angular_distance(candidates[i].bearing_rad, center_rad), 4),
                       -candidates[i].confidence, -candidates[i].bbox_h_frac, i),
    )
    return (ranked[0] if ranked else None), in_gate


def largest(candidates: Sequence[Candidate]) -> Optional[int]:
    """Legacy camera-only choice (no gate): the tallest box."""
    if not candidates:
        return None
    return max(range(len(candidates)), key=lambda i: (candidates[i].bbox_h_frac, -i))
