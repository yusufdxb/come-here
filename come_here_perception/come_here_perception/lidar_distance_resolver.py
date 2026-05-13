"""LiDAR-based person distance resolver for come-here APPROACH phase.

Given a YOLO bearing and a base-frame point cloud, return the person's
range in meters, or None if the quality gate fails (caller falls back
to the YOLO bbox distance).

This class has zero ROS dependencies so it can be unit-tested with
synthetic clouds.
"""

from __future__ import annotations

import numpy as np


class LidarDistanceResolver:
    """Filter a point cloud to a narrow bearing wedge at torso height and
    return a robust person range.

    Args:
        cone_half_rad: Half-angle of the bearing cone (default ±8°).
        z_min, z_max: Torso/legs height band in meters (base_link frame).
        x_min: Minimum forward distance to reject self-returns.
        max_range_m: Discard points beyond this horizontal range.
        min_points: Gate — minimum points required in the wedge.
        min_vertical_extent_m: Gate — minimum (z.max - z.min) across wedge points.
        percentile: Percentile of horizontal range taken as the person distance.
    """

    def __init__(
        self,
        cone_half_rad: float = 0.14,
        z_min: float = 0.3,
        z_max: float = 1.8,
        x_min: float = 0.2,
        max_range_m: float = 5.0,
        min_points: int = 5,
        min_vertical_extent_m: float = 0.6,
        percentile: float = 10.0,
    ):
        self._cone_half_rad = float(cone_half_rad)
        self._z_min = float(z_min)
        self._z_max = float(z_max)
        self._x_min = float(x_min)
        self._max_range_sq = float(max_range_m) ** 2
        self._min_points = int(min_points)
        self._min_vertical_extent_m = float(min_vertical_extent_m)
        self._percentile = float(percentile)

    def refine(
        self,
        bearing_rad: float,
        cloud_xyz: np.ndarray | None,
    ) -> float | None:
        """Return person range (m) or None if the gate fails.

        cloud_xyz: (N, 3) float array in base_link frame (+x forward).
        """
        if cloud_xyz is None or cloud_xyz.shape[0] == 0:
            return None

        x = cloud_xyz[:, 0]
        y = cloud_xyz[:, 1]
        z = cloud_xyz[:, 2]

        az = np.arctan2(y, x)
        in_cone = np.abs(az - float(bearing_rad)) < self._cone_half_rad
        in_height = (z > self._z_min) & (z < self._z_max)
        in_front = x > self._x_min
        in_range = (x * x + y * y) < self._max_range_sq

        mask = in_cone & in_height & in_front & in_range
        if int(mask.sum()) < self._min_points:
            return None

        zs = z[mask]
        if float(zs.max() - zs.min()) < self._min_vertical_extent_m:
            return None

        rs = np.hypot(x[mask], y[mask])
        return float(np.percentile(rs, self._percentile))
