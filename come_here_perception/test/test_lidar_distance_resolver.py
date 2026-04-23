"""Unit tests for LidarDistanceResolver (pure numpy, no ROS)."""

import numpy as np
import pytest

from come_here_perception.lidar_distance_resolver import LidarDistanceResolver


def _human_column(x_m: float, y_m: float = 0.0,
                  z_lo: float = 0.4, z_hi: float = 1.6,
                  n_points: int = 30,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Return an (n,3) float32 array of points on a human-sized vertical column."""
    rng = rng or np.random.default_rng(42)
    xs = np.full(n_points, x_m, dtype=np.float32) + rng.normal(0, 0.02, n_points).astype(np.float32)
    ys = np.full(n_points, y_m, dtype=np.float32) + rng.normal(0, 0.02, n_points).astype(np.float32)
    zs = np.linspace(z_lo, z_hi, n_points, dtype=np.float32)
    return np.stack([xs, ys, zs], axis=-1)


def test_person_dead_ahead_at_two_meters():
    resolver = LidarDistanceResolver()
    cloud = _human_column(x_m=2.0, y_m=0.0)
    dist = resolver.refine(bearing_rad=0.0, cloud_xyz=cloud)
    assert dist is not None
    assert 1.9 <= dist <= 2.1
