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


def test_person_outside_cone_returns_none():
    resolver = LidarDistanceResolver()
    cloud = _human_column(x_m=2.0, y_m=0.0)
    # Person dead ahead but we ask about bearing 0.5 rad (~28°) — out of ±8° cone.
    dist = resolver.refine(bearing_rad=0.5, cloud_xyz=cloud)
    assert dist is None


def test_too_few_points_returns_none():
    resolver = LidarDistanceResolver()  # min_points=5
    cloud = _human_column(x_m=2.0, n_points=3)
    dist = resolver.refine(bearing_rad=0.0, cloud_xyz=cloud)
    assert dist is None


def test_short_column_returns_none():
    # 30 points but column spans only 0.4 m vertically — fails extent gate (0.6 m).
    resolver = LidarDistanceResolver()
    cloud = _human_column(x_m=2.0, z_lo=0.8, z_hi=1.2, n_points=30)
    dist = resolver.refine(bearing_rad=0.0, cloud_xyz=cloud)
    assert dist is None


def test_floor_only_returns_none():
    # 30 floor points all below z_min — should fail after height mask.
    rng = np.random.default_rng(7)
    n = 30
    xs = rng.uniform(1.0, 3.0, n).astype(np.float32)
    ys = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    zs = np.full(n, 0.05, dtype=np.float32)
    floor = np.stack([xs, ys, zs], axis=-1)
    resolver = LidarDistanceResolver()
    dist = resolver.refine(bearing_rad=0.0, cloud_xyz=floor)
    assert dist is None


def test_background_clutter_does_not_dominate():
    # Person at 2.0 m plus scattered background returns at 4.5 m in cone.
    rng = np.random.default_rng(11)
    person = _human_column(x_m=2.0, n_points=30, rng=rng)
    bg_xs = np.full(40, 4.5, dtype=np.float32)
    bg_ys = rng.uniform(-0.2, 0.2, 40).astype(np.float32)
    bg_zs = rng.uniform(0.4, 1.6, 40).astype(np.float32)
    background = np.stack([bg_xs, bg_ys, bg_zs], axis=-1)
    cloud = np.concatenate([person, background], axis=0)
    resolver = LidarDistanceResolver()
    dist = resolver.refine(bearing_rad=0.0, cloud_xyz=cloud)
    assert dist is not None
    # 10th percentile should land on the person column, not the background.
    assert 1.9 <= dist <= 2.2


def test_empty_cloud_returns_none():
    resolver = LidarDistanceResolver()
    assert resolver.refine(bearing_rad=0.0, cloud_xyz=None) is None
    assert resolver.refine(bearing_rad=0.0,
                           cloud_xyz=np.zeros((0, 3), dtype=np.float32)) is None
