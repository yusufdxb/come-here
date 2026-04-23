# Come-Here LiDAR Distance Source — Design

**Date:** 2026-04-23
**Branch:** `lab/2026-04-16-approach-fix`
**Context:** Resolves blocker #1 from the 2026-04-21 hardware session — pinhole/bbox distance estimate underestimates person range by ~2× when only legs are in frame, causing the GO2 to overshoot `approach_stop_distance_m = 0.8 m` and stop at ~0.4 m.

## Goal

Replace the pinhole/bbox-height distance estimate in `yolo_person_detector.py` with a LiDAR-based range for the detected person, published through the existing `/come_here/person_detection` contract so the behavior FSM is unchanged.

Non-goals:
- No change to bearing estimation (YOLO bbox center stays authoritative for "which direction").
- No change to message contract, launch files, or `behavior_node.py`.
- No camera intrinsics calibration or camera↔lidar TF work.
- No change to the EMA bearing smoothing (blocker #2 — separate future change).

## Hardware context (verified 2026-04-23 on Jetson `192.168.0.2`)

| Fact | Value |
|---|---|
| LiDAR topic | `/utlidar/cloud_base` |
| Message type | `sensor_msgs/msg/PointCloud2` |
| Frame | `base_link` (axis-aligned with YOLO bearing, **no TF needed**) |
| Rate | 15.4 Hz |
| Points / frame | ~871 (already filtered/downsampled by Unitree stack) |
| QoS | RELIABLE, KEEP_LAST, depth 1 |
| Fields | `x, y, z, intensity, ring, time` (point_step 32 bytes) |

Camera stays on `/camera/image_raw` published by standalone `go2_video_publisher.py` (must be launched manually — see RUN_NOTES).

## Approach — bearing cone + torso-height gate

YOLO answers "where is the person" (bearing). LiDAR answers "how far is that bearing" (range). YOLO's bearing filters LiDAR's points, so the returned range is person-specific, not nearest-anything.

Filter points in the most recent cached `/utlidar/cloud_base`:

```
in_cone   = abs(atan2(y, x) − bearing_rad) < 0.14   # ±8° absorbs bbox jitter
in_height = (z > 0.3) & (z < 1.8)                   # torso/legs; rejects floor, ceiling, tables
in_front  = x > 0.2                                 # rejects self-returns
in_range  = (x*x + y*y) < 25                        # ≤5 m (demo room)
wedge = in_cone & in_height & in_front & in_range
```

Gate (must all pass, else fall back to YOLO bbox distance):
- ≥5 points in wedge
- Vertical extent `z.max − z.min ≥ 0.6 m` — requires a human-shaped column, not a chair or low obstacle

Distance = `percentile(sqrt(x² + y²), 10)` over wedge points — robust to stray background returns.

## Architecture

One new file, one modified node.

**New:** `come_here_perception/come_here_perception/lidar_distance_resolver.py`

Pure numpy class. Zero ROS deps. Unit-testable with synthetic point clouds.

```python
class LidarDistanceResolver:
    def __init__(self, cone_half_rad=0.14, z_min=0.3, z_max=1.8,
                 x_min=0.2, max_range_m=5.0, min_points=5,
                 min_vertical_extent_m=0.6, percentile=10.0):
        ...

    def refine(self, bearing_rad: float, cloud_xyz: np.ndarray | None
               ) -> float | None:
        """Return person distance in meters, or None if gate fails.

        cloud_xyz: (N, 3) float32 array in base_link frame, or None.
        """
```

**Modified:** `come_here_perception/come_here_perception/perception_node.py`

- Adds param `use_lidar_distance` (default `True`) — kill-switch if hardware misbehaves.
- Adds subscriber on `/utlidar/cloud_base` with matching RELIABLE / KEEP_LAST / depth=1 QoS.
- `_on_cloud_base` parses `msg.data` once with a pre-built structured numpy dtype (zero-copy view), extracts the `xyz` columns into a contiguous `(N, 3) float32` array, stores as `self._latest_cloud_xyz` plus timestamp. That's it — no work in the callback path beyond parse+store.
- In `_tick`, if `detected` and `use_lidar_distance` and cached cloud is not None and <0.5 s old: call `resolver.refine(bearing, cloud_xyz)`. On `None` return, log once at level INFO ("lidar gate failed — falling back to bbox distance") and use YOLO distance. On success, overwrite YOLO distance with LiDAR distance before publishing.

No other files touched. `behavior_node.py`, launch files, message definitions all unchanged.

## Latency

| Stage | Cost |
|---|---|
| `np.frombuffer(msg.data, dtype=..)` + `xyz` view (per cloud arrival) | ~0.2 ms |
| Wedge masks + percentile on ~871 pts (per tick) | ~0.1 ms |
| Total added CPU vs. today | **<1 ms/tick** |
| Worst-case cache staleness (cloud just before next frame arrives) | ≤67 ms |
| End-to-end: image → YOLO → cloud wedge → publish | YOLO (~30-50 ms) + <1 ms = unchanged within noise |

Callback groups: unchanged. `rclpy.spin` single-threaded serializes `_on_image`, `_on_cloud_base`, `_tick` → no race, no locks.

## Fallback & failure modes

| Situation | Behavior |
|---|---|
| No cloud received yet | YOLO bbox distance (pre-existing path) |
| Cloud stale >0.5 s | YOLO bbox distance, log warning once |
| Gate fails (too few points / too short column) | YOLO bbox distance, log INFO once per transition |
| `use_lidar_distance = False` param | YOLO bbox distance always (quick hardware kill-switch) |
| Multi-person same bearing | Returns closer one (acceptable for single-operator demo) |

## Testing

**Unit tests** (`come_here_perception/test/test_lidar_distance_resolver.py`, mock-safe, CI-runnable):

- Synthetic cloud with one human-shaped column at 2.0 m, bearing 0 → resolver returns ~2.0 m.
- Same cloud, bearing 0.5 rad → returns `None` (person out of cone).
- Column too short (0.4 m tall) → returns `None` (gate fails).
- Only 3 points in wedge → returns `None`.
- Floor-only points (z < 0.3) → returns `None`.
- Background clutter at 4.5 m + person at 2.0 m → returns ~2.0 m (percentile robust).

**Integration test** (mock mode, no hardware):
- `PerceptionNode` with `use_mock=False` and `use_lidar_distance=True`, replay a synthetic PointCloud2 + Image pair → verify published distance comes from lidar path.

**Hardware validation** (Jetson, one session):
1. Start `go2_video_publisher.py`; `ros2 launch come_here_bringup come_here.launch.py use_mock:=false`.
2. Echo `/come_here/person_detection` at 3 m / 2 m / 1 m with operator standing full-body, legs-only, and upper-body-only.
3. **Success criteria:** published distance matches tape-measure to ±0.15 m in all three postures (vs. today's 2× underestimate for legs-only).
4. Run full come-here flow: wake phrase (injected via topic), APPROACH, confirm robot stops inside 0.8 ± 0.1 m instead of 0.4 m.

## Rollback

One-line rollback: launch with `use_lidar_distance:=false` param. No code revert needed.

For full revert: `git revert <commit>` on `lab/2026-04-16-approach-fix` — single commit, touches only `perception_node.py` + new file.

## Out of scope (deferred)

- **Blocker #2** (YOLO bearing jitter EMA smoothing) — separate change.
- TF-based camera→lidar fusion — current body-frame approach is sufficient without intrinsics.
- Multi-person disambiguation via clustering — not needed for single-operator demo.
- Merging `lab/2026-04-16-approach-fix` to `main` — blocked until both blockers resolved.
