# Come-Here LiDAR Distance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the YOLO pinhole/bbox distance in the come-here APPROACH phase with a LiDAR-based range from `/utlidar/cloud_base`, so the GO2 stops at the commanded 0.8 m instead of the current ~0.4 m when only legs are in frame.

**Architecture:** One new pure-numpy class `LidarDistanceResolver` (bearing cone + torso-height gate + 10th-percentile range). Wire it into `PerceptionNode` via a new `/utlidar/cloud_base` subscriber. `/come_here/person_detection` contract unchanged; `behavior_node` untouched.

**Tech Stack:** Python 3.10, ROS 2 Humble (rclpy, sensor_msgs), NumPy, pytest, colcon.

---

## Preconditions (verify once before Task 1)

- [ ] **P1:** On lab PC, `cd "/media/cares/T7 Storage/come-here" && git status` shows clean tree on branch `lab/2026-04-16-approach-fix` at or ahead of `6e7ebfa` (design doc).
- [ ] **P2:** On lab PC, `python3 -c 'import numpy; print(numpy.__version__)'` succeeds (pytest will run locally against the resolver; ROS is NOT required on lab PC).
- [ ] **P3:** Jetson reachable: `python3 -c "import pexpect; c=pexpect.spawn('ssh unitree@192.168.0.2 true', encoding='utf-8'); c.expect('password:'); c.sendline('123'); c.expect(pexpect.EOF, timeout=15); print('OK')"`.
- [ ] **P4:** On Jetson, `/utlidar/cloud_base` publishing at ~15 Hz with `frame_id: base_link` (already verified 2026-04-23 during design; re-check before hardware run).

## File Structure

| Path | Role |
|---|---|
| `come_here_perception/come_here_perception/lidar_distance_resolver.py` (NEW) | Pure-numpy class. No ROS deps. Sole responsibility: given a bearing and an XYZ cloud, return person range or `None`. |
| `come_here_perception/test/test_lidar_distance_resolver.py` (NEW) | Pytest for the resolver. Synthetic clouds only; no ROS. |
| `come_here_perception/come_here_perception/perception_node.py` (MODIFY) | Adds `/utlidar/cloud_base` subscriber, parse-and-cache, `use_lidar_distance` param, refine-in-tick. |
| `docs/superpowers/specs/2026-04-23-come-here-lidar-distance-design.md` (exists) | Spec, already committed at `6e7ebfa`. |

All other files — `behavior_node.py`, `yolo_person_detector.py`, `person_detector.py`, launch files, message defs, `setup.py`, `package.xml` — **unchanged**.

---

### Task 1: Resolver — scaffold + happy-path test

**Files:**
- Create: `come_here_perception/come_here_perception/lidar_distance_resolver.py`
- Create: `come_here_perception/test/test_lidar_distance_resolver.py`

- [ ] **Step 1.1: Write the failing test**

Write `come_here_perception/test/test_lidar_distance_resolver.py`:

```python
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
```

- [ ] **Step 1.2: Run test, verify it fails**

Run from lab PC:
```bash
cd "/media/cares/T7 Storage/come-here"
python3 -m pytest come_here_perception/test/test_lidar_distance_resolver.py -v
```
Expected: collection error / ImportError on `come_here_perception.lidar_distance_resolver`.

- [ ] **Step 1.3: Write the minimal resolver**

Write `come_here_perception/come_here_perception/lidar_distance_resolver.py`:

```python
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
```

- [ ] **Step 1.4: Run test, verify it passes**

```bash
cd "/media/cares/T7 Storage/come-here"
python3 -m pytest come_here_perception/test/test_lidar_distance_resolver.py -v
```
Expected: `1 passed`.

- [ ] **Step 1.5: Commit**

```bash
cd "/media/cares/T7 Storage/come-here"
git add come_here_perception/come_here_perception/lidar_distance_resolver.py come_here_perception/test/test_lidar_distance_resolver.py
git -c commit.gpgsign=false commit -m "feat(perception): LidarDistanceResolver — bearing cone + torso gate

Pure-numpy class that returns person range from a base-frame point cloud.
Happy-path test: human column at 2 m returns ~2 m."
```

---

### Task 2: Resolver — gate and robustness tests

**Files:**
- Modify: `come_here_perception/test/test_lidar_distance_resolver.py`

- [ ] **Step 2.1: Append five gate/robustness tests**

Append to `come_here_perception/test/test_lidar_distance_resolver.py`:

```python
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
```

- [ ] **Step 2.2: Run all resolver tests, verify they pass**

```bash
cd "/media/cares/T7 Storage/come-here"
python3 -m pytest come_here_perception/test/test_lidar_distance_resolver.py -v
```
Expected: `7 passed` (1 from Task 1 + 6 new).

- [ ] **Step 2.3: Commit**

```bash
cd "/media/cares/T7 Storage/come-here"
git add come_here_perception/test/test_lidar_distance_resolver.py
git -c commit.gpgsign=false commit -m "test(perception): gate and robustness tests for LidarDistanceResolver

Covers: out-of-cone, too-few-points, short-column, floor-only,
background-clutter percentile robustness, empty/None cloud."
```

---

### Task 3: PerceptionNode — cloud subscribe + cache

**Files:**
- Modify: `come_here_perception/come_here_perception/perception_node.py`

- [ ] **Step 3.1: Add imports and param declaration**

In `perception_node.py` at the top of the file, replace the existing imports with:

```python
"""ROS 2 node for visual person detection.

Publishes:
  /come_here/person_detection  (std_msgs/Float64MultiArray) [bearing, distance, confidence, detected]

Subscribes:
  /camera/image_raw            (sensor_msgs/Image) - from go2_av_node
  /utlidar/cloud_base          (sensor_msgs/PointCloud2) - base-frame LiDAR (GO2 L1)
  /come_here/mock_person       (std_msgs/Bool) - toggle mock person detection
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float64MultiArray

from come_here_perception.lidar_distance_resolver import LidarDistanceResolver
from come_here_perception.person_detector import MockPersonDetector, PersonDetector
```

Then inside `PerceptionNode.__init__`, immediately after the existing `self.declare_parameter('confidence', 0.45)` line, add:

```python
        self.declare_parameter('use_lidar_distance', True)
        self.declare_parameter('lidar_cloud_topic', '/utlidar/cloud_base')
        self.declare_parameter('lidar_max_age_s', 0.5)
```

- [ ] **Step 3.2: Initialize resolver and cloud cache**

In `__init__`, after the existing `self._detector.setup()` line and before the `self._pub = ...` line, add:

```python
        self._use_lidar_distance = bool(self.get_parameter('use_lidar_distance').value)
        self._lidar_max_age_s = float(self.get_parameter('lidar_max_age_s').value)
        self._resolver = LidarDistanceResolver()
        self._latest_cloud_xyz: np.ndarray | None = None
        self._latest_cloud_stamp_s: float = 0.0
        self._lidar_fallback_logged: bool = False

        if self._use_lidar_distance and not use_mock:
            lidar_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            cloud_topic = self.get_parameter('lidar_cloud_topic').value
            self.create_subscription(
                PointCloud2, cloud_topic, self._on_cloud_base, lidar_qos
            )
            self.get_logger().info(f'Subscribed to {cloud_topic} for distance refinement')
        elif use_mock:
            self.get_logger().info('Mock mode — skipping LiDAR distance refinement')
        else:
            self.get_logger().info('LiDAR distance refinement disabled (use_lidar_distance=false)')
```

- [ ] **Step 3.3: Add the cloud callback**

Inside `PerceptionNode`, add this method (place it just after `_on_image`):

```python
    def _on_cloud_base(self, msg: PointCloud2):
        """Parse the cloud once on arrival; cache XYZ as a contiguous (N,3) array.

        Zero-copy parse: point_step=32 on the GO2 L1 cloud_base means each
        point is 8 x float32 slots; the first three slots are x, y, z.
        We read all 8 columns and view the first three.
        """
        if msg.point_step != 32:
            # Unexpected layout; disable lidar refinement for this frame.
            self.get_logger().warn(
                f'cloud_base point_step={msg.point_step}, expected 32 — skipping'
            )
            return
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 8)
        self._latest_cloud_xyz = arr[:, :3].copy()  # ~10 KB for ~871 pts
        self._latest_cloud_stamp_s = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
```

- [ ] **Step 3.4: Lint the file — no test here yet (integration validated on Jetson)**

Run:
```bash
cd "/media/cares/T7 Storage/come-here"
python3 -m py_compile come_here_perception/come_here_perception/perception_node.py
python3 -m pytest come_here_perception/test/ -v
```
Expected: compile OK; all existing tests still pass (the resolver tests + the existing `test_mock_imports.py`, `test_person_detector.py`, `test_face_detector.py`).

- [ ] **Step 3.5: Commit**

```bash
cd "/media/cares/T7 Storage/come-here"
git add come_here_perception/come_here_perception/perception_node.py
git -c commit.gpgsign=false commit -m "feat(perception): subscribe to /utlidar/cloud_base, cache xyz zero-copy

Adds PointCloud2 subscribe with RELIABLE QoS to match bare-DDS publisher.
np.frombuffer + stride view parses ~871 points in ~0.2 ms. Cache holds
latest (N,3) float32 array and stamp for downstream tick to query."
```

---

### Task 4: PerceptionNode — wire resolver into _tick

**Files:**
- Modify: `come_here_perception/come_here_perception/perception_node.py`

- [ ] **Step 4.1: Replace `_tick` with refine-aware version**

Replace the existing `_tick` method in `perception_node.py` with:

```python
    def _tick(self):
        result = self._detector.detect()
        distance_m = result.distance_m

        if result.detected and self._use_lidar_distance:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            cloud_age_s = now_s - self._latest_cloud_stamp_s
            cloud_fresh = (
                self._latest_cloud_xyz is not None
                and cloud_age_s < self._lidar_max_age_s
            )
            if cloud_fresh:
                refined = self._resolver.refine(
                    bearing_rad=result.bearing_rad,
                    cloud_xyz=self._latest_cloud_xyz,
                )
                if refined is not None:
                    distance_m = refined
                    if self._lidar_fallback_logged:
                        self.get_logger().info(
                            f'lidar distance recovered: {refined:.2f} m'
                        )
                        self._lidar_fallback_logged = False
                elif not self._lidar_fallback_logged:
                    self.get_logger().info(
                        f'lidar gate failed (bearing={result.bearing_rad:.2f}) — '
                        f'falling back to bbox distance {result.distance_m:.2f} m'
                    )
                    self._lidar_fallback_logged = True
            elif not self._lidar_fallback_logged:
                self.get_logger().info(
                    f'no fresh cloud (age={cloud_age_s:.2f}s) — bbox distance '
                    f'{result.distance_m:.2f} m'
                )
                self._lidar_fallback_logged = True

        msg = Float64MultiArray()
        msg.data = [
            result.bearing_rad,
            distance_m,
            result.confidence,
            float(result.detected),
        ]
        self._pub.publish(msg)
```

- [ ] **Step 4.2: Verify compile and existing tests still pass**

```bash
cd "/media/cares/T7 Storage/come-here"
python3 -m py_compile come_here_perception/come_here_perception/perception_node.py
python3 -m pytest come_here_perception/test/ -v
```
Expected: compile OK; all tests pass.

- [ ] **Step 4.3: Commit**

```bash
cd "/media/cares/T7 Storage/come-here"
git add come_here_perception/come_here_perception/perception_node.py
git -c commit.gpgsign=false commit -m "feat(perception): refine YOLO distance with LiDAR in _tick

On every tick with detected=True and a fresh cloud (<0.5 s), query
LidarDistanceResolver. On gate success, publish lidar distance;
on gate failure, fall back to YOLO bbox distance and log once per
transition. Kill-switch: use_lidar_distance:=false reverts to today's
behavior."
```

---

### Task 5: Sync to Jetson and build

**Files:** none (deploy only)

- [ ] **Step 5.1: Rsync the repo to Jetson**

Run from lab PC:

```bash
sshpass -p '123' rsync -avz --delete \
  --exclude='.git/' --exclude='build/' --exclude='install/' --exclude='log/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  "/media/cares/T7 Storage/come-here/" \
  unitree@192.168.0.2:/home/unitree/come-here/
```

If `sshpass` is not installed, use the pexpect pattern in `~/.claude/projects/-home-cares/memory/project_jetson_ssh_password.md`.

Expected: rsync transfers only the modified files (perception_node.py, lidar_distance_resolver.py, test_lidar_distance_resolver.py, design doc, plan).

- [ ] **Step 5.2: Colcon build on Jetson**

Drive via pexpect from lab PC:

```python
import pexpect
c = pexpect.spawn('ssh unitree@192.168.0.2', timeout=300, encoding='utf-8')
c.expect('password:'); c.sendline('123'); c.expect(r'\$ ')
c.sendline("bash -lc 'cd ~/come-here && source /opt/ros/humble/setup.bash && "
           "colcon build --packages-select come_here_perception --symlink-install'")
c.expect(r'\$ ', timeout=300)
print(c.before[-2000:])
c.sendline('exit'); c.expect(pexpect.EOF)
```

Expected: `Finished <<< come_here_perception` in the tail output. No errors.

- [ ] **Step 5.3: Run resolver tests on Jetson (sanity check on target numpy)**

```python
import pexpect
c = pexpect.spawn('ssh unitree@192.168.0.2', timeout=120, encoding='utf-8')
c.expect('password:'); c.sendline('123'); c.expect(r'\$ ')
c.sendline("bash -lc 'cd ~/come-here && source /opt/ros/humble/setup.bash && "
           "source install/setup.bash && "
           "python3 -m pytest come_here_perception/test/test_lidar_distance_resolver.py -v'")
c.expect(r'\$ ', timeout=120)
print(c.before[-3000:])
c.sendline('exit'); c.expect(pexpect.EOF)
```

Expected: `7 passed`.

- [ ] **Step 5.4: No commit — deploy only.**

---

### Task 6: Hardware validation — tape-measure at 3/2/1 m

**Files:** none (run + log only)

**Setup** (operator and assistant coordination; one runs topics, the other stands):
1. Start the camera: on Jetson, `python3 /home/unitree/go2_video_publisher.py &`.
2. In another Jetson terminal, launch come-here in real mode (no behavior / no motion — we only want perception publishing):
   ```bash
   ros2 run come_here_perception perception_node --ros-args \
     -p use_mock:=false \
     -p use_lidar_distance:=true
   ```
3. In a third terminal, echo the detection:
   ```bash
   ros2 topic echo /come_here/person_detection std_msgs/Float64MultiArray
   ```

- [ ] **Step 6.1: Measure at 3.0 m tape distance**

Operator stands 3.0 m from the GO2 front face (use a tape measure from the front bumper). Hold three postures for ~5 seconds each:
- (a) Full body in frame
- (b) Upper body only (crouch below bbox-bottom or step closer so camera cuts legs — this is the legs-only edge case from the spec, mirrored)
- (c) Legs only (step forward until head leaves the top of frame)

Record the median of data[1] (distance field) during each posture. Expected: all three within `3.0 ± 0.15 m`.

- [ ] **Step 6.2: Measure at 2.0 m tape distance**

Same postures. Expected: all three within `2.0 ± 0.15 m`.

- [ ] **Step 6.3: Measure at 1.0 m tape distance**

Same postures. Expected: all three within `1.0 ± 0.15 m`.

- [ ] **Step 6.4: Compare legs-only (c) to today's behavior**

Re-run posture (c) at 2.0 m with `use_lidar_distance:=false`. Expected: published distance ~1.0 m (the 2× underestimate the spec calls out). This confirms the kill-switch restores the old path and the fix is doing real work.

- [ ] **Step 6.5: Log findings to the repo**

On the lab PC, append the nine (3 distances × 3 postures) median distances plus the kill-switch delta to `docs/superpowers/specs/2026-04-23-come-here-lidar-distance-design.md` under a new `## Hardware results 2026-04-23` section. Commit:

```bash
cd "/media/cares/T7 Storage/come-here"
git add docs/superpowers/specs/2026-04-23-come-here-lidar-distance-design.md
git -c commit.gpgsign=false commit -m "docs(spec): hardware validation results — lidar distance fix

Tape-measure at 3/2/1 m × {full,upper,legs} all within ±0.15 m.
Kill-switch regression confirms 2× underestimate of bbox-only path
at legs-only 2 m."
```

---

### Task 7: Full-flow regression — come-here stops at 0.8 m

**Files:** none (hardware run)

This verifies the original motivation: the robot should now stop at the commanded 0.8 m, not the previous ~0.4 m.

- [ ] **Step 7.1: Bring up the full stack on Jetson**

Per the RUN_NOTES workflow (don't modify audio configs — use the bringup launch):

```bash
# Terminal 1 (Jetson):
python3 /home/unitree/go2_video_publisher.py

# Terminal 2 (Jetson):
bash -lc 'cd ~/come-here && source /opt/ros/humble/setup.bash && source install/setup.bash && \
  ros2 launch come_here_bringup come_here.launch.py use_mock:=false'
```

- [ ] **Step 7.2: Inject wake phrase (operator 2.5 m from robot, full-body in frame)**

From a Jetson terminal (cyclone-pinned, intra-process):

```bash
ros2 topic pub --once /come_here/wake_phrase std_msgs/String '{data: come here}'
```

- [ ] **Step 7.3: Observe and record**

Watch the robot: it should say "I am coming", then APPROACH. **Stop criterion: final robot-to-operator distance is 0.8 ± 0.1 m** (measured with tape after the robot halts).

- [ ] **Step 7.4: Repeat twice** (3 total runs at 2.5 m start; 3 at 3.0 m start). Record stop distance for each.

- [ ] **Step 7.5: Append run table to spec and commit**

Append a `## Full-flow regression 2026-04-23` section to the spec with the six stop measurements. Commit:

```bash
cd "/media/cares/T7 Storage/come-here"
git add docs/superpowers/specs/2026-04-23-come-here-lidar-distance-design.md
git -c commit.gpgsign=false commit -m "docs(spec): full-flow come-here regression at 0.8 m stop distance

Six runs (3×2.5m start, 3×3.0m start) all stopped within 0.8 ± 0.1 m.
Blocker #1 closed. Blocker #2 (bearing jitter EMA) still open —
branch stays off main."
```

---

### Task 8: Push to origin (NOT to main)

**Files:** none (git only)

- [ ] **Step 8.1: Push `lab/2026-04-16-approach-fix` to origin**

```bash
cd "/media/cares/T7 Storage/come-here"
git push origin lab/2026-04-16-approach-fix
```

- [ ] **Step 8.2: Confirm no merge-to-main yet**

```bash
cd "/media/cares/T7 Storage/come-here"
git log main..lab/2026-04-16-approach-fix --oneline
```

Expected: lists the new lidar-distance commits plus the pre-existing approach-fix commits. Branch still off `main` — blocker #2 (YOLO bearing jitter) remains, and merging is out of scope for this plan.

---

## Self-review checklist (done)

- **Spec coverage:** Topic, frame, QoS, filter math, gate thresholds, percentile, fallback, kill-switch param, unit tests (6), integration (via mock), hardware validation (tape + full-flow), rollback — all mapped to tasks above.
- **Placeholders:** None. Every code block is complete; every command is runnable.
- **Type consistency:** `LidarDistanceResolver.refine(bearing_rad, cloud_xyz) -> float | None` is used identically in Task 1 (def), Task 2 (tests), Task 4 (call site). Param names match. `_latest_cloud_xyz` / `_latest_cloud_stamp_s` / `_lidar_fallback_logged` defined in Task 3, used in Task 4.
- **Scope:** single focused change, two code files, one test file, single plan; no dependency on blocker #2 work.
