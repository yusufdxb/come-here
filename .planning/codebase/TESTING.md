# Testing Patterns

**Analysis Date:** 2026-04-06

## Test Framework

**Runner:**
- pytest (implicit, no config file found)
- Invoked via `python -m pytest test/` per README.md

**Assertion Library:**
- Plain Python `assert` statements (standard pytest assertions)

**Run Commands:**
```bash
cd come_here_audio && python -m pytest test/
cd come_here_perception && python -m pytest test/
cd come_here_behavior && python -m pytest test/
```

## Test File Organization

**Location:**
- Co-located: Each package has a `test/` directory at package level
- Structure:
  - `come_here_audio/test/test_audio_provider.py`
  - `come_here_perception/test/test_person_detector.py`
  - `come_here_behavior/test/test_state_machine.py`

**Naming:**
- Pattern: `test_*.py` (e.g., `test_audio_provider.py`, `test_person_detector.py`, `test_state_machine.py`)
- Test functions: `test_*` prefix (e.g., `test_mock_provider_returns_estimate()`, `test_mock_detector_default_no_person()`)

**Structure:**
```
come_here_audio/
├── come_here_audio/          # Package code
├── test/
│   ├── __init__.py           # Empty marker file
│   └── test_*.py             # Test modules
├── setup.py
└── setup.cfg
```

**__init__.py in tests:**
- Present but empty (marks directory as package, required by older pytest versions)

## Test Structure and Organization

**Suite Organization:**

Each test file groups related test functions without explicit class-based organization. Tests are simple functions at module level.

Example from `test_audio_provider.py`:
```python
"""Unit tests for audio direction provider abstraction."""

from come_here_audio.audio_direction_provider import DirectionEstimate
from come_here_audio.mock_audio_provider import MockAudioProvider


def test_mock_provider_returns_estimate():
    provider = MockAudioProvider(fixed_azimuth_rad=0.5)
    provider.setup()
    result = provider.get_direction()
    assert result is not None
    assert isinstance(result, DirectionEstimate)
    assert 0.0 <= result.confidence <= 1.0
    provider.teardown()


def test_mock_provider_inactive_returns_none():
    provider = MockAudioProvider()
    result = provider.get_direction()
    assert result is None
```

**Patterns:**
- Setup: Instantiate mock object, call `.setup()`
- Exercise: Call method being tested
- Assert: Use inline `assert` statements
- Teardown: Call `.teardown()` explicitly (lifecycle management)

**Lifecycle Pattern:**

All testable objects implement three-phase lifecycle:
1. **Instantiate:** Constructor with optional parameters (no I/O yet)
2. **Setup:** Call `.setup()` to initialize resources
3. **Teardown:** Call `.teardown()` to release resources

Tests must follow this pattern:
```python
def test_something():
    obj = SomeClass()
    obj.setup()
    # ... test code
    obj.teardown()
```

## Mocking

**Framework:** 
- No external mocking library (unittest.mock not used)
- Custom mock implementations that are testable abstractions

**Patterns:**

Mocks are real classes implementing the abstract interface, not library-generated mocks. Located in same package as abstractions.

Example pattern from `mock_audio_provider.py`:
```python
class MockAudioProvider(AudioDirectionProvider):
    def __init__(self, fixed_azimuth_rad: float = 0.0, noise_std: float = 0.05):
        self._fixed_azimuth = fixed_azimuth_rad
        self._noise_std = noise_std
        self._active = False

    def setup(self) -> None:
        self._active = True

    def get_direction(self) -> DirectionEstimate | None:
        if not self._active:
            return None
        noise = random.gauss(0, self._noise_std)
        return DirectionEstimate(
            azimuth_rad=self._fixed_azimuth + noise,
            confidence=max(0.0, min(1.0, 0.85 + random.gauss(0, 0.05))),
        )

    def teardown(self) -> None:
        self._active = False
```

**Simulating Changes:**

For testable state changes, mocks include setter methods:
- `MockAudioProvider`: No state setter (fixed direction)
- `MockPersonDetector.set_detected(detected: bool, bearing_rad: float = 0.0, distance_m: float = 2.0)` (line 51-55 of `person_detector.py`)
- `MockWakePhraseDetector.set_triggered(triggered: bool = True)` (line 53-55 of `wake_phrase_detector.py`)

Tests call these setters to simulate different conditions:
```python
def test_mock_detector_set_detected():
    det = MockPersonDetector()
    det.setup()
    det.set_detected(True, bearing_rad=0.3, distance_m=1.5)
    result = det.detect()
    assert result.detected
    assert result.bearing_rad == 0.3
    assert result.distance_m == 1.5
    det.teardown()
```

**What to Mock:**
- Hardware providers: All hardware-dependent components have mock implementations (audio, camera, mic)
- File I/O: Not present in core; `WhisperPhraseDetector` uses local model cache (lines 113-119) which gracefully falls back to remote

**What NOT to Mock:**
- Data structures: No mocking of dataclasses; they are assertions targets
- Abstract base classes: Cannot instantiate; define the contract
- ROS 2 infrastructure: Not tested in unit tests (full integration would require ROS 2 runtime)

## Fixtures and Factories

**Test Data:**
- No pytest fixtures defined
- No factory pattern; direct instantiation with parameters

**Parameterization:**
- Not used; separate test functions for each case
- Example: `test_mock_provider_returns_estimate()` and `test_mock_provider_inactive_returns_none()` are two separate functions, not parameterized

**Location:**
- Not applicable; no shared test data files

## Coverage

**Requirements:** 
- Not enforced (no coverage config found)
- No coverage targets specified

**View Coverage:**
```bash
# Not configured
```

**Current Coverage:**
- Abstractions: Not directly tested (interfaces only)
- Mock implementations: Well covered (see test files)
- ROS 2 nodes: Not covered by unit tests (require ROS 2 runtime, would need integration tests)
- Real implementations: `WhisperPhraseDetector` not unit tested (audio/ML complexity)

## Test Types

**Unit Tests:**
- **Scope:** Mock implementations of abstract providers
- **Approach:** Test lifecycle (setup/teardown), return types, edge cases (inactive states, no detection)
- **Files:** `test_audio_provider.py`, `test_person_detector.py`, `test_state_machine.py`
- **Example:** `test_mock_provider_returns_estimate()` validates return type and confidence bounds

**Integration Tests:**
- **Status:** Not automated
- **Manual approach:** Per README.md lines 78-89, use `ros2 topic pub` to simulate triggers:
  ```bash
  ros2 topic pub --once /come_here/mock_trigger std_msgs/Bool "data: true"
  ros2 topic pub --once /come_here/mock_person std_msgs/Bool "data: true"
  ros2 topic echo /come_here/state
  ```
- **What's tested:** End-to-end state transitions with full node stack running

**E2E Tests:**
- Not implemented
- Hardware validation would require Unitree GO2 with actual mic and camera

## Testable Components

**Directly Testable (Mock Abstractions):**

1. **`AudioDirectionProvider` implementations:**
   - `MockAudioProvider` (tested in `test_audio_provider.py`)
   - Lifecycle: setup → get_direction() → teardown
   - Validates: return type, confidence bounds, None when inactive

2. **`WakePhraseDetector` implementations:**
   - `MockWakePhraseDetector` (tested in `test_audio_provider.py`)
   - Lifecycle: setup → set_triggered() → check() → teardown
   - Validates: return type, phrase content, auto-reset behavior

3. **`PersonDetector` implementations:**
   - `MockPersonDetector` (tested in `test_person_detector.py`)
   - Lifecycle: setup → set_detected() → detect() → teardown
   - Validates: detection state, bearing/distance accuracy

4. **`State` enum:**
   - Tested in `test_state_machine.py` for completeness
   - Validates: all expected states exist

**Not Testable Without ROS 2 Runtime:**

1. **ROS 2 Nodes** (`AudioNode`, `PerceptionNode`, `BehaviorNode`):
   - Require `rclpy.init()` and ROS 2 DDS infrastructure
   - Could test with `rclpy` test utilities (see ROS 2 documentation), but not currently done
   - Comment in `test_state_machine.py` line 1: "Basic state enum tests. Full integration tests require ROS 2 runtime."

2. **Real implementations:**
   - `WhisperPhraseDetector`: Requires audio hardware/mock audio
   - Real `PersonDetector`: Would require camera
   - Real `AudioDirectionProvider`: Would require mic array

## Common Test Patterns

**Setup/Teardown:**
```python
def test_something():
    obj = SomeClass()
    obj.setup()
    try:
        # test assertions
        assert result is not None
    finally:
        obj.teardown()
```

Actually used pattern (no try/finally; assumes no exceptions):
```python
def test_mock_provider_returns_estimate():
    provider = MockAudioProvider(fixed_azimuth_rad=0.5)
    provider.setup()
    result = provider.get_direction()
    assert result is not None
    assert isinstance(result, DirectionEstimate)
    assert 0.0 <= result.confidence <= 1.0
    provider.teardown()
```

**Type Validation:**
```python
assert isinstance(result, DirectionEstimate)
assert isinstance(result, PersonEstimate)
```

**Dataclass Field Validation:**
```python
assert result.detected
assert result.bearing_rad == 0.3
assert result.distance_m == 1.5
assert result.confidence == 0.9
assert 0.0 <= result.confidence <= 1.0
```

**State Transitions (Enum):**
```python
def test_all_states_exist():
    expected = {'IDLE', 'LISTENING', 'TURN_TO_SOUND', 'SEARCH_FOR_PERSON', 'APPROACH_PERSON', 'STOP'}
    actual = {s.name for s in State}
    assert actual == expected
```

**None Checking:**
```python
def test_mock_provider_inactive_returns_none():
    provider = MockAudioProvider()
    result = provider.get_direction()
    assert result is None

def test_mock_wake_phrase_detector():
    det = MockWakePhraseDetector()
    det.setup()
    assert det.check() is None
    det.set_triggered(True)
    result = det.check()
    assert result is not None
    assert result.phrase == "come here"
    assert det.check() is None  # auto-reset
    det.teardown()
```

## Testing Philosophy

**What's Tested:**
- Mock implementations: These are testable because they don't depend on external hardware
- Data contracts: Dataclass instances are validated for type and field values
- Lifecycle: Setup/teardown are called and state changes are validated
- Enum completeness: All expected states/values exist

**What's Not Tested (By Design):**
- ROS 2 node behavior: Requires full ROS 2 environment
- Hardware integration: Deferred until hardware is available
- ML model behavior: `WhisperPhraseDetector` is integration-tested only
- Full state machine transitions: Would require mocked ROS 2 node with pub/sub

**Why:**
- Unit tests must be fast and dependency-free
- Hardware tests are manual until real hardware available (see README.md "Assumptions" section)
- ROS 2 nodes tested via integration tests with topic pub/sub (documented in README.md)

---

*Testing analysis: 2026-04-06*
