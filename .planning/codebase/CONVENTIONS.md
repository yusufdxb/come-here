# Coding Conventions

**Analysis Date:** 2026-04-06

## Naming Patterns

**Files:**
- Module files: `snake_case.py` (e.g., `audio_node.py`, `person_detector.py`, `wake_phrase_detector.py`)
- Abstract base classes: `abstraction_name.py` (e.g., `audio_direction_provider.py`, `person_detector.py`)
- Mock implementations: `mock_*.py` (e.g., `mock_audio_provider.py`)
- Specific implementations: `specific_name_*.py` (e.g., `whisper_phrase_detector.py`)
- Launch files: `module_name.launch.py` (e.g., `audio.launch.py`, `come_here.launch.py`)

**Classes:**
- PascalCase for all classes (both abstract and concrete)
- Abstract base classes: descriptive names without "Abstract" prefix (e.g., `AudioDirectionProvider`, `WakePhraseDetector`, `PersonDetector`)
- Concrete implementations: descriptive with purpose (e.g., `MockAudioProvider`, `WhisperPhraseDetector`, `AudioNode`)
- ROS 2 nodes: suffix `Node` (e.g., `AudioNode`, `PerceptionNode`, `BehaviorNode`)

**Functions and Methods:**
- snake_case for all function and method names
- Private methods: prefix with `_` (e.g., `_tick()`, `_wake_cb()`, `_setup()`)
- Callback methods: suffix with `_cb` (e.g., `_wake_cb()`, `_direction_cb()`, `_person_cb()`, `_mock_trigger_cb()`)
- Getter/setter pattern: direct naming without "get_" prefix when returning single value (e.g., `check()`, `detect()`, `get_direction()`)

**Variables:**
- snake_case for all local and instance variables
- Private instance variables: prefix with `_` (e.g., `self._active`, `self._fixed_azimuth`, `self._state`)
- Constants: UPPER_CASE (not explicitly found in codebase but follows Python convention)
- Dataclass fields: descriptive snake_case with type hints (e.g., `azimuth_rad`, `confidence`, `bearing_rad`, `distance_m`)

**Types:**
- Use Python type hints throughout
- Optional[Type] or Type | None for nullable values (codebase uses union syntax `Type | None`)
- Dataclasses for simple data structures (see `DirectionEstimate`, `PersonEstimate`, `PhraseDetection`)

## Code Style

**Formatting:**
- No explicit linting config found (no `.flake8`, `.pylintrc`, or `pyproject.toml`)
- Line length: Appears to follow ~88 character limit (Black default)
- Indentation: 4 spaces
- Blank lines: 2 lines between top-level definitions, 1 line between methods

**Imports:**
- Standard library imports first
- Third-party imports (rclpy, numpy, etc.) second
- Local/relative imports last
- No wildcard imports observed; all imports are explicit
- Guarded imports for optional dependencies: wrapped in try/except with `_AVAILABLE` flags (see `whisper_phrase_detector.py` lines 31-44)

## Import Organization

**Order:**
1. Standard library (`import os`, `import math`, `import random`, `from enum import`, `from abc import`, `from dataclasses import`, `from typing import`)
2. Third-party ROS 2 packages (`import rclpy`, `from rclpy.node import Node`, `from launch import`, `from launch_ros.actions import`, `from std_msgs.msg import`)
3. Third-party ML/audio packages (`import numpy`, `import torch`, `from transformers import`, `from faster_whisper import`)
4. Local package imports (relative, using package name: `from come_here_audio.audio_direction_provider import`)

**Path Aliases:**
- No alias system detected (using fully qualified imports from package root)
- Import style: `from come_here_audio.module import ClassName`

## Code Structure and Patterns

**Module Docstrings:**
- Every module starts with a docstring explaining purpose and usage
- ROS 2 nodes document their published/subscribed topics and parameters in docstring
- Abstractions document their interface and subclassing requirements

**Class Design:**

**Abstract Base Classes:**
- Use `ABC` and `@abstractmethod` decorator (e.g., `AudioDirectionProvider`, `WakePhraseDetector`, `PersonDetector`)
- Document interface contract in docstring
- Include guidance for implementing subclasses in docstrings

**ROS 2 Nodes:**
- Inherit from `rclpy.node.Node`
- `__init__()` declares parameters, creates pub/sub, starts timer
- Use descriptive node name (lowercase, underscores, e.g., `'audio_node'`, `'perception_node'`)
- Implement `destroy_node()` override to cleanup resources (teardown providers)
- Main entry point pattern: `def main(args=None): ... finally: node.destroy_node(); rclpy.shutdown()`

**Concrete Implementations:**
- Subclass abstract interface
- Implement all abstract methods
- Add `setup()` and `teardown()` lifecycle methods for resource management
- For mock implementations, include settable state methods (e.g., `set_detected()`, `set_triggered()`)

**Dataclasses:**
- Use `@dataclass` for simple data structures
- Include docstring describing the class purpose
- Annotate all fields with types
- No methods (pure data carriers)

## Error Handling

**Patterns:**
- Raise `NotImplementedError` with descriptive message for unimplemented features (e.g., `audio_node.py` lines 55-58, `perception_node.py` lines 32-34)
- Import errors wrapped with informative messages (e.g., `whisper_phrase_detector.py` lines 86-94)
- Thread safety: exception swallowing with sleep retry in background threads (e.g., `whisper_phrase_detector.py` lines 194-198)
- Graceful shutdown: try/except KeyboardInterrupt in main() with finally block to cleanup

**None-Checking:**
- Check for None before dereferencing (e.g., `behavior_node.py` lines 114-120: `if estimate is not None:`)
- Return None for "not found" or "no result" cases (not exceptions)

## Logging

**Framework:** ROS 2 logger via `self.get_logger()`

**Patterns:**
- `self.get_logger().info()` for startup, state transitions, detections
- `self.get_logger().warn()` for timeouts, recoverable errors
- Log human-readable info with format strings (e.g., `f'Wake phrase detected: "{detection.phrase}" (confidence={detection.confidence:.2f})'`)
- Avoid logging in tight loops; log state changes only

## Comments

**When to Comment:**
- Module docstrings (required on every file)
- Class docstrings for abstractions and complex classes (required)
- Method docstrings for public abstract methods (required)
- TODO comments for incomplete/unimplemented work (e.g., `audio_node.py` line 23, 48, 80; `behavior_node.py` line 124, 156)
- Inline comments for non-obvious logic (e.g., `whisper_phrase_detector.py` lines 113-119: explain local model resolution)

**Documentation Style:**
- Docstrings use triple-quoted format with description, then blank line, then details
- Parameter documentation: use ROS 2 node docstring format listing Publishes/Subscribes/Parameters
- Avoid redundant comments (code is self-documenting when naming is clear)

## Function Design

**Size:** 
- Most methods 5-30 lines; state machine callbacks typically 3-10 lines
- Longer methods (40+ lines) found in `WhisperPhraseDetector._listen_loop()` due to threaded event loop complexity

**Parameters:**
- Use type hints on all parameters
- ROS 2 node constructors take no required args (`__init__(self)`)
- Provider initialization accepts configuration: `__init__(self, model_size: str = ..., device: str = ..., adapter_path: Optional[str] = None)`
- Callback signatures match ROS 2 pattern: `def callback(self, msg: MessageType)`

**Return Values:**
- Always annotated with type hints (including `-> None`)
- Return dataclass instances for structured results (e.g., `DirectionEstimate`, `PersonEstimate`, `PhraseDetection`)
- Return None for "no result" rather than raising exceptions
- Interface methods document return contract in docstring (e.g., "Returns PhraseDetection if detected, None otherwise")

## Module Design

**Exports:**
- No explicit `__all__` found; all public classes are importable
- Mock implementations and abstractions co-located in same file when tightly coupled (e.g., `mock_audio_provider.py` in separate file but `mock_audio_provider.py` imports from `audio_direction_provider.py`)
- Each package's `__init__.py` is empty (no re-exports)

**Barrel Files:**
- Not used; clients import directly from module files (e.g., `from come_here_audio.audio_node import AudioNode`)

## ROS 2 Specific Conventions

**Topic Naming:**
- Use leading slash: `/come_here/wake_phrase`, `/come_here/audio_direction`
- Use snake_case for topic names
- Organize by subsystem prefix: `/come_here/`

**Parameter Naming:**
- Declare with snake_case: `self.declare_parameter('use_mock', True)`
- Retrieve with same name: `self.get_parameter('use_mock').value`
- Document defaults and types in docstring

**Message Handling:**
- Publish data packed into Float64MultiArray with documented element order (e.g., `behavior_node.py` line 14 documents `[azimuth, confidence]`)
- Unpack with length check and indexing (e.g., `if len(msg.data) >= 2: self._last_azimuth = msg.data[0]`)

---

*Convention analysis: 2026-04-06*
