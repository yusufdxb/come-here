# AGENTS.md

This repository is a ROS 2 workspace for a Unitree GO2 audio-visual approach system ("come here"). The robot hears a voice command, estimates the sound direction, rotates toward it, detects the speaker visually, and approaches them.

## Repo-Specific Working Norms

1. **Inspect before editing.** Read the target package, its package.xml/setup.py, nearby tests, and config files before making changes.

2. **No fake completion.** Do not claim something works on hardware unless it has been tested on hardware. Always separate `verified`, `inferred`, and `not yet validated`.

3. **Microphone hardware is unknown.** The audio input layer is abstracted behind `AudioDirectionProvider`. Do not assume any specific microphone model or driver. When the mic is chosen, implement a new provider subclass.

4. **Mock-first development.** All sensor inputs have mock providers. Use `use_mock:=true` (the default) for local development. Real providers raise `NotImplementedError` until implemented.

5. **Keep scope tight.** Avoid unrelated edits. Do not touch `build/`, `install/`, or `log/`.

## Public Interfaces (Topics)

- `/come_here/wake_phrase` - wake phrase detection events
- `/come_here/audio_direction` - sound source direction estimates
- `/come_here/person_detection` - visual person detection results
- `/come_here/cmd_rotate` - rotation commands from behavior
- `/come_here/cmd_move` - forward motion commands from behavior
- `/come_here/state` - current behavior state name
- `/come_here/mock_trigger` - mock wake phrase trigger (testing only)
- `/come_here/mock_person` - mock person detection toggle (testing only)

## Validation Expectations

1. Unit tests for provider abstractions: `pytest` in each package's `test/` directory.
2. Integration tests require ROS 2 runtime and are not yet implemented.
3. Hardware tests require the GO2 robot and real sensors.
