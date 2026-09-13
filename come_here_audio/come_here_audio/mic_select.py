"""Microphone selection by name, preferring the far-field array.

Passing ``device=None`` to sounddevice selects the PulseAudio "default" source,
which can return SILENCE rather than an error: the stream reports active,
Whisper transcribes nothing forever, and every log line looks healthy. So the
device is resolved by NAME, the far-field array is preferred, and a specifically
requested microphone that is absent RAISES instead of falling back to a source
that may be mute. The card index changes on every USB replug, so it cannot be
pinned in a config.
"""

# Substrings that identify a far-field array. The ReSpeaker Mic Array v2.0
# (USB 2886:0018) presents 6 input channels: 0 = beamformed DSP output,
# 1-4 = raw capsules, 5 = playback reference.
FAR_FIELD_HINTS = ('respeaker', 'mic array', 'seeed')
FAR_FIELD_MIN_CHANNELS = 6


class MicNotFound(RuntimeError):
    """A specific microphone was requested and is not attached."""


def describe_inputs(devices) -> list:
    """[(index, name, max_input_channels)] for everything that can record."""
    out = []
    for index, device in enumerate(devices):
        channels = int(device.get('max_input_channels', 0) or 0)
        if channels > 0:
            out.append((index, str(device.get('name', '')), channels))
    return out


def is_far_field(name: str, channels: int) -> bool:
    lowered = (name or '').lower()
    return (any(hint in lowered for hint in FAR_FIELD_HINTS)
            and channels >= FAR_FIELD_MIN_CHANNELS)


def select_input_device(devices, requested: str = '', prefer_far_field: bool = True):
    """Resolve a microphone to (index, name, channels, far_field).

    ``requested`` wins: digits are an index, anything else is a
    case-insensitive name substring, and a miss raises MicNotFound. With no
    request, a far-field array is preferred, otherwise the first real capture
    device that is not the pulse/default source. Returns
    (None, 'system default', 0, False) only when nothing can record, which the
    caller must treat as a hard failure.
    """
    inputs = describe_inputs(devices)
    want = (requested or '').strip()

    if want:
        if want.isdigit():
            index = int(want)
            for i, name, channels in inputs:
                if i == index:
                    return i, name, channels, is_far_field(name, channels)
            raise MicNotFound(f'no capture device at index {index}')
        matches = [(i, n, c) for i, n, c in inputs if want.lower() in n.lower()]
        if matches:
            # Several entries can carry the array's name; take the one that is
            # actually the full far-field array when there is one.
            for i, name, channels in matches:
                if is_far_field(name, channels):
                    return i, name, channels, True
            i, name, channels = matches[0]
            return i, name, channels, False
        attached = ', '.join(f'{i}:{n}' for i, n, _ in inputs) or 'none'
        raise MicNotFound(f'no capture device matching {want!r}; attached: {attached}')

    if prefer_far_field:
        for i, name, channels in inputs:
            if is_far_field(name, channels):
                return i, name, channels, True

    for i, name, channels in inputs:
        lowered = name.lower()
        if lowered.startswith('default') or lowered.startswith('pulse'):
            continue
        return i, name, channels, False

    return None, 'system default', 0, False


def resolve(requested: str = '', prefer_far_field: bool = True):
    """select_input_device against the live sounddevice list."""
    import sounddevice as sd
    return select_input_device(sd.query_devices(), requested, prefer_far_field)
