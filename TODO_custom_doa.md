# TODO: Custom GCC-PHAT DOA (Replace Firmware DOA)

## Problem

ReSpeaker Mic Array v2.0 onboard XMOS DOA gives false directions — visible on
LED ring pointing the wrong way. The firmware's basic DOA algorithm gets confused
by room reflections, low SNR, and transient sounds. Current interim fix is
stricter median filtering + outlier rejection in `respeaker_doa_provider.py`, but
this is a band-aid — garbage in, filtered garbage out.

## Plan: Custom GCC-PHAT DOA

Replace firmware DOAANGLE register reads with custom Python DOA using the 4 raw
mic channels:

1. **Read 4 raw mics** via ALSA (ch1-4 from `hw:0,0`, 6-channel device)
2. **GCC-PHAT** (Generalized Cross-Correlation with Phase Transform) between mic
   pairs to compute time-delay-of-arrival (TDOA)
3. **Triangulate angle** from TDOA values using known mic geometry
4. Alternative: **SRP-PHAT** (Steered Response Power) for more robust estimation

## Key Details

- ReSpeaker v2.0 mic geometry: 4 mics in a square, ~46mm between adjacent mics
- 6-channel ALSA: ch0=beamformed, ch1-4=raw mics, ch5=playback
- Sample rate: 16000 Hz
- Implement as a new provider class implementing `AudioDirectionProvider` ABC
  (`come_here_audio/audio_direction_provider.py`)
- Keep firmware DOA provider as fallback option

## Interim Fix (2026-04-07)

Added iterative outlier rejection + circular median + VAD gating to
`respeaker_doa_provider.py:get_latched_direction()`. Filters worst outliers but
doesn't fix root cause.
