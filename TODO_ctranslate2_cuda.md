# TODO: Build CTranslate2 with CUDA on Jetson

**Status:** Skipped for now (2026-04-07)

**Problem:** Whisper inference on CPU takes ~3s per 3s chunk. This causes ~3-6s latency in wake phrase detection.

**Fix:** Build CTranslate2 from source with CUDA support on the Jetson Orin NX.
- Jetson has GPU (Orin NX, CUDA 12.6, driver 540.4.0)
- Current pip CTranslate2 4.7.1 is CPU-only (reports 0 CUDA devices)
- With CUDA FP16, inference should drop to <0.5s

**Steps:**
1. Install CUDA toolkit on Jetson (nvcc not currently available)
2. Build CTranslate2 from source with `-DWITH_CUDA=ON`
3. Switch faster-whisper to `device="cuda", compute_type="float16"`
4. Also transfer tiny.en model (75MB, partially copied to ~/come-here/models/faster-whisper-tiny.en)

**Also consider:** Once CUDA works, switch from base.en to tiny.en for even faster inference.
