# WEDOAUDIO 4.2.2-rc1 — verification

Candidate prepared 2026-09-11. Not published. Build target: TouchDesigner **2025.32460**.

The 22-channel public interface is preserved. The portable component contains the analyzer only, with no network service, camera or performance scene.

## Reproducible offline checks

```text
python src/_wedoaudio_test.py
python src/_wedoaudio_timbre_test.py
python src/_wedoaudio_bench.py
python src/test_release_controls.py
```

Results: core 14/14; timbre 14/14; lifecycle/validation 5/5. Full synthetic music benchmark: kick 18/18 and snare 18/18 at F >= 0.80; tempo 12/15, unchanged from the recorded baseline. This is not a claim of universal music accuracy.

An additional run used the exact-click tests and a private 30-second recording of system music. Exact clicks: 6/6. Tempo estimates for the real excerpt were 102.9 BPM (WEDOAUDIO) and 103.4 BPM (librosa): agreement, not ground truth. Six alternate encodings retained tempo; onset agreement F ranged 0.92–1.00. Private audio is not distributed.

## Native acceptance

Tested by loading the compiled component in a clean container with a different name, without manually priming its graph. Seven checks passed:

1. Cold load produces 22 named channels.
2. A stereo signal present only on the right channel is detected.
3. Internal stereo downmix produces one channel.
4. Disconnect produces neutral values while preserving the 22-channel interface.
5. Reconnect resumes analysis.
6. Input resampled from 44.1 to 48 kHz updates the analyzer sample rate.
7. Reset clears analyzer state.

Native Selftest returned `(True, [])`; no operator errors. The small native test project ran at about 60.1 FPS for 14 seconds. This does not establish performance of a complete show rig. The embedded DSP source was compared with the source file.

## Known limits

Drone/ambient false onsets, tempo ambiguity and short-loop accuracy remain limitations. Time-dependent history at varying analysis rates still deserves a separate benchmark. Timbre/key/MFCC source is available but is not embedded in the portable 22-channel component. Instrument separation is not included. Only the 2025.32460 build of this candidate was checked.
