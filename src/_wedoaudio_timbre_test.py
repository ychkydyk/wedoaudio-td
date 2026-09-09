"""Tests for WEDOAUDIO 4.3 additions. Run: python src/_wedoaudio_timbre_test.py
Every test states what it proves and prints the number it measured, so a failure
tells you the value, not just the word FAIL."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedoaudio_timbre import (mel_filterbank, mfcc, chroma, estimate_key,
                              loudness_lu, k_weight_gain, tempo_octave_fix, NOTES)

SR, N = 44100, 2048
FREQS = np.linspace(0, SR / 2, N)
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}  {detail}")
    else: fail += 1; print(f"  FAIL  {name}  {detail}")


def tone(freqs_hz, amps=None, width=3):
    """Magnitude spectrum with narrow peaks at the given frequencies."""
    m = np.zeros(N)
    amps = amps or [1.0] * len(freqs_hz)
    for f, a in zip(freqs_hz, amps):
        i = int(round(f / (SR / 2) * (N - 1)))
        for d in range(-width, width + 1):
            j = i + d
            if 0 <= j < N: m[j] += a * np.exp(-(d ** 2) / 2.0)
    return m


def harmonics(f0, n=6, roll=0.7):
    return tone([f0 * k for k in range(1, n + 1)], [roll ** (k - 1) for k in range(1, n + 1)])


print("MFCC")
fb = mel_filterbank(N, SR)
c_sine = mfcc(harmonics(220), fb)
c_noise = mfcc(np.ones(N) * 0.05, fb)
d = float(np.linalg.norm(c_sine[1:] - c_noise[1:]))
check("tone and noise differ in shape", d > 5.0, f"distance {d:.1f}")
loud = mfcc(harmonics(220) * 4.0, fb)
check("mfcc[0] tracks level", loud[0] > c_sine[0] + 2.0, f"{c_sine[0]:.1f} -> {loud[0]:.1f}")
check("shape is level-invariant", float(np.linalg.norm(loud[1:] - c_sine[1:])) < 1.5,
      f"shape drift {np.linalg.norm(loud[1:] - c_sine[1:]):.2f}")

print("\nCHROMA / KEY")
# C major triad C4 E4 G4 with harmonics, plus the tonic doubled — a plain tonal frame
cmaj = harmonics(261.63) + harmonics(329.63) * 0.8 + harmonics(392.00) * 0.8 + harmonics(523.25) * 0.6
ch = chroma(cmaj, SR)
top = NOTES[int(np.argmax(ch))]
top3 = [NOTES[i] for i in np.argsort(ch)[::-1][:3]]
check("C is among the strongest pitch classes", "C" in top3,
      f"top three {top3} — a major triad often peaks on its fifth, that is physics, not a bug")
k, mode, conf = estimate_key(ch)
check("key detected as C major", k == "C" and mode == "major", f"got {k} {mode}, confidence {conf:.2f}")
amin = harmonics(220.00) + harmonics(261.63) * 0.8 + harmonics(329.63) * 0.8 + harmonics(440.0) * 0.6
k2, mode2, conf2 = estimate_key(chroma(amin, SR))
check("A minor triad detected", k2 == "A" and mode2 == "minor", f"got {k2} {mode2}, confidence {conf2:.2f}")
k3, m3, conf3 = estimate_key(chroma(np.ones(N) * 0.05, SR))
check("noise scores far below a real triad", conf3 < 0.15 and conf3 < conf / 3,
      f"noise {conf3:.2f} vs C major {conf:.2f}")

print("\nLOUDNESS (K-weighted, not certified R128)")
a = harmonics(1000, n=1)
l1 = loudness_lu(a, SR)
l2 = loudness_lu(a * 2.0, SR)
check("doubling amplitude adds ~6 LU", 5.5 < (l2 - l1) < 6.5, f"delta {l2 - l1:.2f} LU")
low = loudness_lu(tone([40], [1.0]), SR)
mid = loudness_lu(tone([1000], [1.0]), SR)
check("K-weighting attenuates 40 Hz vs 1 kHz", low < mid - 6.0, f"40Hz {low:.1f} vs 1kHz {mid:.1f} LU")
hi = loudness_lu(tone([4000], [1.0]), SR)
check("K-weighting lifts 4 kHz above 1 kHz", hi > mid, f"4kHz {hi:.1f} vs 1kHz {mid:.1f} LU")
g = k_weight_gain(np.array([38.0, 1000.0, 4000.0]))
check("gain curve is monotonic in this range", g[0] < g[1] < g[2], f"{g.round(3).tolist()}")

print("\nTEMPO OCTAVE")
dt = 1 / 60.0
beats = np.zeros(240)                      # one pulse per 30 frames = 120 BPM at 60 fps
beats[::30] = 1.0
fixed, mult = tempo_octave_fix(60.0, beats, dt)
check("60 BPM on a 120 BPM pulse is doubled", abs(fixed - 120.0) < 1.0, f"-> {fixed:.0f} BPM (x{mult})")
fixed2, mult2 = tempo_octave_fix(240.0, beats, dt)
check("240 BPM is halved into range", abs(fixed2 - 120.0) < 1.0, f"-> {fixed2:.0f} BPM (x{mult2})")
fixed3, _ = tempo_octave_fix(120.0, beats, dt)
check("correct tempo is left alone", abs(fixed3 - 120.0) < 1.0, f"-> {fixed3:.0f} BPM")

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
