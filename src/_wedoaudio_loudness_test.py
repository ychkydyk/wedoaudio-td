"""Conformance bench for wedoaudio_loudness. Run: python src/_wedoaudio_loudness_test.py

The signals are the SYNTHETIC test cases of EBU Tech 3341 (loudness, true peak) and
Tech 3342 (loudness range): tones whose level is known by construction, so the truth
is exact and nobody's opinion is required. Tolerances are the ones the documents set:
+-0.1 LU for loudness, +0.2/-0.4 dB for true peak, +-1 LU for range.

Not covered, and said so: 3341 cases 6 (5.0 surround), 7-8 (authentic programme,
needs the EBU files) and 20-23 (decimated clicks). Mono and stereo only.

Every test prints the number it measured, so a failure tells you the value."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedoaudio_loudness import (LoudnessMeter, k_coefficients, k_impulse, _StreamFIR,
                                tp_phases, _tp_factor)

ok = fail = skipped = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}  {detail}")
    else: fail += 1; print(f"  FAIL  {name}  {detail}")


def fmt(v):
    return "None" if v is None else f"{v:+.2f}"


def amp(dbfs):
    return 10.0 ** (dbfs / 20.0)


def tone(sr, seconds, dbfs, freq=1000.0, phase=0.0, channels=2):
    n = int(round(seconds * sr))
    x = amp(dbfs) * np.sin(2 * np.pi * freq * np.arange(n) / sr + phase)
    return np.tile(x, (channels, 1))


def run(sr, signal, block=800, channels=2, collect=None, truepeak=False):
    """Feed `signal` (channels, n) the way a frame loop would; return last reading."""
    m = LoudnessMeter(sr, channels, truepeak=truepeak)
    out, i, n = None, 0, signal.shape[1]
    while i < n:
        out = m.process(signal[:, i:i + block])
        if collect is not None:
            collect.append((i + block, out))
        i += block
    return out, m


# ---------------------------------------------------------------------------------
print("K-WEIGHTING")
(b1, a1), (b2, a2) = k_coefficients(48000)
ref = ([1.53512485958697, -2.69169618940638, 1.19839281085285],
       [1.0, -1.69065929318241, 0.73248077421585],
       [1.0, -1.99004745483398, 0.99007225036621])
err = max(np.abs(b1 - ref[0]).max(), np.abs(a1 - ref[1]).max(), np.abs(a2 - ref[2]).max())
check("48 kHz coefficients reproduce the BS.1770-4 table", err < 1e-9, f"max err {err:.1e}")

(c1, d1), _ = k_coefficients(44100)
check("44.1 kHz coefficients are re-derived, not reused", np.abs(c1 - b1).max() > 1e-3,
      f"shelf b0 {c1[0]:.6f} vs {b1[0]:.6f} at 48k")

try:
    from scipy.signal import lfilter
    worst = 0.0
    for sr in (44100, 48000, 96000):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((1, sr * 2)) * 0.2
        (p1, q1), (p2, q2) = k_coefficients(sr)
        yref = lfilter(p2, q2, lfilter(p1, q1, x[0]))
        f = _StreamFIR(k_impulse(sr), 1)
        parts, i = [], 0
        while i < x.shape[1]:
            n = int(rng.integers(64, 3000))          # ragged blocks, like dropped frames
            parts.append(f.process(x[:, i:i + n])[0]); i += n
        y = np.concatenate(parts)
        worst = max(worst, abs(10 * np.log10(np.mean(y ** 2) / np.mean(yref ** 2))))
    check("streamed FIR equals the recursive filter (scipy reference)", worst < 1e-6,
          f"worst loudness difference {worst:.1e} LU over 44.1/48/96 kHz, ragged blocks")
except ImportError:
    skipped += 1
    print("  SKIP  streamed FIR vs recursive filter  (scipy not installed)")

# ---------------------------------------------------------------------------------
print("EBU TECH 3341 - INTEGRATED")
for sr in (48000, 44100):
    out, _ = run(sr, tone(sr, 20, -23.0))
    check(f"case 1: stereo 1 kHz -23 dBFS 20 s @ {sr}", abs(out["integrated"] + 23.0) <= 0.1,
          f"I {fmt(out['integrated'])}  M {fmt(out['momentary'])}  S {fmt(out['shortterm'])}")
    check(f"        momentary and short-term agree @ {sr}",
          abs(out["momentary"] + 23.0) <= 0.1 and abs(out["shortterm"] + 23.0) <= 0.1)

sr = 48000
out, _ = run(sr, tone(sr, 20, -33.0))
check("case 2: -33 dBFS", abs(out["integrated"] + 33.0) <= 0.1, f"I {fmt(out['integrated'])}")

sig = np.concatenate([tone(sr, 10, -36), tone(sr, 60, -23), tone(sr, 10, -36)], axis=1)
out, _ = run(sr, sig)
check("case 3: -36 / -23 / -36, relative gate drops the quiet ends",
      abs(out["integrated"] + 23.0) <= 0.1, f"I {fmt(out['integrated'])}")

sig = np.concatenate([tone(sr, 10, -72), tone(sr, 10, -36), tone(sr, 60, -23),
                      tone(sr, 10, -36), tone(sr, 10, -72)], axis=1)
out, _ = run(sr, sig)
check("case 4: adds -72 dBFS ends, absolute gate drops them",
      abs(out["integrated"] + 23.0) <= 0.1, f"I {fmt(out['integrated'])}")

sig = np.concatenate([tone(sr, 20, -26), tone(sr, 20.1, -20), tone(sr, 20, -26)], axis=1)
out, _ = run(sr, sig)
check("case 5: -26 / -20 / -26", abs(out["integrated"] + 23.0) <= 0.1, f"I {fmt(out['integrated'])}")

out, _ = run(sr, tone(sr, 20, -23.0, channels=1), channels=1)
check("mono -23 dBFS reads 3 dB below the stereo pair", abs(out["integrated"] + 26.01) <= 0.1,
      f"I {fmt(out['integrated'])}")

# ---------------------------------------------------------------------------------
print("EBU TECH 3341 - SHORT-TERM AND MOMENTARY")
seg = np.concatenate([tone(sr, 1.34, -20), tone(sr, 1.66, -30)], axis=1)
log = []
run(sr, np.tile(seg, (1, 5)), collect=log)
vals = [o["shortterm"] for t, o in log if t >= 3 * sr and o["shortterm"] is not None]
check("case 9: 1.34 s -20 / 1.66 s -30 repeated, S stays at -23 after 3 s",
      max(abs(v + 23.0) for v in vals) <= 0.1, f"S range {min(vals):+.2f} .. {max(vals):+.2f}")

seg = np.concatenate([tone(sr, 0.18, -20), tone(sr, 0.22, -30)], axis=1)
log = []
run(sr, np.tile(seg, (1, 25)), collect=log)
vals = [o["momentary"] for t, o in log if t >= 1 * sr and o["momentary"] is not None]
check("case 12: 0.18 s -20 / 0.22 s -30 repeated, M stays at -23 after 1 s",
      max(abs(v + 23.0) for v in vals) <= 0.1, f"M range {min(vals):+.2f} .. {max(vals):+.2f}")

# ---------------------------------------------------------------------------------
print("EBU TECH 3341 - TRUE PEAK  (tolerance +0.2 / -0.4 dB)")


def ideal_peak(x, up=64):
    """Peak of the ideal band-limited reconstruction: zero-padding in the spectrum.
    An independent reference - it shares no code with the meter's interpolator."""
    n = len(x)
    X = np.fft.rfft(x)
    Y = np.zeros(n * up // 2 + 1, dtype=complex)
    Y[:len(X)] = X
    return float(np.abs(np.fft.irfft(Y, n * up) * up).max())


def steady_tone(freq_div, phase_deg, peak, seconds=1.0, fade=0.01):
    """The Tech 3341 tone WITHOUT a gating artefact. A sine switched on at 60 degrees is
    not that sine: the cut itself overshoots between samples (+0.4 dB at fs/6, +0.7 dB
    at fs/8 - measured against the ideal reconstruction below), and a correct meter has
    to report it. The expected -6.0 dBTP belongs to the tone, so the tone is faded in."""
    n = np.arange(int(seconds * sr))
    x = peak * np.sin(2 * np.pi * n / freq_div + np.radians(phase_deg))
    k = int(fade * sr)
    r = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    x[:k] *= r
    x[-k:] *= r[::-1]
    return x


def tp_case(name, freq_div, phase_deg, peak, expect):
    x = steady_tone(freq_div, phase_deg, peak)
    out, _ = run(sr, x[np.newaxis, :], channels=1, truepeak=True)
    e = out["truepeak_max"] - expect
    check(name, -0.4 <= e <= 0.2, f"read {out['truepeak_max']:+.2f} dBTP, expected {expect:+.1f} "
          f"(sample peak {20 * np.log10(np.abs(x).max()):+.2f})")


tp_case("case 15: fs/4,  0 deg, 0.50 FS", 4, 0.0, 0.5, -6.0)
tp_case("case 16: fs/4, 45 deg, 0.50 FS", 4, 45.0, 0.5, -6.0)
tp_case("case 17: fs/6, 60 deg, 0.50 FS", 6, 60.0, 0.5, -6.0)
tp_case("case 18: fs/8, 67.5 deg, 0.50 FS", 8, 67.5, 0.5, -6.0)
tp_case("case 19: fs/4, 45 deg, 1.41 FS (samples at full scale)", 4, 45.0, 1.41, +3.0)

worst = 0.0
P = tp_phases(_tp_factor(sr))
for fr in (0.05, 0.1, 1 / 6, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
    for p0 in np.linspace(0, np.pi, 25):
        x = np.sin(2 * np.pi * fr * np.arange(4096) + p0)
        win = np.lib.stride_tricks.sliding_window_view(x, P.shape[1])
        e = 20 * np.log10(np.abs(win @ P[:, ::-1].T).max())
        if abs(e) > abs(worst): worst = e
check("interpolator error over 0.05..0.45 fs, all phases", -0.4 <= worst <= 0.2,
      f"worst {worst:+.2f} dB")

worst = 0.0
for div, ph in ((6, 60.0), (8, 67.5), (4, 45.0), (5, 30.0)):
    n = np.arange(4800)
    gated = np.concatenate([np.zeros(2400), 0.5 * np.sin(2 * np.pi * n / div + np.radians(ph)), np.zeros(2400)])
    out, _ = run(sr, gated[np.newaxis, :], channels=1, truepeak=True)
    e = out["truepeak_max"] - 20 * np.log10(ideal_peak(gated))
    if abs(e) > abs(worst): worst = e
check("abruptly gated tones: meter agrees with the ideal reconstruction", -0.4 <= worst <= 0.2,
      f"worst {worst:+.2f} dB against 64x spectral zero-padding")

# ---------------------------------------------------------------------------------
print("EBU TECH 3342 - LOUDNESS RANGE  (tolerance +-1 LU)")
for name, lo, hi, want in (("case 1: -20 / -30", -20, -30, 10.0), ("case 2: -20 / -15", -20, -15, 5.0)):
    sig = np.concatenate([tone(sr, 20, lo), tone(sr, 20, hi)], axis=1)
    out, _ = run(sr, sig)
    check(name, out["lra"] is not None and abs(out["lra"] - want) <= 1.0, f"LRA {fmt(out['lra'])} LU")

# ---------------------------------------------------------------------------------
print("THIRD STATE AND ROBUSTNESS")
m = LoudnessMeter(sr, 2)
o = m.process(tone(sr, 0.35, -23))
check("momentary is None before 400 ms of audio", o["momentary"] is None, f"M {fmt(o['momentary'])}")
o = m.process(tone(sr, 1.0, -23))
check("short-term is None before 3 s of audio", o["momentary"] is not None and o["shortterm"] is None,
      f"M {fmt(o['momentary'])}  S {fmt(o['shortterm'])}")

out, _ = run(sr, np.zeros((2, sr * 3)))
check("silence gives no integrated value, not a number", out["integrated"] is None and out["momentary"] is None,
      f"I {fmt(out['integrated'])}")

full = tone(sr, 12, -23.0) + 0.3 * np.random.default_rng(3).standard_normal((2, sr * 12)) * amp(-23)
a, _ = run(sr, full, block=800)
b, _ = run(sr, full, block=735)
c, _ = run(sr, full, block=sr * 12)
d = max(abs(a["integrated"] - b["integrated"]), abs(a["integrated"] - c["integrated"]))
check("result does not depend on how samples are cut into blocks", d < 1e-9, f"max difference {d:.1e} LU")

bad = tone(sr, 5, -23.0); bad[:, 1000:1010] = np.nan; bad[0, 2000] = np.inf
out, _ = run(sr, bad)
check("NaN and inf in the input do not poison the meter",
      out["integrated"] is not None and np.isfinite(out["integrated"]) and abs(out["integrated"] + 23.0) < 0.2,
      f"I {fmt(out['integrated'])}")

out, m = run(sr, tone(sr, 5, -10.0), truepeak=True)
m.reset()
o = m.read()
check("reset clears every reading", all(v is None for v in o.values()), str({k: fmt(v) for k, v in o.items()}))

try:
    LoudnessMeter(sr, 2).process(np.zeros((3, 100)))
    check("wrong channel count is refused", False)
except ValueError as e:
    check("wrong channel count is refused", True, str(e))

print(f"\n{ok} passed, {fail} failed" + (f", {skipped} skipped" if skipped else ""))
sys.exit(1 if fail else 0)
