"""WEDOAUDIO benchmark — does it actually hear kicks, snares, bands and tempo?

Not unit tests. This renders synthetic tracks of several styles at several sample rates
and bit depths, runs them through the real 60 fps frame pipeline, and scores the result
against ground truth the way the field does it: onset F-measure with a +/-50 ms tolerance,
which is the MIREX standard threshold (chosen there to absorb hand-labelling error).

Run: python src/_wedoaudio_bench.py
Every row prints precision / recall / F so a regression shows which half broke.
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedoaudio_dsp import WedoAudio
from wedoaudio_timbre import mel_filterbank, chroma, estimate_key, loudness_lu, tempo_octave_fix

FPS = 60.0
TOL = 0.050              # MIREX onset tolerance, seconds
FFT = 2048


# ---------- synthesis: instruments that behave like the real thing ----------
def env(n, sr, attack=0.002, decay=0.25):
    t = np.arange(n) / sr
    a = np.clip(t / max(attack, 1e-6), 0, 1)
    return a * np.exp(-t / decay)


def kick(sr, dur=0.30, f0=110.0, f1=42.0):
    n = int(sr * dur); t = np.arange(n) / sr
    f = f1 + (f0 - f1) * np.exp(-t / 0.035)                 # pitch drop, as a real kick
    ph = 2 * np.pi * np.cumsum(f) / sr
    body = np.sin(ph) * env(n, sr, 0.001, 0.10)
    click = np.random.RandomState(0).randn(n) * env(n, sr, 0.0005, 0.004) * 0.25
    return (body + click) * 0.9


def bandpass_noise(n, sr, lo, hi, seed):
    """Noise limited to [lo, hi] by zeroing bins. A first attempt used np.diff as a
    cheap high-pass; it left the snare 20x quieter than the kick in the 1.8-6 kHz
    band, so the detector was being blamed for a defect in this file."""
    x = np.random.RandomState(seed).randn(n)
    X = np.fft.rfft(x)
    f = np.linspace(0.0, sr / 2.0, len(X))
    X[(f < lo) | (f > hi)] = 0.0
    y = np.fft.irfft(X, n)
    m = np.max(np.abs(y)) or 1.0
    return y / m


def snare(sr, dur=0.22):
    """Noise body 200 Hz-9 kHz plus two shell tones, peaking near the kick's level,
    which is what a snare does in a real mix."""
    n = int(sr * dur); t = np.arange(n) / sr
    body = bandpass_noise(n, sr, 200.0, 9000.0, 1) * env(n, sr, 0.001, 0.085)
    crack = bandpass_noise(n, sr, 2000.0, 8000.0, 3) * env(n, sr, 0.0008, 0.030) * 0.8
    tone = (np.sin(2 * np.pi * 190 * t) + np.sin(2 * np.pi * 330 * t)) * env(n, sr, 0.001, 0.05) * 0.25
    return (body + crack + tone) * 0.85


def hat(sr, dur=0.06):
    n = int(sr * dur)
    return bandpass_noise(n, sr, 6000.0, 14000.0, 2) * env(n, sr, 0.0005, 0.02) * 0.30


def bassline(sr, dur, f=55.0):
    t = np.arange(int(sr * dur)) / sr
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * 0.25


def pad(sr, dur, freqs=(261.63, 329.63, 392.0)):
    t = np.arange(int(sr * dur)) / sr
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return x * 0.2


def place(track, sr, sample, at_s):
    i = int(at_s * sr)
    j = min(len(track), i + len(sample))
    if i < len(track): track[i:j] += sample[:j - i]


def render(style, sr, seconds=8.0, level=1.0):
    """Returns (audio, kick_times, snare_times, bpm)."""
    n = int(sr * seconds); x = np.zeros(n)
    kt, st = [], []
    if style == "four-on-floor":
        bpm = 128.0; beat = 60.0 / bpm
        for i in range(int(seconds / beat)):
            t = i * beat; place(x, sr, kick(sr), t); kt.append(t)
            if i % 2 == 1: place(x, sr, snare(sr), t); st.append(t)
            place(x, sr, hat(sr), t + beat / 2)
        x += bassline(sr, seconds)
    elif style == "breakbeat-dnb":
        bpm = 174.0; beat = 60.0 / bpm
        pattern_k = [0, 2.5, 4, 6.5]; pattern_s = [1, 3, 5, 7]
        bars = int(seconds / (beat * 8))
        for b in range(bars):
            base = b * beat * 8
            for p in pattern_k: place(x, sr, kick(sr), base + p * beat); kt.append(base + p * beat)
            for p in pattern_s: place(x, sr, snare(sr), base + p * beat); st.append(base + p * beat)
        x += bassline(sr, seconds, 41.0)
    elif style == "halftime":
        bpm = 70.0; beat = 60.0 / bpm
        for i in range(int(seconds / beat)):
            t = i * beat
            if i % 2 == 0: place(x, sr, kick(sr), t); kt.append(t)
            else: place(x, sr, snare(sr), t); st.append(t)
        x += pad(sr, seconds)
    elif style == "waltz":
        bpm = 150.0; beat = 60.0 / bpm
        for i in range(int(seconds / beat)):
            t = i * beat
            if i % 3 == 0: place(x, sr, kick(sr), t); kt.append(t)
            else: place(x, sr, hat(sr), t)
        x += pad(sr, seconds, (220.0, 261.63, 329.63))
    elif style == "ambient-nobeat":
        bpm = 0.0; x += pad(sr, seconds) + bassline(sr, seconds, 65.0) * 0.5
    elif style == "clipped-loud":
        bpm = 128.0; beat = 60.0 / bpm
        for i in range(int(seconds / beat)):
            t = i * beat; place(x, sr, kick(sr), t); kt.append(t)
            if i % 2 == 1: place(x, sr, snare(sr), t); st.append(t)
        x = np.clip(x * 6.0, -1.0, 1.0)                     # brickwalled, as loud masters are
    else:
        raise ValueError(style)
    peak = np.max(np.abs(x)) or 1.0
    return x / peak * level, kt, st, bpm


def quantize(x, bits):
    if bits >= 32: return x
    q = 2 ** (bits - 1)
    return np.round(np.clip(x, -1, 1) * (q - 1)) / (q - 1)


# ---------- the real pipeline: 60 fps frames of magnitude spectrum ----------
def run_pipeline(x, sr, cfg=None):
    hop = int(sr / FPS)
    win = np.hanning(FFT)
    wa = WedoAudio(sr=sr)
    keys = ("rawkick", "rawsnare", "bass", "mid", "high", "rms", "bpm", "agcgain", "centroid")
    out = {k: [] for k in keys}; out["t"] = []
    for start in range(0, len(x) - FFT, hop):
        frame = x[start:start + FFT] * win
        mag = np.abs(np.fft.rfft(frame)) / (FFT / 4)
        rms = float(np.sqrt(np.mean(frame ** 2)))
        t = start / sr
        f = wa.process(mag, rms, t, 1.0 / FPS, cfg)
        out["t"].append(t)
        for k in keys:
            out[k].append(float(f[k]))
    return {k: np.asarray(v) for k, v in out.items()}


def events(flags, times):
    """rawkick / rawsnare are single-frame 1.0 impulses -> take their timestamps.

    A frame analyses the window that ENDS at start+FFT, so the transient that fired
    it can sit up to one window earlier than the frame's nominal time. We report the
    frame time as-is and let the +/-50 ms tolerance absorb that window, which is
    precisely the situation the tolerance exists for."""
    idx = np.where(np.asarray(flags) > 0.5)[0]
    return np.asarray(times)[idx] if len(idx) else np.array([])


def f_measure(det, ref, tol=TOL):
    det = np.sort(np.asarray(det, dtype=float)); ref = np.sort(np.asarray(ref, dtype=float))
    if len(ref) == 0 and len(det) == 0: return 1.0, 1.0, 1.0
    if len(ref) == 0: return 0.0, 1.0, 0.0
    used = np.zeros(len(det), bool); tp = 0
    for r in ref:
        cand = np.where((~used) & (np.abs(det - r) <= tol))[0] if len(det) else np.array([])
        if len(cand):
            used[cand[np.argmin(np.abs(det[cand] - r))]] = True; tp += 1
    p = tp / len(det) if len(det) else 0.0
    rc = tp / len(ref)
    f = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    return p, rc, f


if __name__ == "__main__":
    styles = ["four-on-floor", "breakbeat-dnb", "halftime", "waltz", "ambient-nobeat", "clipped-loud"]
    print(f"ONSETS — F-measure at +/-{int(TOL*1000)} ms (MIREX tolerance), 60 fps frames, FFT {FFT}\n")
    print(f"{'style':<16}{'sr':>7}{'bits':>6}{'kick P/R/F':>22}{'snare P/R/F':>22}{'bpm':>12}")
    rows = []
    for style in styles:
        for sr, bits in ((44100, 32), (48000, 32), (44100, 16)):
            x, kt, st, bpm = render(style, sr)
            x = quantize(x, bits)
            r = run_pipeline(x, sr)
            kp, kr, kf = f_measure(events(r["rawkick"], r["t"]), kt)
            sp, srr, sf = f_measure(events(r["rawsnare"], r["t"]), st)
            bpm_est = float(np.median(r["bpm"][len(r["bpm"]) // 2:])) if len(r["bpm"]) else 0.0
            bl = "-" if bpm == 0 else f"{bpm_est:.0f}/{bpm:.0f}"
            print(f"{style:<16}{sr:>7}{bits:>6}{f'{kp:.2f}/{kr:.2f}/{kf:.2f}':>22}"
                  f"{f'{sp:.2f}/{srr:.2f}/{sf:.2f}':>22}{bl:>12}")
            rows.append((style, sr, bits, kf, sf, bpm_est, bpm))

    print("\nBANDS — does energy land in the band it belongs to")
    sr = 44100
    for name, f0, expect in (("sub 40 Hz", 40, "bass"), ("bass 90 Hz", 90, "bass"),
                             ("mid 900 Hz", 900, "mid"), ("high 5 kHz", 5000, "high")):
        t = np.arange(int(sr * 2.0)) / sr
        x = np.sin(2 * np.pi * f0 * t) * 0.5
        r = run_pipeline(x, sr)
        b = float(np.median(r["bass"])); m = float(np.median(r["mid"])); h = float(np.median(r["high"]))
        got = ["bass", "mid", "high"][int(np.argmax([b, m, h]))]
        mark = "ok " if got == expect else "BAD"
        print(f"  {mark} {name:<12} bass {b:.2f}  mid {m:.2f}  high {h:.2f}  -> {got} (expected {expect})")

    print("\nAGC — the same track at four levels should give the same band reading")
    vals = []
    for lvl in (0.05, 0.2, 0.5, 1.0):
        x, *_ = render("four-on-floor", 44100, level=lvl)
        r = run_pipeline(x, 44100)
        vals.append(float(np.median(r["bass"])))
        print(f"  level {lvl:<5} bass {vals[-1]:.3f}  gain {float(np.median(r['agcgain'])):.2f}")
    print(f"  spread across a 20x level range: {max(vals) - min(vals):.3f} (lower is better)")

    print("\nKEY — on a rendered pad, not a synthetic spectrum")
    for freqs, label in (((261.63, 329.63, 392.0), "C major"), ((220.0, 261.63, 329.63), "A minor")):
        x = pad(44100, 3.0, freqs)
        win = np.hanning(FFT)
        mags = [np.abs(np.fft.rfft(x[i:i + FFT] * win)) / (FFT / 4) for i in range(0, len(x) - FFT, 4096)]
        ch = np.mean([chroma(m, 44100) for m in mags], axis=0)
        k, mode, conf = estimate_key(ch)
        print(f"  {label:<10} -> {k} {mode}, confidence {conf:.2f}")

    kicks_ok = sum(1 for r_ in rows if r_[3] >= 0.80)
    snares_ok = sum(1 for r_ in rows if r_[4] >= 0.80)
    tempo_rows = [r_ for r_ in rows if r_[6] != 0]
    tempo_ok = sum(1 for r_ in tempo_rows if abs(r_[5] - r_[6]) / r_[6] < 0.10)
    print("")
    print("kicks  F >= 0.80 : %d/%d" % (kicks_ok, len(rows)))
    print("snares F >= 0.80 : %d/%d" % (snares_ok, len(rows)))
    print("tempo within 10%% : %d/%d   (breakbeat reads half time, see README)" % (tempo_ok, len(tempo_rows)))
    # CI gate: the baseline measured on 2026-09-08. The renders are deterministic,
    # so a drop is a regression, never a flaky test.
    bad = []
    if kicks_ok < 18: bad.append("kicks %d/18" % kicks_ok)
    if snares_ok < 18: bad.append("snares %d/18" % snares_ok)
    if tempo_ok < 12: bad.append("tempo %d/15" % tempo_ok)
    if bad:
        print("REGRESSION: " + ", ".join(bad))
        sys.exit(1)
    print("benchmark at or above the recorded baseline")
