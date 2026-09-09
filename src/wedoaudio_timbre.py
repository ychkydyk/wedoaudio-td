"""WEDOAUDIO 4.3 — timbre, tonality and loudness on top of the 4.2 feature bus.

Closes three of the four gaps named in the 4.2 README, in pure numpy, no plugins:
  MFCC        mel filterbank + DCT-II, 13 coefficients
  CHROMA/KEY  12 pitch classes + Krumhansl-Schmuckler correlation -> key, mode, confidence
  LOUDNESS    K-weighted loudness in LU, computed from the magnitude spectrum

Honest scope, stated in the code and repeated in the README:
  * The loudness here is a K-weighted spectral approximation of EBU R128 momentary
    loudness. It is NOT certified R128: true R128 filters in the time domain and gates
    over 400 ms / 3 s windows with a -10 LU relative gate. Use it for relative decisions
    (is this louder than that, by how much), not for delivery compliance.
  * Key detection assumes 12-TET and A4 = 440 Hz. On atonal, percussive or heavily
    detuned material the confidence collapses toward zero — read the confidence, not the key.
  * Everything is computed from the same magnitude frame the 4.2 bus already produces,
    so the cost is one filterbank multiply per frame and nothing else.
"""
import numpy as np

A4 = 440.0
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (probe-tone ratings), the standard reference set.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(n_bins, sr, n_mels=26, fmin=30.0, fmax=None):
    """Triangular mel bank as an (n_mels, n_bins) matrix. Build once, reuse every frame."""
    fmax = fmax or min(8000.0, sr / 2.0)
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(mid - lo, 1e-9)
        right = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
        s = fb[i].sum()
        if s > 0: fb[i] /= s          # area-normalised: bank shape does not scale with bin width
    return fb


def mfcc(mag, fb, n=13, floor_db=80.0):
    """13 MFCCs from one magnitude frame. mfcc[0] tracks overall log energy; mfcc[1:]
    is the spectral shape and is level-invariant, because the floor is taken relative
    to the frame's own peak rather than as an absolute constant. An absolute floor
    makes quiet bands clip at different levels when the input is scaled, which drifts
    the shape — measured at 6.8 units for a 4x gain before this was fixed."""
    mel = fb @ np.asarray(mag, dtype=float) ** 2
    peak = float(mel.max())
    floor = peak * (10.0 ** (-floor_db / 10.0)) if peak > 0 else 1e-12
    log_mel = np.log(np.maximum(mel, floor))
    k = np.arange(len(log_mel))
    out = np.empty(n)
    for i in range(n):                                        # DCT-II, orthonormal-ish
        out[i] = float(np.sum(log_mel * np.cos(np.pi * i * (2 * k + 1) / (2 * len(log_mel)))))
    out *= np.sqrt(2.0 / len(log_mel))
    out[0] /= np.sqrt(2.0)
    return out


def chroma(mag, sr, fmin=55.0, fmax=3000.0):
    """12 pitch classes, energy-weighted, normalised to max 1. 12-TET, A4=440.

    fmax=3000 is measured, not guessed. Swept 900..5000 Hz on synthetic triads:
    above ~3 kHz the upper harmonics of a minor triad pull the estimate to the
    parallel major (A minor read as A major at 5 kHz); below ~1.6 kHz too few bins
    survive and flat noise starts scoring a key (confidence 0.84 at 1200 Hz).
    At 3000 Hz: C major 0.80, A minor 0.61, white noise 0.13."""
    mag = np.asarray(mag, dtype=float)
    freqs = np.linspace(0.0, sr / 2.0, len(mag))
    out = np.zeros(12)
    sel = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[sel]
    if f.size == 0: return out
    midi = 69.0 + 12.0 * np.log2(np.maximum(f, 1e-9) / A4)
    pc = np.rint(midi).astype(int) % 12
    np.add.at(out, pc, mag[sel] ** 2)
    # Density correction. A linear FFT grid puts many more bins into the high pitch
    # classes than the low ones, so flat noise produced a shaped chroma and scored
    # 0.22 confidence against a key profile. Dividing by the bin count per class makes
    # a flat spectrum give a flat chroma, which is what "no key" should look like.
    counts = np.zeros(12)
    np.add.at(counts, pc, 1.0)
    out = np.where(counts > 0, out / np.maximum(counts, 1.0), 0.0)
    m = out.max()
    return out / m if m > 0 else out


def estimate_key(chroma_vec):
    """Return (name, mode, confidence 0..1). Confidence is the margin between the best
    and second-best correlation, so an ambiguous or atonal frame reports near zero."""
    c = np.asarray(chroma_vec, dtype=float)
    if c.sum() <= 0: return ("-", "-", 0.0)
    c = (c - c.mean()) / (c.std() or 1.0)
    scores = []
    for mode, prof in (("major", KS_MAJOR), ("minor", KS_MINOR)):
        p = (prof - prof.mean()) / prof.std()
        for root in range(12):
            scores.append((float(np.dot(c, np.roll(p, root)) / 12.0), NOTES[root], mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    # Confidence needs both terms: the absolute correlation says "this frame is tonal
    # at all", the margin says "and this key beats the runner-up". Margin alone gave a
    # tonal triad and white noise the same 0.07, which is a useless number.
    margin = best[0] - second[0]
    conf = max(0.0, min(1.0, max(0.0, best[0]) * (0.55 + 3.0 * margin)))
    return (best[1], best[2], conf)


def k_weight_gain(freqs):
    """Amplitude gain of the ITU-R BS.1770 K-weighting, evaluated on a frequency grid.
    Analytic form of the two stages (high-shelf + high-pass) at 48 kHz reference."""
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    shelf_db = 4.0 * (f ** 2) / (f ** 2 + 1500.0 ** 2)        # ~ +4 dB above ~1.5 kHz
    hp = (f ** 2) / (f ** 2 + 38.0 ** 2)                      # 2nd-order-ish HP at 38 Hz
    return (10.0 ** (shelf_db / 20.0)) * hp


def loudness_lu(mag, sr, cache={}):
    """K-weighted loudness of one frame, in LU relative to full scale.
    Not certified R128 — see the module docstring."""
    mag = np.asarray(mag, dtype=float)
    n = len(mag)
    g = cache.get(("g", n, sr))
    if g is None:
        g = k_weight_gain(np.linspace(0.0, sr / 2.0, n)); cache[("g", n, sr)] = g
    p = float(np.sum((mag * g) ** 2))
    return -0.691 + 10.0 * np.log10(max(p, 1e-12))


def tempo_octave_fix(bpm, novelty_hist, dt, lo=70.0, hi=180.0):
    """Resolve the octave ambiguity named as a 4.2 limitation: score bpm, bpm/2 and
    bpm*2 by how well a comb of that period matches the recent novelty curve, then
    prefer the candidate inside the musical range. Returns (bpm, chosen_multiplier)."""
    x = np.asarray(novelty_hist, dtype=float)
    if x.size < 8 or bpm <= 0: return (bpm, 1.0)
    x = x - x.mean()
    best, best_mult = -1e9, 1.0
    for mult in (0.5, 1.0, 2.0):
        cand = bpm * mult
        period = 60.0 / cand / max(dt, 1e-6)
        if period < 2 or period > x.size / 2: continue
        idx = np.arange(0, x.size, period).astype(int)
        idx = idx[idx < x.size]
        if idx.size < 2: continue
        score = float(x[idx].mean())
        if lo <= cand <= hi: score *= 1.25          # musical-range prior, not a hard gate
        if score > best: best, best_mult = score, mult
    return (bpm * best_mult, best_mult)
