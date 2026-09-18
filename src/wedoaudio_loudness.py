# wedoaudio_loudness.py - sample-domain loudness per ITU-R BS.1770-4 / EBU R128.
#
# Why a second module: the analyzer core works on magnitude spectra at frame rate,
# and loudness measured there is an approximation (see wedoaudio_timbre.loudness_lu).
# BS.1770 is defined on SAMPLES: two biquads, mean square over 400 ms blocks hopped
# by 100 ms, two gates. This module does exactly that, on the samples themselves.
#
# Pure numpy - TouchDesigner ships numpy and no scipy. The K-weighting IIR is applied
# as its own impulse response, truncated where the tail is below -110 dB, so the
# result matches the recursive filter to better than 1e-7 LU (the test bench checks
# this against scipy.signal.lfilter when scipy is present).
#
# What is measured:        momentary (400 ms), short-term (3 s), integrated (gated,
#                          -70 LUFS absolute / -10 LU relative), loudness range LRA
#                          (EBU Tech 3342), true peak (8x oversampled below 96 kHz, dBTP).
# What is NOT claimed:     certification. Mono and stereo are verified against the
#                          synthetic EBU Tech 3341 signals; surround weights can be
#                          passed in but have not been verified here.
# Third state:             a value that cannot be measured yet is None, not a
#                          plausible number. Momentary needs 400 ms of audio,
#                          short-term 3 s, integrated one gating block above -70.

import numpy as np

LUFS_OFFSET = -0.691          # BS.1770: makes a 997 Hz full-scale sine read -3.01 LKFS
ABS_GATE = -70.0              # LUFS
REL_GATE = -10.0              # LU below the absolute-gated mean (integrated)
LRA_REL_GATE = -20.0          # LU (EBU Tech 3342)
MAX_BLOCKS = 72000            # gating blocks kept: 2 h at 10 per second


# ----- K-weighting ------------------------------------------------------------------

def k_coefficients(sr):
    """BS.1770-4 K-weighting biquads for ANY sample rate.

    The standard tabulates coefficients for 48 kHz only. Reusing them at 44.1 kHz
    shifts both corner frequencies by 8.8 percent - the same class of mistake as a
    hard-coded sample rate elsewhere in this analyzer. The filters are therefore
    re-derived from their analog prototypes (bilinear transform with pre-warping);
    at 48 kHz this reproduces the published table to 1e-9 (asserted in the tests).
    Returns ((b, a) shelf, (b, a) highpass), a[0] == 1."""
    sr = float(sr)
    # stage 1: high shelf, +4 dB
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b1 = np.array([(Vh + Vb * K / Q + K * K) / a0,
                   2.0 * (K * K - Vh) / a0,
                   (Vh - Vb * K / Q + K * K) / a0])
    a1 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    # stage 2: RLB high-pass
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + K / Q + K * K
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    return (b1, a1), (b2, a2)


def _biquad(b, a, x):
    """Reference recursion, direct form I. Used once, to build the impulse response."""
    y = np.zeros(len(x))
    x1 = x2 = y1 = y2 = 0.0
    b0, b1_, b2_ = b
    _, a1_, a2_ = a
    for i in range(len(x)):
        xi = x[i]
        yi = b0 * xi + b1_ * x1 + b2_ * x2 - a1_ * y1 - a2_ * y2
        x2, x1 = x1, xi
        y2, y1 = y1, yi
        y[i] = yi
    return y


def k_impulse(sr, seconds=0.06):
    """Impulse response of the K-weighting cascade. The slow part is the 38 Hz
    high-pass (time constant 4.2 ms); after 60 ms the tail is below -110 dB and the
    loudness differs from the recursive filter by under 1e-8 LU (measured)."""
    n = int(round(seconds * sr))
    d = np.zeros(n)
    d[0] = 1.0
    (b1, a1), (b2, a2) = k_coefficients(sr)
    return _biquad(b2, a2, _biquad(b1, a1, d))


class _StreamFIR:
    """y = x * h over a stream of blocks of any length, all channels at once.
    Convolution runs through the FFT; outputs at index >= len(h)-1 of a circular
    convolution equal the linear one, so only those are kept."""

    def __init__(self, h, channels=1):
        self.h = np.asarray(h, dtype=float)
        self.L = len(self.h)
        self.channels = int(channels)
        self.hist = np.zeros((self.channels, self.L - 1))
        self._H = {}

    def reset(self):
        self.hist[:] = 0.0

    def process(self, x):
        """x: (channels, n) -> (channels, n)."""
        n = x.shape[1]
        if n == 0:
            return x
        buf = np.concatenate((self.hist, x), axis=1)
        N = 1 << int(np.ceil(np.log2(buf.shape[1])))
        H = self._H.get(N)
        if H is None:
            H = np.fft.rfft(self.h, N)
            self._H[N] = H
        y = np.fft.irfft(np.fft.rfft(buf, N, axis=1) * H, N, axis=1)[:, self.L - 1:self.L - 1 + n]
        self.hist = buf[:, -(self.L - 1):]
        return y


# ----- true peak --------------------------------------------------------------------

def _tp_factor(sr):
    # BS.1770 Annex 2 asks for at least 192 kHz. 4x at 48 kHz meets that, but a sine
    # locked to the sample grid (fs/4, 2fs/5) is then under-read by up to 0.44 dB no
    # matter how good the filter is - the grid itself misses the crest. 8x brings the
    # worst case across 0.05..0.45 fs to -0.15 dB (measured), at no real cost.
    return 8 if sr < 96000 else (4 if sr < 192000 else 2)


def tp_phases(factor, taps_per_phase=16, beta=8.0):
    """Polyphase interpolator: windowed sinc, generated rather than tabulated, so it
    carries no hand-copied coefficients and is verified by measurement in the tests.
    BS.1770 permits any suitable interpolator. Returns (factor, taps) matrix."""
    n = factor * taps_per_phase
    t = (np.arange(n) - (n - 1) / 2.0) / factor
    h = np.sinc(t) * np.kaiser(n, beta)
    ph = np.array([h[p::factor] for p in range(factor)])
    return ph / ph.sum(axis=1, keepdims=True)      # each phase passes DC at unity


class _TruePeak:
    def __init__(self, sr, channels=1):
        self.P = tp_phases(_tp_factor(sr))[:, ::-1].T.copy()     # (taps, factor)
        self.L = self.P.shape[0]
        self.hist = np.zeros((int(channels), self.L - 1))

    def reset(self):
        self.hist[:] = 0.0

    def process(self, x):
        """x: (channels, n) -> peak of the oversampled block over all channels, linear."""
        if x.shape[1] == 0:
            return 0.0
        buf = np.concatenate((self.hist, x), axis=1)
        win = np.lib.stride_tricks.sliding_window_view(buf, self.L, axis=1)   # (C, n, taps)
        pk = float(np.abs(win @ self.P).max())
        self.hist = buf[:, -(self.L - 1):]
        return pk


# ----- meter ------------------------------------------------------------------------

def _lufs(power):
    return LUFS_OFFSET + 10.0 * np.log10(power) if power > 0.0 else None


class LoudnessMeter:
    """Feed it samples; read momentary / short-term / integrated / LRA / true peak.

        m = LoudnessMeter(sr=48000, channels=2)
        out = m.process(block)      # block: (n,) mono or (channels, n), floats in -1..1

    `out` is a dict; a value that is not measurable yet is None."""

    def __init__(self, sr, channels=1, weights=None, truepeak=True):
        self.sr = float(sr)
        self.channels = int(channels)
        self.weights = (np.ones(self.channels) if weights is None
                        else np.asarray(weights, dtype=float))
        if len(self.weights) != self.channels:
            raise ValueError("weights must have one entry per channel")
        self.hop = int(round(0.1 * self.sr))            # 100 ms, in samples
        if self.hop < 1:
            raise ValueError("sample rate too low: %r" % sr)
        self._k = _StreamFIR(k_impulse(self.sr), self.channels)
        self._tp = _TruePeak(self.sr, self.channels) if truepeak else None
        self.reset()

    def reset(self):
        self._k.reset()
        if self._tp:
            self._tp.reset()
        self._acc = 0.0                 # weighted sum of squares in the open 100 ms slot
        self._acc_n = 0
        self._sub = []                  # closed 100 ms slots, mean square (last 30 kept)
        self._blocks = []               # 400 ms gating blocks, hop 100 ms
        self._st = []                   # short-term powers, hop 1 s (for LRA)
        self._sub_count = 0
        self._tp_max = 0.0
        self._tp_recent = []            # per-slot peaks, last 4 -> 400 ms window
        self._tp_slot = 0.0
        self.samples = 0
        self._integrated = None
        self._lra = None

    # -- feeding ---------------------------------------------------------------------
    def process(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[0] != self.channels:
            if x.shape[1] == self.channels:
                x = x.T
            else:
                raise ValueError("expected %d channel(s), got shape %r" % (self.channels, x.shape))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n = x.shape[1]
        if n:
            y = self._k.process(x)
            sq = (self.weights[:, np.newaxis] * y * y).sum(axis=0)
            if self._tp:
                pk = self._tp.process(x)
                if pk > self._tp_slot:
                    self._tp_slot = pk
                if pk > self._tp_max:
                    self._tp_max = pk
            self.samples += n
            self._accumulate(sq)
        return self.read()

    def _accumulate(self, sq):
        i, n = 0, len(sq)
        while i < n:
            take = min(self.hop - self._acc_n, n - i)
            self._acc += float(sq[i:i + take].sum())
            self._acc_n += take
            i += take
            if self._acc_n == self.hop:
                self._close_slot(self._acc / self.hop)
                self._acc, self._acc_n = 0.0, 0

    def _close_slot(self, ms):
        self._sub.append(ms)
        if len(self._sub) > 30:
            del self._sub[0]
        self._sub_count += 1
        self._tp_recent.append(self._tp_slot)
        if len(self._tp_recent) > 4:
            del self._tp_recent[0]
        self._tp_slot = 0.0
        if len(self._sub) >= 4:
            self._blocks.append(sum(self._sub[-4:]) / 4.0)
            if len(self._blocks) > MAX_BLOCKS:
                del self._blocks[0]
            self._integrated = self._gate(self._blocks, REL_GATE)[0]
        if len(self._sub) >= 30 and (self._sub_count - 30) % 10 == 0:
            self._st.append(sum(self._sub) / 30.0)
            if len(self._st) > MAX_BLOCKS // 10:
                del self._st[0]
            self._lra = self._range()

    # -- gating ----------------------------------------------------------------------
    @staticmethod
    def _gate(powers, rel):
        z = np.asarray(powers, dtype=float)
        if z.size == 0:
            return None, z
        abs_thr = 10.0 ** ((ABS_GATE - LUFS_OFFSET) / 10.0)
        za = z[z > abs_thr]
        if za.size == 0:
            return None, za
        rel_thr = za.mean() * 10.0 ** (rel / 10.0)
        zr = za[za > rel_thr]
        if zr.size == 0:
            return None, zr
        return _lufs(float(zr.mean())), zr

    def _range(self):
        _, z = self._gate(self._st, LRA_REL_GATE)
        if z.size < 2:
            return None
        l = np.sort(LUFS_OFFSET + 10.0 * np.log10(z))
        lo = l[int(round((len(l) - 1) * 0.10))]
        hi = l[int(round((len(l) - 1) * 0.95))]
        return float(hi - lo)

    # -- reading ---------------------------------------------------------------------
    def read(self):
        m = _lufs(sum(self._sub[-4:]) / 4.0) if len(self._sub) >= 4 else None
        s = _lufs(sum(self._sub) / 30.0) if len(self._sub) >= 30 else None
        tp_now = max(self._tp_recent + [self._tp_slot]) if self._tp else 0.0
        return {
            "momentary": m,
            "shortterm": s,
            "integrated": self._integrated,
            "lra": self._lra,
            "truepeak": (20.0 * np.log10(tp_now) if tp_now > 0.0 else None) if self._tp else None,
            "truepeak_max": (20.0 * np.log10(self._tp_max) if self._tp_max > 0.0 else None) if self._tp else None,
        }
