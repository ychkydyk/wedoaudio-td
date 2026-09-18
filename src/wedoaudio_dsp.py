# wedoaudio_dsp.py - WEDOAUDIO v4.2 core DSP (pure functions + a stateful analyzer).
#
# Designed to run in TWO places with identical math:
#   1) OFFLINE tests (_wedoaudio_test.py) on synthetic spectra -> verify behavior
#      against analytic ground truth BEFORE any TD integration (no cook-thread risk).
#   2) Inside a lightweight TD Script CHOP (imports this module) once verified.
#
# Input contract: a magnitude spectrum array `mag` (linear FFT bins, index i ->
# frequency i*(sr/2)/len(mag) Hz), the current rms scalar, time t (s), dt (s).
# Everything is numpy; TD 2023 ships numpy. Per-frame cost is O(nbins) with small
# constant work, so it stays inside the 60fps cook budget when nbins<=512.

import numpy as np

# ----- stateless spectral descriptors -------------------------------------------

def bin_range(n, sr, f_lo, f_hi):
    nyq = sr * 0.5
    lo = max(0, int(f_lo / nyq * n))
    hi = min(n, max(lo + 1, int(f_hi / nyq * n)))
    return lo, hi

def band_peak(mag, sr, f_lo, f_hi, gain=1.0):
    lo, hi = bin_range(len(mag), sr, f_lo, f_hi)
    if hi <= lo:
        return 0.0
    return float(mag[lo:hi].max()) * gain

def band_energy(mag, sr, f_lo, f_hi):
    lo, hi = bin_range(len(mag), sr, f_lo, f_hi)
    if hi <= lo:
        return 0.0
    return float(mag[lo:hi].sum())

def spectral_centroid(mag, sr):
    # energy-weighted mean frequency, normalized 0..1 over nyquist
    s = mag.sum()
    if s <= 1e-9:
        return 0.0
    idx = np.arange(len(mag))
    c = float((idx * mag).sum() / s) / max(1, len(mag) - 1)
    return min(1.0, max(0.0, c))

def spectral_flatness(mag):
    # geometric mean / arithmetic mean (Wiener entropy). 0=tonal, 1=noise.
    m = mag[mag > 1e-9]
    if m.size < 4:
        return 0.0
    gm = np.exp(np.log(m).mean())
    am = m.mean()
    return float(min(1.0, max(0.0, gm / (am + 1e-12))))

def spectral_rolloff(mag, sr, pct=0.85):
    tot = mag.sum()
    if tot <= 1e-9:
        return 0.0
    cum = np.cumsum(mag)
    k = int(np.searchsorted(cum, pct * tot))
    return min(1.0, k / max(1, len(mag) - 1))

def spectral_flux(mag, prev):
    # rectified spectral flux: sum of positive bin-to-bin increases, normalized.
    if prev is None or prev.shape != mag.shape:
        return 0.0
    d = mag - prev
    d[d < 0] = 0.0
    return float(d.sum() / max(1, len(mag)))

def mel_bank(n_bins, sr, n_mels=40, fmin=30.0, _cache={}):
    """Area-normalised triangular mel bank, built once per (bins, rate)."""
    key = (n_bins, sr, n_mels, fmin)
    fb = _cache.get(key)
    if fb is not None:
        return fb
    fmax = min(8000.0, sr / 2.0)
    def to_mel(f): return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)
    def to_hz(m): return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    edges = to_hz(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(mid - lo, 1e-9)
        right = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
        ssum = fb[i].sum()
        if ssum > 0:
            fb[i] /= ssum
    _cache[key] = fb
    return fb


def tempo_autocorr(novelty_hist, dt, lo=68.0, hi=200.0, centre=120.0, spread=1.0):
    """Tempo straight from the novelty curve: normalised autocorrelation over every lag
    in the supported range, weighted by a log-normal preference around 120 BPM. This is
    the standard construction (Ellis-style onset-strength autocorrelation with a tempo
    prior), and it is the primary estimator because the inter-onset median proved
    fragile on real material.

    The supported range is declared, not implied: 68-200 BPM. A track slower than that
    is reported at double time, which is also how such music is usually counted."""
    x = np.asarray(novelty_hist, dtype=float)
    if x.size < 64:
        return 0.0
    x = x - x.mean()
    var = float(np.dot(x, x)) / x.size
    if var <= 1e-12:
        return 0.0
    lag_lo = max(2, int(round(60.0 / hi / max(dt, 1e-6))))
    lag_hi = min(x.size - 8, int(round(60.0 / lo / max(dt, 1e-6))))
    if lag_hi <= lag_lo:
        return 0.0
    def ac(l):
        if l < 2 or l >= x.size - 8:
            return 0.0
        return float(np.dot(x[:-l], x[l:])) / ((x.size - l) * var)

    best, best_lag = -1e9, 0
    for lag in range(lag_lo, lag_hi + 1):
        r = ac(lag) + 0.5 * ac(2 * lag) + 0.25 * ac(3 * lag)   # harmonic reinforcement
        cand = 60.0 / (lag * dt)
        r *= float(np.exp(-0.5 * (np.log2(cand / centre) / spread) ** 2))
        if r > best:
            best, best_lag = r, lag
    return 60.0 / (best_lag * dt) if best_lag else 0.0


def tempo_multiple_fix(bpm, novelty_hist, dt, lo=70.0, hi=180.0,
                       mults=(1.0 / 3.0, 0.5, 1.0, 2.0, 3.0)):
    """Pick the metrical level. The IOI median measures the ANCHOR's period, which is
    not the beat: a breakbeat kick pattern repeats every two beats and reads 87 for a
    174 BPM track, a half-time kick reads 35 for 70. Each candidate multiple is scored
    by the normalised autocorrelation of the recent novelty curve at that lag, and a
    candidate inside the musical range gets a 25% prior.

    Autocorrelation, not a comb of samples: a comb takes fewer samples as the period
    grows, so a lucky landing on two peaks beats an honest match at the true period.
    Measured — the comb walked a correct 128 BPM down to 46. Triple metre needs 3 and
    1/3 as well as the octave, because a waltz kick on beat one is a 3:1 relation and
    no power of two reaches it."""
    x = np.asarray(novelty_hist, dtype=float)
    if x.size < 32 or bpm <= 0:
        return (bpm, 1.0)
    x = x - x.mean()
    var = float(np.dot(x, x)) / x.size
    if var <= 1e-12:
        return (bpm, 1.0)
    # RESCUE, NOT RE-DECISION. If the anchor period is already musically plausible it
    # is left alone, and the multiples are only searched when it is not. The reason is
    # measured: on a track with a kick every beat and a snare every other beat, the
    # novelty curve is honestly periodic at TWO beats (autocorrelation 1.00 there
    # against 0.12 at the beat), so any scoring rule that is free to overrule a
    # correct 120 BPM will report 60. No prior bridges an eight-fold gap. The anchor
    # is trusted when it is believable, and second-guessed only when it cannot be
    # right — which is exactly the half-time and waltz case this was written for.
    if lo <= bpm <= hi:
        return (bpm, 1.0)
    cands = sorted(((bpm * m, m) for m in mults), key=lambda cm: -cm[0])
    best, best_mult = -1e9, 1.0
    for cand, mult in cands:
        if cand < 40.0 or cand > 240.0:
            continue
        lag = int(round(60.0 / cand / max(dt, 1e-6)))
        if lag < 2 or lag >= x.size - 8:
            continue
        r = float(np.dot(x[:-lag], x[lag:])) / ((x.size - lag) * var)
        if lo <= cand <= hi:
            r *= 1.25
        # A log-normal preference around 120 BPM was tried here instead of this flat
        # bonus. It fixed the one remaining octave error (a brickwalled master, whose
        # kick+snare pattern genuinely repeats every two beats) and broke two others:
        # the waltz went 148 -> 67 and the half-time 77 -> 64. Trading one failure for
        # two is where tuning stops and the limitation gets written down instead.
        if r > best * (1.10 if best > 0 else 1.0) or best <= -1e8:
            best, best_mult = r, mult
    return (bpm * best_mult, best_mult)


# ----- onset detector: adaptive median+delta peak-pick with refractory ----------

class OnsetState:
    def __init__(self, hist=43, k=1.6, delta=0.02, refractory_s=0.11):
        self.hist = hist
        self.k = k
        self.delta = delta
        self.refr = refractory_s
        self.buf = []
        self.last_t = -9.0
        self.env = 0.0

    def step(self, novelty, t, dt, level=1.0, floor=0.0, decay_s=0.12):
        # level/floor: absolute gate so the adaptive threshold can't fire on an
        # AGC-amplified noise floor (the proven web 'kickFloor' guard). The onset
        # needs BOTH a novelty spike AND the band to actually be loud enough.
        self.buf.append(novelty)
        if len(self.buf) > self.hist:
            self.buf.pop(0)
        med = float(np.median(self.buf)) if self.buf else 0.0
        thr = med * self.k + self.delta
        raw = 0
        if novelty > thr and level > floor and (t - self.last_t) > self.refr:
            raw = 1
            self.last_t = t
        dec = np.exp(-dt / max(0.02, decay_s))
        self.env = max(self.env * dec, float(raw))
        return raw, self.env, thr

# ----- AGC: rolling-percentile auto gain (fast attack, slow release) ------------

class AGC:
    def __init__(self, win=240, pct=95, target=0.7, attack=0.5, release=0.03, maxgain=12.0):
        self.win = win
        self.pct = pct
        self.target = target
        self.attack = attack
        self.release = release
        self.maxgain = maxgain
        self.buf = []
        self.gain = 1.0

    def step(self, level):
        self.buf.append(level)
        if len(self.buf) > self.win:
            self.buf.pop(0)
        ref = float(np.percentile(self.buf, self.pct)) if len(self.buf) > 8 else level
        want = self.target / max(1e-3, ref)
        want = min(want, self.maxgain)
        a = self.attack if want < self.gain else self.release  # fast DOWN, slow UP
        self.gain += (want - self.gain) * a
        return self.gain

# ----- the analyzer: holds all state, returns the full feature bus --------------

class WedoAudio:
    def __init__(self, sr=44100):
        self.sr = sr
        self.prev = None
        # 4.4 defaults, chosen by sweeping the whole benchmark rather than by ear.
        # They belong to the relative-onset path; cfg onset_rel=0 restores the 4.3
        # numbers below in process(), so the old behaviour is one flag away.
        self.kick = OnsetState(refractory_s=0.20, k=1.5, delta=0.40)
        self.snare = OnsetState(refractory_s=0.09, k=1.6, delta=0.50)
        self.beat = OnsetState(refractory_s=0.10, k=1.7, delta=0.40)
        self.agc = AGC()
        self.ioi = []
        self.last_kick_t = -9.0
        self.last_tempo_t = -9.0
        self.bpm = 120.0
        self.bpm_raw = 120.0    # anchor period straight from the IOI median
        self.phase = 0.0
        self.sustain = 0.0
        self.silent = 0
        # 4.4 relative-onset state: slow follower + decaying peak per onset band
        self.kick_ref = 0.0
        self.snare_ref = 0.0
        self.kick_peak = 1e-6
        self.snare_peak = 1e-6
        self.nov_hist = []      # recent broadband novelty, for the autocorrelation tempo
        self.bpm_ac = 0.0       # last autocorrelation estimate
        self.ac_countdown = 0
        self.mel_fb = None      # mel bank for the tempo novelty, built on first frame
        self.prev_mel = None
        self.flux_ref = 0.0     # slow follower of broadband flux, for the beat detector

    def process(self, mag, rms, t, dt, cfg=None):
        # CONTRACT FOR `mag`. It must be a LINEAR magnitude spectrum: bin i covers
        # frequency i * (sr/2) / len(mag). Roughly 1025 bins (a 2048-point FFT),
        # normalised so a full-scale sine reads about 1.0. Every absolute guard in
        # here was measured on that geometry.
        # This is not pedantry. TouchDesigner's Audio Spectrum CHOP ships configured
        # FOR DISPLAY -- logarithmic frequency axis plus a high-frequency boost. Fed
        # straight in, the 40-120 Hz kick band read 0.0025 against a 0.05 guard: the
        # kick detector was dead, with no error anywhere, while the offline bench
        # kept reporting 18/18. A spectrum on the wrong axis is not a smaller signal,
        # it is a different measurement.
        #
        # cfg: optional dict of live tunables (from the WEDOAUDIO param page). Any
        # missing key / cfg=None -> the exact verified default behavior, so offline
        # tests and the DSP contract are unchanged unless a knob is explicitly set.
        c = cfg or {}
        gate = c.get('gate', 0.001)
        mag = np.asarray(mag, dtype=float)
        if mag.ndim != 1 or not np.isfinite(mag).all() or (mag < 0).any():
            raise ValueError("mag must be a finite, non-negative 1D magnitude spectrum")
        if not all(np.isfinite(v) for v in (rms, t, dt, self.sr)) or rms < 0 or dt <= 0 or self.sr <= 0:
            raise ValueError("rms/time/dt/sample rate must be finite; rms >= 0, dt and rate > 0")
        sr = self.sr
        self.silent = 1 if rms < gate else 0

        # AGC reference from broadband energy; target/maxgain tunable; agc_on=0 -> unity
        if 'agc_target' in c: self.agc.target = c['agc_target']
        if 'agc_max' in c:    self.agc.maxgain = c['agc_max']
        g = self.agc.step(max(rms, band_energy(mag, sr, 20, 20000) / max(1, len(mag))))
        if not c.get('agc_on', 1):
            g = 1.0

        # band crossovers + per-band makeup gains (defaults = current edges / unity)
        b_hi = c.get('bass_hi', 250); m_hi = c.get('mid_hi', 2000); h_hi = c.get('high_hi', 20000)
        bass = min(1.0, band_peak(mag, sr, 20,   b_hi) * g * c.get('bass_gain', 1.0))
        mid  = min(1.0, band_peak(mag, sr, b_hi, m_hi) * g * c.get('mid_gain', 1.0))
        high = min(1.0, band_peak(mag, sr, m_hi, h_hi) * g * c.get('high_gain', 1.0))
        centroid = spectral_centroid(mag, sr)
        flatness = spectral_flatness(mag)
        rolloff = spectral_rolloff(mag, sr)
        flux = spectral_flux(mag, self.prev) * g

        # per-band novelty for kick/snare via band flux
        # Band edges are tunable. A present-but-zero key must fall back, not pass
        # through: the rig shipped these as parameters whose default was 0.0, so
        # "reset to defaults" narrowed the kick band to 0-0 Hz and the detector
        # went dead with no error anywhere. c.get(k, 40) does not save you when
        # the key IS there and holds 0 -- hence `or`, not a get-default.
        k_lo = c.get('kick_lo') or 40.0
        k_hi = c.get('kick_hi') or 120.0
        s_lo = c.get('snare_lo') or 1800.0
        s_hi = c.get('snare_hi') or 6000.0
        kick_band = band_peak(mag, sr, k_lo, k_hi) * g
        snare_band = band_peak(mag, sr, s_lo, s_hi) * g
        kf_abs = 0.0 if self.prev is None else max(0.0, kick_band - (band_peak(self.prev, sr, k_lo, k_hi) * g))
        sf_abs = 0.0 if self.prev is None else max(0.0, snare_band - (band_peak(self.prev, sr, s_lo, s_hi) * g))

        # --- 4.4 relative onsets (cfg 'onset_rel'; set 0 to restore 4.3 exactly) ---
        # Two defects found by the benchmark, both from measuring in absolute units:
        #  1. Absolute band flux fires whenever a band is merely LOUD. A sustained
        #     bass note inside 40-120 Hz gave 43 kick events for 17 real kicks
        #     (precision 0.37), and moving 44.1 -> 48 kHz shifted the 41 Hz
        #     fundamental across the band edge, changing the count 19 -> 47.
        #     Dividing the rise by a slow follower of the same band asks "did this
        #     band jump relative to itself", which a held note does not do.
        #  2. The level gate compared a peak-BIN measure to a fixed fraction of full
        #     scale. A kick concentrates its energy in two bins and reads 0.64; a
        #     snare spreads the same energy over ~180 bins and reads 0.04. So the
        #     0.35 snare floor was unreachable by any real snare — measured across
        #     two styles, snare hits peaked at 0.08-0.16. The floor now means
        #     "this fraction of the band's own recent peak", which is scale-free.
        af = min(1.0, dt / 0.5)
        self.kick_ref += (kick_band - self.kick_ref) * af
        self.snare_ref += (snare_band - self.snare_ref) * af
        pd = np.exp(-dt / 2.0)
        self.kick_peak = max(self.kick_peak * pd, kick_band)
        self.snare_peak = max(self.snare_peak * pd, snare_band)
        rel = bool(c.get('onset_rel', 1))
        # The broadband beat detector had the same disease as the other two, in its
        # worst form: spectral_flux divides by the number of bins, so it lives around
        # 0.004-0.03, while the detector's own delta floor was 0.02. The threshold was
        # therefore almost never crossed and rawbeat was effectively dead - measured
        # zero beats on a clean click train, zero on a four-on-floor render, and two
        # in thirty seconds of real drum and bass. It also silently disabled the tempo
        # fallback for kickless material, which is anchored on rawbeat.
        self.flux_ref += (flux - self.flux_ref) * min(1.0, dt / 0.5)
        bflux = (flux / max(1e-4, self.flux_ref)) if rel else flux
        if rel:
            kf = kf_abs / max(0.05, self.kick_ref)
            sf = sf_abs / max(0.01, self.snare_ref)
            # Both gates are needed, and each fails alone. The absolute gate alone made
            # the snare unreachable (defect 2). The relative gate alone cannot tell a
            # band holding only hi-hats from a band holding snares, because it
            # normalises to whatever is present: on a hat-only waltz every hat scored
            # 100% of the band's recent peak and fired as a snare. So: loud for this
            # band right now, AND above a small absolute guard.
            live = 0.0 if self.silent else 1.0
            k_ok = 1.0 if kick_band > c.get('kick_abs', 0.05) else 0.0
            s_ok = 1.0 if snare_band > c.get('snare_abs', 0.055) else 0.0
            kick_level = kick_band / max(self.kick_peak, 1e-6) * live * k_ok
            snare_level = snare_band / max(self.snare_peak, 1e-6) * live * s_ok
        else:
            kf, sf = kf_abs, sf_abs
            kick_level = min(1.0, kick_band)
            snare_level = min(1.0, snare_band)

        # onset internals (k / refractory) tunable; floors default to the verified
        # values (kick 0.22 / snare 0.35 / beat 0.10 — snare gate rejects AGC-lifted
        # hats). decay = onset envelope release (LED fade time).
        if rel:
            # Re-apply the selected mode before explicit UI overrides. Otherwise
            # switching legacy -> relative leaves the legacy thresholds latched.
            self.kick.k, self.kick.delta, self.kick.refr = 1.5, 0.40, 0.20
            self.snare.k, self.snare.delta, self.snare.refr = 1.6, 0.50, 0.09
            self.beat.k, self.beat.delta, self.beat.refr = 1.7, 0.40, 0.10
        else:
            # exact 4.3 detector constants, so onset_rel=0 is a true rollback
            self.kick.k, self.kick.delta, self.kick.refr = 1.5, 0.015, 0.12
            self.snare.k, self.snare.delta, self.snare.refr = 1.6, 0.02, 0.09
            self.beat.k, self.beat.delta, self.beat.refr = 1.7, 0.02, 0.10
        if 'kick_k'   in c: self.kick.k = c['kick_k']
        if 'kick_refr' in c: self.kick.refr = c['kick_refr']
        if 'snare_k'  in c: self.snare.k = c['snare_k']
        if 'snare_refr' in c: self.snare.refr = c['snare_refr']
        if 'beat_k'   in c: self.beat.k = c['beat_k']
        if 'beat_refr' in c: self.beat.refr = c['beat_refr']
        dec = c.get('decay', 0.12)
        k_floor = c.get('kick_floor', 0.40 if rel else 0.22)
        s_floor = c.get('snare_floor', 0.35)
        rk, kick_env, _ = self.kick.step(kf, t, dt, level=kick_level, floor=k_floor, decay_s=dec)
        rs, snare_env, _ = self.snare.step(sf, t, dt, level=snare_level, floor=s_floor, decay_s=dec)
        rb, beat_env, _ = self.beat.step(bflux, t, dt, level=(0.0 if rel and self.silent else min(1.0, rms * g)), floor=c.get('beat_floor', 0.10), decay_s=dec)

        # tempo via inter-onset intervals from a SINGLE anchor: kick preferred
        # (most reliable in 4-on-floor); fall back to broadband beat only after
        # ~2s with no kick (kickless ambient). Mixing kick+snare onsets corrupts
        # the IOI median, so only one source drives tempo at a time.
        if rk:
            self.last_kick_t = t
        anchor = rk or (rb and (t - self.last_kick_t) > 2.0)
        if anchor:
            if self.last_tempo_t > 0:
                iv = t - self.last_tempo_t
                if 0.20 < iv < 2.60:      # 4.4: was 0.25-1.5, which could not represent
                                          # a slow anchor at all — a kick every two
                                          # beats at 70 BPM is 1.71 s and was dropped,
                                          # leaving bpm frozen at its initial 120.
                    self.ioi.append(iv)
                    if len(self.ioi) > 8:
                        self.ioi.pop(0)
                    med = float(np.median(self.ioi))
                    self.bpm_raw = self.bpm_raw * 0.7 + (60.0 / med) * 0.3
            self.last_tempo_t = t

        # 4.4: resolve the metrical level. The IOI median measures the ANCHOR's period,
        # which is not the beat: a breakbeat kick pattern repeats every two beats and
        # reads 87 instead of 174, a half-time kick every two beats reads 35 instead of
        # 70. Score the candidates against the novelty curve and prefer the musical
        # range. Triple metre needs 3 and 1/3 as well as the octave, because a waltz
        # kick on beat one is a 3:1 relation and no power of two reaches it.
        # A log-compressed mel difference was tried here as the tempo novelty, the way
        # reference onset-strength envelopes are built. On an eight-loop probe it looked
        # like a clear win (3/8 against 0/8 for the linear flux). On the full 65-loop
        # library it changed nothing (18/65 either way) while costing the benchmark
        # 12/15 -> 9/15 and real-music agreement 4/4 -> 3/4. A probe is not a measurement.
        self.nov_hist.append(flux)
        if len(self.nov_hist) > 480:            # 8 s of history at 60 fps
            self.nov_hist.pop(0)
        # The multiplier is applied to the RAW anchor period every frame, never fed
        # back into itself: an earlier version wrote the corrected value back and the
        # multipliers compounded, walking a correct 128 BPM down to 112.
        # 4.4b: tempo from the AUTOCORRELATION of the novelty curve, with the inter-onset
        # median kept as the fallback. Measured reason: the IOI median is fragile on real
        # music — it agreed with librosa on one track out of four, and merely re-encoding
        # the same file to 48 kHz moved it by 5.9% while 95% of the kick onsets stayed
        # identical. Autocorrelation looks at the whole curve instead of the spacing
        # between a handful of picked events, which is what every reference beat tracker
        # does and why they are stable.
        self.ac_countdown -= 1
        if c.get('tempo_ac', 1) and len(self.nov_hist) >= 240 and self.ac_countdown <= 0:
            self.ac_countdown = 15              # four times a second is plenty
            est = tempo_autocorr(self.nov_hist, dt)
            if est > 0:
                self.bpm_ac = est if self.bpm_ac <= 0 else self.bpm_ac * 0.6 + est * 0.4
        if c.get('tempo_ac', 1) and self.bpm_ac > 0:
            self.bpm = self.bpm_ac
        elif c.get('tempo_fix', 1) and len(self.nov_hist) >= 60:
            self.bpm, _ = tempo_multiple_fix(self.bpm_raw, self.nov_hist, dt)
        else:
            self.bpm = self.bpm_raw
        # continuous beat-grid phase (movement between onsets)
        period = 60.0 / max(40.0, min(220.0, self.bpm))
        self.phase = (self.phase + dt / period) % 1.0

        # transient / sustain (HPSS-lite): slow follower = sustain, residual = transient
        e = band_energy(mag, sr, 20, 20000) / max(1, len(mag)) * g
        self.sustain += (e - self.sustain) * min(1.0, dt / 0.25)
        transient = max(0.0, e - self.sustain)

        self.prev = mag.copy()
        return {
            'rms': min(1.0, rms * g), 'bass': bass, 'mid': mid, 'high': high,
            'centroid': centroid, 'flatness': flatness, 'rolloff': rolloff, 'flux': min(1.0, flux),
            'kick': kick_env, 'rawkick': float(rk), 'kickband': min(1.0, kick_band),
            'snare': snare_env, 'rawsnare': float(rs), 'snareband': min(1.0, snare_band),
            'beat': beat_env, 'rawbeat': float(rb),
            'bpm': self.bpm, 'beatphase': self.phase,
            'transient': min(1.0, transient * 3.0), 'sustain': min(1.0, self.sustain * 3.0),
            'agcgain': g, 'silence': float(self.silent),
        }
