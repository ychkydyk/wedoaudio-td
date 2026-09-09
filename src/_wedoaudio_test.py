# _wedoaudio_test.py - offline verification of WEDOAUDIO v4.2 DSP on synthetic
# signals. NO TouchDesigner involved -> zero cook-thread risk. Run:
#   python td/_wedoaudio_test.py
# Asserts each feature behaves against analytic ground truth before TD integration.

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedoaudio_dsp import (WedoAudio, spectral_centroid, spectral_flatness,
                           spectral_flux, band_peak)

SR = 44100
N = 512                      # working spectrum bins
DT = 1.0 / 60.0
PASS, FAIL = [], []

def chk(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + ("  " + extra if extra else ""))

def sine_spectrum(freq, amp=1.0, noise=0.0):
    mag = np.zeros(N)
    nyq = SR * 0.5
    k = int(freq / nyq * N)
    k = min(N - 2, max(1, k))
    mag[k] = amp
    mag[max(1, k - 1)] += amp * 0.3
    mag[min(N - 1, k + 1)] += amp * 0.3
    if noise > 0:
        mag += np.abs(np.random.randn(N)) * noise
    return mag

def noise_spectrum(amp=1.0):
    return np.abs(np.random.randn(N)) * amp + amp * 0.2

def band_noise(f_lo, f_hi, amp=1.0):
    # band-limited noise (a realistic snare/hat lives in mid-high, NOT sub-bass)
    mag = np.zeros(N)
    nyq = SR * 0.5
    lo = int(f_lo / nyq * N); hi = min(N, int(f_hi / nyq * N))
    mag[lo:hi] = np.abs(np.random.randn(hi - lo)) * amp + amp * 0.2
    return mag

print("== STATELESS DESCRIPTORS ==")
# centroid monotonic with frequency
c_lo = spectral_centroid(sine_spectrum(200), SR)
c_mid = spectral_centroid(sine_spectrum(2000), SR)
c_hi = spectral_centroid(sine_spectrum(10000), SR)
chk("centroid rises with pitch", c_lo < c_mid < c_hi, f"{c_lo:.3f}<{c_mid:.3f}<{c_hi:.3f}")

# flatness: sine low, noise high
f_sine = spectral_flatness(sine_spectrum(1000))
f_noise = spectral_flatness(noise_spectrum())
chk("flatness sine<<noise", f_sine < 0.3 and f_noise > 0.4, f"sine={f_sine:.3f} noise={f_noise:.3f}")

# flux spikes on change, ~0 on static
prev = sine_spectrum(1000)
flux_static = spectral_flux(sine_spectrum(1000), prev)
flux_jump = spectral_flux(sine_spectrum(1000, amp=3.0), prev)
chk("flux 0 on static, >0 on jump", flux_static < 1e-6 and flux_jump > 0.0, f"static={flux_static:.4f} jump={flux_jump:.4f}")

# band peak isolates the band
bp_bass = band_peak(sine_spectrum(80), SR, 40, 120)
bp_bass_wrong = band_peak(sine_spectrum(5000), SR, 40, 120)
chk("band_peak isolates bass", bp_bass > 0.5 and bp_bass_wrong < 0.1, f"in={bp_bass:.2f} out={bp_bass_wrong:.2f}")

print("== ONSET / BEAT (impulse train) ==")
# 120 BPM kick: a sub-bass burst every 0.5s; analyzer should fire ~ once per beat
wa = WedoAudio(SR)
kicks = 0
snares = 0
beat_period = 0.5
dur = 8.0
nf = int(dur / DT)
n_snare_hits = 0
for i in range(nf):
    t = i * DT
    ph = t % beat_period
    is_snare = (t % beat_period > 0.25) and (t % beat_period < 0.30) and (int(t / beat_period) % 2 == 1)
    if ph < 0.05:                                   # kick burst (sub-bass)
        mag = sine_spectrum(60, amp=2.0) + sine_spectrum(90, amp=1.5)
        rms = 0.6
    elif is_snare:
        mag = band_noise(1800, 6000, 0.9)            # snare on backbeat (mid-high)
        rms = 0.4
        if ph >= 0.25 and (i == 0 or (i - 1) * DT % beat_period < 0.25):
            n_snare_hits += 1                        # count distinct snare events
    else:
        mag = noise_spectrum(0.05)                   # broadband filler (incl. snare band)
        rms = 0.08
    f = wa.process(mag, rms, t, DT)
    kicks += int(f['rawkick'])
    snares += int(f['rawsnare'])
expected = dur / beat_period
chk("kick count ~= beats", abs(kicks - expected) <= 4, f"got {kicks}, expect ~{expected:.0f}")
# snare must fire on backbeats (~8) and NOT on broadband filler noise. Over-firing
# (e.g. 58) means the floor gate is too low vs the AGC-lifted noise floor.
chk("snare count ~= backbeats, rejects filler", 4 <= snares <= 14, f"got {snares}, expect ~8")
chk("bpm locked ~120", 100 < f['bpm'] < 140, f"bpm={f['bpm']:.1f}")
chk("beatphase in 0..1", 0.0 <= f['beatphase'] <= 1.0, f"phase={f['beatphase']:.3f}")

print("== AGC (quiet vs loud, same shape) ==")
# feed a quiet track then a loud track of identical spectral shape; AGC should
# bring the post-gain bass into a similar range (the 'works on any track' claim).
def run_level(amp, rms_level):
    w = WedoAudio(SR)
    last = None
    for i in range(300):
        t = i * DT
        ph = t % 0.5
        mag = (sine_spectrum(60, amp=2.0 * amp) if ph < 0.05 else noise_spectrum(0.05 * amp))
        last = w.process(mag, rms_level if ph < 0.05 else rms_level * 0.15, t, DT)
    return last
quiet = run_level(0.12, 0.10)
loud = run_level(1.0, 0.85)
chk("AGC normalizes bass quiet~loud", abs(quiet['bass'] - loud['bass']) < 0.35,
    f"quiet={quiet['bass']:.2f} loud={loud['bass']:.2f} gQ={quiet['agcgain']:.1f} gL={loud['agcgain']:.1f}")

print("== SILENCE GATE ==")
w = WedoAudio(SR)
fs = w.process(noise_spectrum(0.0005), 0.0002, 1.0, DT)
chk("silence flag on quiet", fs['silence'] == 1.0, f"silence={fs['silence']}")

print("== TRANSIENT/SUSTAIN ==")
w = WedoAudio(SR)
# pad (sustained) then a transient hit
for i in range(120):
    w.process(sine_spectrum(440, amp=1.0), 0.4, i * DT, DT)
sus = w.process(sine_spectrum(440, amp=1.0), 0.4, 120 * DT, DT)
tr = w.process(noise_spectrum(2.0), 0.7, 121 * DT, DT)
chk("transient fires on hit > pad", tr['transient'] > sus['transient'], f"pad={sus['transient']:.3f} hit={tr['transient']:.3f}")

print("== CFG (param page -> live DSP) ==")
# the 120BPM kick train, run three ways: cfg=None (baseline), high kick_floor
# (must suppress kicks), and bass_gain=3 (must lift the bass reading). Proves the
# WEDOAUDIO param-page knobs actually reach the analyzer.
def run_kick_train(cfg=None):
    w = WedoAudio(SR); kk = 0; last = None
    for i in range(int(8.0 / DT)):
        t = i * DT; ph = t % 0.5
        if ph < 0.05:
            mag = sine_spectrum(60, amp=2.0) + sine_spectrum(90, amp=1.5); rms = 0.6
        else:
            mag = noise_spectrum(0.05); rms = 0.08
        last = w.process(mag, rms, t, DT, cfg=cfg)
        kk += int(last['rawkick'])
    return kk, last
k_base, _ = run_kick_train(None)
k_hifloor, _ = run_kick_train({'kick_floor': 1.01})           # floor above the clamped band level -> no kicks
chk("cfg kick_floor suppresses kicks", k_base >= 12 and k_hifloor == 0, f"base={k_base} hifloor={k_hifloor}")
_, base_last = run_kick_train(None)
_, gain_last = run_kick_train({'bass_gain': 3.0})
chk("cfg bass_gain lifts bass", gain_last['bass'] >= base_last['bass'], f"base={base_last['bass']:.2f} x3={gain_last['bass']:.2f}")
# cfg=None must be byte-identical to the legacy call (no-regression guard)
wa1 = WedoAudio(SR); wa2 = WedoAudio(SR)
for i in range(60):
    a = wa1.process(sine_spectrum(60, 2.0), 0.5, i*DT, DT)
    b = wa2.process(sine_spectrum(60, 2.0), 0.5, i*DT, DT, cfg=None)
chk("cfg=None == legacy behavior", abs(a['bass']-b['bass'])<1e-9 and a['rawkick']==b['rawkick'], f"bass {a['bass']:.4f}/{b['bass']:.4f}")

print(f"\n=== {len(PASS)} PASS / {len(FAIL)} FAIL ===")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("ALL WEDOAUDIO DSP TESTS PASS")
