"""WEDOAUDIO false-kick bench — material with NO drums must produce no kicks.

The rhythm bench (_wedoaudio_bench.py) has an 'ambient-nobeat' style and it always passed. It is synthetic and too
smooth: real recordings of wind, rain, engines and sea exposed 2.0 invented kicks per second. This bench keeps that
failure visible.

Material:
  * eight synthesised drones, 30 s each, rendered here (nothing is downloaded, nothing is shipped):
    beating pairs inside the kick band, a sub drone with vibrato, a tone cluster, rumble with a slow envelope,
    gusty rumble like wind on a microphone, a swelling pad, a riser crossing the kick band;
  * optionally your own non-rhythmic recordings: set WEDOAUDIO_NONRHYTHMIC=<folder with wav files>.
    The numbers in CHANGELOG for field recordings were measured on 96 five-second clips of the ESC-50 dataset
    (CC BY-NC, so it is not part of this repository).

Run: python src/_wedoaudio_falsekick.py
Prints false kicks per second for the current detector and for kick_spread=0, which is the 4.3.0 detector exactly.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedoaudio_dsp import WedoAudio

FPS, FFT = 60, 2048
BASELINE_SYNTH = 0.25      # false kicks per second on the synthesised set; the run fails above this


def _lowpass(x, fc, sr):
    a = np.exp(-2 * np.pi * fc / sr); y = np.empty_like(x); s = 0.0
    for i, v in enumerate(x):
        s = a * s + (1 - a) * v; y[i] = s
    return y


def drones(sr=44100, seconds=30):
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(t))
    out = {
        'beat_55_57': .35 * (np.sin(2 * np.pi * 55 * t) + np.sin(2 * np.pi * 57.3 * t)),
        'beat_82_83': .35 * (np.sin(2 * np.pi * 82 * t) + np.sin(2 * np.pi * 83.1 * t)),
        'sub_vibrato': .5 * np.sin(2 * np.pi * (48 * t + 2.5 * np.sin(2 * np.pi * .3 * t) / (2 * np.pi * .3))),
        'cluster': .2 * sum(np.sin(2 * np.pi * f * t + p) for f, p in zip((61, 73.4, 92.5, 110, 146.8), rng.random(5) * 6.28)),
        'rumble_lfo': .6 * _lowpass(_lowpass(noise, 90, sr), 90, sr) * 6 * (.55 + .45 * np.sin(2 * np.pi * .4 * t)),
        'rumble_gusts': .6 * _lowpass(_lowpass(noise, 120, sr), 120, sr) * 6
                        * np.clip(_lowpass(np.abs(rng.standard_normal(len(t))), .8, sr) * 3, 0, 1.5),
        'pad_swell': .25 * sum(np.sin(2 * np.pi * f * t) for f in (65.4, 98, 130.8, 196)) * (.5 - .5 * np.cos(2 * np.pi * t / 10)),
        'riser': .4 * np.sin(2 * np.pi * (40 * t + 60 * t * t / (2 * seconds))) * (t / seconds),
    }
    return [(k, np.clip(v, -1, 1), float(sr)) for k, v in out.items()]


def false_kicks(x, sr, cfg=None):
    hop = int(sr / FPS); win = np.hanning(FFT); wa = WedoAudio(sr=sr); n = 0
    for s in range(0, max(0, len(x) - FFT), hop):
        fr = x[s:s + FFT] * win
        mag = np.abs(np.fft.rfft(fr)) / (FFT / 4)
        n += int(wa.process(mag, float(np.sqrt(np.mean(fr ** 2))), s / sr, 1.0 / FPS, cfg)['rawkick'])
    return n


def rate(items, cfg=None):
    return sum(false_kicks(x, sr, cfg) for _, x, sr in items) / sum(len(x) / sr for _, x, sr in items)


def main():
    synth = drones()
    now, was = rate(synth), rate(synth, {'kick_spread': 0.0})
    print('synthesised drones, %d files, %.0f s' % (len(synth), sum(len(x) / sr for _, x, sr in synth)))
    print('  4.3.0 detector (kick_spread=0): %.2f false kicks per second' % was)
    print('  current detector              : %.2f false kicks per second' % now)
    folder = os.environ.get('WEDOAUDIO_NONRHYTHMIC', '')
    if folder and os.path.isdir(folder):
        import soundfile as sf
        own = []
        for f in sorted(glob.glob(os.path.join(folder, '*.wav'))):
            x, sr = sf.read(f, dtype='float64', always_2d=True); own.append((os.path.basename(f), x.mean(axis=1), float(sr)))
        if own:
            print('your recordings, %d files, %.0f s' % (len(own), sum(len(x) / sr for _, x, sr in own)))
            print('  4.3.0 detector (kick_spread=0): %.2f' % rate(own, {'kick_spread': 0.0}))
            print('  current detector              : %.2f' % rate(own))
    ok = now <= BASELINE_SYNTH
    print('false-kick bench %s the recorded baseline (%.2f)' % ('at or below' if ok else 'ABOVE', BASELINE_SYNTH))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
