"""Tune the onset thresholds by measurement over every style at once."""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wedoaudio_bench import render, events, f_measure, FFT, FPS
from wedoaudio_dsp import WedoAudio

STYLES = ["four-on-floor", "breakbeat-dnb", "halftime", "waltz", "ambient-nobeat", "clipped-loud"]
CACHE = {}


def audio(style, sr):
    if (style, sr) not in CACHE:
        CACHE[(style, sr)] = render(style, sr)
    return CACHE[(style, sr)]


def score(kk, kd, kr, kfl, sk, sd, sr_refr, sfl):
    kfs, sfs = [], []
    for style in STYLES:
        for sr in (44100, 48000):
            x, kt, st, _ = audio(style, sr)
            hop = int(sr / FPS); win = np.hanning(FFT)
            wa = WedoAudio(sr=sr)
            wa.kick.k, wa.kick.delta, wa.kick.refr = kk, kd, kr
            wa.snare.k, wa.snare.delta, wa.snare.refr = sk, sd, sr_refr
            cfg = {"kick_floor": kfl, "snare_floor": sfl}
            ts, rk, rs = [], [], []
            for s in range(0, len(x) - FFT, hop):
                fr = x[s:s + FFT] * win
                mag = np.abs(np.fft.rfft(fr)) / (FFT / 4)
                f = wa.process(mag, float(np.sqrt(np.mean(fr ** 2))), s / sr, 1 / FPS, cfg)
                ts.append(s / sr); rk.append(f["rawkick"]); rs.append(f["rawsnare"])
            kfs.append(f_measure(events(rk, ts), kt)[2])
            sfs.append(f_measure(events(rs, ts), st)[2])
    return float(np.mean(kfs)), float(np.mean(sfs))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "kick"
    best = None
    if which == "kick":
        grid = itertools.product([1.5, 2.5, 4.0, 6.0], [0.02, 0.15, 0.4], [0.12, 0.20, 0.30], [0.22, 0.40])
        for kk, kd, kr, kfl in grid:
            k, s = score(kk, kd, kr, kfl, 1.6, 0.02, 0.09, 0.35)
            print("kick k=%.1f d=%.2f refr=%.2f floor=%.2f -> F %.3f" % (kk, kd, kr, kfl, k))
            if best is None or k > best[0]: best = (k, (kk, kd, kr, kfl))
    else:
        grid = itertools.product([1.6, 3.0, 5.0], [0.02, 0.2, 0.5], [0.09, 0.15], [0.35, 0.55, 0.75])
        for sk, sd, srr, sfl in grid:
            k, s = score(4.0, 0.15, 0.20, 0.22, sk, sd, srr, sfl)
            print("snare k=%.1f d=%.2f refr=%.2f floor=%.2f -> F %.3f" % (sk, sd, srr, sfl, s))
            if best is None or s > best[0]: best = (s, (sk, sd, srr, sfl))
    print("BEST", best)
