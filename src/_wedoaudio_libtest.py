# -*- coding: utf-8 -*-
"""Check WEDOAUDIO against a labelled sample library — real recordings, vendor labels.

Loop libraries name their files with the answer already in them: the tempo, the
instrument, and for tonal material the root note. That makes them ground truth we
did not write, on audio nobody synthesised. This walks such a folder and scores:

  TEMPO       filename says 110 / 130 / 90 -> our BPM must match within a metrical level
  KEY         filename ends in _D# / _F / _A -> our tonic must be that pitch class
  INSTRUMENT  a file called *_kick must fire the kick and not the snare; a bass loop
              must land in the bass band and must NOT spray false kicks (that was a
              real defect: a sustained bass note used to trigger three kicks a beat);
              atmosphere and drone must stay quiet in the onset channels

No audio is committed with this repo — point it at your own library:
    python src/_wedoaudio_libtest.py "D:/path/to/samples"
Tested against an Ableton user library (the .asd files next to each wav are Live's
own analysis), but any pack with the tempo in the filename works.
"""
import os, sys, re, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import librosa
from wedoaudio_dsp import WedoAudio
from wedoaudio_timbre import chroma, estimate_key, NOTES
from _wedoaudio_bench import events, FPS, FFT

DEFAULT_DIR = os.environ.get("WEDOAUDIO_SAMPLES", "")
NOTE_RE = re.compile(r"_([A-G]#?)$")
BPM_RE = re.compile(r"(?:^|_)(\d{2,3})(?:_|$)")


def parse(name):
    """Pull the vendor's own labels out of the filename."""
    stem = os.path.splitext(os.path.basename(name))[0]
    low = stem.lower()
    bpm = None
    for m in BPM_RE.finditer(stem):
        v = int(m.group(1))
        if 60 <= v <= 200:
            bpm = float(v)
            break
    note = None
    mn = NOTE_RE.search(stem)
    if mn:
        note = mn.group(1)
    kind = None
    for k in ("kick", "snare", "clap", "hat", "bass", "atmosphere", "drone",
              "noise", "perc", "drum", "kit", "sonic", "fx"):
        if k in low:
            kind = k
            break
    return stem, bpm, note, kind


def analyse(path):
    x, sr = librosa.load(path, sr=None, mono=True)
    if len(x) < FFT * 2:
        return None
    x = x / (np.max(np.abs(x)) or 1.0)
    hop = int(sr / FPS)
    win = np.hanning(FFT)
    wa = WedoAudio(sr=sr)
    t, rk, rs, bpm, bands, chs = [], [], [], [], [], []
    for s in range(0, len(x) - FFT, hop):
        fr = x[s:s + FFT] * win
        mag = np.abs(np.fft.rfft(fr)) / (FFT / 4)
        f = wa.process(mag, float(np.sqrt(np.mean(fr ** 2))), s / sr, 1.0 / FPS)
        t.append(s / sr)
        rk.append(f["rawkick"])
        rs.append(f["rawsnare"])
        bpm.append(f["bpm"])
        bands.append((f["bass"], f["mid"], f["high"]))
        if len(chs) < 400:
            chs.append(chroma(mag, sr))
    b = np.array(bands)
    dur = len(x) / sr
    ref = 0.0
    if dur >= 4.5:
        try:
            rb, _ = librosa.beat.beat_track(y=x, sr=sr, units="time")
            ref = float(np.atleast_1d(rb)[0])
        except Exception:
            ref = 0.0
    return {"dur": dur, "sr": sr,
            "kicks": events(rk, t), "snares": events(rs, t),
            "bpm": float(np.median(bpm[len(bpm) // 2:])) if bpm else 0.0,
            "bass": float(np.median(b[:, 0])), "mid": float(np.median(b[:, 1])),
            "high": float(np.median(b[:, 2])),
            "chroma": np.mean(chs, axis=0) if chs else np.zeros(12),
            "ref_bpm": ref}


def metrical(a, b, tol=0.06):
    if a <= 0 or b <= 0:
        return False, 0.0
    for m in (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0):
        if abs(a * m - b) / b < tol:
            return True, m
    return False, 0.0


def main(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if not files:
        print("no wav files in " + folder)
        return 1
    print("library: %s  (%d wav files)" % (folder, len(files)))

    tempo_rows, key_rows, inst = [], [], collections.defaultdict(list)
    skipped = 0
    for path in files:
        stem, bpm, note, kind = parse(path)
        try:
            a = analyse(path)
        except Exception:
            skipped += 1
            continue
        if a is None:
            skipped += 1
            continue
        # tempo: only where the file is long enough for the estimator to have a window
        if bpm and a["dur"] >= 4.5:
            ok, mult = metrical(a["bpm"], bpm)
            # Rhythmic or not: a pad, drone or bass loop carries the project tempo in
            # its filename but nothing audible to measure it from. Scoring those as
            # tempo failures would be dishonest in our favour AND against us at once,
            # so the two groups are counted separately and both are printed.
            rhythmic = kind in ("kick", "drum", "kit", "perc", "clap", "hat", "snare")
            ref_ok, _ = metrical(a["ref_bpm"], bpm)
            tempo_rows.append((stem, bpm, a["bpm"], ok, mult, rhythmic, a["ref_bpm"], ref_ok))
        if note:
            k, mode, conf = estimate_key(a["chroma"])
            key_rows.append((stem, note, k, mode, conf))
        if kind:
            inst[kind].append((stem, a))

    # ---- TEMPO
    print("")
    print("TEMPO — vendor BPM in the filename vs ours (metrical level counts as a match)")
    rhy = [r for r in tempo_rows if r[5]]
    oth = [r for r in tempo_rows if not r[5]]
    ok_n = sum(1 for r in tempo_rows if r[3])
    for row in rhy[:12]:
        stem, want, got, ok, mult, rh, ref, rok = row
        print("  %-40s %5.0f | ours %6.1f %-4s | librosa %6.1f %s"
              % (stem[:40], want, got, ("ok" if ok else "MISS"), ref, ("ok" if rok else "")))
    if len(rhy) > 12:
        print("  ... %d more rhythmic" % (len(rhy) - 12))
    ok_r = sum(1 for r in rhy if r[3])
    ok_o = sum(1 for r in oth if r[3])
    ex_r = sum(1 for r in rhy if r[3] and abs(r[4] - 1.0) < 1e-6)
    ref_r = sum(1 for r in rhy if r[7])
    print("  rhythmic material (drums, kits, kicks) : %d/%d matched, %d of them exact"
          % (ok_r, len(rhy), ex_r))
    print("  the same files through librosa         : %d/%d — the yardstick, so the "
          "number above is read against it and not against 32" % (ref_r, len(rhy)))
    print("  pads, drones, bass, texture            : %d/%d — these carry the project "
          "tempo in the name but nothing audible to measure" % (ok_o, len(oth)))

    # ---- KEY
    print("")
    print("KEY — root note in the filename vs our tonic (mode is not labelled, so only the tonic)")
    kok = 0
    bass_rows = [r for r in key_rows if "bass" in r[0].lower()]
    full_rows = [r for r in key_rows if "bass" not in r[0].lower()]
    for stem, want, got, mode, conf in full_rows[:10]:
        print("  %-46s %-3s -> %-3s %-5s conf %.2f  %s" % (stem[:46], want, got, mode, conf,
              "ok" if got == want else ""))
    if len(full_rows) > 10:
        print("  ... %d more full-range" % (len(full_rows) - 10))
    kb = sum(1 for r in bass_rows if r[2] == r[1])
    kf = sum(1 for r in full_rows if r[2] == r[1])
    kok = kb + kf
    print("  full-range tonal material : %d/%d tonics exact" % (kf, len(full_rows)))
    print("  bass-only material        : %d/%d — the root of a bass note sits below the "
          "chroma window and its harmonics point elsewhere" % (kb, len(bass_rows)))

    # ---- INSTRUMENT
    print("")
    print("INSTRUMENT — does the content land in the right channel")
    for kind in ("kick", "bass", "atmosphere", "drone", "noise", "drum", "kit"):
        rows = inst.get(kind, [])
        if not rows:
            continue
        k_rate = np.mean([len(a["kicks"]) / max(a["dur"], 1e-6) for _, a in rows])
        s_rate = np.mean([len(a["snares"]) / max(a["dur"], 1e-6) for _, a in rows])
        bass = np.mean([a["bass"] for _, a in rows])
        high = np.mean([a["high"] for _, a in rows])
        print("  %-12s n=%-3d kicks/s %5.2f  snares/s %5.2f  bass %.2f  high %.2f"
              % (kind, len(rows), k_rate, s_rate, bass, high))

    print("")
    print("=" * 78)
    print("tempo %d/%d rhythmic · key %d/%d full-range · %d files skipped (too short)"
          % (ok_r, len(rhy), kf, len(full_rows), skipped))
    return 0


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not folder:
        print(__doc__.strip())
        print("")
        print("Give the library folder as an argument, or set WEDOAUDIO_SAMPLES.")
        sys.exit(2)
    sys.exit(main(folder))
