# -*- coding: utf-8 -*-
"""Check WEDOAUDIO against audio it did not generate itself.

Three levels, and they are not equally strong — the labels say which is which:

  A. EXACT GROUND TRUTH. Audio built by librosa's click synthesis at times we
     choose, so the answer is known by construction and the generator is not ours.
     A failure here is a real failure.

  B. REAL MUSIC vs AN INDEPENDENT REFERENCE. Actual tracks off the disk, compared
     with librosa's beat tracker and onset detector. librosa is a peer-reviewed
     reference implementation, NOT ground truth: where we disagree, either of us
     can be the one who is wrong. Reported as agreement, never as accuracy.

  C. FORMAT ROBUSTNESS. The same track through ffmpeg into wav16 / wav24 / flac /
     mp3 / ogg / opus and 44.1 vs 48 kHz. Here the ground truth IS exact, because
     the answer must not depend on the container: it is the same music.

Run: python src/_wedoaudio_realtest.py
Needs librosa and ffmpeg; skips level C if ffmpeg is absent.
"""
import os, sys, glob, shutil, subprocess, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import librosa
from wedoaudio_dsp import WedoAudio
from _wedoaudio_bench import f_measure, events, FPS, FFT

def _ffmpeg():
    """ffmpeg из PATH или из переменной WEDOAUDIO_FFMPEG.

    Абсолютный путь к конкретной установке в открытом коде — это не настройка,
    а отпечаток чужой машины: он называет и диск, и версию сборки."""
    c = os.environ.get("WEDOAUDIO_FFMPEG", "")
    if c and os.path.isfile(c):
        return c
    return shutil.which("ffmpeg")


FFMPEG = _ffmpeg()
EXCERPT = 30.0          # seconds analysed per track


def analyse(x, sr):
    """Run the real 60 fps pipeline over a mono float signal."""
    hop = int(sr / FPS)
    win = np.hanning(FFT)
    wa = WedoAudio(sr=sr)
    t, rk, rs, rb, bpm = [], [], [], [], []
    for s in range(0, len(x) - FFT, hop):
        fr = x[s:s + FFT] * win
        mag = np.abs(np.fft.rfft(fr)) / (FFT / 4)
        f = wa.process(mag, float(np.sqrt(np.mean(fr ** 2))), s / sr, 1.0 / FPS)
        t.append(s / sr)
        rk.append(f["rawkick"])
        rs.append(f["rawsnare"])
        rb.append(f["rawbeat"])
        bpm.append(f["bpm"])
    return np.array(t), np.array(rk), np.array(rs), np.array(rb), np.array(bpm)


def median_bpm(b):
    return float(np.median(b[len(b) // 2:])) if len(b) else 0.0


def octave_equal(a, b, tol=0.06):
    """Same tempo up to a metrical level (1/3, 1/2, 1, 2, 3)."""
    if a <= 0 or b <= 0:
        return False, 0.0
    for m in (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0):
        if abs(a * m - b) / b < tol:
            return True, m
    return False, 0.0


# ---------------------------------------------------------------- A
def level_a():
    print("A. EXACT GROUND TRUTH - audio synthesised by librosa, times known by construction")
    print("")
    sr = 44100
    rows = []
    # Click frequencies must land in the band the detector actually covers: kick is
    # 40-120 Hz, snare is 1800-6000 Hz. A first version put 1 kHz clicks in the snare
    # column, which is neither band, and scored 0.00 twice - a defect in this file,
    # not in the detector. The out-of-band case is kept below on purpose.
    cases = (("kick clicks 90 BPM", 90.0, 60.0, "kick"),
             ("kick clicks 128 BPM", 128.0, 60.0, "kick"),
             ("kick clicks 174 BPM", 174.0, 80.0, "kick"),
             ("snare clicks 120 BPM", 120.0, 3000.0, "snare"),
             ("snare clicks 96 BPM", 96.0, 4000.0, "snare"),
             ("1 kHz clicks 120 BPM", 120.0, 1000.0, "beat"))
    for name, bpm, click_hz, want in cases:
        dur = 16.0
        times = np.arange(0.0, dur - 0.5, 60.0 / bpm)
        x = librosa.clicks(times=times, sr=sr, click_freq=click_hz,
                           click_duration=0.05, length=int(dur * sr))
        x = x / (np.max(np.abs(x)) or 1.0)
        t, rk, rs, rb, b = analyse(x, sr)
        src = {"kick": rk, "snare": rs, "beat": rb}[want]
        p, r, f = f_measure(events(src, t), times)
        est = median_bpm(b)
        same, mult = octave_equal(est, bpm)
        verdict = ("ok x%.2f" % mult) if same else "MISMATCH"
        print("  %-22s [%-5s] P/R/F %.2f/%.2f/%.2f   bpm %6.1f vs %5.1f  %s"
              % (name, want, p, r, f, est, bpm, verdict))
        rows.append((f, same, want))
    return rows


# ---------------------------------------------------------------- B
def level_b(tracks):
    print("")
    print("B. REAL MUSIC vs librosa - agreement between two estimators, not accuracy")
    print("")
    rows = []
    for path in tracks:
        try:
            x, sr = librosa.load(path, sr=None, mono=True, duration=EXCERPT)
        except Exception as e:
            print("  %-28s could not load: %s" % (os.path.basename(path), e))
            continue
        x = x / (np.max(np.abs(x)) or 1.0)
        ref_bpm, ref_beats = librosa.beat.beat_track(y=x, sr=sr, units="time")
        ref_bpm = float(np.atleast_1d(ref_bpm)[0])
        ref_on = librosa.onset.onset_detect(y=x, sr=sr, units="time")
        t, rk, rs, rb, b = analyse(x, sr)
        est = median_bpm(b)
        same, mult = octave_equal(est, ref_bpm)
        ours = np.sort(np.concatenate([events(rk, t), events(rs, t)]))
        p, r, f = f_measure(ours, ref_on)
        verdict = ("agree x%.2f" % mult) if same else "DIFFER"
        print("  %-28s sr %d  our bpm %6.1f | librosa %6.1f  %-12s "
              "onset overlap P/R/F %.2f/%.2f/%.2f  (%d ours / %d librosa)"
              % (os.path.basename(path), sr, est, ref_bpm, verdict, p, r, f, len(ours), len(ref_on)))
        rows.append((same, f))
    return rows


# ---------------------------------------------------------------- C
def level_c(track):
    print("")
    print("C. FORMAT ROBUSTNESS - the same music must give the same answer in any container")
    print("")
    if not os.path.exists(FFMPEG):
        print("  ffmpeg not found, skipped")
        return []
    tmp = tempfile.mkdtemp(prefix="wa_fmt_")
    variants = (("wav 16 bit 44.1k", ["-ar", "44100", "-c:a", "pcm_s16le", "-f", "wav"]),
                ("wav 24 bit 44.1k", ["-ar", "44100", "-c:a", "pcm_s24le", "-f", "wav"]),
                ("wav 32 float 48k", ["-ar", "48000", "-c:a", "pcm_f32le", "-f", "wav"]),
                ("flac 44.1k", ["-ar", "44100", "-c:a", "flac", "-f", "flac"]),
                ("mp3 128 kbps", ["-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"]),
                ("ogg vorbis q4", ["-ar", "44100", "-c:a", "libvorbis", "-q:a", "4", "-f", "ogg"]),
                ("opus 96 kbps", ["-ar", "48000", "-c:a", "libopus", "-b:a", "96k", "-f", "opus"]))
    base = None
    rows = []
    for label, args in variants:
        ext = args[-1]
        out = os.path.join(tmp, label.replace(" ", "_").replace(".", "") + "." + ext)
        cmd = [FFMPEG, "-y", "-v", "error", "-i", track, "-t", str(EXCERPT), "-ac", "1"] + args + [out]
        if subprocess.run(cmd, capture_output=True).returncode != 0 or not os.path.exists(out):
            print("  %-20s transcode failed, skipped" % label)
            continue
        x, sr = librosa.load(out, sr=None, mono=True)
        x = x / (np.max(np.abs(x)) or 1.0)
        t, rk, rs, rb, b = analyse(x, sr)
        est = median_bpm(b)
        kicks = events(rk, t)
        if base is None:
            base = (est, kicks, label)
            print("  %-20s bpm %6.1f  kicks %4d   (reference for this block)" % (label, est, len(kicks)))
            rows.append((label, 1.0, 0.0))
            continue
        p, r, f = f_measure(kicks, base[1])
        drift = abs(est - base[0]) / max(base[0], 1e-6) * 100.0
        flag = "" if (f >= 0.85 and drift < 5.0) else "   <-- differs"
        print("  %-20s bpm %6.1f  kicks %4d   kick agreement F %.2f vs reference, bpm drift %.1f%%%s"
              % (label, est, len(kicks), f, drift, flag))
        rows.append((label, f, drift))
    return rows


if __name__ == "__main__":
    a = level_a()
    cand = []
    # Материал для уровня B задаёт тот, кто запускает: аргументы командной
    # строки или WEDOAUDIO_TRACKS (папка либо маска). Пути чужой машины в
    # открытом коде называют её устройство, а не помогают воспроизвести замер.
    источники = sys.argv[1:] or [x for x in (os.environ.get("WEDOAUDIO_TRACKS", ""),) if x]
    for pat in источники:
        if os.path.isdir(pat):
            for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a"):
                cand += sorted(glob.glob(os.path.join(pat, ext)))
        else:
            cand += sorted(glob.glob(pat))
    if not cand:
        print("Уровень B пропущен: задайте треки аргументом или WEDOAUDIO_TRACKS")
    b = level_b(cand)
    c = level_c(cand[0]) if cand else []

    print("")
    print("=" * 78)
    a_ok = sum(1 for f, s, w in a if f >= 0.90 and s)
    print("A exact ground truth : %d/%d passed (onset F >= 0.90 and tempo within a metrical level)"
          % (a_ok, len(a)))
    if b:
        agree = sum(1 for s, f in b if s)
        print("B agreement with librosa on real music : tempo %d/%d within a metrical level"
              % (agree, len(b)))
    if c:
        stable = sum(1 for lbl, f, d in c[1:] if f >= 0.85 and d < 5.0)
        print("C format robustness  : %d/%d containers matched the reference" % (stable, max(len(c) - 1, 1)))
    sys.exit(0 if a_ok == len(a) else 1)
