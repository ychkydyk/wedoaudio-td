# -*- coding: utf-8 -*-
"""ФОРМАТЫ. Одна музыка — один ответ, чем бы её ни закодировали.

Зачем отдельный стенд. Библиотеку лупов мы уже гоняли, но только как wav.
Пользователь приносит материал в чём угодно: flac из архива, mp3 из мессенджера,
ogg из браузера, opus из голосового. Если анализатор меняет ответ от кодека —
это дефект анализатора, а не свойство музыки.

Истина здесь ТОЧНАЯ и не требует ничьего мнения: исходный wav и его перекодировки
содержат одну и ту же музыку, значит темп и рисунок ударов обязаны совпасть.
Расхождение измеряется, а не оценивается на слух.

Второй раздел — ЛОЖНЫЕ СРАБАТЫВАНИЯ. Дроны, эмбиенс, полевые записи: правильный
ответ «ударов нет». Это единственный раздел, где хороший результат — ноль.

Запуск:
    set WEDOAUDIO_SAMPLES=D:\\путь\\к\\библиотеке
    python src/_wedoaudio_formats.py [--fast]

Нужны ffmpeg и soundfile (librosa не обязателен).
"""
import os, re, sys, json, glob, shutil, subprocess, tempfile, collections

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wedoaudio_dsp import WedoAudio
from _wedoaudio_bench import f_measure, events, FPS, FFT

import soundfile as sf

LIB = os.environ.get("WEDOAUDIO_SAMPLES", "")
FAST = "--fast" in sys.argv
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ffmpeg():
    # ffmpeg берётся из PATH или из WEDOAUDIO_FFMPEG. Абсолютный путь к чужой
    # установке в открытом коде — отпечаток машины, а не настройка.
    c = os.environ.get("WEDOAUDIO_FFMPEG", "")
    if c and os.path.isfile(c):
        return c
    return shutil.which("ffmpeg")


FF = ffmpeg()

# кодек, расширение, аргументы. Порядок — от точного к разрушительному.
FORMATS = [
    ("wav16",   ".wav",  ["-c:a", "pcm_s16le"]),
    ("wav24",   ".wav",  ["-c:a", "pcm_s24le"]),
    ("wav32f",  ".wav",  ["-c:a", "pcm_f32le"]),
    ("flac",    ".flac", ["-c:a", "flac"]),
    ("mp3-320", ".mp3",  ["-c:a", "libmp3lame", "-b:a", "320k"]),
    ("mp3-128", ".mp3",  ["-c:a", "libmp3lame", "-b:a", "128k"]),
    ("ogg-q5",  ".ogg",  ["-c:a", "libvorbis", "-q:a", "5"]),
    ("opus-96", ".opus", ["-c:a", "libopus", "-b:a", "96k"]),
    ("wav16@48", ".wav", ["-c:a", "pcm_s16le", "-ar", "48000"]),
]

BPM_IN_NAME = re.compile(r"(?:^|[_-])(\d{2,3})(?:[_-])")
РИТМ = ("drum", "kit_loop", "perc", "beat", "kick_snare", "rhythm", "hats", "hat_")
ТИХО = ("drone", "ambian", "ambien", "atmosphere", "texture", "pad", "field_recording",
        "riser", "swarm", "hum")


def подпись(name):
    """Темп из имени файла — это разметка производителя, а не наша догадка."""
    m = BPM_IN_NAME.search(name)
    if not m:
        return 0
    v = int(m.group(1))
    return v if 60 <= v <= 200 else 0


def читать(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), float(sr)


def анализ(x, sr):
    """Тот же конвейер 60 кадров/с, что и в остальных стендах."""
    hop = int(sr / FPS)
    win = np.hanning(FFT)
    wa = WedoAudio(sr=sr)
    t, rk, rs, rb, bpm = [], [], [], [], []
    for s in range(0, max(0, len(x) - FFT), hop):
        fr = x[s:s + FFT] * win
        mag = np.abs(np.fft.rfft(fr)) / (FFT / 4)
        f = wa.process(mag, float(np.sqrt(np.mean(fr ** 2))), s / sr, 1.0 / FPS)
        t.append(s / sr); rk.append(f["rawkick"]); rs.append(f["rawsnare"])
        rb.append(f["rawbeat"]); bpm.append(f["bpm"])
    return (np.array(t), np.array(rk), np.array(rs), np.array(rb), np.array(bpm))


def темп(bpm):
    return float(np.median(bpm[len(bpm) // 2:])) if len(bpm) else 0.0


def октавно_равны(a, b, tol=0.06):
    if a <= 0 or b <= 0:
        return False
    return any(abs(a * m - b) / b < tol for m in (1 / 3.0, 0.5, 1.0, 2.0, 3.0))


def перекодировать(src, имя, ext, args, tmp):
    dst = os.path.join(tmp, "x_%s%s" % (имя.replace("@", "at"), ext))
    p = subprocess.run([FF, "-y", "-v", "error", "-i", src] + args + [dst],
                       capture_output=True)
    return dst if p.returncode == 0 and os.path.isfile(dst) else None


def главное():
    if not LIB or not os.path.isdir(LIB):
        print("Укажите библиотеку: set WEDOAUDIO_SAMPLES=<папка>")
        return 1
    if not FF:
        print("ffmpeg не найден — стенд по форматам невозможен")
        return 1

    все = sorted(glob.glob(os.path.join(LIB, "*.wav")))
    ритм = [f for f in все
            if подпись(os.path.basename(f)) and
            any(k in os.path.basename(f).lower() for k in РИТМ)]
    тихо = [f for f in все
            if any(k in os.path.basename(f).lower() for k in ТИХО) and
            not any(k in os.path.basename(f).lower() for k in РИТМ)]
    if FAST:
        ритм, тихо = ритм[::3][:10], тихо[::3][:8]

    print("библиотека: %s" % LIB)
    print("ритмических с разметкой темпа: %d · нератмичных (дрон/эмбиенс): %d"
          % (len(ритм), len(тихо)))
    print("форматов на файл: %d · всего прогонов: %d\n"
          % (len(FORMATS) + 1, (len(ритм) + len(тихо)) * (len(FORMATS) + 1)))

    строки = []
    tmp = tempfile.mkdtemp(prefix="wedo_fmt_")
    try:
        # ---------------------------------------------------------- ритм
        print("=" * 78)
        print("РИТМИЧЕСКИЙ МАТЕРИАЛ — темп и рисунок ударов не должны зависеть от кодека")
        print("=" * 78)
        for i, src in enumerate(ритм, 1):
            имяф = os.path.basename(src)
            метка = подпись(имяф)
            try:
                x, sr = читать(src)
            except Exception as e:
                print("  пропуск %s: %s" % (имяф[:40], e)); continue
            эт = анализ(x, sr)
            эт_bpm = темп(эт[4])
            эт_kick = events(эт[1], эт[0])
            эт_snare = events(эт[2], эт[0])
            print("\n%2d/%d  %s" % (i, len(ритм), имяф[:66]))
            print("      разметка %d BPM · оригинал wav: %.1f BPM, киков %d, снейров %d"
                  % (метка, эт_bpm, len(эт_kick), len(эт_snare)))
            for имя, ext, args in FORMATS:
                dst = перекодировать(src, имя, ext, args, tmp)
                if not dst:
                    print("      %-10s перекодировка не удалась" % имя); continue
                try:
                    y, sr2 = читать(dst)
                except Exception as e:
                    print("      %-10s не читается: %s" % (имя, e)); continue
                r = анализ(y, sr2)
                b = темп(r[4])
                k = events(r[1], r[0])
                s_ = events(r[2], r[0])
                дрейф = abs(b - эт_bpm) / эт_bpm * 100 if эт_bpm else 0.0
                fk = f_measure(k, эт_kick)[2] if len(эт_kick) else (1.0 if not len(k) else 0.0)
                fs = f_measure(s_, эт_snare)[2] if len(эт_snare) else (1.0 if not len(s_) else 0.0)
                строки.append(dict(группа="ритм", файл=имяф, формат=имя, метка=метка,
                                   bpm=b, bpm_эталон=эт_bpm, дрейф=дрейф,
                                   киков=len(k), снейров=len(s_),
                                   f_кик=fk, f_снейр=fs,
                                   метка_ок=bool(октавно_равны(b, метка)) if метка else None))
                print("      %-10s %6.1f BPM  дрейф %5.2f%%  киков %3d  снейров %3d  "
                      "F кик %.2f  F снейр %.2f"
                      % (имя, b, дрейф, len(k), len(s_), fk, fs))

        # ------------------------------------------------- ложные срабатывания
        print("\n" + "=" * 78)
        print("НЕРИТМИЧЕСКИЙ МАТЕРИАЛ — правильный ответ «ударов нет»")
        print("=" * 78)
        for i, src in enumerate(тихо, 1):
            имяф = os.path.basename(src)
            try:
                x, sr = читать(src)
            except Exception as e:
                print("  пропуск %s: %s" % (имяф[:40], e)); continue
            длит = len(x) / sr
            эт = анализ(x, sr)
            k0, s0 = len(events(эт[1], эт[0])), len(events(эт[2], эт[0]))
            print("\n%2d/%d  %s   (%.1f с)" % (i, len(тихо), имяф[:60], длит))
            print("      оригинал wav: ложных киков %.2f/с, снейров %.2f/с"
                  % (k0 / длит, s0 / длит))
            for имя, ext, args in FORMATS:
                dst = перекодировать(src, имя, ext, args, tmp)
                if not dst:
                    continue
                try:
                    y, sr2 = читать(dst)
                except Exception:
                    continue
                r = анализ(y, sr2)
                k = len(events(r[1], r[0])); s_ = len(events(r[2], r[0]))
                строки.append(dict(группа="тихо", файл=имяф, формат=имя, длит=длит,
                                   киков_с=k / длит, снейров_с=s_ / длит))
                print("      %-10s ложных киков %.2f/с, снейров %.2f/с"
                      % (имя, k / длит, s_ / длит))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ свод
    print("\n" + "=" * 78)
    print("СВОД ПО ФОРМАТАМ")
    print("=" * 78)
    # Среднее здесь врёт: у всех форматов медиана дрейфа 0.00%, а среднее
    # задирают единичные файлы, где темп перескочил на другой метрический
    # уровень. Поэтому главная цифра — медиана, а рядом честно стоит счётчик
    # неустойчивых файлов, а не размазанное среднее.
    print("%-10s %8s %8s %8s %7s %7s %9s" %
          ("формат", "медиана", "среднее", "макс", "F кик", "F снр", "неуст."))
    свод = {}
    for имя, _, _ in FORMATS:
        r = [x for x in строки if x["формат"] == имя and x["группа"] == "ритм"]
        t_ = [x for x in строки if x["формат"] == имя and x["группа"] == "тихо"]
        if not r:
            continue
        d = np.array([x["дрейф"] for x in r], dtype=float)
        свод[имя] = dict(
            медиана=float(np.median(d)), дрейф=float(np.mean(d)), макс=float(np.max(d)),
            неуст=int((d > 5.0).sum()),
            f_кик=float(np.mean([x["f_кик"] for x in r])),
            f_снейр=float(np.mean([x["f_снейр"] for x in r])),
            ложн=float(np.mean([x["киков_с"] for x in t_])) if t_ else 0.0,
            n=len(r))
        s = свод[имя]
        print("%-10s %8.2f %8.2f %8.2f %7.2f %7.2f %5d/%-3d"
              % (имя, s["медиана"], s["дрейф"], s["макс"], s["f_кик"], s["f_снейр"],
                 s["неуст"], s["n"]))

    рит = [x for x in строки if x["группа"] == "ритм" and x["метка"]]
    if рит:
        по_метке = sum(1 for x in рит if x["метка_ок"])
        print("\nсовпадение с разметкой производителя (октавно): %d из %d = %.0f%%"
              % (по_метке, len(рит), по_метке / len(рит) * 100))

    # Имена файлов в выгрузку не идут. Отчёт использует из этого поля только
    # число уникальных значений, а список имён из чужой библиотеки — это опись
    # рабочего места, а не результат замера.
    уник = sorted(set(x["файл"] for x in строки))
    карта = {f: "loop_%03d%s" % (i + 1, os.path.splitext(f)[1]) for i, f in enumerate(уник)}
    обезличенные = [dict(x, файл=карта[x["файл"]]) for x in строки]

    out = os.path.join(HERE, "format_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"строки": обезличенные, "свод": свод}, f, ensure_ascii=False, indent=1)
    print("\nсырые числа: %s" % out)

    # Ворота двойные. Медиана отвечает за типичный файл: кодек не имеет права
    # двигать темп вообще. Счётчик неустойчивых отвечает за хвост: сколько
    # файлов перескочило на другой метрический уровень.
    плохо = [k for k, v in свод.items() if v["медиана"] > 0.5]
    хвост = [k for k, v in свод.items() if v["неуст"] > 0.25 * v["n"]]
    print("ворота, медиана дрейфа ≤ 0.5%%:",
          "пройдены" if not плохо else "НЕ пройдены: " + ", ".join(плохо))
    print("ворота, неустойчивых файлов ≤ 25%%:",
          "пройдены" if not хвост else "НЕ пройдены: " + ", ".join(хвост))
    плохо = плохо + хвост
    return 0 if not плохо else 1


if __name__ == "__main__":
    sys.exit(главное())
