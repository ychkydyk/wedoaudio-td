# -*- coding: utf-8 -*-
"""Нативная приёмка собранного компонента. Запускается ВНУТРИ TouchDesigner.

    import os; os.environ['WEDOAUDIO_REPO'] = r'C:/путь/к/репозиторию'
    exec(open(os.environ['WEDOAUDIO_REPO'] + '/tools/native_acceptance.py', encoding='utf-8').read())

Что делает: грузит записанный .tox в чистый контейнер под ДРУГИМ именем, ничего в нём
не трогает руками и полминуты ведёт его по сценарию — ровный тон, один канал,
разрыв входа, возврат, другая частота дискретизации, выключение громкости, сброс,
две намеренные заминки кадра.
Снимки берутся по кадрам через run(delayFrames), потому что половина проверок про то,
что происходит в первые кадры после события. Итог пишется в native_acceptance.json
рядом с .tox и печатается в Textport.

Источник звука — генератор, а не устройство: проверка не зависит от звуковой карты и
даёт сигнал с известным ответом. Стерео-синус 1 кГц на -23 dBFS обязан читаться как
-23.0 LUFS (EBU Tech 3341, случай 1), тот же тон в одном канале — как -26.0.

Скрипт ничего не удаляет: каждая приёмка создаёт новый контейнер wedoaudio_acceptN.
"""
import os
import glob

REPO = os.environ.get("WEDOAUDIO_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = str(getattr(app, "build", "unknown"))
_found = sorted(glob.glob(os.path.join(REPO, "WEDOAUDIO_*_TD%s.tox" % BUILD)), key=os.path.getmtime)
if not _found:
    raise RuntimeError("в %s нет .tox для сборки %s — сначала tools/build_portable.py" % (REPO, BUILD))
TOX = _found[-1].replace("\\", "/")

ACC = '''
import json, os, time, hashlib, datetime
import numpy as np

REPO = %(repo)r
TOX = %(tox)r
BASE = ("rms bass mid high centroid flatness rolloff flux kick rawkick kickband snare rawsnare "
        "snareband beat rawbeat bpm beatphase transient sustain agcgain silence").split()
LOUD = "lufsm lufss lufsi lra truepeak truepeakmax".split()
AMP = 10 ** (-23 / 20.0)


def zone():
    return me.parent()


def comp():
    return zone().op("wa_cold")


def snap(tag):
    c = comp()
    out = c.op("out_features")
    ext = c.ext.Wedoaudio
    m = getattr(ext, "_loud", None)
    zone().store(tag, {
        "t": absTime.seconds, "frame": absTime.frame,
        "names": [ch.name for ch in out.chans()],
        "vals": {ch.name: float(ch[0]) for ch in out.chans()},
        "status": c.par.Status.eval(),
        "sr": ext.Samplerate(),
        "api": ext.Loudness(),
        "samples": (m.samples if m is not None else None),
        "meter_sr": (m.sr if m is not None else None),
        "mono_chans": c.op("to_mono").numChans,
        "in_chans": c.op("in_audio").numChans,
    })


def act(name):
    z, c = zone(), comp()
    if name == "left_off":      z.op("l44").par.amp = 0
    elif name == "left_on":     z.op("l44").par.amp = AMP
    elif name == "disconnect":  c.inputConnectors[0].disconnect()
    elif name == "to44":        c.inputConnectors[0].connect(z.op("stereo44"))
    elif name == "to48":        c.inputConnectors[0].connect(z.op("stereo48"))
    elif name == "reset_loud":  c.par.Resetloudness.pulse()
    elif name == "loud_off":    c.par.Loudness = 0
    elif name == "loud_on":     c.par.Loudness = 1
    elif name == "tp_off":      c.par.Truepeak = 0
    elif name == "tp_on":       c.par.Truepeak = 1
    elif name == "reset_all":   c.par.Resetstate.pulse()
    elif name == "hitch_small": time.sleep(0.10)
    elif name == "hitch_big":   time.sleep(0.50)
    else: raise ValueError(name)


PLAN = [
    (3, "snap", "cold"), (330, "snap", "steady"),       # диапазону нужны два кратковременных блока: 4 с
    (335, "act", "left_off"), (420, "snap", "right_only"), (425, "act", "left_on"),
    (426, "act", "disconnect"), (450, "snap", "disc"),
    (455, "act", "to44"), (520, "snap", "recon"),
    (525, "act", "to48"), (530, "act", "reset_loud"), (590, "snap", "r48a"), (890, "snap", "r48"),
    (895, "act", "to44"), (896, "act", "loud_off"), (920, "snap", "loudoff"),
    (925, "act", "loud_on"), (926, "act", "tp_off"), (990, "snap", "tpoff"),
    (995, "act", "tp_on"), (1250, "snap", "pre_reset"),
    (1255, "act", "reset_all"), (1258, "snap", "reset3"),
    # Сторож потери. Заминка короче 0.2 с TouchDesigner навёрстывает сам, длиннее -
    # обрезает срез, и звук за остаток выпадает. Первая не должна шуметь, вторая
    # обязана быть замечена. Окно сторожа 5 с, поэтому снимки через 5.5 с.
    (1265, "act", "hitch_small"), (1600, "snap", "after_small"),
    (1605, "act", "hitch_big"), (1960, "snap", "after_big"),
    (1980, "finish", ""),
]
HITCH_BIG_AT = 1605


def accounting():
    """Покадровый учёт из framelog: пока измеритель один и тот же, его счётчик обязан
    расти ровно на длину среза входа, а сумма срезов - сходиться с часами."""
    log = zone().op("framelog").module.LOG
    big = zone().fetch("plan_start", 0) + HITCH_BIG_AT
    gaps = fed_short = frames = 0
    notes = set()
    spans, cur = [], None
    for prev, row in zip(log, log[1:]):
        frames += 1
        if row["frame"] - prev["frame"] != 1:
            gaps += 1
        if "громкост" in row["status"] and row["frame"] < big:
            notes.add(row["status"])
        # Один и тот же измеритель на том же входе: на кадре переключения источника
        # срез в логе уже от нового входа, а накормлен измеритель был ещё старым.
        same = (row["meter"] == prev["meter"] and row["meter"] is not None
                and row["samples"] >= prev["samples"]
                and row["sr"] == prev["sr"] == row["meter_sr"])
        if same:
            if row["samples"] - prev["samples"] != row["slice"]:
                fed_short += 1
            if cur is None:
                cur = [prev, row]
            cur[1] = row
        elif cur is not None:
            spans.append(cur); cur = None
    if cur is not None:
        spans.append(cur)
    worst = 0.0
    for a, b in spans:
        dt = b["t"] - a["t"]
        if dt >= 1.0:
            worst = max(worst, abs((b["samples"] - a["samples"]) / (dt * a["sr"]) - 1.0))
    return {"frames": frames, "frame_gaps": gaps, "frames_fed_wrong": fed_short,
            "spans": len(spans), "worst_span_error_vs_td_clock": round(worst, 5),
            "notes_before_big_hitch": sorted(notes)}


def start():
    here = me.path
    zone().store("plan_start", absTime.frame)
    for frames, kind, arg in PLAN:
        call = "finish()" if kind == "finish" else "%%s(%%r)" %% (kind, arg)
        run("op(%%r).module.%%s" %% (here, call), delayFrames=frames)
    print("[WEDOAUDIO приёмка] сценарий поставлен, итог через %%.0f с" %% (PLAN[-1][0] / 60.0))


def near(v, target, tol):
    return v is not None and abs(v - target) <= tol


def cost():
    """Цена громкости в питоне ЭТОГО TouchDesigner: свежий измеритель и после двух часов."""
    mod = comp().op("wedoaudio_loudness").module
    rng = np.random.default_rng(1)
    blk = rng.standard_normal((2, 735)) * 0.1
    res = {}
    for label, fill in (("fresh", 0), ("two_hours", mod.MAX_BLOCKS + 500)):
        m = mod.LoudnessMeter(44100, 2)
        for _ in range(60): m.process(blk)
        for v in rng.uniform(1e-6, 1e-2, fill): m._blocks.push(v)
        for v in rng.uniform(1e-6, 1e-2, fill // 10): m._st.push(v)
        ts = []
        for _ in range(300):
            t = time.perf_counter(); m.process(blk); ts.append((time.perf_counter() - t) * 1e3)
        res[label] = {"mean_ms": round(float(np.mean(ts)), 3), "max_ms": round(float(np.max(ts)), 3)}
    return res


def finish():
    z, c = zone(), comp()
    S = {k: z.fetch(k, None) for _, kind, k in PLAN if kind == "snap"}
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    missing = [k for k, v in S.items() if v is None]
    check("все снимки сняты", not missing, missing or "%%d снимков" %% len(S))
    if missing:
        return report(checks, {})

    v = lambda tag, k: S[tag]["vals"].get(k)
    L = lambda tag: {k: round(S[tag]["vals"][k], 2) for k in LOUD if k in S[tag]["vals"]}

    check("1. холодная загрузка: 28 именованных каналов на третьем кадре, порядок базовых прежний",
          S["cold"]["names"] == BASE + LOUD, "%%d каналов" %% len(S["cold"]["names"]))
    check("2. стерео 1 кГц -23 dBFS читается как -23.0 LUFS (M, S, I) с допуском 0.1",
          all(near(v("steady", k), -23.0, 0.1) for k in ("lufsm", "lufss", "lufsi")), L("steady"))
    check("3. диапазон ровного тона нулевой, истинный пик -23.0 dBTP",
          near(v("steady", "lra"), 0.0, 0.1) and near(v("steady", "truepeak"), -23.0, 0.1), L("steady"))
    check("4. измеритель, начавший посреди тона, не выдумывает пик",
          v("steady", "truepeakmax") <= -22.95 and v("reset3", "truepeakmax") <= -22.95,
          "после загрузки %%.3f, после сброса %%.3f dBTP" %% (v("steady", "truepeakmax"), v("reset3", "truepeakmax")))
    ratio = v("right_only", "rms") / max(v("steady", "rms"), 1e-9)
    check("5. сигнал только в правом канале слышен; сведение в моно даёт один канал",
          0.4 < ratio < 0.6 and S["right_only"]["mono_chans"] == 1 and S["right_only"]["in_chans"] == 2,
          "rms %%.3f от стерео, каналов после сведения %%d" %% (ratio, S["right_only"]["mono_chans"]))
    check("6. тот же тон в одном канале читается как -26.0 LUFS",
          near(v("right_only", "lufsm"), -26.0, 0.1), "M %%.2f" %% v("right_only", "lufsm"))
    d = S["disc"]
    check("7. разрыв входа: интерфейс цел, значения нейтральные, громкость «не измерена»",
          d["names"] == BASE + LOUD and d["vals"]["rms"] == 0.0 and d["vals"]["silence"] == 1.0
          and d["vals"]["agcgain"] == 1.0 and all(d["vals"][k] == -100.0 for k in LOUD)
          and all(x is None for x in d["api"].values()) and not d["status"].startswith("работает"),
          d["status"])
    check("8. возврат входа возобновляет анализ",
          S["recon"]["status"].startswith("работает") and v("recon", "rms") > 0.3
          and near(v("recon", "lufsm"), -23.0, 0.1), S["recon"]["status"])
    r = S["r48"]
    check("9. вход 48 кГц: частоту узнали и ядро, и измеритель; громкость та же",
          r["sr"] == 48000.0 and r["meter_sr"] == 48000.0 and near(r["vals"]["lufsi"], -23.0, 0.1),
          "%%s · I %%.2f" %% (r["status"], r["vals"]["lufsi"]))
    acct = accounting()
    check("10. сэмплы не теряются и не дублируются: покадровый учёт за весь сценарий",
          acct["frames_fed_wrong"] == 0 and acct["worst_span_error_vs_td_clock"] < 0.005
          and not acct["notes_before_big_hitch"], acct)
    check("10а. сторож молчит на заминке 0.1 с, которую TouchDesigner навёрстывает сам",
          "потеряно" not in S["after_small"]["status"], S["after_small"]["status"])
    check("10б. сторож замечает заминку 0.5 с: срез обрезан на 0.2 с, остаток выпал",
          "потеряно" in S["after_big"]["status"], S["after_big"]["status"])
    check("11. Loudness выключен: шина ровно прежняя, 22 канала",
          S["loudoff"]["names"] == BASE and S["loudoff"]["samples"] is None, S["loudoff"]["status"])
    check("12. Truepeak выключен: пик «не измерен», громкость считается",
          v("tpoff", "truepeak") == -100.0 and v("tpoff", "truepeakmax") == -100.0
          and near(v("tpoff", "lufsm"), -23.0, 0.1), L("tpoff"))
    check("13. Reset очищает состояние: усиление откатилось, интегральная начата заново",
          v("reset3", "agcgain") < 0.5 * v("pre_reset", "agcgain")
          and all(v("reset3", k) == -100.0 for k in ("lufsm", "lufss", "lufsi", "lra")),
          "agcgain %%.2f -> %%.2f · %%s" %% (v("pre_reset", "agcgain"), v("reset3", "agcgain"), L("reset3")))
    check("14. значения API — обычные числа или None",
          all(x is None or type(x) is float for x in S["steady"]["api"].values()),
          {k: type(x).__name__ for k, x in S["steady"]["api"].items()})

    ok, problems = c.ext.Wedoaudio.Selftest()
    check("15. самопроверка компонента под чужим именем в чужом контейнере", ok and not problems, problems or "чисто")
    errs, warns = c.errors(recurse=True), c.warnings(recurse=True)
    check("16. ни ошибок, ни предупреждений у операторов", not errs and not warns, (errs + warns) or "чисто")
    same = []
    for dat, rel in (("wedoaudio_dsp", "src/wedoaudio_dsp.py"), ("wedoaudio_loudness", "src/wedoaudio_loudness.py")):
        disk = open(os.path.join(REPO, rel), encoding="utf-8").read().replace("\\r\\n", "\\n")
        same.append(c.op(dat).text.replace("\\r\\n", "\\n") == disk)
    check("17. встроенные исходники совпадают с файлами репозитория", all(same), same)
    fps = (S["pre_reset"]["frame"] - S["cold"]["frame"]) / (S["pre_reset"]["t"] - S["cold"]["t"])
    check("18. тестовый проект держит кадр", fps > 59.0, "%%.2f к/с за %%.1f с" %% (fps, S["pre_reset"]["t"] - S["cold"]["t"]))
    price = cost()
    check("19. цена громкости в кадре, включая двухчасовой сет",
          price["fresh"]["mean_ms"] < 1.0 and price["two_hours"]["mean_ms"] < 1.5, price)
    return report(checks, {"cost_ms": price, "fps": round(fps, 2), "sample_accounting": acct})


def report(checks, extra):
    passed = sum(1 for x in checks if x["ok"])
    doc = {
        "date": datetime.datetime.now().strftime("%%Y-%%m-%%d %%H:%%M"),
        "touchdesigner_build": str(app.build),
        "tox": os.path.basename(TOX),
        "tox_sha256": hashlib.sha256(open(TOX, "rb").read()).hexdigest(),
        "passed": passed, "total": len(checks), "checks": checks,
    }
    doc.update(extra)
    path = os.path.join(REPO, "native_acceptance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    zone().store("result", doc)
    print("[WEDOAUDIO приёмка] %%d из %%d · %%s" %% (passed, len(checks), path))
    for x in checks:
        print("  %%s  %%s  %%s" %% ("PASS" if x["ok"] else "FAIL", x["name"], x["detail"]))
    return doc
''' % {"repo": REPO.replace("\\", "/"), "tox": TOX}


FRAMELOG = '''
LOG = []

def onFrameEnd(frame):
    c = me.parent().op("wa_cold")
    if c is None or len(LOG) > 3000:
        return
    ext = c.ext.Wedoaudio
    m = getattr(ext, "_loud", None)
    ia = c.op("in_audio")
    LOG.append({"frame": absTime.frame, "t": absTime.seconds, "slice": ia.numSamples,
                "sr": float(ia.rate), "meter": (id(m) if m is not None else None),
                "meter_sr": (m.sr if m is not None else None),
                "samples": (m.samples if m is not None else 0), "status": c.par.Status.eval()})
'''


def _setup():
    root = op("/project1")
    n = 1
    while root.op("wedoaudio_accept%d" % n) is not None:
        n += 1
    zone = root.create(baseCOMP, "wedoaudio_accept%d" % n)
    zone.nodeX, zone.nodeY = 400, -200 * n
    amp = 10 ** (-23 / 20.0)

    def pair(tag, rate, y):
        chans = []
        for i, side in enumerate("lr"):
            o = zone.create(audiooscillatorCHOP, side + tag)
            o.nodeX, o.nodeY = -500, y - 100 * i
            o.par.frequency, o.par.amp, o.par.rate = 1000, amp, rate
            chans.append(o)
        m = zone.create(mergeCHOP, "stereo" + tag)
        m.nodeX, m.nodeY = -300, y - 50
        m.inputConnectors[0].connect(chans[0])
        m.inputConnectors[1].connect(chans[1])
        return m

    s44 = pair("44", 44100, 200)
    pair("48", 48000, -100)
    log = zone.create(executeDAT, "framelog")
    log.nodeX, log.nodeY = 300, -300
    log.text = FRAMELOG
    log.par.frameend = True
    acc = zone.create(textDAT, "acc")
    acc.nodeX, acc.nodeY = 100, -300
    acc.text = ACC
    comp = zone.loadTox(TOX)          # кладёт компонент РЕБЁНКОМ зоны
    comp.name = "wa_cold"             # другое имя: пути внутри обязаны быть относительными
    comp.nodeX, comp.nodeY = 100, 0
    comp.inputConnectors[0].connect(s44)
    acc.module.start()
    return zone


_zone = _setup()
print("[WEDOAUDIO приёмка] %s · %s · сборка %s" % (_zone.path, os.path.basename(TOX), BUILD))
