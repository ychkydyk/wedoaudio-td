# -*- coding: utf-8 -*-
"""Собрать переносимый WEDOAUDIO.tox с нуля. ЗАПУСКАТЬ ВНУТРИ TOUCHDESIGNER.

Зачем скриптом, а не руками: тох, собранный вручную, — это слепок одной машины.
Собранный скриптом воспроизводится у кого угодно, а расхождения видны в диффе.

Запуск (текстпорт TD):
    import os; os.environ['WEDOAUDIO_REPO'] = r'C:/путь/к/wedoaudio'
    exec(open(r'C:/путь/к/wedoaudio/tools/build_portable.py', encoding='utf-8').read())

Что получится: /project1/WEDOAUDIO_build/WEDOAUDIO — самодостаточный компонент,
и рядом файл WEDOAUDIO_4.2.1_portable.tox.

ПРАВИЛА ПЕРЕНОСИМОСТИ, которым следует сборка:
  · ни одного абсолютного пути наружу — только входы и параметры;
  · enableexternaltox выключен: внешний тох это ссылка, а не содержимое;
  · ассеты в VFS, не на диске;
  · ни одного сетевого узла внутри: компонент не занимает портов на чужой машине;
  · частота дискретизации берётся у входа, а не назначается на глаз;
  · вся настройка в одной странице параметров, читается один раз в кадр;
  · у каждого параметра есть подсказка: компонент объясняет себя сам;
  · умолчание каждого параметра — рабочее значение, а не ноль;
  · состояние и публичный API — в расширении, заглавные имена наружу;
  · самопроверка: загрузка в пустой контейнер с ДРУГИМ именем.
"""
import os, td


def T(name):
    """Тип оператора по имени. В TD 2025 регистр не такой, как ждёшь:
    choptoTOP и dattoCHOP пишутся со строчной 'to'."""
    t = getattr(td, name, None)
    if t is None:
        raise RuntimeError("нет типа оператора: %s" % name)
    return t

REPO = os.environ.get("WEDOAUDIO_REPO", "")
if not REPO:
    try:
        REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        REPO = ""
if not REPO or not os.path.isdir(os.path.join(REPO, "src")):
    raise SystemExit("не найден репозиторий: задайте WEDOAUDIO_REPO")

CORE = open(os.path.join(REPO, "src", "wedoaudio_dsp.py"), encoding="utf-8").read()

# --------------------------------------------------------------------------- расширение
EXT = '''
"""Публичный интерфейс компонента. Заглавные имена видны снаружи, строчные нет."""


class Wedoaudio:
    def __init__(self, ownerComp):
        self.comp = ownerComp
        self._core = None
        self._sr = 0.0
        self.Version = "4.2.1"
        self.Status = "ожидание звука"

    # ---- Status держится в параметре, а не в поле объекта --------------------
    # Поле объекта не видно в интерфейсе: компонент мог молча не считать, и
    # снаружи это выглядело как тишина на входе. Один источник правды — параметр.
    @property
    def Status(self):
        return self.comp.par.Status.eval()

    @Status.setter
    def Status(self, v):
        self.comp.par.Status = str(v)[:160]

    # ---- публичное -------------------------------------------------------
    def Reset(self):
        """Сбросить состояние анализатора. Вызывать при смене источника."""
        self._core = None
        self._sr = 0.0
        self.Status = "сброшено"

    def Samplerate(self):
        """Частота дискретизации: из параметра, а 0 значит «взять у входа».

        Назначать её вручную нельзя: интерфейсы сплошь и рядом работают на
        48 кГц, а компонент, собранный под 44.1, раскладывает полосы по частотам
        со сдвигом 8.8% — кик уезжает в подвал, и никакой ошибки при этом нет."""
        p = float(self.comp.par.Samplerate.eval())
        if p > 0:
            return p
        for n in ("to_mono", "in_audio"):
            o = self.comp.op(n)
            if o is not None and getattr(o, "rate", 0):
                return float(o.rate)
        return 44100.0

    def Cfg(self):
        """Все настройки одним словарём, ровно один раз за кадр.
        Если каждый узел дёргает par сам, внутри кадра появляется рассинхрон.

        Умолчания здесь совпадают с проверенными умолчаниями ядра: словарь,
        собранный из нетронутой страницы параметров, обязан вести себя ровно
        как cfg=None, на котором прогнан стенд."""
        P = self.comp.par
        return {
            "gate": P.Inputgate.eval(),
            "agc_on": int(P.Agc.eval()),
            "agc_target": P.Agctarget.eval(),
            "agc_max": P.Agcmaxgain.eval(),
            "bass_hi": P.Bassmaxhz.eval(),
            "mid_hi": P.Midmaxhz.eval(),
            "high_hi": P.Highmaxhz.eval(),
            "bass_gain": P.Bassgain.eval(),
            "mid_gain": P.Midgain.eval(),
            "high_gain": P.Highgain.eval(),
            "kick_lo": P.Kickbandlo.eval(),
            "kick_hi": P.Kickbandhi.eval(),
            "snare_lo": P.Snarebandlo.eval(),
            "snare_hi": P.Snarebandhi.eval(),
            "kick_floor": P.Kickfloor.eval(),
            "snare_floor": P.Snarefloor.eval(),
            "beat_floor": P.Beatfloor.eval(),
            "kick_k": P.Kicksensitivity.eval(),
            "snare_k": P.Snaresensitivity.eval(),
            "beat_k": P.Beatsensitivity.eval(),
            "kick_refr": P.Kickrefractoryms.eval() / 1000.0,
            "snare_refr": P.Snarerefractoryms.eval() / 1000.0,
            "beat_refr": P.Beatrefractoryms.eval() / 1000.0,
            "kick_abs": P.Kickguard.eval(),
            "snare_abs": P.Snareguard.eval(),
            "decay": 0.08 + P.Smoothing.eval() * 0.5,
            "onset_rel": int(P.Onsetrel.eval()),
            "tempo_ac": int(P.Tempoac.eval()),
            "tempo_fix": int(P.Tempofix.eval()),
        }

    def Selftest(self):
        """Проверить себя, не выходя за свои границы. Возвращает (ok, отчёт)."""
        c = self.comp
        problems = []

        # Структурное и рабочее — разные вещи. Сразу после сборки звука на входе
        # ещё нет, таблица признаков пуста, и жалоба на число каналов означала бы
        # «непереносим», хотя переносимость тут ни при чём.
        out = c.op("out_features")
        tbl = c.op("features_table")
        if out is None:
            problems.append("нет выхода out_features")
        elif (tbl is not None and tbl.numRows >= 20 and out.numChans < 20
              and out.cookFrame >= tbl.cookFrame):
            # сравнивать можно только когда CHOP пересчитался ПОСЛЕ таблицы:
            # иначе он честно отстаёт на кадр, и проверка ругается на исправный
            # компонент сразу после сборки.
            problems.append("анализ идёт, но out_features отдаёт %d каналов" % out.numChans)
        if c.par.enableexternaltox.eval():
            problems.append("включён внешний tox — компонент непереносим")
        if self.Samplerate() <= 0:
            problems.append("частота дискретизации нулевая")

        # Геометрия спектра — часть контракта с ядром. Ядро переводит герцы в
        # номер бина линейно; логарифмическая ось или подъём верхов означают,
        # что оно считает не по тем частотам, и молча: ошибки не будет,
        # просто кик перестанет срабатывать.
        sp = c.op("spectrum_raw")
        if sp is None:
            problems.append("нет spectrum_raw")
        else:
            if float(sp.par.frequencylog.eval()) != 0:
                problems.append("у spectrum_raw логарифмическая ось частот — ядро ждёт линейную")
            if float(sp.par.highfreqboost.eval()) != 0:
                problems.append("у spectrum_raw включён подъём верхов — полосы будут врать")
            if sp.numSamples and abs(sp.numSamples - 1024) > 1:
                problems.append("длина спектра %d, ядро проверено на 1024" % sp.numSamples)

        # findChildren(depth=N) отбирает РОВНО глубину N, а не «до N». Пока здесь
        # стояло depth=4, проверка обходила почти весь компонент и всегда была
        # «чисто». Глубина задаётся через maxDepth.
        kids = c.findChildren(maxDepth=10)

        for o in kids:
            for p in o.pars():
                try:
                    if p.mode.name == "EXPRESSION" and p.expr and "/project1/" in p.expr:
                        problems.append("абсолютный путь в выражении: %s.%s" % (o.name, p.name))
                except Exception:
                    pass
                # хранимое значение, а не вычисленное: у ссылочных параметров
                # eval() отдаёт разрешённый абсолютный путь даже у относительной
                # ссылки, и на этом легко насчитать сотню несуществующих бед.
                try:
                    v = p.val
                except Exception:
                    v = None
                if isinstance(v, str) and v.startswith("/") and not v.startswith("/sys"):
                    if not v.startswith(c.path):
                        problems.append("ссылка наружу: %s.%s = %s" % (o.name, p.name, v))

        net = [o.name for o in kids
               if o.type in ("oscin", "oscout", "udpin", "udpout", "tcpip",
                             "webserver", "touchin", "touchout", "ndiin", "ndiout")]
        if net:
            problems.append("сетевые узлы внутри компонента: %s" % ", ".join(net))

        zero = [p.name for p in c.customPars
                if p.style in ("Float", "Int") and p.default == 0 and p.eval() != 0]
        if zero:
            problems.append("умолчание 0 при рабочем значении: %s" % ", ".join(zero))

        self.Status = "самопроверка пройдена" if not problems else "замечаний: %d" % len(problems)
        return (not problems), problems

    # ---- внутреннее ------------------------------------------------------
    def _dsp(self):
        return self.comp.op("wedoaudio_dsp").module

    def core(self, sr):
        if self._core is None or sr != self._sr:
            self._core = self._dsp().WedoAudio(sr=sr)
            self._sr = sr
        return self._core
'''

# --------------------------------------------------------------------------- анализ
ANALYZE = '''
"""Одна точка входа: раз в кадр читаем спектр, считаем признаки, кладём в таблицу.
Никаких op() наружу компонента — всё берётся от me.parent().

Ошибки НЕ глотаются: любая записывается в параметр Status, иначе компонент
молча перестаёт считать и выглядит как тишина. Это уже случалось."""

import numpy as np

_S = {"tick": 0}


def onFrameStart(frame):
    comp = me.parent()
    _S["tick"] += 1
    every = max(1, int(comp.par.Rate.eval()))
    if _S["tick"] % every:
        return

    ext = comp.ext.Wedoaudio
    try:
        spec = comp.op("spectrum_raw")
        if not spec or not spec.numChans:
            ext.Status = "нет спектра — подключите звук ко входу"
            return
        mag = np.array(spec.chans()[0].vals, dtype=float)

        lvl = comp.op("level_named")
        rms = float(lvl["rms"]) if (lvl and lvl.numChans) else 0.0

        sr = ext.Samplerate()
        core = ext.core(sr)
        # dt меряется по часам, а заявленная частота кадров — только запасной
        # вариант на первый кадр. project.cookRate это ЦЕЛЬ, а не факт: под
        # нагрузкой проект идёт медленнее, чем объявил, и записанные грабли
        # проекта ровно об этом. Разница уходит прямо в темп и в скорость
        # спада огибающих, поэтому спрашиваем часы, а не настройку.
        now = absTime.seconds
        last = _S.get("last_t", -1.0)
        _S["last_t"] = now
        dt = (now - last) if last >= 0 else 0.0
        if dt <= 0 or dt > 0.5:
            dt = every / max(1.0, project.cookRate)
        feat = core.process(mag, rms, now, dt, ext.Cfg())

        tbl = comp.op("features_table")
        tbl.clear()
        for k, v in feat.items():
            tbl.appendRow([k, float(v)])
        ext.Status = "работает · %.0f Гц · %d признаков" % (sr, len(feat))
    except Exception as e:
        ext.Status = "ошибка: %s" % e
        debug("[WEDOAUDIO]", e)
'''

# --------------------------------------------------------------------------- сброс
ONPULSE = '''
"""Кнопка Reset обязана что-то делать. Пока её никто не слушал, она была
украшением: пользователь жал, ничего не менялось, и это выглядело как
неисправность анализатора."""


def onPulse(par):
    if par.name == "Resetstate":
        par.owner.ext.Wedoaudio.Reset()
'''

AGENTS = """# WEDOAUDIO 4.2.1

Анализатор звука для TouchDesigner. Самодостаточен: ни одной ссылки наружу.

## Как пользоваться
- вход: подключите аудио-CHOP ко входу компонента (моно сведётся само);
- выходы: `out_features` (CHOP, 22+ именованных канала), `out_spectrum` (TOP);
- настройки: одна страница `WEDOAUDIO`, у каждого параметра есть подсказка;
- `Samplerate = 0` означает «взять частоту у входа» — так и оставьте.

## Публичный интерфейс расширения
- `op('WEDOAUDIO').Reset()` — сбросить состояние при смене источника;
- `op('WEDOAUDIO').Selftest()` — вернуть `(ok, список замечаний)`;
- `op('WEDOAUDIO').Cfg()` — текущие настройки одним словарём;
- `op('WEDOAUDIO').Samplerate()` — действующая частота дискретизации.

## Чего не делать
- не включать `enableexternaltox`: внешний тох — это ссылка, а не содержимое;
- не заводить внутри сетевых узлов: компонент не занимает портов;
- не ставить параметрам умолчание 0, если рабочее значение не ноль:
  ядро получает ключ явно, и запасное значение `c.get(k, 40)` уже не спасает.

Лицензия CC BY-NC · github.com/ychkydyk/wedoaudio
"""


def par_set(res, default=None, help=None):
    """appendFloat(...)[0].default = X задаёт УМОЛЧАНИЕ, но не значение:
    свежесозданный параметр остаётся нулём. Нуль в Samplerate — это нулевая
    частота Найквиста и деление на ноль в первом же кадре. Ставим оба."""
    p = res[0]
    if default is not None:
        p.default = default
        p.val = default
    if help:
        p.help = help
    return p


def build():
    root = op("/project1")
    holder = root.op("WEDOAUDIO_build") or root.create(T("baseCOMP"), "WEDOAUDIO_build")
    old = holder.op("WEDOAUDIO")
    if old:
        old.destroy()
    c = holder.create(T("baseCOMP"), "WEDOAUDIO")
    c.nodeX, c.nodeY = 0, 0
    c.color = (0.08, 0.09, 0.095)

    # --- страница параметров: один порядок — порядок сигнала -----------------
    pg = c.appendCustomPage("WEDOAUDIO")
    par_set(pg.appendStr("Version", label="Version"), "4.2.1",
            "Версия ядра и разводки компонента.")
    c.par.Version.readOnly = True
    par_set(pg.appendStr("Status", label="Status"), "ожидание звука",
            "Что компонент делает прямо сейчас. Сюда же попадают ошибки анализа.")
    c.par.Status.readOnly = True

    par_set(pg.appendFloat("Samplerate", label="Sample Rate (0 = from input)"), 0,
            "0 — взять частоту у входного CHOP. Ставьте число только если знаете, "
            "что вход врёт: на 48 кГц при заданных 44100 все полосы уезжают на 8.8%.")
    par_set(pg.appendInt("Rate", label="Analyse Every N Frames"), 1,
            "Считать не каждый кадр, а каждый N-й. Экономит кадр на слабой машине; "
            "dt пересчитывается, темп не врёт.")
    par_set(pg.appendFloat("Inputgate", label="Input Gate (RMS)"), 0.001,
            "Ниже этого RMS вход считается тишиной и детекторы гасятся.")

    par_set(pg.appendToggle("Agc", label="Auto Gain"), True,
            "Автоматически подтягивать тихий материал до рабочего уровня.")
    par_set(pg.appendFloat("Agctarget", label="AGC Target"), 0.7,
            "Уровень, к которому стремится автоусиление.")
    par_set(pg.appendFloat("Agcmaxgain", label="AGC Max Gain"), 12,
            "Потолок автоусиления. Выше — начинает вытягивать шум.")

    par_set(pg.appendFloat("Bassmaxhz", label="Bass Max Hz"), 250,
            "Верхняя граница полосы баса.")
    par_set(pg.appendFloat("Midmaxhz", label="Mid Max Hz"), 2000,
            "Верхняя граница середины.")
    par_set(pg.appendFloat("Highmaxhz", label="High Max Hz"), 20000,
            "Верхняя граница верха.")
    par_set(pg.appendFloat("Bassgain", label="Bass Gain"), 1,
            "Добавка к готовому значению полосы баса.")
    par_set(pg.appendFloat("Midgain", label="Mid Gain"), 1,
            "Добавка к готовому значению середины.")
    par_set(pg.appendFloat("Highgain", label="High Gain"), 1,
            "Добавка к готовому значению верха.")

    par_set(pg.appendFloat("Kickbandlo", label="Kick Band Lo Hz"), 40,
            "Нижняя граница полосы, в которой ищется кик. Ноль недопустим.")
    par_set(pg.appendFloat("Kickbandhi", label="Kick Band Hi Hz"), 120,
            "Верхняя граница полосы кика.")
    par_set(pg.appendFloat("Snarebandlo", label="Snare Band Lo Hz"), 1800,
            "Нижняя граница полосы снейра.")
    par_set(pg.appendFloat("Snarebandhi", label="Snare Band Hi Hz"), 6000,
            "Верхняя граница полосы снейра.")

    par_set(pg.appendFloat("Kickfloor", label="Kick Floor"), 0.40,
            "Доля собственного затухающего пика полосы, ниже которой удар не считается.")
    par_set(pg.appendFloat("Snarefloor", label="Snare Floor"), 0.35,
            "То же для снейра.")
    par_set(pg.appendFloat("Beatfloor", label="Beat Floor"), 0.10,
            "То же для общей доли.")
    par_set(pg.appendFloat("Kicksensitivity", label="Kick Sensitivity"), 1.6,
            "Во сколько раз всплеск должен превысить недавний фон. Меньше — чувствительнее.")
    par_set(pg.appendFloat("Snaresensitivity", label="Snare Sensitivity"), 1.6,
            "То же для снейра.")
    par_set(pg.appendFloat("Beatsensitivity", label="Beat Sensitivity"), 1.6,
            "То же для доли.")
    par_set(pg.appendFloat("Kickrefractoryms", label="Kick Refractory (ms)"), 110,
            "Сколько миллисекунд после удара новый не засчитывается.")
    par_set(pg.appendFloat("Snarerefractoryms", label="Snare Refractory (ms)"), 110,
            "То же для снейра.")
    par_set(pg.appendFloat("Beatrefractoryms", label="Beat Refractory (ms)"), 110,
            "То же для доли.")
    par_set(pg.appendFloat("Kickguard", label="Kick Absolute Guard"), 0.05,
            "Небольшой абсолютный порог поверх относительного: отсекает срабатывания "
            "на почти тишине. Замеренная полка шума — 0.05.")
    par_set(pg.appendFloat("Snareguard", label="Snare Absolute Guard"), 0.055,
            "То же для снейра. Замеренная полка — 0.055.")

    par_set(pg.appendFloat("Smoothing", label="Envelope Smoothing"), 0.08,
            "Насколько медленно спадает огибающая. Больше — плавнее и вязче.")
    par_set(pg.appendToggle("Onsetrel", label="Onsets 4.2.1 (off = 4.2.0)"), True,
            "Относительные онсеты 4.2.1. Выключить — вернуть поведение 4.2.0 в точности.")
    par_set(pg.appendToggle("Tempoac", label="Tempo by autocorrelation"), True,
            "Темп автокорреляцией новизны. Выключить — вернуть счёт по интервалам.")
    par_set(pg.appendToggle("Tempofix", label="Fix tempo octave"), True,
            "Подтягивать явные удвоения и половины темпа к основному.")
    pg.appendPulse("Resetstate", label="Reset")
    c.par.Resetstate.help = "Сбросить состояние анализатора. Жать при смене источника."

    # --- сеть: вход слева, выход справа, вспомогательное вниз ---------------
    ai = c.create(T("inCHOP"), "in_audio");         ai.nodeX, ai.nodeY = -700, 0
    # сведение в моно ДО анализа: стерео давало chan1/chan2, и чтение "первого
    # канала" тихо означало бы "только левый". Один канал — одна правда.
    mono = c.create(T("mathCHOP"), "to_mono");      mono.nodeX, mono.nodeY = -600, 0
    mono.par.chopop = "average"
    mono.inputConnectors[0].connect(ai)

    lvl = c.create(T("analyzeCHOP"), "level");      lvl.nodeX, lvl.nodeY = -450, -140
    lvl.par.function = "rmspower"
    lvl.inputConnectors[0].connect(mono)
    rn = c.create(T("renameCHOP"), "level_named");  rn.nodeX, rn.nodeY = -320, -140
    rn.par.renamefrom, rn.par.renameto = "*", "rms"
    rn.inputConnectors[0].connect(lvl)

    sp = c.create(T("audiospectrumCHOP"), "spectrum_raw"); sp.nodeX, sp.nodeY = -450, 0
    sp.inputConnectors[0].connect(mono)
    # ГЕОМЕТРИЯ СПЕКТРА — ЧАСТЬ КОНТРАКТА С ЯДРОМ, а не украшение.
    # По умолчанию Audio Spectrum CHOP настроен ДЛЯ ПОКАЗА: frequencylog = 1
    # (логарифмическая ось частот) и highfreqboost = 0.75 (подъём верхов).
    # Ядро же переводит герцы в номер бина ЛИНЕЙНО и держит абсолютные пороги,
    # снятые на спектре из 1025 бинов. Замер на живом сигнале: с умолчаниями
    # полоса кика 40–120 Гц читалась 0.0025 при пороге 0.05 — кик был мёртв
    # структурно, а снейр наоборот срабатывал вдвое чаще нужного (0.44).
    # После приведения к линейной оси на том же сигнале: кик 1.003, снейр 0.078,
    # то есть то же соотношение, что в офлайновом стенде (0.163 против 0.0079).
    sp.par.frequencylog = 0
    sp.par.highfreqboost = 0
    sp.par.fftsize = "2048"
    sp.par.outputmenu = "setmanually"
    sp.par.outlength = 1024

    dsp = c.create(T("textDAT"), "wedoaudio_dsp");  dsp.nodeX, dsp.nodeY = -520, 180
    dsp.text = CORE
    dsp.color = (0.11, 0.12, 0.125)

    tbl = c.create(T("tableDAT"), "features_table"); tbl.nodeX, tbl.nodeY = -200, 0
    d2c = c.create(T("dattoCHOP"), "features_chop"); d2c.nodeX, d2c.nodeY = -40, 0
    d2c.par.dat = "features_table"
    # каждая строка таблицы — отдельный канал, имя берётся из первого столбца.
    # Значение "name" (в единственном числе) не входит в меню и молча
    # откатывается в "ignored" — каналы тогда зовутся chan1..chanN.
    d2c.par.output = "chanperrow"
    d2c.par.firstcolumn = "names"
    d2c.par.firstrow = "values"
    out = c.create(T("outCHOP"), "out_features");   out.nodeX, out.nodeY = 140, 0
    out.inputConnectors[0].connect(d2c)
    out.color = (0.62, 0.64, 0.645)

    spec_tex = c.create(T("choptoTOP"), "spectrum128_tex"); spec_tex.nodeX, spec_tex.nodeY = 140, -180
    spec_tex.par.chop = "spectrum_raw"
    tex_out = c.create(T("outTOP"), "out_spectrum"); tex_out.nodeX, tex_out.nodeY = 300, -180
    tex_out.inputConnectors[0].connect(spec_tex)
    tex_out.color = (0.62, 0.64, 0.645)

    an = c.create(T("executeDAT"), "analyze");      an.nodeX, an.nodeY = -200, 180
    an.text = ANALYZE
    an.par.framestart = True
    an.color = (0.11, 0.12, 0.125)

    pe = c.create(T("parameterexecuteDAT"), "on_reset"); pe.nodeX, pe.nodeY = -40, 180
    pe.text = ONPULSE
    pe.par.op = ".."
    pe.par.pars = "Resetstate"
    # у parexec параметр называется onpulse, а не pulse: на 'pulse' сборка
    # падает с "ParCollection object has no attribute", и без явного custom
    # DAT слушает только встроенные параметры, а Resetstate — свой.
    pe.par.onpulse = True
    pe.par.custom = True
    pe.par.valuechange = False
    pe.color = (0.11, 0.12, 0.125)

    ex = c.create(T("textDAT"), "Wedoaudio_ext");   ex.nodeX, ex.nodeY = -200, 300
    ex.text = EXT
    # ВАЖНО: именно me.op(), а не op(). В параметре расширения голый op('имя')
    # не находит собственных детей компонента — extensions приходит [None],
    # а ошибка нигде не показывается. Проверено на TD 2025.32460.
    c.par.extension1 = "me.op('Wedoaudio_ext').module.Wedoaudio(me)"
    try:
        c.par.promoteextension1 = True
    except Exception:
        pass

    doc = c.create(T("textDAT"), "agents_md");      doc.nodeX, doc.nodeY = -700, 300
    doc.text = AGENTS

    # видимость: узлы, по которым читают состояние, должны быть видны без
    # захода внутрь — иначе отладка начинается с открывания каждого оператора.
    for o in (rn, sp, tbl, out, spec_tex, tex_out):
        try:
            o.viewer = True
        except Exception:
            pass

    # плитка самого компонента показывает свою же шину признаков: поставил в
    # сеть — и сразу видно, шевелится оно или нет. Путь ОБЯЗАН начинаться с
    # './': у ссылочных параметров голое имя ищет СОСЕДА, а не ребёнка, и
    # молча остаётся пустым.
    c.par.opviewer = "./out_features"
    c.viewer = True

    # --- переносимость ------------------------------------------------------
    c.par.enableexternaltox = False
    c.par.externaltox = ""
    c.par.parentshortcut = "Wedoaudio"
    try:
        c.par.extname1 = "Wedoaudio"
    except Exception:
        pass
    c.initializeExtensions()

    return c


comp = build()
print("собран:", comp.path)
try:
    ok, problems = comp.ext.Wedoaudio.Selftest()
    print("самопроверка:", "чисто" if ok else "замечания")
    for p in problems[:12]:
        print("   ·", p)
except Exception as e:
    print("самопроверка не отработала:", e)

# Имя несёт сборку TD, в которой тох записан. Тох, сохранённый в новой сборке,
# в старой открывается с предупреждением или не открывается вовсе, а по имени
# файла это не видно — поэтому сборка пишется в имя, а не в примечание.
build = getattr(app, "build", "unknown")
out_path = os.path.join(REPO, "WEDOAUDIO_4.2.1_TD%s.tox" % build)
comp.save(out_path)
print("сохранён тох:", out_path, "· сборка TD:", build)
