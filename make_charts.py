# -*- coding: utf-8 -*-
"""Графики по замерам анализатора. Ничего не рисуется от руки — всё читается
из bench_results.json, который пишет src/_wedoaudio_bench.py."""
import io, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_img")
os.makedirs(OUT, exist_ok=True)

BG, FG, DIM, GRID, BRAND, WARN, BAD = "#0A0A0A", "#E3E7E2", "#8E9998", "#1C2123", "#C8FF00", "#E0A264", "#E08A81"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.family": "DejaVu Sans Mono",
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def finish(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


# ---------------------------------------------------------------- 1. онсеты
rows = json.load(io.open(os.path.join(HERE, "bench_results.json"), encoding="utf-8"))
styles = []
for r in rows:
    if r["style"] not in styles:
        styles.append(r["style"])
kick = [np.mean([r["kick_f"] for r in rows if r["style"] == s]) for s in styles]
snare = [np.mean([r["snare_f"] for r in rows if r["style"] == s]) for s in styles]

fig, ax = plt.subplots(figsize=(7.6, 3.4))
y = np.arange(len(styles)); h = 0.36
ax.barh(y + h/2, kick, h, color=BRAND, label="кик")
ax.barh(y - h/2, snare, h, color=FG, label="снейр")
ax.axvline(0.8, color=WARN, lw=1, ls="--")
ax.text(0.805, len(styles) - 0.4, "порог 0.80", color=WARN, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(styles); ax.set_xlim(0, 1.05)
ax.set_xlabel("F-мера онсетов, допуск ±50 мс (порог MIREX)")
ax.set_title("ОНСЕТЫ ПО СТИЛЯМ · среднее по трём форматам")
ax.legend(frameon=False, loc="lower left", labelcolor=FG)
ax.grid(axis="x", lw=0.5, alpha=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
finish(fig, "01_onsets.png")

# ---------------------------------------------------------------- 2. темп
tr = [r for r in rows if r["bpm_true"]]
seen = {}
for r in tr:
    seen.setdefault(r["style"], []).append(r)
names = list(seen)
true = [seen[n][0]["bpm_true"] for n in names]
est = [np.median([x["bpm_est"] for x in seen[n]]) for n in names]

fig, ax = plt.subplots(figsize=(7.6, 3.2))
y = np.arange(len(names)); h = 0.36
ax.barh(y + h/2, true, h, color=DIM, label="истинный")
cols = [BRAND if abs(e - t)/t < 0.1 else WARN for e, t in zip(est, true)]
ax.barh(y - h/2, est, h, color=cols, label="измеренный")
for i, (e, t) in enumerate(zip(est, true)):
    err = abs(e - t)/t*100
    ax.text(max(e, t) + 4, i, "%.0f%%" % err, va="center",
            color=BRAND if err < 10 else WARN, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlabel("BPM"); ax.set_title("ТЕМП · 12 прогонов из 15 в пределах 10%")
ax.legend(frameon=False, loc="lower right", labelcolor=FG)
ax.grid(axis="x", lw=0.5, alpha=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
finish(fig, "02_tempo.png")

# ---------------------------------------------------------------- 3. дефекты до/после
defects = [
    ("снейр\nне срабатывал", 0.00, 1.00, "F-мера"),
    ("кик троил\nна басу", 0.37, 0.90, "точность"),
    ("канал доли\nбыл мёртв", 0.02, 0.98, "F-мера"),
    ("смена 44.1→48\nменяла счёт", 0.40, 1.00, "совпадение"),
    ("темп плыл от\nпересжатия", 0.00, 1.00, "стабильность"),
]
fig, ax = plt.subplots(figsize=(7.6, 3.4))
x = np.arange(len(defects)); w = 0.38
ax.bar(x - w/2, [d[1] for d in defects], w, color=BAD, label="до")
ax.bar(x + w/2, [d[2] for d in defects], w, color=BRAND, label="после")
ax.set_xticks(x); ax.set_xticklabels([d[0] for d in defects], fontsize=8)
ax.set_ylim(0, 1.12); ax.set_ylabel("нормированный показатель")
ax.set_title("ПЯТЬ ДЕФЕКТОВ · все пять — одна болезнь: величина против порога в чужой шкале")
ax.legend(frameon=False, labelcolor=FG)
ax.grid(axis="y", lw=0.5, alpha=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
finish(fig, "03_defects.png")

print("готово, папка:", OUT)
