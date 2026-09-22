# WEDOAUDIO — audio analysis for TouchDesigner

One component turns sound into **22 per-frame features plus loudness to ITU-R BS.1770-4 / EBU R128**
and publishes them as a named CHOP bus and a spectrum TOP. The core is plain numpy, embedded inside
the component: no plugins, no binaries, nothing installed into your system.

The component is **self-contained** — no external references, no network nodes, assets inside.
Drop it into your network, feed it audio, it runs.

> This is an analysis instrument. Visuals, shaders and control surfaces are not part of it and are
> not published. The component hands you numbers; what you do with them is yours.

Russian documentation, which is longer and kept in sync first, is in [README.md](README.md).

---

## Version

Released file: `WEDOAUDIO_4.3.0_TD2025.32460.tox`, built and accepted inside
**TouchDesigner 2025.32460** — fresh load under a different name in a foreign container, reference
loudness signal, right stereo channel, input loss and recovery, sample rate switched to 48 kHz,
Reset, deliberate frame hitches. The acceptance run is in `native_acceptance.json`. No other TD
build was tested for 4.3.0.

**Branch `release/4.3.1` is open and is the thing worth testing.** Source only — the `.tox` is not
built and not accepted in TouchDesigner yet, so the released component stays 4.3.0.

---

## Quick start

1. Drag the `.tox` into your network. A `WEDOAUDIO` node appears; the feature bus on its tile moves
   as soon as analysis runs.
2. Feed any audio CHOP into its input. Stereo is summed to mono automatically; sample rate is taken
   from the input.
3. Read `out_features` (CHOP, 22 named channels plus six loudness channels) and `out_spectrum`
   (TOP, for shaders).
4. Read the `Status` parameter. It says what is happening in words, including
   `interval not measured (first frame or gap)`. There are no silent failures: on the first frame
   there is nothing to subtract, and the component will not substitute the declared rate for a
   measurement.

From Python:

```python
a = op('WEDOAUDIO')
a.op('out_features')['kick']    # channel value
a.Reset()                       # drop state when the source changes
ok, notes = a.Selftest()        # self-check: (True, []) when healthy
a.Loudness()                    # loudness as a dict; unmeasured is None, not a number
```

## The feature bus

| group | channels | what it is |
|---|---|---|
| level | `rms` `silence` | level after AGC · silence flag on the raw input |
| bands | `bass` `mid` `high` | peak FFT bin per band, gain applied; edges are parameters |
| timbre | `centroid` `flatness` `rolloff` | spectral centre of gravity · noisiness · roll-off frequency |
| novelty | `flux` | rectified spectral flux, the basis of every detector |
| kick | `kick` `rawkick` `kickband` | decaying envelope · single-frame edge · band energy |
| snare | `snare` `rawsnare` `snareband` | the same for snare |
| beat | `beat` `rawbeat` | wideband onset with an adaptive threshold |
| tempo | `bpm` `beatphase` | tempo by autocorrelation of novelty · beat phase 0…1 |
| dynamics | `transient` `sustain` | percussive against sustained |
| telemetry | `agcgain` | current auto-level gain |

Loudness adds `lufsm` `lufss` `lufsi` `lra` `truepeak` `truepeakmax`, computed on the stereo
**samples** before the mono sum, strictly after the base 22 so existing indices never shift.
**`-100` means "not measured", not "very quiet"** — instantaneous loudness needs 400 ms of audio,
short-term 3 s, range 4 s. From Python the same value is an honest `None`.

---

## Verify it yourself, without us

```
python src/_wedoaudio_test.py          # core, 14 tests
python src/_wedoaudio_timbre_test.py   # timbre and tonality, 14 tests
python src/_wedoaudio_bench.py         # synthesised tracks, 18 runs
python src/_wedoaudio_falsekick.py     # material with no drums must produce no kicks
python src/_wedoaudio_realtest.py      # real music, seven containers (needs librosa + ffmpeg)
```

The first four need only numpy. The bench synthesises its own audio, so a run is deterministic:
a difference is a regression, not a flaky test. Each run prints precision / recall / F and exits
non-zero if it falls below the recorded baseline.

## What is still broken, with numbers

- **We own no real music with hand annotation at all.** Level B of `_realtest.py` compares us to
  librosa, which is a second opinion, not ground truth. This is the largest hole in the whole
  verification, and one annotated track closes more of it than any amount of code.
- Onset resolution is 16 ms because analysis runs on 60 Hz frames. Sample-tight work needs a
  separate path.
- Tempo on breakbeat reads as half (88 instead of 174). Declared range is 68–200 BPM.
- AGC holds about a twelvefold input range; quieter than that, response falls with the input.
- Loudness is measured to BS.1770-4 but the component is not a certified meter.
- **False kicks on non-rhythmic material** — measured on 96 field recordings (wind, rain, sea,
  thunder, engines, a train) and eight synthesised drones: the 4.3.0 detector invented
  **2.33 kicks per second** on recordings and 1.46 on drones. Fixed on `release/4.3.1` down to
  **0.38** and **0.18** by adding the *signed* band movement to the threshold — noise moves a band
  up and down every frame, a drum track between hits barely moves at all. Cost: at 48 kHz recall on
  one real-kick set fell 0.94 → 0.82 while F rose 0.86 → 0.90. `kick_spread = 0` restores the
  4.3.0 detector exactly. What remains is thunder and sea, which contain genuine low-frequency hits.

---

## How to contribute

Pull request or issue at **github.com/ychkydyk/wedoaudio-td**. Attach the output of the commands
above before and after your change; no separate description is needed.

**Changes come back here, into this repository.** Not because of ownership theatre — because the
released artifact is a single binary `.tox` built by `tools/build_portable.py` from `src/`, and a
`.tox` edited in your copy cannot be merged, diffed or reviewed by anyone. Source in one place is
the only arrangement where your fix reaches everyone else's component. Fork freely, run it anywhere,
use it commercially-free under the licence — but send the fix as a patch to `src/`, and it ships in
the next build with your name on the commit.

The most valuable contribution is **a track on which the bench is wrong** — not "it feels off", but
a reproducible case.

Four rules, so there are no surprises:

- **Nothing is fixed by eye.** A threshold change without a measurement is not accepted, even when
  it is obviously right. A measurement means the whole bench, not the one style the change was for.
- **A regression outweighs an improvement.** If a change fixes one style and breaks another, it
  waits until the trade is shown to be worth it. Two such stopping points are written up in the
  Russian README.
- **Default behaviour changes only behind a flag.** Everything new lives in `cfg`, the old path
  stays one line away. Rigs in production must not lose an evening.
- **A limitation that could not be closed goes into the README,** in plain words, with a number.
  An open limitation is not a disgrace; an open limitation left unsaid is.

Licence CC BY-NC. Made by WEDOIT, a media installation and realtime graphics studio; the agent
publishing here runs with a human operator in the loop, disclosed.
