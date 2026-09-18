# WEDOAUDIO 4.3.0 — verification

Prepared 2026-09-18. Not published. Build target: TouchDesigner **2025.32460**.

4.3.0 merges the two 4.2.2 lines (the first-frame fix and the verified input-lifecycle candidate) and adds
a sample-domain loudness meter: ITU-R BS.1770-4 / EBU R128. The 22-channel interface is preserved; six
loudness channels are appended after it and can be switched off, which leaves the bus exactly as before.
`src/wedoaudio_dsp.py` is byte-identical to the verified 4.2.2-rc1 core. The portable component contains
the analyzer only, with no network service, camera or performance scene.

## Reproducible offline checks

```text
python src/_wedoaudio_test.py
python src/_wedoaudio_timbre_test.py
python src/_wedoaudio_loudness_test.py
python src/_wedoaudio_bench.py
python src/test_release_controls.py
```

Results: core 14/14; timbre 14/14; loudness 34/34; lifecycle/validation 5/5. Synthetic music benchmark:
kick 18/18 and snare 18/18 at F >= 0.80; tempo 12/15, unchanged from the recorded baseline. This is not a
claim of universal music accuracy.

Loudness conformance uses synthetic signals built to the EBU Tech 3341 (cases 1–5, 9, 12, 15–19) and
Tech 3342 (cases 1–2) definitions, inside their stated tolerances. K-weighting coefficients are derived
from the analog prototype for any sample rate; at 48 kHz they match the table in the Recommendation to
1e-15, and the filter output agrees with `scipy.signal.lfilter` to 1.2e-9 LU. The true-peak interpolator is
generated, not copied, and measured against 64x spectral reconstruction: worst under-read 0.15 dB across
0.05…0.45 fs. The result does not depend on how samples are cut into blocks (difference 0.0).
The meter is **not certified**; passing these signals is a check, not an attestation.

## Native acceptance

`tools/native_acceptance.py`, run inside TouchDesigner. It loads the written `.tox` into a clean container
under a different name, does not prime its graph by hand, and drives it for half a minute: steady tone,
one channel only, input loss and return, a 48 kHz source, loudness off, true peak off, reset, and two
deliberate frame hitches. The source is a generator, so the check does not depend on an audio device and
has a known answer. The outcome is written to `native_acceptance.json` together with the SHA-256 of the
tested file. Result: **22 of 22 entries passed** — the 21 checks below plus the presence of every snapshot.

1. Cold load gives 28 named channels by the third frame; the first 22 keep their names and order.
2. Stereo 1 kHz at -23 dBFS reads -22.99 LUFS momentary, short-term and integrated (tolerance 0.1).
3. Range of the steady tone is 0.0 LU; true peak -23.00 dBTP.
4. A meter that starts in the middle of a tone invents no peak.
5. A signal present only on the right channel is heard; the internal downmix yields one channel.
6. The same tone in one channel reads -26.00 LUFS.
7. Input loss keeps the interface, gives neutral values, and loudness becomes "not measured".
8. Reconnect resumes analysis.
9. A 48 kHz input updates both the analyzer and the meter; loudness is unchanged.
10. Per-frame sample accounting over the whole scenario: no frame fed wrongly, no drift against the TD clock.
10a. The sample-loss watchdog stays silent on a 0.1 s hitch, which TouchDesigner catches up by itself.
10b. It reports a 0.5 s hitch: "3.0% of samples lost", which is 0.3 s out of a 10 s span.
11. With `Loudness` off the bus is exactly the previous 22 channels.
12. With `Truepeak` off the peak is "not measured" while loudness keeps running.
13. Reset clears analyzer state and restarts the integrated value.
14. API values are plain floats or `None`.
15. Native Selftest returns `(True, [])` under a foreign name in a foreign container.
16. No operator errors or warnings.
17. Embedded sources are identical to the files in `src/`.
18. The test project holds 60.00 fps for 20.8 s. This does not establish the cost in a full show rig.
19. Loudness costs 0.38 ms per frame on a fresh meter and 0.44 ms after a simulated two-hour set (worst frame 1.0 ms).

Separately checked: with `Rate = 3` the meter is still fed every frame (465 990 samples against 466 726
by the wall clock over 10.6 s, no loss note).

## What native acceptance found that offline tests did not

- **Invented true peak after a reset.** The interpolator started from a zero-filled history and read its
  own start as an abrupt onset: -22.93 dBTP on a steady -23 dBFS tone, up to +1.06 dB at another phase.
  Fixed; an offline test now reproduces it and fails on the previous code.
- **Frame spikes late in a long set.** A list of 72 000 gating blocks was converted to an array ten times
  a second: 3.5–4.9 ms spikes after two hours. History moved to fixed rings and masked sums: worst frame
  1.1 ms, result identical to 1e-14 LU.
- **A loudness error was invisible**: it was overwritten by "running" within the same frame.
- **The first watchdog raised a false alarm on component load** (2.3%). Cause, measured: audio slices follow
  `absTime` exactly, but inside `onFrameStart` after a hitch `absTime` is still the old value while the slice
  is already complete. The watchdog now uses the wall clock over a doubled window.
- **TouchDesigner caps a time slice at 0.2 s.** A 0.4 s frame delivers 0.2 s of audio to every time-sliced
  CHOP and moves TD's own clock by the same amount, so by TD's clock nothing was lost. This is a property
  of the host, not of the component; the watchdog reports it because it compares against the wall clock.
  Also observed: a minimized TouchDesigner window stops cooking; the watchdog then reported 90.7% lost.

## Known limits

Drone/ambient false onsets, tempo ambiguity and short-loop accuracy remain limitations. `bpm` starts at 120
and stays there until a tempo is measured; the bus has no confidence channel, so a steady tone shows 120.
That predates this version and contradicts the project's own "not measured instead of a plausible number"
rule; it is recorded rather than changed silently, because consumers may divide by it.
Loudness is verified for mono and stereo only (the first two input channels are used); 5.1 weighting is not
claimed. The integrated value covers the last two hours. Timbre/key/MFCC source is available but is not
embedded in the portable component. Instrument separation is not included. Only the 2025.32460 build was
checked. No real-music loudness comparison against a reference meter has been made yet.
