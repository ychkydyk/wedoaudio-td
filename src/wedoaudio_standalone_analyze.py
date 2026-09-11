# LEGACY 4.2 wrapper, retained for history. The current portable component is built
# from ANALYZE in tools/build_portable.py. Do not wire this file into the 4.2.2 node.
# WEDOAUDIO standalone analyzer — Execute DAT (onFrameStart) embedded INSIDE the
# WEDOAUDIO .tox. Self-contained: reads the component's own audio_in + spectrum_raw,
# runs the embedded wedoaudio_dsp module, stores the feature dict on the component
# (parent().store('wedo', ...)). out_features / spectrum128 read that store. No
# dependency on any external rig. ~0.3 ms/frame, 30Hz throttle (cook-thread safe).
import numpy as np, math

_S = {'wa': None, 'last_t': -1.0, 'tick': 0, 'edges': None}

def _log_edges(n):
    lo, hi = math.log(20.0), math.log(20000.0)
    return [int(math.exp(lo + (hi - lo) * (i / 128.0))) for i in range(129)]

def _log128(arr):
    n = len(arr)
    if _S['edges'] is None:
        _S['edges'] = _log_edges(n)
    e = _S['edges']; out = []
    for i in range(128):
        a = max(0, e[i]); b = min(n, max(a + 1, e[i + 1]))
        out.append(round(float(arr[a:b].max()), 5) if b > a else 0.0)
    return out

def _cfg(comp):
    try:
        P = comp.par
        return {
            'gate': P.Inputgate.eval(), 'agc_on': int(P.Agc.eval()),
            'agc_target': P.Agctarget.eval(), 'agc_max': P.Agcmaxgain.eval(),
            'bass_hi': P.Bassmaxhz.eval(), 'mid_hi': P.Midmaxhz.eval(), 'high_hi': P.Highmaxhz.eval(),
            'bass_gain': P.Bassgain.eval(), 'mid_gain': P.Midgain.eval(), 'high_gain': P.Highgain.eval(),
            'kick_floor': P.Kickfloor.eval(), 'kick_k': 1.0 + P.Kickthreshold.eval() * 1.5, 'kick_refr': P.Kickrefractoryms.eval() / 1000.0,
            'snare_floor': P.Snarefloor.eval(), 'snare_k': 1.0 + P.Snarethreshold.eval() * 1.5, 'snare_refr': P.Snarerefractoryms.eval() / 1000.0,
            'beat_floor': P.Beatthreshold.eval(), 'beat_refr': P.Beatrefractoryms.eval() / 1000.0,
            'decay': 0.08 + P.Smoothing.eval() * 0.5,
        }
    except Exception:
        return None

def onFrameStart(frame):
    comp = me.parent()
    _S['tick'] += 1
    if _S['tick'] % 2:
        return  # 30Hz
    spec = comp.op('spectrum_raw')
    if not spec or not spec.numSamples:
        return
    try:
        dsp = comp.op('wedoaudio_dsp').module
        if _S['wa'] is None:
            _S['wa'] = dsp.WedoAudio(sr=44100)
        arr = np.abs(np.asarray(spec.numpyArray()[0], dtype=float))
        per = max(1, arr.shape[0] // 512)
        mag = arr[:per * 512].reshape(512, per).max(axis=1)
        # rms from the time-domain audio input
        rms = 0.0
        ain = comp.op('audio_in')
        if ain and ain.numSamples:
            a = np.asarray(ain.numpyArray()[0], dtype=float)
            rms = float(np.sqrt((a * a).mean()))
        t = absTime.frame / 60.0
        last = _S['last_t']
        dt = (t - last) if last >= 0 else (1.0 / 30.0)
        if dt <= 0 or dt > 0.5:
            dt = 1.0 / 30.0
        _S['last_t'] = t
        feat = _S['wa'].process(mag, rms, t, dt, cfg=_cfg(comp))
        feat['spectrum'] = _log128(arr)
        comp.store('wedo', feat)
    except Exception:
        pass
