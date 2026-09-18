"""Release lifecycle tests: actual operator actions, not threshold tuning."""
import ast
import pathlib
import unittest
import numpy as np
from wedoaudio_dsp import WedoAudio, spectral_flatness

class ReleaseControls(unittest.TestCase):
    def test_invalid_input_is_rejected_instead_of_a_fake_peak(self):
        for invalid in [np.array([float('nan')]),np.array([float('inf')]),np.array([-1.]),np.zeros((2,20))]:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                WedoAudio().process(invalid,.1,0,1/60)
        with self.assertRaises(ValueError):WedoAudio().process(np.zeros(1024),float('nan'),0,1/60)

    def test_disable_autocorrelation_really_uses_anchor(self):
        w=WedoAudio();w.bpm_ac=180.;w.bpm_raw=100.
        f=w.process(np.zeros(1025),0,1,1/60,{'tempo_ac':0,'tempo_fix':0})
        self.assertAlmostEqual(f['bpm'],100.)

    def test_onset_mode_roundtrip_restores_live_parameters(self):
        w=WedoAudio();mag=np.zeros(1025)
        w.process(mag,0,0,1/60,{'onset_rel':0})
        w.process(mag,0,1/60,1/60,{'onset_rel':1})
        fresh=WedoAudio()
        for name in ('kick','snare','beat'):
            a,b=getattr(w,name),getattr(fresh,name)
            self.assertEqual((a.k,a.delta,a.refr),(b.k,b.delta,b.refr),name)

    def test_gate_mutes_all_onset_triggers(self):
        w=WedoAudio();count=0
        for i in range(240):
            mag=np.ones(1025)*(.10 if i%30==0 else .001)
            f=w.process(mag,.05,i/60,1/60,{'gate':.1})
            self.assertEqual(f['silence'],1)
            count+=f['rawkick']+f['rawsnare']+f['rawbeat']
        self.assertEqual(count,0)

    def test_builder_defaults_match_core_detectors(self):
        path=pathlib.Path(__file__).resolve().parents[1]/'tools'/'build_portable.py'
        tree=ast.parse(path.read_text(encoding='utf-8-sig'));defaults={}
        for call in ast.walk(tree):
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='par_set' or len(call.args)<2:continue
            inner=call.args[0]
            if isinstance(inner,ast.Call) and inner.args and isinstance(inner.args[0],ast.Constant):
                try:defaults[inner.args[0].value]=ast.literal_eval(call.args[1])
                except ValueError:pass
        w=WedoAudio()
        for prefix,detector in [('Kick',w.kick),('Snare',w.snare),('Beat',w.beat)]:
            self.assertEqual(defaults[prefix+'sensitivity'],detector.k,prefix+' sensitivity')
            self.assertEqual(defaults[prefix+'refractoryms']/1000,detector.refr,prefix+' refractory')

if __name__=='__main__':unittest.main(verbosity=2)
