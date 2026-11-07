import sys, tempfile, unittest, json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import model_diagnosis as d

class DiagnosisTests(unittest.TestCase):
    def test_rotated_ellipse_and_projections(self):
        region=dict(center_yz_m=[0,0],covariance_m2=[[2.5,1.5],[1.5,2.5]],threshold=1)
        a=d.region_diagnostics(region,[1,1])
        self.assertTrue(a['covered']); self.assertAlmostEqual(a['minor_semiaxis_m'],1)
        self.assertAlmostEqual(a['major_semiaxis_m'],2)
        b=d.region_diagnostics(region,[1,-1]); self.assertFalse(b['covered'])
        self.assertTrue(b['miss_minor_component'])
        self.assertFalse(b['outside_y_projection'])
    def test_missing_region_counts_as_miss(self):
        rows=[dict(method='S',source_id='a',sigma=.01,scenario='base',truth_event='goal',status='failure',covered=False),
              dict(method='S',source_id='b',sigma=.01,scenario='base',truth_event='goal',status='ok',covered=True)]
        m=next(iter(d.aggregate(rows,bootstrap=20)['overall'].values()))
        self.assertEqual(m['coverage'],.5); self.assertEqual(m['direct_n'],2)
    def test_repeated_sources_do_not_get_independent_binomial_interval(self):
        rows=[dict(method='S',source_id='same',sigma=sigma,scenario='base',truth_event='goal',status='ok',covered=True)
              for sigma in (.005,.02)]
        summary=d.aggregate(rows,bootstrap=0)
        overall=next(iter(summary['overall'].values()))
        self.assertNotIn('coverage_wilson_95_independent_sources',overall)
        for subgroup in summary['noise'].values():
            self.assertIn('coverage_wilson_95_independent_sources',subgroup)
    def test_prepared_pairing_and_single_factor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'; d.prepare(out,count=2)
            truth=d.read_jsonl(out/'truth_development.jsonl'); obs=d.read_jsonl(out/'observations.jsonl')
            self.assertEqual(len(obs),24); self.assertEqual(len(json.loads((out/'tasks.json').read_text())),44)
            self.assertTrue(all(o['split']=='development' for o in obs))
            group=[t for t in truth if t['group_id']=='group000']; base=group[0]
            for t in group:
                self.assertEqual(t['parameters']['initial_velocity_m_s'],base['parameters']['initial_velocity_m_s'])
            vert=group[1]
            self.assertEqual(vert['parameters']['drag_coefficient'],.22)
            self.assertEqual(vert['parameters']['transverse_vector'][2],base['parameters']['transverse_vector'][2])
            for sigma in (.005,.02):
                noises=[]
                for t in group:
                    o=next(o for o in obs if o['source_id']==t['source_id'] and o['measurement']['sigma_m']==sigma)
                    if o['observation_available']:
                        noises.append((np.array(o['case']['positions_m'])-t['truth']['clean_positions_m'])/sigma)
                for n in noises[1:]: np.testing.assert_allclose(n,noises[0],atol=1e-12)
    def test_independent_clean_generator_restores_goal(self):
        truth=d.generate_truth(d.TruthParameters(mechanism='constant',drag_coefficient=.22,
                              transverse_vector=(0.,.08,-.1)))
        obs=d.observation(truth,'clean',.01,np.zeros((2,3)))
        fit=d.fit_case('F',d.observation_case(obs),mode='ls',sigma_m=.01,starts=3)
        self.assertTrue(fit['success'])
        self.assertLess(fit['rms_residual_m'],1e-6)
        self.assertEqual(fit['simulation']['event'],truth['event'])
        self.assertLess(np.linalg.norm(np.array(fit['simulation']['event_position_m'])-
                                      truth['event_position_m']),1e-4)
    def test_non_development_rejected(self):
        with self.assertRaises(ValueError):
            d.diagnostic_rows([],[],[{'split':'test'}])
    def test_tampered_seal_rejected_before_truth_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'predictions.jsonl').write_text('changed')
            d.write_json(p/'prediction_manifest.json',{'predictions_sha256':'wrong'})
            with self.assertRaisesRegex(ValueError,'seal changed'): d.summarize(p)
if __name__=='__main__': unittest.main()
