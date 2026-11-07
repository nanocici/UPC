"""Numerical equivalence gates for the optional batched F ODE evaluator."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import batch_f_sampler as batch
from config import CASE, ENVIRONMENT
from final_config import FINAL
from physics import simulate
from uncertainty import summarize_candidates


class BatchSamplerTests(unittest.TestCase):
    def test_forward_events_and_positions_match_reference(self):
        rng=np.random.default_rng(20261108)
        params=np.column_stack([rng.uniform(1,60,60),rng.uniform(-30,30,60),
                                rng.uniform(.001,30,60),rng.uniform(-.3,.3,(60,2))])
        for environment in (ENVIRONMENT,replace(ENVIRONMENT,wind_m_s=(2,-1,.3))):
            obs,event,yzt=batch.propagate(params,CASE,environment)
            for i,p in enumerate(params):
                ref=simulate('F',dict(initial_velocity_m_s=p[:3],b_y=p[3],b_z=p[4],drag_coefficient=.22),CASE.times_s,CASE,environment)
                name=batch.EVENTS[event[i]]
                if name.startswith('goal_'):name='goal'
                self.assertEqual(name,ref['event'])
                for j,r in enumerate(ref['positions_m']):
                    if r is not None:np.testing.assert_allclose(obs[i,3*j:3*j+3],r,atol=1e-6,rtol=0)
                if ref['event']=='goal':
                    np.testing.assert_allclose(yzt[i,:2],ref['event_position_m'][1:],atol=1e-4,rtol=0)
                    self.assertLess(abs(yzt[i,2]-ref['event_time_s']),1e-6)

    def test_weights_and_array_summaries_match_reference(self):
        fit,p,d,w=batch.prepare(CASE,ENVIRONMENT,FINAL.inference_settings(),.01)
        self.assertTrue(fit['success'])
        data=batch.sample(p,d,w,20261109,count=1024)
        raw=[]
        for i in range(1024):
            if i<50:
                ref=p.log_density(data['parameters'][i]/p.scales)
                self.assertTrue(np.isclose(ref,data['log_target'][i],atol=1e-5) or ref==data['log_target'][i])
            raw.append(dict(weight=float(data['weight'][i]),event=batch.EVENTS[data['event'][i]],
                            goal_yz_m=data['goal_yz_m'][i].tolist(),arrival_time_s=float(data['arrival_time_s'][i])))
        a=batch.summarize(data,CASE);b=summarize_candidates(raw,CASE)
        for condition in ('direct_crossing','inside_conditioned'):
            np.testing.assert_allclose(a[condition]['mean_yz_m'],b[condition]['mean_yz_m'],atol=1e-12)
            self.assertAlmostEqual(a[condition]['region_area_m2'],b[condition]['region_area_m2'],places=10)

    def test_step_halving_agrees(self):
        p=np.array([[30,9,7,.05,.12],[60,-15,15,.3,-.3],[22,0,.01,-.1,0]])
        a=batch.propagate(p,CASE,ENVIRONMENT,step=.005)
        b=batch.propagate(p,CASE,ENVIRONMENT,step=.0025)
        np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[0],b[0],atol=1e-6,rtol=0)
        np.testing.assert_allclose(a[2],b[2],atol=1e-5,rtol=0)


if __name__=='__main__':unittest.main()
