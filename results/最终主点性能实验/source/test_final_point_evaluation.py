"""Failure denominators, pairing and no-candidate mode certification."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import final_point_evaluation as experiment
from config import CASE


def row(source='a',peak=.2,g=.3,stable=True,covered=True):
    return dict(case_id=source,source_id=source,mechanism='constant',sigma_m=.01,
                truth_inside=True,truth_direct=True,observation_available=True,
                stable_inside=stable,peak_error_m=peak,g_error_m=g,mean_error_m=.25,
                raw_peak_error_m=.2,direct_peak_error_m=peak,direct_mean_error_m=.25,
                region_covered=covered,region_available=covered,region_area_m2=1. if covered else None,
                failure_reason='' if stable else 'unstable')


class FinalPointTests(unittest.TestCase):
    def test_failures_remain_in_release_and_coverage_denominators(self):
        result=experiment.metrics([row(),row('b',peak=None,stable=False,covered=False)])
        self.assertEqual(result['stable_inside_rate'],.5)
        self.assertEqual(result['region_coverage'],.5)
        self.assertEqual(result['g_all_eligible']['n'],2)
        self.assertEqual(result['paired_g']['n'],1)
        self.assertEqual(result['stable_and_error_below_05_rate'],.5)

    def test_bootstrap_pairs_sources_without_counting_missing_peaks(self):
        rows=[row('a'),row('a'),row('b',peak=None,stable=False)]
        result=experiment.bootstrap(rows,n=40)
        np.testing.assert_allclose(result['paired_median_difference_f1_minus_g_ci95_m'],[-.1,-.1])

    def test_no_inside_candidates_cannot_publish_peak(self):
        data=dict(event=np.full(20,3),weight=np.full(20,.05),goal_yz_m=np.zeros((20,2)))
        summary=dict(success=True,summary={'inside_conditioned':None},first_half={'inside_conditioned':None})
        result=experiment.mode_analysis([data]*4,[summary]*4,'inside_conditioned',CASE)
        self.assertFalse(result['stable'])
        self.assertIsNone(result['point_yz_m'])


if __name__=='__main__':unittest.main()
