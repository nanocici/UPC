"""权重和区域用手算/解析分布核验，集成测试阈值明确放宽以控制耗时。"""

import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import multivariate_t, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import uncertainty as uq
from config import CASE, ENVIRONMENT, INFERENCE, SOLVER
from evaluate import evaluate_predictions
from inverse import fit_case


def candidate(weight, event, point=None, time=.4):
    return {"weight": weight, "event": event, "goal_yz_m": point,
            "arrival_time_s": time if point is not None else None}


TEST_SETTINGS = replace(uq.UNCERTAINTY, base_samples=120, min_effective_samples=20,
    min_conditional_effective_samples=10, event_probability_tolerance=.25,
    mean_shift_tolerance_m=.4, relative_area_tolerance=.8)


class WeightedTests(unittest.TestCase):
    def test_log_weights_are_stable_and_normalized(self):
        weights = uq.normalize_log_weights([-10000, -10000, -np.inf])
        np.testing.assert_allclose(weights, [.5, .5, 0])
        self.assertEqual(uq.effective_sample_size(weights), 2)
        with self.assertRaises(ValueError):
            uq.normalize_log_weights([-np.inf, -np.inf])

    def test_weighted_cdf_quantile(self):
        self.assertEqual(uq.weighted_quantile([3, 1, 2], [.1, .8, .1], .5), 1)
        self.assertEqual(uq.weighted_quantile([1, 2], [0, 1], 0), 2)
        self.assertEqual(uq.weighted_quantile([1, 2], [.5, .5], 1), 2)

    def test_grid_mode_uses_mass_not_sample_count(self):
        mode = uq.grid_mode([[0, 1], [0, 1.01], [2, 1]], [.1, .1, .8], width=.1)
        self.assertAlmostEqual(mode["cell_probability"], .8)
        self.assertLess(abs(mode["center_yz_m"][0]-2), .1)

    def test_mixture_density_correction_recovers_analytic_normal(self):
        distributions = [multivariate_t(loc=[0], shape=[[.6]], df=5),
                         multivariate_t(loc=[0], shape=[[5.4]], df=5)]
        points, log_q = uq.draw_proposal(distributions, [.85, .15], 20000, np.random.default_rng(7))
        weights = uq.normalize_log_weights(norm.logpdf(points[:, 0])-log_q)
        mean = weights@points[:, 0]
        variance = weights@((points[:, 0]-mean)**2)
        self.assertAlmostEqual(mean, 0, delta=.035)
        self.assertAlmostEqual(variance, 1, delta=.05)

    def test_full_hessian_known_quadratic(self):
        precision = np.array([[3., .4], [.4, 2.]])
        result = uq._hessian(lambda q: .5*q@precision@q, np.array([.2, -.3]))
        np.testing.assert_allclose(result, precision, atol=1e-7)

    def test_event_failure_mass_not_deleted(self):
        settings = replace(TEST_SETTINGS, min_conditional_effective_samples=1)
        draws = [candidate(.4, "goal_inside", [0, 1]), candidate(.1, "goal_outside", [4, 1]),
                 candidate(.3, "ground"), candidate(.2, "failure")]
        summary = uq.summarize_candidates(draws, settings=settings)
        self.assertAlmostEqual(summary["event_probabilities"]["ground"], .3)
        np.testing.assert_allclose(summary["direct_inside_probability_bounds"], [.4, .6])
        self.assertAlmostEqual(summary["direct_crossing"]["posterior_mass"], .5)
        np.testing.assert_allclose(summary["direct_crossing"]["mean_yz_m"], [.8, 1])

    def test_area_and_empirical_region_mass(self):
        draws = [candidate(1, "goal_inside", p) for p in ((1, 1), (-1, 1), (0, 2), (0, 0))]
        settings = replace(TEST_SETTINGS, min_conditional_effective_samples=3)
        direct = uq.summarize_candidates(draws, settings=settings)["direct_crossing"]
        np.testing.assert_allclose(direct["mean_yz_m"], [0, 1])
        np.testing.assert_allclose(direct["region"]["covariance_m2"], [[.5, 0], [0, .5]])
        self.assertAlmostEqual(direct["region"]["threshold"], 2)
        self.assertAlmostEqual(direct["region_area_m2"], np.pi)

    def test_all_ground_has_no_conditional_location(self):
        summary = uq.summarize_candidates([candidate(.5, "ground"), candidate(.5, "ground")])
        self.assertIsNone(summary["direct_crossing"])
        self.assertEqual(summary["direct_inside_probability_bounds"], [0, 0])

    def test_mean_stability_does_not_imply_mode_stability(self):
        first = {"event_probabilities": {e: float(e == "goal_inside") for e in uq.EVENT_CLASSES},
                 "direct_crossing": {"mean_yz_m": [0, 1], "region_area_m2": 1,
                     "most_likely_cell": {"center_yz_m": [0, 1]}}}
        full = {**first, "direct_crossing": {**first["direct_crossing"],
                    "most_likely_cell": {"center_yz_m": [1, 1]}}}
        result = uq._convergence(first, full, uq.UNCERTAINTY)
        self.assertTrue(result["sampling_stable"])
        self.assertFalse(result["mode_sampling_stable"])

    def test_singular_region_is_not_fabricated(self):
        draws = [candidate(1, "goal_inside", [1, 1]) for _ in range(20)]
        direct = uq.summarize_candidates(draws, settings=TEST_SETTINGS)["direct_crossing"]
        self.assertIsNone(direct["region"])


class PosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = fit_case("S", starts=1)

    def test_target_matches_map_likelihood_and_prior(self):
        posterior = uq._Posterior("S", self.fit, CASE, ENVIRONMENT, SOLVER, INFERENCE)
        self.assertAlmostEqual(posterior.log_density(posterior.center), -self.fit["objective"], places=8)
        covariance, kind = uq.proposal_covariance(posterior)
        np.linalg.cholesky(covariance)
        self.assertIn(kind, ("full_posterior_hessian", "gauss_newton_plus_prior_proposal_only"))

    def test_out_of_bounds_prior_is_zero_density(self):
        posterior = uq._Posterior("S", self.fit, CASE, ENVIRONMENT, SOLVER, INFERENCE)
        q = posterior.center.copy()
        q[0] = -1
        self.assertEqual(posterior.log_density(q), -np.inf)

    def test_likelihood_failure_aborts_instead_of_assigning_zero(self):
        failed = {"event": "failure", "positions_m": (None, None)}
        with patch.object(uq, "simulate", return_value=failed):
            result = uq.sample_case("S", uncertainty=TEST_SETTINGS, starts=1)
        self.assertFalse(result["success"])
        self.assertIsNone(result["summary"])
        self.assertEqual(result["diagnostics"]["unresolved_likelihood_failures"], 1)

    def test_prediction_failure_keeps_known_likelihood_weight(self):
        original = uq.simulate
        def forward(model, params, times, case, environment, solver):
            if solver.maximum_horizon_s > .2:
                return {"event": "failure", "positions_m": (None, None)}
            return original(model, params, times, case, environment, solver)
        with patch.object(uq, "simulate", side_effect=forward):
            result = uq.sample_case("S", uncertainty=TEST_SETTINGS, starts=1)
        self.assertFalse(result["success"])
        self.assertAlmostEqual(result["summary"]["unknown_event_mass"], 1)
        np.testing.assert_allclose(result["summary"]["direct_inside_probability_bounds"], [0, 1])

    def test_actual_sampling_is_reproducible(self):
        a = uq.sample_case("S", uncertainty=TEST_SETTINGS, seed=3, starts=1)
        b = uq.sample_case("S", uncertainty=TEST_SETTINGS, seed=3, starts=1)
        self.assertIsNotNone(a["summary"])
        self.assertEqual(a["draws"], b["draws"])
        self.assertGreater(a["summary"]["effective_samples"], 20)

    def test_invalid_sampler_settings(self):
        for settings in (replace(TEST_SETTINGS, base_samples=1), replace(TEST_SETTINGS, student_df=2),
                         replace(TEST_SETTINGS, broad_component_weight=0)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                uq.sample_case(uncertainty=settings)

    def test_unavailable_record_keeps_mean_label_and_nominal_mass(self):
        record = {"case_id": "c", "source_id": "s", "split": "development", "observation_available": False}
        prediction, result = uq.uncertainty_observation(record, "S")
        self.assertEqual(prediction["status"], "unavailable")
        self.assertEqual(prediction["point_kind"], "mean")
        self.assertEqual(prediction["region_mass"], .9)

    def test_public_prediction_evaluation_contract(self):
        record = {"case_id": "c", "source_id": "s", "split": "development", "observation_available": True,
            "case": asdict(CASE), "known_environment": asdict(ENVIRONMENT), "wind_known": False,
            "measurement": {"mode": "gaussian", "sigma_m": .01, "frame_correlation": 0}}
        prediction, result = uq.uncertainty_observation(record, "S", uncertainty=TEST_SETTINGS, seed=3, starts=1)
        truth = {"source_id": "s", "split": "development", "parameters": {"mechanism": "constant"},
            "truth": {"event": "goal", "event_position_m": [11, 3, 1.9],
                      "event_time_s": .4, "inside_goal_opening": True}}
        report = evaluate_predictions([record], [truth], [prediction], bootstrap_samples=0)
        self.assertEqual(report["methods"]["S_cd0.22_posterior"]["case_count"], 1)
        self.assertEqual(len(result["draws"]), 2*TEST_SETTINGS.base_samples)


if __name__ == "__main__":
    unittest.main()
