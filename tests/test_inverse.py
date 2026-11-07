"""固定阻力反演与评价接口检查，不把同模型恢复当成外部精度证据。"""

import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import inverse
from config import CASE, ENVIRONMENT, INFERENCE
from physics import simulate
from evaluate import evaluate_predictions


def parameters(model="F"):
    return {"initial_velocity_m_s": (30., 5., 6.), "drag_coefficient": .22,
            "b_y": .04 if model == "F" else 0., "b_z": -.15}


def simulated_case(model="F"):
    result = simulate(model, parameters(model), CASE.times_s)
    return replace(CASE, positions_m=result["positions_m"])


def record(case=CASE, available=True, source="source_0"):
    return {"schema_version": 1, "source_id": source, "case_id": source+"_noise_0",
        "split": "development", "observation_available": available,
        "case": asdict(case), "known_environment": asdict(ENVIRONMENT), "wind_known": False,
        "measurement": {"mode": "gaussian", "sigma_m": .01, "frame_correlation": 0.,
                        "rounding_step_m": None}}


class InverseTests(unittest.TestCase):
    def test_unavailable_and_successful_records_share_point_kind(self):
        missing = record(available=False)
        available = record(source="source_1")
        a, _ = inverse.fit_observation(missing, "S", starts=1)
        b, _ = inverse.fit_observation(available, "S", starts=1)
        self.assertEqual(a["point_kind"], b["point_kind"])

    def test_ls_recovers_known_s_ball(self):
        fit = inverse.fit_case("S", simulated_case("S"), mode="ls", starts=1)
        self.assertTrue(fit["success"])
        self.assertLess(fit["rms_residual_m"], 1e-5)
        np.testing.assert_allclose(fit["parameters"]["initial_velocity_m_s"],
                                   parameters("S")["initial_velocity_m_s"], atol=.002)
        self.assertAlmostEqual(fit["parameters"]["b_z"], -.15, delta=.002)
        self.assertEqual(fit["parameters"]["b_y"], 0)

    def test_ls_recovers_known_f_ball(self):
        fit = inverse.fit_case("F", simulated_case("F"), mode="ls", starts=1)
        self.assertTrue(fit["success"])
        self.assertLess(fit["rms_residual_m"], 1e-5)
        for key in ("b_y", "b_z"):
            self.assertAlmostEqual(fit["parameters"][key], parameters()[key], delta=.002)
        expected = simulate("F", parameters(), CASE.times_s)
        np.testing.assert_allclose(fit["simulation"]["event_position_m"], expected["event_position_m"], atol=.005)

    def test_map_objective_is_likelihood_plus_prior(self):
        fit = inverse.fit_case("F", starts=1)
        self.assertTrue(fit["success"])
        self.assertAlmostEqual(fit["objective"], .5*fit["data_chi_square"]+fit["prior_penalty"], places=9)
        self.assertEqual(len(fit["parameter_names"]), 5)
        self.assertEqual(np.asarray(fit["diagnostics"]["data_jacobian_scaled"]).shape, (6, 5))

    def test_strong_prior_shrinks_transverse_coefficients(self):
        case = simulated_case("F")
        ls = inverse.fit_case("F", case, mode="ls", starts=1)
        strong = inverse.fit_case("F", case, settings=replace(INFERENCE, transverse_prior_std=.01), starts=1)
        def norm(f):
            return np.hypot(f["parameters"]["b_y"], f["parameters"]["b_z"])
        self.assertTrue(strong["success"])
        self.assertLess(norm(strong), norm(ls))

    def test_s_prior_is_f_matching_marginal(self):
        p = inverse.transverse_prior_penalty
        self.assertAlmostEqual(p("S", [0]), 0)
        self.assertGreater(p("S", [.4]), p("F", [0, .4]))
        self.assertAlmostEqual(p("S", [.2]), p("S", [-.2]))
        self.assertTrue(np.isinf(p("S", [.5])))

    def test_correlated_covariance_and_whitening(self):
        covariance = inverse.observation_covariance(CASE, sigma_m=.02, frame_correlation=.5)
        self.assertAlmostEqual(covariance[0, 3], .02**2*.5)
        self.assertEqual(covariance[0, 4], 0)
        fit = inverse.fit_case("S", covariance_m2=covariance, starts=1)
        self.assertTrue(fit["success"])
        error = np.asarray(fit["residuals_m"]).ravel()
        self.assertAlmostEqual(fit["data_chi_square"], error@np.linalg.solve(covariance, error), places=8)

    def test_disk_and_velocity_constraints(self):
        fit = inverse.fit_case("F", starts=1)
        p = fit["parameters"]
        self.assertLessEqual(np.hypot(p["b_y"], p["b_z"]), INFERENCE.transverse_norm_max+1e-8)
        for value, bounds in zip(p["initial_velocity_m_s"], (INFERENCE.vx_bounds_m_s,
                               INFERENCE.vy_bounds_m_s, INFERENCE.vz_bounds_m_s)):
            self.assertGreaterEqual(value, bounds[0])
            self.assertLessEqual(value, bounds[1])

    def test_landing_after_observations_is_valid_fit(self):
        p = {"initial_velocity_m_s": (10., 0., 1.), "drag_coefficient": .22, "b_y": 0., "b_z": 0.}
        result = simulate("S", p, CASE.times_s)
        case = replace(CASE, positions_m=result["positions_m"])
        fit = inverse.fit_case("S", case, mode="ls", starts=1)
        self.assertTrue(fit["success"])
        self.assertEqual(fit["simulation"]["event"], "ground")

    def test_nonconvergence_not_exported_as_valid_prediction(self):
        settings = replace(INFERENCE, max_iterations_per_start=1)
        fit = inverse.fit_case("F", settings=settings, starts=1)
        self.assertFalse(fit["success"])
        self.assertIsNone(fit["parameters"])

    def test_numerical_failure_not_a_fake_fit(self):
        failed = {"event": "failure", "positions_m": (None, None), "message": "test"}
        with patch.object(inverse, "simulate", return_value=failed):
            fit = inverse.fit_case("F", starts=1)
        self.assertFalse(fit["success"])
        self.assertGreater(fit["numerical_failure_calls"], 0)

    def test_invalid_settings_rejected(self):
        calls = [lambda: inverse.fit_case("G"), lambda: inverse.fit_case("F", mode="unknown"),
            lambda: inverse.fit_case("S", sigma_m=0), lambda: inverse.fit_case("S", starts=0),
            lambda: inverse.fit_case("S", drag_coefficient=-1),
            lambda: inverse.fit_case("S", frame_correlation=1),
            lambda: inverse.fit_case("S", covariance_m2=np.ones((6, 6))),
            lambda: inverse.fit_case("S", initial_guesses=[[1, 2]]),
            lambda: inverse.fit_case("S", replace(CASE, times_s=(.1, .05)))]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_noiseless_data_requires_explicit_sigma(self):
        observation = record(simulated_case("S"))
        observation["measurement"].update(mode="none", sigma_m=0)
        pred, _ = inverse.fit_observation(observation, "S", starts=1)
        self.assertEqual(pred["status"], "failure")
        pred, fit = inverse.fit_observation(observation, "S", sigma_override_m=.01, starts=1)
        self.assertEqual(pred["status"], "ok")
        self.assertTrue(fit["success"])

    def test_prediction_evaluation_contract(self):
        case = simulated_case("S")
        observation = record(case)
        missing = record(available=False, source="missing")
        predictions = [inverse.fit_observation(row, "S", starts=1)[0] for row in (observation, missing)]
        truth_result = simulate("S", parameters("S"), CASE.times_s)
        truths = [{"source_id": row["source_id"], "split": "development", "parameters": {"mechanism": "constant"},
                   "truth": {**truth_result, "inside_goal_opening": True}} for row in (observation, missing)]
        report = evaluate_predictions([observation, missing], truths, predictions, bootstrap_samples=0)
        summary = next(iter(report["methods"].values()))
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["status_counts"]["unavailable"], 1)
        self.assertIsNone(summary["coverage"]["rate"])

    def test_repeatability_and_multiple_starts(self):
        a = inverse.fit_case("S", starts=2)
        b = inverse.fit_case("S", starts=2)
        self.assertEqual(len(a["optimizer_runs"]), 2)
        self.assertEqual(a["parameters"], b["parameters"])
        self.assertEqual(a["objective"], b["objective"])


if __name__ == "__main__":
    unittest.main()
