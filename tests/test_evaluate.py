"""评价指标用已知答案手算验证，独立于待实现的反演与不确定性程序。"""

import copy
import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import evaluate
from config import CASE, ENVIRONMENT


def observation(index, source=None, available=True):
    return {"schema_version": 1, "case_id": f"case_{index}",
        "source_id": source or f"source_{index}", "split": "development",
        "observation_available": available, "case": asdict(CASE),
        "known_environment": asdict(ENVIRONMENT), "wind_known": False,
        "measurement": {"mode": "gaussian", "sigma_m": .01,
                        "frame_correlation": 0, "rounding_step_m": None}}


def truth(source, event="goal"):
    return {"source_id": source, "split": "development", "parameters": {"mechanism": "spin"},
        "truth": {"event": event, "event_position_m": [11, 0, 1] if event == "goal" else [2, 0, .11],
                  "event_time_s": .4, "inside_goal_opening": True if event == "goal" else None}}


def prediction(obs, method="F", *, point=(0, 1), region_mass=None, region=None):
    return {"schema_version": 1, "case_id": obs["case_id"], "source_id": obs["source_id"],
        "split": obs["split"], "method": method, "status": "ok", "event": "goal",
        "point_kind": "mean", "point_yz_m": point, "arrival_time_s": .5,
        "region_mass": region_mass, "region": region, "goal_probability": None, "message": ""}


def region(center=(0, 1), covariance=((1, 0), (0, 1)), threshold=1):
    return {"center_yz_m": center, "covariance_m2": covariance, "threshold": threshold}


class EvaluationTests(unittest.TestCase):
    def score(self, observations, truths, predictions, bootstrap=0):
        return evaluate.evaluate_predictions(observations, truths, predictions,
                                              bootstrap_samples=bootstrap, seed=12)

    def test_yz_distance_and_arrival_error(self):
        obs = observation(0)
        p = prediction(obs, point=(3, 5))  # 真值 (0,1)，距离 sqrt(3²+4²)=5。
        s = self.score([obs], [truth(obs["source_id"])], [p])["methods"]["F"]
        self.assertEqual(s["position_error_m"]["median"], 5)
        self.assertAlmostEqual(s["arrival_time_absolute_error_s"]["median"], .1)
        self.assertEqual(s["position_error_m"]["n"], 1)

    def test_missing_prediction_keeps_coverage_denominator(self):
        observations = [observation(i) for i in range(3)]
        truths = [truth(o["source_id"]) for o in observations]
        first = prediction(observations[0], region_mass=.9, region=region())
        failed = prediction(observations[1], region_mass=.9)
        failed.update(status="failure", event="failure", point_yz_m=None, arrival_time_s=None)
        s = self.score(observations, truths, [first, failed])["methods"]["F"]
        self.assertEqual(s["coverage"]["denominator"], 3)
        self.assertAlmostEqual(s["coverage"]["rate"], 1/3)
        self.assertEqual(s["status_counts"], {"ok": 1, "failure": 1, "missing": 1})
        self.assertAlmostEqual(s["failure_or_missing_rate"], 2/3)

    def test_no_regions_is_not_a_fake_zero_coverage(self):
        obs = observation(0)
        report = self.score([obs], [truth(obs["source_id"])], [prediction(obs)])
        self.assertIsNone(report["methods"]["F"]["coverage"]["rate"])
        claimed = prediction(obs, region_mass=.9)
        report = self.score([obs], [truth(obs["source_id"])], [claimed])
        self.assertEqual(report["methods"]["F"]["coverage"]["rate"], 0)

    def test_ellipse_area_and_boundary(self):
        obs = observation(0)
        p = prediction(obs, region_mass=.9, region=region(center=(2, 1), threshold=4))
        s = self.score([obs], [truth(obs["source_id"])], [p])["methods"]["F"]
        self.assertAlmostEqual(s["region_area_on_true_direct_m2"]["median"], 4*math.pi)
        self.assertEqual(s["coverage"]["rate"], 1)
        p["region"] = region(center=(2.01, 1), threshold=4)
        self.assertEqual(self.score([obs], [truth(obs["source_id"])], [p])["methods"]["F"]["coverage"]["rate"], 0)

    def test_singular_or_invalid_region_is_miss_not_dropped(self):
        obs = observation(0)
        for candidate in (region(covariance=((1, 0), (0, 0))), region(threshold=-1),
                          region(covariance=((1, 2), (0, 1)))):
            with self.subTest(region=candidate):
                p = prediction(obs, region_mass=.9, region=candidate)
                s = self.score([obs], [truth(obs["source_id"])], [p])["methods"]["F"]
                self.assertEqual(s["invalid_region_count"], 1)
                self.assertEqual(s["coverage"]["rate"], 0)
                self.assertEqual(s["position_error_m"]["n"], 1)

    def test_unavailable_input_separately_reported(self):
        observations = [observation(0), observation(1, available=False)]
        predictions = [prediction(observations[0], region_mass=.9, region=region())]
        report = self.score(observations, [truth(o["source_id"]) for o in observations], predictions)
        s = report["methods"]["F"]
        self.assertEqual(s["coverage"]["rate"], .5)
        self.assertEqual(s["coverage_available_inputs"]["rate"], 1)
        self.assertEqual(s["observation_unavailable_count"], 1)

    def test_brier_known_outcomes_and_unknown_exclusion(self):
        observations = [observation(i) for i in range(3)]
        truths = [truth(observations[i]["source_id"], event) for i, event in enumerate(("goal", "ground", "boundary"))]
        predictions = [prediction(o) for o in observations]
        for p, probability in zip(predictions, (.8, .2, .99)):
            p["goal_probability"] = probability
        s = self.score(observations, truths, predictions)["methods"]["F"]
        self.assertEqual(s["brier_score"]["n"], 2)
        self.assertAlmostEqual(s["brier_score"]["mean"], .04)
        self.assertEqual(s["brier_known_outcome_count"], 2)
        self.assertEqual(s["position_error_m"]["n"], 1)

    def test_missing_probability_is_counted(self):
        obs = observation(0)
        s = self.score([obs], [truth(obs["source_id"])], [prediction(obs)])["methods"]["F"]
        self.assertEqual(s["probability_missing_on_known_outcome_count"], 1)
        self.assertIsNone(s["brier_score"]["mean"])

    def test_duplicate_or_wrong_prediction_ids_rejected(self):
        obs = observation(0)
        p = prediction(obs)
        with self.assertRaises(ValueError):
            self.score([obs], [truth(obs["source_id"])], [p, p])
        for field, value in (("case_id", "wrong"), ("source_id", "wrong"), ("split", "test")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.score([obs], [truth(obs["source_id"])], [{**p, field: value}])

    def test_missing_duplicate_or_wrong_split_truth_rejected(self):
        obs = observation(0)
        t = truth(obs["source_id"])
        for truths in ([], [t, t], [{**t, "split": "test"}]):
            with self.subTest(truths=truths), self.assertRaises(ValueError):
                self.score([obs], truths, [prediction(obs)])

    def test_mixed_method_definitions_rejected(self):
        observations = [observation(i) for i in range(2)]
        predictions = [prediction(o) for o in observations]
        predictions[1]["point_kind"] = "mode"
        with self.assertRaises(ValueError):
            self.score(observations, [truth(o["source_id"]) for o in observations], predictions)

    def test_nonfinite_prediction_rejected(self):
        obs = observation(0)
        with self.assertRaises(ValueError):
            self.score([obs], [truth(obs["source_id"])], [prediction(obs, point=(np.nan, 1))])

    def test_failed_prediction_cannot_sneak_in_a_point(self):
        obs = observation(0)
        p = prediction(obs)
        p["status"] = "failure"
        with self.assertRaises(ValueError):
            self.score([obs], [truth(obs["source_id"])], [p])

    def test_empty_point_samples_serialize_as_null(self):
        obs = observation(0)
        report = self.score([obs], [truth(obs["source_id"], "ground")], [prediction(obs)])
        self.assertIsNone(report["methods"]["F"]["position_error_m"]["median"])
        json.dumps(report, allow_nan=False)

    def test_cluster_bootstrap_keeps_source_replicates_together(self):
        observations = [observation(i, source=f"source_{i//4}") for i in range(8)]
        truths = [truth("source_0"), truth("source_1")]
        predictions = [prediction(o, point=(1 if i < 4 else 3, 1)) for i, o in enumerate(observations)]
        report = self.score(observations, truths, predictions, bootstrap=100)
        s = report["methods"]["F"]
        self.assertEqual(s["source_count"], 2)
        interval = s["cluster_bootstrap_95_intervals"]["position_error_median_m"]
        self.assertEqual((interval["lower"], interval["upper"]), (1, 3))
        self.assertEqual(report, self.score(observations, truths, predictions, bootstrap=100))

    def test_paired_comparison_uses_common_cases(self):
        observations = [observation(0), observation(1)]
        truths = [truth(o["source_id"]) for o in observations]
        predictions = [prediction(o, "F", point=(1, 1)) for o in observations]
        predictions.append(prediction(observations[0], "S", point=(3, 1)))
        report = self.score(observations, truths, predictions, bootstrap=10)
        paired = report["paired_comparisons"][0]
        self.assertEqual((paired["method_a"], paired["method_b"]), ("F", "S"))
        self.assertEqual(paired["common_case_count"], 1)
        self.assertEqual(paired["median_error_a_minus_b_m"], -2)
        self.assertEqual(report["methods"]["S"]["status_counts"]["missing"], 1)

    def test_strata_counts(self):
        observations = [observation(0), observation(1)]
        observations[1]["measurement"]["sigma_m"] = .02
        truths = [truth(o["source_id"]) for o in observations]
        truths[1]["parameters"]["mechanism"] = "wind"
        report = self.score(observations, truths, [prediction(o) for o in observations])
        self.assertEqual(set(report["strata"]["F"]["noise_sigma_m"]), {"0.01", "0.02"})
        self.assertEqual(report["strata"]["F"]["mechanism"]["wind"]["source_count"], 1)

    def test_save_load_and_overwrite_refusal(self):
        obs = observation(0)
        predictions = [prediction(obs)]
        report = self.score([obs], [truth(obs["source_id"])], predictions)
        with tempfile.TemporaryDirectory() as temp:
            output = evaluate.save_evaluation(report, Path(temp)/"evaluation", predictions)
            loaded = evaluate.load_predictions(output/"predictions.jsonl")
            self.assertEqual(loaded[0]["case_id"], obs["case_id"])
            self.assertTrue((output/"per_case.csv").exists())
            self.assertTrue((output/"metrics.csv").exists())
            with self.assertRaises(FileExistsError):
                evaluate.save_evaluation(report, output)

    def test_baselines_do_not_need_truth(self):
        obs = observation(0)
        unavailable = observation(1, available=False)
        predictions = evaluate.generate_baseline_predictions([obs, unavailable])
        self.assertEqual(len(predictions), 4)
        self.assertEqual([p["status"] for p in predictions[-2:]], ["unavailable", "unavailable"])
        self.assertTrue(all(p["region"] is None and p["goal_probability"] is None for p in predictions))
        self.assertEqual({p["method"] for p in predictions}, {"G", "A"})


if __name__ == "__main__":
    unittest.main()
