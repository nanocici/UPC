"""开发区域检验的划分隔离、预测封存、失败分母与源级配对检查。"""

import copy
import gzip
import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import development_regions as dev
from config import CASE, ENVIRONMENT
from evaluate import evaluate_predictions


def observation(i=0, available=True):
    return {"schema_version": 1, "case_id": f"case_{i}", "source_id": "source_a" if i < 2 else "source_b",
        "split": "development", "observation_available": available, "case": asdict(CASE),
        "known_environment": asdict(ENVIRONMENT), "wind_known": False,
        "measurement": {"mode": "gaussian", "sigma_m": .01, "frame_correlation": 0., "rounding_step_m": None}}


def prediction(o, model, status="ok", threshold=1.):
    return {"schema_version": 1, "case_id": o["case_id"], "source_id": o["source_id"], "split": "development",
        "method": f"{model}_cd0.22_posterior", "status": status, "event": None, "point_kind": "mean",
        "point_yz_m": [0, 1] if status == "ok" else None, "arrival_time_s": .4 if status == "ok" else None,
        "region_mass": .9, "region": {"center_yz_m": [0, 1], "covariance_m2": [[1, 0], [0, 1]],
                                     "threshold": threshold} if status == "ok" else None,
        "goal_probability": .8 if status == "ok" else None, "message": "test"}


def example():
    observations = [observation(0), observation(1), observation(2, False)]
    truths = [{"source_id": source, "split": "development", "parameters": {"mechanism": "constant"},
               "truth": {"event": "goal", "event_position_m": [11, 0, 1], "event_time_s": .4,
                         "inside_goal_opening": True}} for source in ("source_a", "source_b")]
    predictions = [prediction(observations[0], "S"), prediction(observations[1], "S", "failure"),
        prediction(observations[2], "S", "unavailable"), prediction(observations[0], "F", threshold=4),
        prediction(observations[1], "F", threshold=4), prediction(observations[2], "F", "unavailable")]
    return observations, truths, predictions


class DevelopmentRegionTests(unittest.TestCase):
    def test_only_development_is_accepted(self):
        for rows in ([], [{**observation(), "split": "test"}], [{**observation(), "split": "calibration"}],
                     [observation(), observation()]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                dev.validate_development(rows)

    def test_missing_or_duplicate_task_blocks_assembly(self):
        obs = observation()
        entry = {"observation_index": 0, "model": "S", "case_id": obs["case_id"],
                 "prediction": prediction(obs, "S"), "map_prediction": prediction(obs, "S")}
        for records in ([], [entry, entry], [{**entry, "case_id": "wrong"}]):
            with self.subTest(records=records), patch.object(dev, "generate_baseline_predictions") as baseline:
                with self.assertRaises(ValueError):
                    dev.assemble_predictions([obs], records, ("S",))
                baseline.assert_not_called()

    def test_complete_tasks_are_assembled_once_with_baselines(self):
        obs = observation()
        entries = [{"observation_index": 0, "model": m, "case_id": obs["case_id"],
                    "prediction": prediction(obs, m), "map_prediction": prediction(obs, m)} for m in ("S", "F")]
        with patch.object(dev, "generate_baseline_predictions", return_value=[{"method": "G"}]):
            result = dev.assemble_predictions([obs], entries, ("S", "F"))
        self.assertEqual(len(result), 5)

    def test_unavailable_task_retains_probability_claim_and_map_status(self):
        with tempfile.TemporaryDirectory() as directory:
            record = dev.predict_task(observation(2, False), 2, "S", directory, base_samples=120, starts=1)
            self.assertEqual(record["prediction"]["status"], "unavailable")
            self.assertEqual(record["prediction"]["region_mass"], .9)
            self.assertEqual(record["map_prediction"]["status"], "unavailable")
            self.assertEqual(record["draw_count"], 0)
            with gzip.open(Path(directory)/"case_0002_S/candidates.jsonl.gz", "rt", encoding="utf-8") as f:
                self.assertEqual(f.read(), "")

    def test_real_small_sample_failure_is_preserved_not_relaxed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = dev.predict_task(observation(), 0, "S", directory, base_samples=120, starts=1, seed=7)
            # 240 次提议不可能满足默认 ESS>=300；点反演仍可作为 MAP 对照。
            self.assertEqual(result["prediction"]["status"], "failure")
            self.assertIsNone(result["prediction"]["region"])
            self.assertEqual(result["prediction"]["region_mass"], .9)
            self.assertEqual(result["map_prediction"]["status"], "ok")
            self.assertEqual(result["map_prediction"]["point_kind"], "map_trajectory")
            self.assertEqual(result["draw_count"], 240)
            with self.assertRaises(FileExistsError):
                dev.predict_task(observation(), 0, "S", directory, base_samples=120, starts=1)

    def test_paired_coverage_keeps_failures_and_unavailable_in_denominator(self):
        obs, truths, predictions = example()
        report = evaluate_predictions(obs, truths, predictions, bootstrap_samples=0)
        paired = dev.paired_region_comparison(report["rows"], bootstrap_samples=0)
        self.assertEqual(paired["true_direct_count"], 3)
        self.assertEqual(paired["common_point_count"], 1)
        self.assertEqual(paired["common_region_count"], 1)
        self.assertAlmostEqual(paired["estimates"]["coverage_F_minus_S"], 1/3)
        self.assertAlmostEqual(paired["estimates"]["available_coverage_F_minus_S"], .5)
        self.assertAlmostEqual(paired["estimates"]["failure_rate_F_minus_S"], -1/3)
        self.assertAlmostEqual(paired["estimates"]["median_area_F_minus_S_m2"], 3*math.pi)

    def test_source_bootstrap_preserves_correlated_copies(self):
        obs, truths, predictions = example()
        report = evaluate_predictions(obs, truths, predictions, bootstrap_samples=0)
        paired = dev.paired_region_comparison(report["rows"], bootstrap_samples=500, seed=9)
        self.assertEqual(paired["source_count"], 2)
        interval = paired["cluster_bootstrap_95_intervals"]["coverage_F_minus_S"]
        self.assertEqual(interval["valid_resamples"], 500)
        self.assertEqual(interval["lower"], 0.)
        self.assertEqual(interval["upper"], .5)

    def test_changed_prediction_seal_blocks_truth_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"predictions.jsonl"
            path.write_text("changed", encoding="utf-8")
            with patch.object(dev, "read_jsonl") as truth_reader:
                with self.assertRaises(ValueError):
                    dev.score_development(directory, [observation()], path, "wrong_hash", Path(directory)/"scores")
                truth_reader.assert_not_called()

    def test_evaluation_reads_only_development_after_seal(self):
        obs, truths, predictions = example()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"truth").mkdir()
            truth_path = root/"truth/development.jsonl"
            truth_path.write_text("".join(json.dumps(t)+"\n" for t in truths), encoding="utf-8")
            prediction_path = root/"predictions.jsonl"
            prediction_path.write_text("".join(json.dumps(p)+"\n" for p in predictions), encoding="utf-8")
            with patch.object(dev, "read_jsonl", wraps=dev.read_jsonl) as reader:
                report, _ = dev.score_development(root, obs, prediction_path, dev.file_hash(prediction_path),
                                                  root/"evaluation", bootstrap_samples=0)
            reader.assert_called_once_with(truth_path)
            self.assertEqual(report["split"], "development")
            self.assertEqual(report["methods"]["F_cd0.22_posterior"]["coverage"]["denominator"], 3)
            self.assertTrue((root/"evaluation/paired_regions.json").exists())


if __name__ == "__main__":
    unittest.main()
