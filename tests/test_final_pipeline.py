"""最终交付入口的冻结常量、区域/时间后处理与输出标签。"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import final_config
import final_pipeline


class FinalPipelineTests(unittest.TestCase):
    def test_frozen_constants(self):
        cfg = final_config.FINAL
        self.assertEqual(cfg.model, "F")
        self.assertAlmostEqual(cfg.b_y_prior_std, 0.10)
        self.assertAlmostEqual(cfg.b_z_prior_std, 0.20)
        self.assertAlmostEqual(cfg.drag_coefficient, 0.22)
        self.assertAlmostEqual(cfg.time_q_s, 0.015188910308210842)
        self.assertAlmostEqual(cfg.spatial_threshold, 3.083525450931334)
        self.assertAlmostEqual(cfg.mode_bandwidth_m, 0.10)
        self.assertEqual(cfg.smooth_mode_yz_m, (2.875, 1.925))
        self.assertEqual(cfg.raw_grid_mode_yz_m, (2.8975, 1.8808))
        self.assertEqual(cfg.mean_auxiliary_yz_m, (2.9057531544942057, 1.9285065612977923))

    def test_inference_settings_use_f1_prior(self):
        settings = final_config.FINAL.inference_settings()
        self.assertAlmostEqual(settings.vertical_force_prior_std, 0.10)
        self.assertAlmostEqual(settings.transverse_prior_std, 0.20)

    def test_calibrated_time_interval_formula(self):
        median = 0.38768793313392635
        interval = (0.38273881673645554, 0.39430455129178865)
        low, high = final_pipeline.calibrated_time_interval(
            median, interval, final_config.FINAL.time_q_s)
        self.assertAlmostEqual(low, 0.3714354087053343, places=12)
        self.assertAlmostEqual(high, 0.4039404575625184, places=12)
        self.assertAlmostEqual(high - low, 0.03250504885718403, places=12)

    def test_calibrated_region_area(self):
        summary = json.loads((ROOT / "results/final_case_f1/summary.json").read_text())
        region = summary["summary"]["direct_crossing"]["region"]
        calibrated = final_pipeline.calibrated_region(region, final_config.FINAL.spatial_threshold)
        self.assertAlmostEqual(calibrated["threshold"], 3.083525450931334)
        self.assertAlmostEqual(calibrated["area_m2"], 0.7658899468897252, places=12)
        self.assertEqual(calibrated["center_yz_m"], region["center_yz_m"])

    def test_assemble_case_labels_and_three_points(self):
        summary = json.loads((ROOT / "results/final_case_f1/summary.json").read_text())
        mode = json.loads(final_pipeline.MODE_PATH.read_text())
        case = final_pipeline.assemble_case(summary, mode)
        self.assertEqual(case["primary_point"]["kind"], "fixed_bandwidth_smooth_mode")
        self.assertAlmostEqual(case["primary_point"]["y_m"], 2.875)
        self.assertAlmostEqual(case["primary_point"]["z_m"], 1.925)
        self.assertEqual(case["mean_auxiliary"]["kind"], "conditional_mean_auxiliary")
        self.assertEqual(case["raw_grid_64k"]["kind"], "raw_grid_mode")
        self.assertAlmostEqual(case["raw_grid_64k"]["y_m"], 2.8975)
        self.assertAlmostEqual(case["raw_grid_64k"]["z_m"], 1.8808)
        self.assertIn("model_internal", case["spatial"])
        self.assertIn("calibrated", case["spatial"])
        self.assertAlmostEqual(case["spatial"]["calibrated"]["area_m2"], 0.7658899468897252, places=12)
        self.assertAlmostEqual(case["time"]["calibrated_interval_s"][0], 0.3714354087053343, places=12)
        self.assertAlmostEqual(case["time"]["calibrated_interval_s"][1], 0.4039404575625184, places=12)

    def test_mean_is_never_primary(self):
        summary = json.loads((ROOT / "results/final_case_f1/summary.json").read_text())
        mode = json.loads(final_pipeline.MODE_PATH.read_text())
        case = final_pipeline.assemble_case(summary, mode)
        self.assertNotEqual(case["primary_point"]["kind"], "conditional_mean")
        self.assertNotEqual(case["primary_point"]["kind"], "conditional_mean_auxiliary")


if __name__ == "__main__":
    unittest.main()
