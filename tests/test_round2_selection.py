"""第二轮选择实验的指标口径测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import round2_selection as r2


def row(method, *, direct=True, covered=True, provided=True, point=True,
        error=0.10, area=1.0, failure=False, failure_class="none",
        mechanism="constant", time_covered=True):
    return {
        "method": method, "direct": direct, "provided": provided,
        "point_valid": point, "covered": covered if direct and provided else False,
        "covered_all": covered if direct else None,
        "axis_y_covered_all": covered if direct else None,
        "axis_z_covered_all": covered if direct else None,
        "axis_y_covered_provided": covered if direct and provided else None,
        "axis_z_covered_provided": covered if direct and provided else None,
        "primary_failure": failure, "failure_class": failure_class,
        "error_m": error if point else None, "area_m2": area if provided else None,
        "time_valid": direct, "time_covered": time_covered if direct else None,
        "time_error_s": 0.005 if direct else None, "time_width_s": 0.02 if direct else None,
        "scenario": mechanism, "source_id": "source_0", "case_id": "case_0",
    }


class Round2MetricTests(unittest.TestCase):
    def test_stable_seed_depends_on_case_not_order(self):
        self.assertEqual(r2.stable_seed("case_a"), r2.stable_seed("case_a"))
        self.assertNotEqual(r2.stable_seed("case_a"), r2.stable_seed("case_b"))

    def test_paired_metrics_use_common_cases(self):
        pairs = []
        for i, (c0, c1, f0, f1, a0, a1, e0, e1) in enumerate([
            (True, True, False, False, 1.0, 0.5, 0.20, 0.10),
            (True, True, False, False, 1.0, 0.5, 0.20, 0.10),
            (False, True, True, False, 1.0, 0.5, 0.20, 0.10),
            (False, False, True, True, 1.0, 0.5, 0.20, 0.10),
        ]):
            a = row(r2.METHODS["F0"], covered=c0, area=a0, error=e0, failure=f0)
            b = row(r2.METHODS["F1"], covered=c1, area=a1, error=e1, failure=f1)
            a["case_id"] = b["case_id"] = f"case_{i}"
            pairs.append((a, b))
        metrics = r2._bootstrap_metrics(pairs)
        self.assertAlmostEqual(metrics["coverage_diff"], 0.25)
        self.assertAlmostEqual(metrics["failure_diff"], -0.25)
        self.assertAlmostEqual(metrics["area_ratio"], 0.5)
        self.assertAlmostEqual(metrics["median_error_diff"], -0.10)
        self.assertAlmostEqual(metrics["p90_error_diff"], -0.10)

    def test_missing_common_cases_are_not_fabricated(self):
        a = row(r2.METHODS["F0"], provided=False, point=False)
        b = row(r2.METHODS["F1"], provided=False, point=False)
        a["case_id"] = b["case_id"] = "case_0"
        metrics = r2._bootstrap_metrics([(a, b)])
        self.assertIsNone(metrics["area_ratio"])
        self.assertIsNone(metrics["median_error_diff"])
        self.assertIsNone(metrics["p90_error_diff"])

    def test_method_summary_has_two_denominators_and_failure_class(self):
        rows = [row(r2.METHODS["F0"], covered=True, provided=True),
                row(r2.METHODS["F0"], covered=False, provided=True),
                row(r2.METHODS["F0"], covered=False, provided=False, point=False,
                    failure=True, failure_class="sampling")]
        rows[2]["failure_class"] = "sampling"
        summary = r2._method_summary(rows)
        self.assertEqual(summary["direct_n"], 3)
        self.assertAlmostEqual(summary["coverage_all"], 1 / 3)
        self.assertAlmostEqual(summary["coverage_provided"], 1 / 2)
        self.assertAlmostEqual(summary["publishable_rate"], 2 / 3)
        self.assertEqual(summary["failure_counts"]["sampling"], 1)
        self.assertAlmostEqual(summary["primary_failure_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
