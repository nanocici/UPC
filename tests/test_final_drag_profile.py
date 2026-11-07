"""Checks for profile bookkeeping and refusal to publish unconfirmed goal points."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import final_drag_profile as profile
from config import CASE
from inverse import fit_case
from physics import simulate


class FinalDragProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = fit_case("F", settings=profile.FINAL.inference_settings(), starts=3)

    def test_grid_and_current_variants(self):
        grid = profile.drag_grid()
        self.assertEqual(len(grid), 26)
        self.assertEqual((grid[0], grid[-1]), (.08, .55))
        self.assertEqual(grid.count(.22), 1)
        self.assertEqual(profile.variants()["F1_MAP"][2].vertical_force_prior_std, .10)
        self.assertEqual(profile.variants()["S_MAP"][2], profile.INFERENCE)

    def test_only_ls_is_pure_data_profile(self):
        self.assertTrue(self.fit["success"])
        record = {"variant": "F1_MAP", "cd": .22, "fit": self.fit}
        self.assertIsNone(profile.flat_row(record)["pure_data_profile_R"])
        record = copy.deepcopy(record)
        record["fit"]["mode"] = "ls"
        self.assertEqual(profile.flat_row(record)["pure_data_profile_R"], self.fit["data_chi_square"])

    def test_non_goal_event_never_has_goal_coordinates(self):
        for event in ("ground", "boundary", "failure", "unresolved"):
            self.assertEqual(profile.goal_values({"event": event, "event_position_m": [11, 2, 0]}),
                             {"y_m": None, "z_m": None, "time_s": None})

    def test_verification_rejects_position_drift_and_failed_fit(self):
        a = {"fit": self.fit}
        self.assertTrue(profile.compare_fits(a, a)["pass"])
        b = copy.deepcopy(a)
        b["fit"]["simulation"]["event_position_m"] = list(b["fit"]["simulation"]["event_position_m"])
        b["fit"]["simulation"]["event_position_m"][1] += .01
        self.assertFalse(profile.compare_fits(a, b)["pass"])
        b["fit"]["success"] = False
        self.assertFalse(profile.compare_fits(a, b)["pass"])

    def test_forward_scan_does_not_refit_or_change_other_parameters(self):
        records = [{"variant": v, "cd": .22, "fit": self.fit} for v in ("F0_MAP", "F1_MAP", "S_MAP")]
        # S requires by=0; supply its actual nominal fit as the runner does.
        records[-1]["fit"] = fit_case("S", starts=1)
        with patch.object(profile, "fit_case", side_effect=AssertionError("forward scan must not refit")):
            rows = profile.forward_scan(records, [.08, .22, .55])
        for row in rows:
            nominal = next(r for r in records if r["variant"] == row["variant"])["fit"]["parameters"]
            for key in ("initial_velocity_m_s", "b_y", "b_z"):
                self.assertEqual(row["parameters"][key], nominal[key])
            self.assertEqual(row["parameters"]["drag_coefficient"], row["cd"])

    def test_ls_residual_not_worse_than_feasible_fixed_trajectory(self):
        p = dict(self.fit["parameters"], drag_coefficient=.4)
        sim = simulate("F", p, CASE.times_s)
        chi = np.sum(((np.asarray(sim["positions_m"])-CASE.positions_m)/.01)**2)
        ls = fit_case("F", mode="ls", drag_coefficient=.4, starts=3)
        self.assertTrue(ls["success"])
        self.assertLessEqual(ls["data_chi_square"], chi+1e-5)


if __name__ == "__main__":
    unittest.main()
