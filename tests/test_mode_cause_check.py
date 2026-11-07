"""C 检查的固定格网、KDE、竞争峰与水平集定义。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import mode_cause_check as m


def draw(weight, point, event="goal_inside"):
    return {"weight": weight, "event": event, "goal_yz_m": list(point),
            "arrival_time_s": 0.4}


class ModeCauseCheckTests(unittest.TestCase):
    def test_grid_variants_are_three_widths_by_four_shifts(self):
        draws = [draw(0.6, (2.90, 1.95)), draw(0.4, (2.80, 1.90))]
        rows = m.grid_variants(draws)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["grid_width_m"] for r in rows}), 3)
        self.assertEqual(len({(r["shift_y_m"], r["shift_z_m"]) for r in rows}), 4)

    def test_kde_is_normalized_and_peak_is_near_center(self):
        rng = np.random.default_rng(7)
        points = rng.normal((2.5, 1.5), 0.05, size=(2000, 2))
        weights = np.full(2000, 1 / 2000)
        y, z, density = m.kde(points, weights, 0.10)
        mass = float(density.sum() * m.KDE_RESOLUTION ** 2)
        self.assertAlmostEqual(mass, 1.0, places=6)
        iy, iz = np.unravel_index(np.argmax(density), density.shape)
        self.assertLess(abs(y[iy] - 2.5), 0.05)
        self.assertLess(abs(z[iz] - 1.5), 0.05)

    def test_competing_peaks_use_ratio_and_separation(self):
        y = np.arange(1.5, 3.51, 0.01)
        z = np.arange(0.5, 3.01, 0.01)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        density = np.exp(-((yy - 2.0) ** 2 + (zz - 1.5) ** 2) / (2 * 0.08 ** 2))
        density += 0.9 * np.exp(-((yy - 2.4) ** 2 + (zz - 1.5) ** 2) / (2 * 0.08 ** 2))
        peaks = m.local_peaks(y, z, density)
        self.assertEqual(len(peaks), 2)
        density -= 0.5 * np.exp(-((yy - 2.4) ** 2 + (zz - 1.5) ** 2) / (2 * 0.08 ** 2))
        peaks = m.local_peaks(y, z, density)
        self.assertEqual(len(peaks), 1)

    def test_high_density_regions_are_connected_components(self):
        y = np.arange(1.5, 3.51, 0.01)
        z = np.arange(0.5, 3.01, 0.01)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        density = np.exp(-((yy - 2.0) ** 2 + (zz - 1.5) ** 2) / (2 * 0.08 ** 2))
        density += 0.7 * np.exp(-((yy - 2.4) ** 2 + (zz - 1.5) ** 2) / (2 * 0.08 ** 2))
        regions = m.high_density_regions(y, z, density)
        self.assertEqual(len(regions), 1)  # 第二峰低于 0.8 阈值
        density = np.exp(-((yy - 2.0) ** 2 + (zz - 1.5) ** 2) / (2 * 0.05 ** 2))
        density += np.exp(-((yy - 2.8) ** 2 + (zz - 1.5) ** 2) / (2 * 0.05 ** 2))
        regions = m.high_density_regions(y, z, density)
        self.assertEqual(len(regions), 2)
        self.assertGreater(regions[0]["mass"], 0)

    def test_prefix_keeps_positive_direct_candidates(self):
        draws = [draw(0.5, (2.9, 1.9)), draw(0.0, (3.0, 2.0)),
                 draw(0.5, (2.8, 1.8), "goal_outside"), draw(0.5, (2.7, 1.7))]
        selected = m.prefix(draws, 3)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item["weight"] > 0 for item in selected))


if __name__ == "__main__":
    unittest.main()
