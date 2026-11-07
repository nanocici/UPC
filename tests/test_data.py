"""独立生成端、误差、数据分组、答案隔离及读取检查。"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import data
from config import CASE, ENVIRONMENT, SOLVER
from physics import simulate  # 仅测试端对照同方程极限；生成模块不依赖它。


class TruthTests(unittest.TestCase):
    def test_constant_generator_matches_forward_limit(self):
        parameters = data.TruthParameters(mechanism="constant")
        truth = data.generate_truth(parameters)
        predicted = simulate("F", {
            "initial_velocity_m_s": parameters.initial_velocity_m_s,
            "drag_coefficient": parameters.drag_coefficient,
            "b_y": parameters.transverse_vector[1], "b_z": parameters.transverse_vector[2],
        }, CASE.times_s)
        np.testing.assert_allclose(truth["clean_positions_m"], predicted["positions_m"], atol=1e-7)
        self.assertEqual(truth["event"], predicted["event"])
        self.assertAlmostEqual(truth["event_time_s"], predicted["event_time_s"], places=7)

    def test_zero_air_matches_ballistic_solution(self):
        parameters = data.TruthParameters(mechanism="constant", drag_coefficient=0,
                                           transverse_vector=(0, 0, 0))
        truth = data.generate_truth(parameters)
        for t, position in zip(CASE.times_s, truth["clean_positions_m"]):
            expected = np.asarray(CASE.initial_position_m) + t*np.asarray(parameters.initial_velocity_m_s)
            expected[2] -= .5*ENVIRONMENT.gravity_m_s2*t*t
            np.testing.assert_allclose(position, expected, atol=1e-9)

    def test_zero_spin_and_parallel_spin_have_no_magnus_force(self):
        for omega in ((0, 0, 0), (50, 0, 0)):
            parameters = data.TruthParameters(mechanism="spin", spin_rad_s=omega)
            acceleration = data.truth_acceleration((30, 0, 0), parameters)
            self.assertAlmostEqual(acceleration[1], 0, places=12)
            self.assertAlmostEqual(acceleration[2], -ENVIRONMENT.gravity_m_s2, places=12)
        environment = replace(ENVIRONMENT, wind_m_s=(30, 0, 0))
        np.testing.assert_allclose(data.truth_acceleration((30, 0, 0), parameters, environment),
                                   (0, 0, -ENVIRONMENT.gravity_m_s2), atol=1e-12)

    def test_three_dimensional_spin_does_no_relative_work(self):
        environment = replace(ENVIRONMENT, wind_m_s=(1, -2, .5))
        p = data.TruthParameters(mechanism="spin", spin_rad_s=(30, -40, 50))
        v = np.array((30, 8, 6))
        u = v-np.asarray(environment.wind_m_s)
        acceleration = data.truth_acceleration(v, p, environment)
        air = acceleration + np.array((0, 0, environment.gravity_m_s2))
        power = environment.mass_kg*np.dot(u, air)
        expected = -.5*environment.air_density_kg_m3*environment.area_m2*p.drag_coefficient*np.linalg.norm(u)**3
        self.assertAlmostEqual(power, expected, places=9)

    def test_variable_drag_curve_is_bounded(self):
        p = data.TruthParameters(mechanism="variable_drag", transverse_vector=(0, 0, 0))
        coefficients = []
        for speed in (1, 20, 50):
            a = data.truth_acceleration((speed, 0, 0), p)
            cd = -a[0]/(ENVIRONMENT.aerodynamic_k_per_m*speed**2)
            self.assertGreaterEqual(cd, p.drag_low)
            self.assertLessEqual(cd, p.drag_high)
            coefficients.append(cd)
        self.assertGreater(coefficients[0], coefficients[-1])

    def test_ground_before_second_frame_is_retained(self):
        p = data.TruthParameters(initial_velocity_m_s=(5, 0, .3), mechanism="constant",
                                drag_coefficient=0, transverse_vector=(0, 0, 0))
        truth = data.generate_truth(p)
        self.assertEqual(truth["event"], "ground")
        self.assertIsNotNone(truth["clean_positions_m"][0])
        self.assertIsNone(truth["clean_positions_m"][1])
        self.assertFalse(truth["observation_available"])
        self.assertLessEqual(max(truth["trajectory_times_s"]), truth["event_time_s"])

    def test_initial_contact_and_boundary(self):
        p = data.TruthParameters(initial_velocity_m_s=(5, 0, 0))
        self.assertEqual(data.generate_truth(p)["event_time_s"], 0)
        p = data.TruthParameters(initial_velocity_m_s=(22, 0, ENVIRONMENT.gravity_m_s2*.5/2),
                                mechanism="constant", drag_coefficient=0, transverse_vector=(0, 0, 0))
        self.assertEqual(data.generate_truth(p)["event"], "boundary")

    def test_unresolved_and_failure_states(self):
        case = replace(CASE, initial_position_m=(0, 0, 1))
        environment = replace(ENVIRONMENT, gravity_m_s2=0)
        p = data.TruthParameters(initial_velocity_m_s=(0, 0, 1), mechanism="constant",
                                drag_coefficient=0, transverse_vector=(0, 0, 0))
        truth = data.generate_truth(p, case, environment)
        self.assertEqual(truth["event"], "unresolved")
        self.assertTrue(truth["observation_available"])
        with patch.object(data, "solve_ivp", side_effect=FloatingPointError("test")):
            failed = data.generate_truth(p, case, environment)
        self.assertEqual(failed["event"], "failure")
        self.assertFalse(failed["observation_available"])

    def test_tighter_generator_is_converged_on_spin_case(self):
        p = data.TruthParameters()
        base = data.generate_truth(p)
        tighter = replace(SOLVER, relative_tolerance=SOLVER.relative_tolerance/10,
                          max_step_s=SOLVER.max_step_s/2)
        refined = data.generate_truth(p, solver=tighter)
        self.assertEqual(base["event"], refined["event"])
        np.testing.assert_allclose(base["event_position_m"], refined["event_position_m"], atol=.005)
        self.assertLess(abs(base["event_time_s"]-refined["event_time_s"]), .001)


class MeasurementTests(unittest.TestCase):
    def test_reproducible_gaussian_and_preserved_missing_frame(self):
        clean = ((1, 2, 3), None)
        first = data.apply_measurement_error(clean, np.random.default_rng(8))
        second = data.apply_measurement_error(clean, np.random.default_rng(8))
        self.assertEqual(first, second)
        self.assertIsNone(first[1])
        self.assertNotEqual(first[0], list(clean[0]))

    def test_rounding_bound_and_no_noise(self):
        clean = np.array(((1.234, -2.127, .114), (3.149, 2.111, .871)))
        rounded = data.apply_measurement_error(clean, np.random.default_rng(1), mode="rounding")
        self.assertLessEqual(np.max(np.abs(np.asarray(rounded)-clean)), .005+1e-12)
        exact = data.apply_measurement_error(clean, np.random.default_rng(1), mode="none")
        np.testing.assert_array_equal(exact, clean)

    def test_shared_bias_at_full_correlation(self):
        clean = np.array(((1, 2, 3), (2, 4, 6)), dtype=float)
        observed = data.apply_measurement_error(clean, np.random.default_rng(4), frame_correlation=1)
        errors = np.asarray(observed)-clean
        np.testing.assert_allclose(errors[0], errors[1], atol=1e-14)

    def test_invalid_settings(self):
        for kwargs in ({"sigma_m": -1}, {"frame_correlation": 2}, {"mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                data.apply_measurement_error([], np.random.default_rng(0), **kwargs)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = data.build_dataset(15, seed=42)

    def test_source_grouping_no_split_leakage(self):
        groups = {}
        for row in self.dataset["observations"]:
            groups.setdefault(row["source_id"], set()).add(row["split"])
        self.assertEqual(len(groups), 15)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(len(self.dataset["observations"]), 15*4)
        self.assertEqual(self.dataset["manifest"]["source_counts_by_split"],
                         {"development": 9, "calibration": 3, "test": 3})
        self.assertEqual(sum(self.dataset["manifest"]["event_counts"].values()), 15)

    def test_development_only_split_is_supported(self):
        dataset = data.build_dataset(5, seed=42, split_fractions=(1, 0, 0))
        self.assertEqual(dataset["manifest"]["source_counts_by_split"],
                         {"development": 5, "calibration": 0, "test": 0})
        self.assertTrue(all(row["split"] == "development" for row in dataset["observations"]))
        with self.assertRaises(ValueError):
            data.build_dataset(5, seed=42, split_fractions=(0, 0, 0))

    def test_inputs_exclude_truth_and_hide_unknown_wind(self):
        for row in self.dataset["observations"]:
            self.assertNotIn("parameters", row)
            self.assertNotIn("truth", row)
            self.assertNotIn("source_seed", row)
            self.assertNotIn("mechanism", row)
            self.assertEqual(row["known_environment"]["wind_m_s"], (0, 0, 0))
        wind_truth = [row for row in self.dataset["truth"] if row["parameters"]["mechanism"] == "wind"]
        self.assertTrue(any(np.linalg.norm(row["true_environment"]["wind_m_s"]) > 0 for row in wind_truth))

    def test_reproducibility(self):
        first = data.build_dataset(3, seed=7)
        second = data.build_dataset(3, seed=7)
        self.assertEqual(first, second)
        changed = data.build_dataset(3, seed=8)
        self.assertNotEqual(first["truth"], changed["truth"])

    def test_roundtrip_and_refuse_truth_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = data.save_dataset(self.dataset, Path(temp)/"dataset")
            rows = data.load_observations(output/"observations/development.jsonl")
            self.assertEqual(len(rows), 9*4)
            available = next(row for row in rows if row["observation_available"])
            case = data.observation_case(available)
            self.assertEqual(case.times_s, CASE.times_s)
            self.assertEqual(len(case.positions_m), 2)
            with self.assertRaises(ValueError):
                data.load_observations(output/"truth/development.jsonl")
            with self.assertRaises(FileExistsError):
                data.save_dataset(self.dataset, output)

    def test_explicit_unavailable_rejection(self):
        with self.assertRaises(ValueError):
            data.observation_case({"observation_available": False})

    def test_invalid_dataset_settings(self):
        for kwargs in ({"source_count": 0}, {"seed": -1}, {"mechanisms": ()},
                       {"noise_sigmas_m": (-1,)}, {"split_fractions": (.5, .5, .5)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                data.build_dataset(**kwargs)

    def test_rounding_mode_has_one_copy_and_known_wind(self):
        dataset = data.build_dataset(2, noise_mode="rounding", mechanisms=("wind",), wind_known=True)
        self.assertEqual(len(dataset["observations"]), 2)
        for observed, truth in zip(dataset["observations"], dataset["truth"]):
            self.assertEqual(observed["known_environment"], truth["true_environment"])
            self.assertEqual(observed["measurement"]["sigma_m"], 0)


class CSVTests(unittest.TestCase):
    def test_explicit_units_and_frame_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"trajectory.csv"
            path.write_text("time,x,y,z\n0,0,0,1.1\n10,15,4.5,4.4\n", encoding="utf-8")
            trajectory = data.load_trajectory_csv(path, length_unit="dm", time_unit="frame", fps=200)
            np.testing.assert_allclose(trajectory["times_s"], (0, .05))
            np.testing.assert_allclose(trajectory["positions_m"], ((0, 0, .11), (1.5, .45, .44)))
            with self.assertRaises(ValueError):
                data.load_trajectory_csv(path, length_unit="dm", time_unit="frame")

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"trajectory.csv"
            path.write_text("time,x,y,z\n0,0,0,1\n0,1,0,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                data.load_trajectory_csv(path, length_unit="m", time_unit="s")


if __name__ == "__main__":
    unittest.main()
