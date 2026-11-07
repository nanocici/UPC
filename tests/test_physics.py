"""物理与事件检查；不代表已验证两帧反演的预测精度。"""

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import physics
from config import CASE, ENVIRONMENT, SOLVER


def parameters(velocity=(30.0, 9.0, 7.0), cd=0.22, by=0.05, bz=-0.15):
    return {"initial_velocity_m_s": velocity, "drag_coefficient": cd,
            "b_y": by, "b_z": bz}


class PhysicsTests(unittest.TestCase):
    def test_ballistic_limit_against_closed_form(self):
        p = parameters(cd=0, by=0, bz=0)
        result = physics.simulate("F", p, CASE.times_s)
        v = np.asarray(p["initial_velocity_m_s"])
        for t, position in zip(CASE.times_s, result["positions_m"]):
            expected = np.asarray(CASE.initial_position_m) + v * t
            expected[2] -= 0.5 * ENVIRONMENT.gravity_m_s2 * t**2
            np.testing.assert_allclose(position, expected, atol=1e-9, rtol=0)
        self.assertEqual(result["event"], "goal")
        self.assertAlmostEqual(result["event_time_s"], CASE.goal_x_m / v[0], places=10)

    def test_pure_drag_against_closed_form(self):
        environment = replace(ENVIRONMENT, gravity_m_s2=0)
        case = replace(CASE, initial_position_m=(0.0, 0.0, 1.0))
        p = parameters(velocity=(30.0, 0.0, 0.0), cd=0.3, by=0, bz=0)
        result = physics.simulate("F", p, CASE.times_s, case, environment)
        kcd = environment.aerodynamic_k_per_m * p["drag_coefficient"]
        for t, position in zip(CASE.times_s, result["positions_m"]):
            self.assertAlmostEqual(position[0], math.log1p(kcd * 30 * t) / kcd, places=7)
        expected_time = math.expm1(kcd * case.goal_x_m) / (kcd * 30)
        self.assertAlmostEqual(result["event_time_s"], expected_time, places=8)

    def test_no_force_uniform_motion(self):
        environment = replace(ENVIRONMENT, gravity_m_s2=0, air_density_kg_m3=0)
        case = replace(CASE, initial_position_m=(0.0, 0.0, 1.0))
        result = physics.simulate("F", parameters(velocity=(20.0, 1.0, 0.0)),
                                  (0.1, 0.2), case, environment)
        np.testing.assert_allclose(result["positions_m"], ((2, .1, 1), (4, .2, 1)), atol=1e-10)
        self.assertAlmostEqual(result["event_time_s"], 11 / 20, places=10)

    def test_lift_perpendicular_and_drag_power_in_wind(self):
        for wind in ((0.0, 0.0, 0.0), (1.0, -2.0, 0.5)):
            with self.subTest(wind=wind):
                environment = replace(ENVIRONMENT, wind_m_s=wind)
                velocity = np.array((29.0, 6.0, 4.0))
                p = parameters(by=0.2, bz=-0.3)
                a = physics.acceleration(velocity, p, environment)
                u = velocity - wind
                gravity = np.array((0.0, 0.0, -environment.gravity_m_s2))
                speed = np.linalg.norm(u)
                drag = -environment.aerodynamic_k_per_m * speed * p["drag_coefficient"] * u
                lift = a - gravity - drag
                self.assertAlmostEqual(float(np.dot(lift, u)), 0, places=10)
                power = environment.mass_kg * np.dot(u, a) + (
                    environment.mass_kg * environment.gravity_m_s2 * u[2])
                expected = -0.5 * environment.air_density_kg_m3 * environment.area_m2 * p["drag_coefficient"] * speed**3
                self.assertAlmostEqual(float(power), expected, places=9)

    def test_horizontal_direction_under_pure_drag(self):
        p = parameters(by=0, bz=0)
        result = physics.simulate("F", p, CASE.times_s)
        for x, y, z in result["positions_m"]:
            self.assertAlmostEqual(y / x, 9 / 30, places=10)

    def test_s_matches_restricted_f(self):
        p = parameters(by=0)
        s = physics.simulate("S", p, CASE.times_s)
        f = physics.simulate("F", p, CASE.times_s)
        self.assertEqual(s, f)

    def test_mirror_symmetry_with_wind(self):
        p = parameters()
        environment = replace(ENVIRONMENT, wind_m_s=(1, 2, 0))
        normal = physics.simulate("F", p, CASE.times_s, environment=environment)
        reflected_p = parameters(velocity=(30, -9, 7), by=p["b_y"], bz=-p["b_z"])
        reflected = physics.simulate("F", reflected_p, CASE.times_s,
                                     environment=replace(environment, wind_m_s=(1, -2, 0)))
        expected = np.asarray(normal["event_position_m"]) * (1, -1, 1)
        np.testing.assert_allclose(reflected["event_position_m"], expected, atol=1e-8)
        self.assertAlmostEqual(normal["event_time_s"], reflected["event_time_s"], places=9)

    def test_tighter_tolerance_and_half_step(self):
        p = parameters()
        normal = physics.simulate("F", p, CASE.times_s)
        tighter = replace(SOLVER, relative_tolerance=SOLVER.relative_tolerance/10,
                          position_absolute_tolerance_m=SOLVER.position_absolute_tolerance_m/10,
                          velocity_absolute_tolerance_m_s=SOLVER.velocity_absolute_tolerance_m_s/10,
                          max_step_s=SOLVER.max_step_s/2)
        refined = physics.simulate("F", p, CASE.times_s, solver=tighter)
        self.assertEqual(normal["event"], refined["event"])
        self.assertLess(np.linalg.norm(np.asarray(normal["event_position_m"]) - refined["event_position_m"]), .005)
        self.assertLess(abs(normal["event_time_s"] - refined["event_time_s"]), .001)

    def test_launch_zero_is_not_landing(self):
        p = parameters(velocity=(5, 0, 1), cd=0, by=0, bz=0)
        result = physics.simulate("G", p, (0, .05, .3))
        self.assertEqual(result["event"], "ground")
        self.assertAlmostEqual(result["event_time_s"], 2/ENVIRONMENT.gravity_m_s2, places=9)
        self.assertEqual(result["positions_m"][0], CASE.initial_position_m)
        self.assertIsNotNone(result["positions_m"][1])
        self.assertIsNone(result["positions_m"][2])

    def test_very_short_hop_is_not_zero_root(self):
        result = physics.simulate("G", parameters(velocity=(1, 0, .001)), (0, .001))
        self.assertEqual(result["event"], "ground")
        self.assertAlmostEqual(result["event_time_s"], .002/ENVIRONMENT.gravity_m_s2, delta=1e-8)
        self.assertIsNone(result["positions_m"][1])

    def test_initial_contact(self):
        for vz in (0, -1):
            result = physics.simulate("G", parameters(velocity=(30, 0, vz)), (0, .05))
            self.assertEqual(result["event"], "ground")
            self.assertEqual(result["event_time_s"], 0)
            self.assertIsNone(result["positions_m"][1])

    def test_goal_plane_is_not_necessarily_inside(self):
        result = physics.simulate("G", parameters(velocity=(30, 15, 7)), CASE.times_s)
        self.assertEqual(result["event"], "goal")
        self.assertFalse(physics.is_in_goal_opening(result["event_position_m"]))
        for position in ((11, 3.66, 1), (11, -3.66, 1), (11, 0, 0), (11, 0, 2.44)):
            self.assertFalse(physics.is_in_goal_opening(position))

    def test_simultaneous_goal_and_ground(self):
        g = ENVIRONMENT.gravity_m_s2
        result = physics.simulate("G", parameters(velocity=(22, 0, g * .5/2)), (.1, .5, .6))
        self.assertEqual(result["event"], "boundary")
        self.assertAlmostEqual(result["event_time_s"], .5, places=9)
        self.assertIsNone(result["positions_m"][-1])

    def test_nearly_simultaneous_events(self):
        landing_time = .5 - SOLVER.event_time_tolerance_s/2
        result = physics.simulate("G", parameters(velocity=(22, 0, ENVIRONMENT.gravity_m_s2*landing_time/2)), ())
        self.assertEqual(result["event"], "boundary")

    def test_initial_goal_and_initial_boundary(self):
        case = replace(CASE, initial_position_m=(11, 0, 1))
        goal = physics.simulate("G", parameters(), (0, .05), case)
        self.assertEqual(goal["event"], "goal")
        boundary_case = replace(CASE, initial_position_m=(11, 0, ENVIRONMENT.radius_m))
        boundary = physics.simulate("G", parameters(velocity=(30, 0, -1)), (), boundary_case)
        self.assertEqual(boundary["event"], "boundary")

    def test_extend_time_window(self):
        environment = replace(ENVIRONMENT, gravity_m_s2=0)
        case = replace(CASE, initial_position_m=(0, 0, 1))
        result = physics.simulate("G", parameters(velocity=(4, 0, 0)), (.1, 2.5, 2.8), case, environment)
        self.assertEqual(result["event"], "goal")
        self.assertAlmostEqual(result["event_time_s"], 2.75, places=9)
        np.testing.assert_allclose(result["positions_m"][1], (10, 0, 1), atol=1e-9)
        self.assertIsNone(result["positions_m"][2])

    def test_unresolved(self):
        environment = replace(ENVIRONMENT, gravity_m_s2=0)
        case = replace(CASE, initial_position_m=(0, 0, 1))
        result = physics.simulate("G", parameters(velocity=(0, 0, 1)), (0.1, 3), case, environment)
        self.assertEqual(result["event"], "unresolved")
        self.assertIsNone(result["event_time_s"])
        self.assertIsNone(result["event_position_m"])
        self.assertIsNotNone(result["positions_m"][-1])

    def test_numeric_failure_is_explicit(self):
        with patch.object(physics, "solve_ivp", side_effect=FloatingPointError("test")):
            result = physics.simulate("F", parameters(), (0, .05))
        self.assertEqual(result["event"], "failure")
        self.assertIsNone(result["event_time_s"])
        self.assertEqual(result["positions_m"], (CASE.initial_position_m, None))

    def test_invalid_inputs_rejected(self):
        trials = [
            lambda: physics.simulate("S", parameters(by=.1), CASE.times_s),
            lambda: physics.simulate("H", parameters(), CASE.times_s),
            lambda: physics.simulate("F", parameters(cd=-1), CASE.times_s),
            lambda: physics.simulate("F", parameters(), (.1, .1)),
            lambda: physics.simulate("F", parameters(), (-.1,)),
            lambda: physics.simulate("F", parameters(), (float("nan"),)),
            lambda: physics.simulate("F", parameters(), (5,)),
            lambda: physics.simulate("F", parameters(), (), environment=replace(ENVIRONMENT, mass_kg=0)),
            lambda: physics.simulate("F", parameters(), (), case=replace(CASE, initial_position_m=(0, 0, 0))),
            lambda: physics.simulate("F", parameters(), (), solver=replace(SOLVER, method="typo")),
        ]
        for trial in trials:
            with self.subTest(trial=trial), self.assertRaises(ValueError):
                trial()

    def test_baselines_fit_and_choose_first_root(self):
        velocity, a = physics.constant_acceleration_parameters()
        for t, position in zip(CASE.times_s, CASE.positions_m):
            np.testing.assert_allclose(np.asarray(CASE.initial_position_m)
                                       + np.asarray(velocity)*t + .5*np.asarray(a)*t*t,
                                       position, atol=1e-10)
        result = physics.predict_baseline("A")
        roots = np.roots((a[0]/2, velocity[0], -CASE.goal_x_m))
        expected = min(float(t) for t in roots if t > 0)
        self.assertEqual(result["event"], "goal")
        self.assertAlmostEqual(result["event_time_s"], expected, places=8)
        self.assertGreater(velocity[0] + a[0]*expected, 0)
        gravity = physics.gravity_initial_velocity()
        self.assertAlmostEqual(physics.predict_baseline("G")["event_time_s"], 11/gravity[0], places=9)

    def test_nonzero_initial_time(self):
        p = parameters()
        normal = physics.simulate("F", p, CASE.times_s)
        shifted_case = replace(CASE, initial_time_s=1, times_s=(1.05, 1.10))
        shifted = physics.simulate("F", p, shifted_case.times_s, shifted_case)
        np.testing.assert_allclose(shifted["positions_m"], normal["positions_m"], atol=1e-8)
        self.assertAlmostEqual(shifted["event_time_s"], normal["event_time_s"]+1, places=8)


if __name__ == "__main__":
    unittest.main()
