"""Acceptance and observer regressions; no ROS or hardware required."""
import dataclasses
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hri_legible_motion import legibility_optimizer as L


class PlanarAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.start = np.array([.256, -.152, .113])
        self.goals = np.array([[.618, .626, .113], [.721, .555, .113],
                               [.782, .470, .113], [.833, .371, .113]])
        self.t, self.xyz = L.sample(L.baseline_path(self.start, self.goals[0]), 5, rate=20)
        self.cfg = L.PlanarConfig(length_ratio=1.04, n_starts=2)
        d = (self.goals[0] - self.start)[:2]
        side = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        self.good = np.zeros(12)
        self.good[0], self.good[6] = .02 * side

    def solve(self, results, **kwargs):
        with patch.object(L, "minimize", side_effect=results):
            return L.optimize_planar_reach(self.t, self.xyz, self.goals, 0, self.cfg, **kwargs)

    def result(self, coefficients, success=True, fun=-999):
        return SimpleNamespace(x=coefficients, success=success, fun=fun, nit=4, message="test result")

    def test_failed_lower_objective_cannot_displace_converged_candidate(self):
        out = self.solve([self.result(self.good), self.result(self.good * 2, False, -1e9)])
        self.assertTrue(out.success, out.diagnostics)
        self.assertTrue(out.numerical_success)
        self.assertTrue(out.geometric_valid)
        self.assertEqual(out.physical_validity, "not_checked")
        np.testing.assert_array_equal(out.xyz[[0, -1]], self.xyz[[0, -1]])
        np.testing.assert_array_equal(out.times, self.t)
        self.assertGreater(out.optimized_score, out.baseline_score)
        self.assertGreater(out.diagnostics["optimized_objective"], -1)

    def test_convergence_is_not_acceptance(self):
        out = self.solve([self.result(np.zeros(12)), self.result(np.zeros(12))])
        self.assertTrue(out.numerical_success)
        self.assertFalse(out.success)
        np.testing.assert_array_equal(out.xyz, self.xyz)

    def test_nan_and_excessive_coefficients_rejected(self):
        out = self.solve([self.result(np.full(12, np.nan)), self.result(self.good * 100)])
        self.assertFalse(out.success)
        self.assertFalse(out.geometric_valid)

    def test_budget_checked_independently_of_solver_flag(self):
        self.cfg = dataclasses.replace(self.cfg, length_ratio=1.00001)
        out = self.solve([self.result(self.good), self.result(self.good)])
        self.assertFalse(out.success)

    def test_malformed_timestamps_rejected_before_solver(self):
        for t in (self.t[:-1], self.t[::-1], np.zeros(len(self.t)),
                  np.r_[np.nan, self.t[1:]], self.t - 10):
            with self.subTest(times=t[:3]), patch.object(L, "minimize") as solver:
                with self.assertRaises(ValueError):
                    L.optimize_planar_reach(t, self.xyz, self.goals, 0, self.cfg)
                solver.assert_not_called()

    def test_invalid_geometry_and_goal_rejected(self):
        for xyz in (np.full_like(self.xyz, np.nan), self.xyz[:, :2], np.tile(self.xyz[0], (len(self.xyz), 1))):
            with self.assertRaises(ValueError):
                L.optimize_planar_reach(self.t, xyz, self.goals, 0, self.cfg)
        with self.assertRaises(ValueError):
            L.optimize_planar_reach(self.t, self.xyz, self.goals, -1, self.cfg)

    def test_nonuniform_time_quadrature_and_offset(self):
        xyz = self.xyz[[0, 30, 100]]
        times = np.array([0., .2, 5.])
        p = L.planar_posterior(xyz, self.goals)[:, 0]
        w = np.array([.1, 2.5, 2.4]) * (5 - times)
        expected = float(w @ p / w.sum())
        self.assertAlmostEqual(L.planar_legibility(xyz, self.goals, 0, times=times), expected)
        self.assertAlmostEqual(L.planar_legibility(xyz, self.goals, 0, times=times + 100), expected)

    def test_posterior_normalization_and_instantaneous_ceiling(self):
        rng = np.random.default_rng(42)
        points = np.vstack([self.start, rng.uniform([.2, -.8, .113], [.95, .7, .113], (100, 3))])
        posterior = L.planar_posterior(points, self.goals)
        np.testing.assert_allclose(posterior.sum(axis=1), 1)
        metrics = L.planar_metrics(self.t, self.xyz, self.goals, 0)
        self.assertLessEqual(posterior[:, 0].max(), metrics["posterior_ceiling"] + 1e-12)

    def test_invalid_configuration_fails_explicitly(self):
        for change in ({"efficiency_weight": -1}, {"beta": float("nan")}, {"length_ratio": .9},
                       {"n_starts": 0}, {"xy_min": (1, 1), "xy_max": (0, 0)}, {"max_tcp_speed": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(self.cfg, **change)

    def test_middle_targets_do_not_acquire_a_wrong_goal_preference(self):
        cfg = dataclasses.replace(self.cfg, efficiency_weight=.2, extension_radius=.8,
                                  xy_min=(.2, -.8), xy_max=(.95, .7), max_tcp_speed=.5)
        for target in (1, 2):
            with self.subTest(target=target):
                times, xyz = L.sample(L.baseline_path(self.start, self.goals[target]), 5, rate=20)
                out = L.optimize_planar_reach(times, xyz, self.goals, target, cfg)
                self.assertTrue(out.success, out.diagnostics)
                before = L.planar_posterior(xyz, self.goals, cfg.beta)
                after = L.planar_posterior(out.xyz, self.goals, cfg.beta)
                others = np.arange(len(self.goals)) != target
                floor = np.minimum(before[:, target, None] - before[:, others], 0)
                self.assertGreaterEqual(float((after[:, target, None] - after[:, others] - floor).min()), -1e-8)
                self.assertLessEqual(out.length_ratio, 1.04 + 1e-6)
                self.assertEqual(out.physical_validity, "not_checked")


if __name__ == "__main__":
    unittest.main()
