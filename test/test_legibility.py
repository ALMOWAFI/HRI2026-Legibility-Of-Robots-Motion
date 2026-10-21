#!/usr/bin/env python3
"""Offline unit tests for legibility.py: python3 -m unittest discover -s test"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hri_legible_motion import legibility as L  # noqa: E402


class TestLegibilityPaths(unittest.TestCase):
    def setUp(self):
        self.p0 = np.array([0.434, -0.134, 0.245])
        self.p1 = np.array([0.804, 0.122, -0.074])
        self.bend = np.array([-0.09, 0.18, 0.0])

    def test_baseline_endpoints(self):
        base = L.baseline_path(self.p0, self.p1)
        self.assertTrue(np.allclose(base(0.0), self.p0))
        self.assertTrue(np.allclose(base(1.0), self.p1))
        mid = base(0.5)
        self.assertTrue(np.allclose(mid, 0.5 * (self.p0 + self.p1)))

    def test_legible_endpoints_and_bend(self):
        leg = L.legible_path(self.p0, self.p1, self.bend)
        # Endpoints must strictly match start and goal
        self.assertTrue(np.allclose(leg(0.0), self.p0))
        self.assertTrue(np.allclose(leg(1.0), self.p1))
        
        # At midpoint, offset must exactly equal bend
        mid = leg(0.5)
        expected_mid = 0.5 * (self.p0 + self.p1) + self.bend
        self.assertTrue(np.allclose(mid, expected_mid))

    def test_min_jerk(self):
        self.assertAlmostEqual(float(L.min_jerk(0.0)), 0.0)
        self.assertAlmostEqual(float(L.min_jerk(1.0)), 1.0)
        self.assertAlmostEqual(float(L.min_jerk(0.5)), 0.5)
        # Monotonic
        tau = np.linspace(0.0, 1.0, 50)
        s = L.min_jerk(tau)
        self.assertTrue(np.all(np.diff(s) >= 0.0))

    def test_sample(self):
        path = L.baseline_path(self.p0, self.p1)
        times, xyz = L.sample(path, duration=4.0, rate=100.0)
        self.assertEqual(len(times), 401)
        self.assertEqual(xyz.shape, (401, 3))
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 4.0)
        self.assertTrue(np.allclose(xyz[0], self.p0))
        self.assertTrue(np.allclose(xyz[-1], self.p1))


class TestGoalInference(unittest.TestCase):
    def setUp(self):
        self.p0 = np.array([0.434, -0.134, 0.245])
        self.goals = np.array([
            [0.804,  0.122, -0.074],
            [0.875, -0.044, -0.074],
            [0.875, -0.224, -0.074],
            [0.804, -0.389, -0.074],
        ])

    def test_goal_posterior_sums_to_one(self):
        probs = L.goal_posterior(self.p0, self.p0, self.goals)
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0)

    def test_goal_posterior_increases_toward_target(self):
        # Initial probabilities should be uniform at start
        p_start = L.goal_posterior(self.p0, self.p0, self.goals)
        self.assertTrue(np.allclose(p_start, 0.25))

        # As EE moves toward goal 0, target 0 probability must strictly increase
        approach = self.p0 + 0.8 * (self.goals[0] - self.p0)
        probs = L.goal_posterior(approach, self.p0, self.goals)
        self.assertGreater(probs[0], p_start[0])
        self.assertEqual(int(np.argmax(probs)), 0)

    def test_legibility_score_comparison(self):
        bend = np.array([-0.09, 0.18, 0.0])  # bend outward toward B1
        pred = L.baseline_path(self.p0, self.goals[0])
        leg = L.legible_path(self.p0, self.goals[0], bend)

        score_pred = L.legibility_score(pred, self.goals, 0)
        score_leg = L.legibility_score(leg, self.goals, 0)

        # Legible path should produce higher goal posterior along reach
        self.assertGreater(score_leg, score_pred)


class TestKinematics(unittest.TestCase):
    def test_fk_position(self):
        # Ready pose
        q = [-3.1420, -1.6302, -2.1798, -0.8981, 1.5700, 0.0001]
        ee = L.ur5e_fk_position(q)
        self.assertEqual(len(ee), 3)
        self.assertAlmostEqual(ee[2], 0.245, places=2)


if __name__ == "__main__":
    unittest.main()
