#!/usr/bin/env python3
"""Offline unit tests for legibility_optimizer.py: python3 -m unittest discover -s test"""
import dataclasses
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hri_legible_motion import legibility_optimizer as L  # noqa: E402


class TestLegibilityPaths(unittest.TestCase):
    def setUp(self):
        self.p0 = np.array([0.434, -0.134, 0.245])
        self.p1 = np.array([0.804, 0.122, -0.074])

    def test_baseline_endpoints(self):
        base = L.baseline_path(self.p0, self.p1)
        self.assertTrue(np.allclose(base(0.0), self.p0))
        self.assertTrue(np.allclose(base(1.0), self.p1))
        mid = base(0.5)
        self.assertTrue(np.allclose(mid, 0.5 * (self.p0 + self.p1)))

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


class TestPlanarOptimizer(unittest.TestCase):
    """The fixed-height optimizer on the 2026-10-21 layout (tool0 plane z = 0.113)."""
    Z = 0.113
    P0 = np.array([0.256, -0.152, Z])
    GOALS = np.array([[0.618, 0.626, Z], [0.721, 0.555, Z], [0.782, 0.470, Z], [0.833, 0.371, Z]])
    CFG = L.PlanarConfig(xy_min=(0.20, -0.80), xy_max=(0.95, 0.70))

    def reach(self, target):
        t, xyz = L.sample(L.baseline_path(self.P0, self.GOALS[target]), 5.0)
        return t, xyz, L.optimize_planar_reach(t, xyz, self.GOALS, target, self.CFG)

    def test_keeps_ends_height_and_timing(self):
        t, xyz, out = self.reach(0)
        self.assertTrue(out.success, out.message)
        self.assertTrue(np.array_equal(out.times, t))
        self.assertTrue(np.allclose(out.xyz[0], self.P0) and np.allclose(out.xyz[-1], self.GOALS[0]))
        self.assertLess(np.abs(out.xyz[:, 2] - self.Z).max(), 1e-9)

    def test_more_legible_within_limits(self):
        for target in range(4):
            _, _, out = self.reach(target)
            self.assertTrue(out.success, out.message)
            self.assertGreater(out.optimized_score, out.baseline_score + 1e-3)
            self.assertLessEqual(out.length_ratio, self.CFG.length_ratio + 1e-3)
            r = np.hypot(out.xyz[:, 0], out.xyz[:, 1])
            self.assertGreaterEqual(r.min(), self.CFG.min_radius - 1e-3)
            self.assertLessEqual(out.xyz[:, 1].max(), self.CFG.xy_max[1] + 1e-6)

    def test_edge_cubes_bend_outward(self):
        for target, sign in ((0, 1), (3, -1)):   # B1 on the left of its reach, B4 on the right
            _, _, out = self.reach(target)
            mid = out.xyz[len(out.xyz) // 2] - self.P0
            direction = (self.GOALS[target] - self.P0)[:2]
            self.assertEqual(np.sign(direction[0] * mid[1] - direction[1] * mid[0]), sign)

    def test_higher_lambda_shorter_path(self):
        t, xyz = L.sample(L.baseline_path(self.P0, self.GOALS[1]), 5.0)
        ratios = [L.optimize_planar_reach(t, xyz, self.GOALS, 1, dataclasses.replace(self.CFG, efficiency_weight=lam))
                  .length_ratio for lam in (0.0, 0.3, 1.0)]
        self.assertGreater(ratios[0], ratios[1])
        self.assertGreater(ratios[1], ratios[2])

    def test_refuses_non_planar(self):
        t, xyz = L.sample(L.baseline_path(self.P0, self.GOALS[0] + [0, 0, 0.05]), 5.0)
        with self.assertRaises(ValueError):
            L.optimize_planar_reach(t, xyz, self.GOALS, 0, self.CFG)

    def test_planar_score_ignores_height(self):
        _, xyz, _ = self.reach(1)
        self.assertAlmostEqual(L.planar_legibility(xyz, self.GOALS, 1),
                               L.planar_legibility(xyz + [0, 0, 0.3], self.GOALS, 1))


if __name__ == "__main__":
    unittest.main()
