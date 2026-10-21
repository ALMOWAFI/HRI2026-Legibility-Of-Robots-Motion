#!/usr/bin/env python3
"""Offline tests for legibility.py (no ROS needed): python3 -m unittest discover -s test"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hri_legible_motion import legibility as L  # noqa: E402
from hri_legible_motion import trajectories as T  # noqa: E402

PLANE_Z = 0.101
P0 = np.array([0.434, -0.134, PLANE_Z])
GOALS = [[0.804, 0.122], [0.875, -0.044], [0.875, -0.224], [0.804, -0.389]]
SIDES = [1, 1, -1, -1]


def reach(target, bend):
    p1 = np.array(GOALS[target] + [PLANE_Z])
    v = T.bend_vector(P0, p1, SIDES[target], bend)
    path = L.baseline_path(P0, p1) if bend == 0 else L.legible_path(P0, p1, v)
    return p1, path


class TestLegiblePath(unittest.TestCase):
    def test_same_endpoints_and_planar(self):
        for target in range(4):
            p1, path = reach(target, 0.20)
            _, xyz = L.sample(path, 4.0)
            self.assertTrue(np.allclose(xyz[0], P0, atol=1e-6))
            self.assertTrue(np.allclose(xyz[-1], p1, atol=1e-6))
            self.assertLess(np.abs(xyz[:, 2] - PLANE_Z).max(), 1e-9)

    def test_peak_offset_and_side(self):
        for target in range(4):
            p1, path = reach(target, 0.20)
            xyz = path(np.linspace(0, 1, 2001))
            d = (p1 - P0)[:2] / np.linalg.norm((p1 - P0)[:2])
            rel = xyz[:, :2] - P0[:2]
            lateral = rel[:, 1] * d[0] - rel[:, 0] * d[1]   # + = left of the reach direction
            peak = lateral[np.abs(lateral).argmax()]
            self.assertAlmostEqual(abs(peak), 0.20, places=3)
            n = T.bend_vector(P0, p1, SIDES[target], 1.0)
            self.assertEqual(np.sign(peak), np.sign(n[0] * -d[1] + n[1] * d[0]))

    def test_arc_length_parameter(self):
        _, path = reach(1, 0.20)
        xyz = path(np.linspace(0, 1, 501))
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        self.assertLess(step.max() / step.min(), 1.01)


class TestScore(unittest.TestCase):
    def score(self, target, bend):
        _, path = reach(target, bend)
        _, xyz = L.sample(path, 4.0)
        return L.legibility_score(xyz, GOALS, target)

    def test_in_range(self):
        for target in range(4):
            for bend in (0.0, 0.20):
                self.assertTrue(0.0 <= self.score(target, bend) <= 1.0)

    def test_edge_bend_more_legible(self):
        for target in (0, 3):
            self.assertGreater(self.score(target, 0.20), self.score(target, 0.0))

    def test_middle_bend_toward_neighbour_less_legible(self):
        # Middle cubes have to bend toward a neighbour (plan §2.2); the model then scores the
        # bend below the straight reach, which is what H3 predicts.
        for target in (1, 2):
            self.assertLess(self.score(target, 0.20), self.score(target, 0.0))

    def test_ignores_height(self):
        _, path = reach(0, 0.20)
        _, xyz = L.sample(path, 4.0)
        lifted = xyz + [0, 0, 0.3]
        self.assertAlmostEqual(L.legibility_score(xyz, GOALS, 0), L.legibility_score(lifted, GOALS, 0))


if __name__ == "__main__":
    unittest.main()
