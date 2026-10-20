#!/usr/bin/env python3
"""Offline tests for trajectories.py (no ROS needed): python3 -m unittest discover -s test"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hri_legible_motion import trajectories as T  # noqa: E402

NAMES = ["j%d" % i for i in range(6)]
START = {n: 0.1 * i for i, n in enumerate(NAMES)}


def reach(n=5):
    t = np.linspace(0, 4, n)
    return {"times": t, "positions": np.random.rand(n, 6), "velocities": np.random.rand(n, 6),
            "xyz": np.random.rand(n, 3)}


class TestBend(unittest.TestCase):
    P0 = np.array([0.434, -0.134, 0.245])

    def test_perpendicular_and_sided(self):
        for side in (1, -1):
            p1 = np.array([0.875, -0.044, 0.053])
            v = T.bend_vector(self.P0, p1, side, 0.2)
            self.assertAlmostEqual(np.linalg.norm(v), 0.2)
            self.assertAlmostEqual(float(np.dot(v[:2], (p1 - self.P0)[:2])), 0.0)
            self.assertEqual(np.sign(v[1]), side)
            self.assertEqual(v[2], 0.0)

    def test_zero_amount(self):
        self.assertTrue(np.allclose(T.bend_vector(self.P0, [0.8, 0.1, 0.05], 1, 0.0), 0))


class TestFile(unittest.TestCase):
    def test_round_trip(self):
        reaches = {T.key("predictable", i): reach() for i in range(4)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            meta = T.save(path, NAMES, START, reaches, {"duration": 4.0})
            names, start, loaded, meta2 = T.load(path)
            self.assertEqual(names, NAMES)
            self.assertEqual(start, START)
            self.assertEqual(meta2["hash"], meta["hash"])
            self.assertEqual(meta2["keys"], sorted(reaches))
            self.assertTrue(np.array_equal(loaded["predictable_2"]["positions"],
                                           reaches["predictable_2"]["positions"]))
            self.assertEqual(T.read_meta(path)["duration"], 4.0)

    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            T.save(path, NAMES, START, {"legible_0": reach()}, {})
            with np.load(path) as f:
                arrays = {k: f[k] for k in f.files}
            arrays["legible_0/positions"][0, 0] += 0.01
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                T.load(path)


if __name__ == "__main__":
    unittest.main()
