"""Offline comparison provenance, target mapping, and output contracts (no ROS)."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_optimizer", ROOT / "tools" / "compare_optimizer.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class TestOfflineComparison(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "frame_id": "test_base", "motion": {"touch_z": 0.1, "hover_height": 0.0, "duration": 1.0},
            "blocks": [{"id": 42, "name": "A", "position": [0.7, 0.2, 0.0]},
                       {"id": 7, "name": "B", "position": [0.7, -0.2, 0.0]}],
            "safety": {"workspace_min": [0.0, -1.0, 0.0], "workspace_max": [1.0, 1.0, 1.0]},
            "optimizer": {"beta": 6.0, "length_ratio": 1.04, "efficiency_weight": 0.2},
        }

    def save_cache(self, path):
        _, baselines = comparison.load_baselines(self.cfg, start=[0.3, 0.0, 0.1])
        reaches = {}
        for block, (times, xyz) in zip(self.cfg["blocks"], baselines):
            reaches[comparison.trajectories.key("predictable", block["id"])] = {
                "times": times, "xyz": xyz, "positions": np.zeros((len(times), 1)),
                "velocities": np.zeros((len(times), 1)),
            }
        return comparison.trajectories.save(str(path), ["joint"], {"joint": 0.0}, reaches, {})

    def test_cli_rejects_ambiguous_source_and_bad_lambda(self):
        for args in (["--start", "0", "0", "0", "--cache", "x.npz"], ["--lambdas", "-1"],
                     ["--lambdas", "nan"], ["--lambdas", "inf"], ["--execute"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                comparison.argument_parser().parse_args(args)

    def test_start_is_explicitly_synthetic_and_must_be_planar(self):
        source, _ = comparison.load_baselines(self.cfg, start=[0.3, 0.0, 0.1])
        self.assertEqual(source["kind"], "synthetic_straight_baseline")
        self.assertEqual(source["physical_validity"], "not_checked")
        for start in ([0.3, 0.0, 0.2], [float("nan"), 0.0, 0.1]):
            with self.assertRaises(ValueError):
                comparison.load_baselines(self.cfg, start=start)

    def test_cache_hash_and_nonconsecutive_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            meta = self.save_cache(path)
            source, baselines = comparison.load_baselines(self.cfg, cache=path)
            self.assertEqual(source["content_hash"], meta["hash"])
            self.assertEqual(len(source["file_sha256"]), 64)
            self.assertEqual(source["kind"], "cached_moveit_cartesian_input")
            np.testing.assert_array_equal(baselines[0][1][-1], [0.7, 0.2, 0.1])
            np.testing.assert_array_equal(baselines[1][1][-1], [0.7, -0.2, 0.1])
            self.cfg["blocks"][0]["position"][0] = 0.8
            with self.assertRaisesRegex(ValueError, "endpoint"):
                comparison.load_baselines(self.cfg, cache=path)

    def test_existing_cache_loads_without_synthetic_fallback(self):
        cfg = comparison.yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
        source, baselines = comparison.load_baselines(cfg, cache=ROOT / "config" / "trajectories.npz")
        self.assertEqual(len(baselines), len(cfg["blocks"]))
        self.assertEqual(source["physical_validity"], "not_checked")
        self.assertEqual(len(source["cached_legible"]), len(cfg["blocks"]))
        self.assertIn("weighted_correct_fraction", source["cached_legible"]["1"]["metrics"])
        self.assertTrue(all(len(times) == len(xyz) for times, xyz in baselines))
        with self.assertRaises(OSError):
            comparison.load_baselines(cfg, cache=ROOT / "nonexistent-cache.npz")

    def test_corrupted_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            self.save_cache(path)
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
            arrays["predictable_42/xyz"][1, 0] += 0.01
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "hash"):
                comparison.load_baselines(self.cfg, cache=path)

    def test_bound_covers_straight_path_and_uses_actual_timestamps(self):
        times = np.array([0.0, 0.03, 0.2, 0.6, 1.0])
        goals = comparison.goal_positions(self.cfg)
        xyz = comparison.optimizer.baseline_path([0.3, 0.0, 0.1], goals[0])(comparison.optimizer.min_jerk(times))
        config = comparison.optimizer.PlanarConfig.from_cfg(self.cfg)
        result = comparison.timing_score_bound(times, xyz, goals, 0, config, spacing=0.005)
        metrics = comparison.optimizer.planar_metrics(times, xyz, goals, 0, config.beta)
        self.assertGreaterEqual(result["score_upper_bound"], metrics["score"])
        self.assertLessEqual(result["score_upper_bound"], metrics["posterior_ceiling"] + 1e-12)
        probs = comparison.optimizer.planar_posterior(xyz, goals, config.beta)
        self.assertEqual(result["posterior_upper_bound"][0], probs[0, 0])
        self.assertEqual(result["posterior_upper_bound"][-1], probs[-1, 0])
        dt = np.diff(times)
        values = np.array(result["posterior_upper_bound"]) * (1 - times)
        expected = np.sum(dt * (values[:-1] + values[1:]) / 2) / 0.5
        self.assertAlmostEqual(result["score_upper_bound"], expected)

    def test_goal_margin_exposes_competing_goal_even_if_target_probability_rises(self):
        probabilities = [[0.25, 0.25, 0.25, 0.25], [0.30, 0.35, 0.20, 0.15]]
        self.assertGreater(probabilities[1][0], probabilities[0][0])
        self.assertLess(comparison.goal_margin(probabilities, 0)[1], 0)

    def test_comparison_uses_same_baseline_and_observer_for_each_lambda(self):
        source, baselines = comparison.load_baselines(self.cfg, start=[0.3, 0.0, 0.1])
        calls = []

        def solve(times, xyz, goals, target, config):
            calls.append((target, config.beta, config.length_ratio, times.copy(), xyz.copy()))
            return SimpleNamespace(times=times, xyz=xyz, success=True, numerical_success=True,
                                   geometric_valid=True, physical_validity="not_checked", message="unchanged",
                                   diagnostics={"active_constraints": ["test"]}, length_ratio=1.0, max_lateral=0.0)

        with mock.patch.object(comparison.optimizer, "optimize_planar_reach", side_effect=solve):
            records = comparison.compare(self.cfg, baselines, [0.0, 0.2])
        self.assertEqual([c[0] for c in calls], [0, 0, 1, 1])
        self.assertEqual([r["target_id"] for r in records], [42, 42, 7, 7])
        self.assertTrue(all(c[1:3] == (6.0, 1.04) for c in calls))
        np.testing.assert_array_equal(calls[0][3], calls[1][3])
        np.testing.assert_array_equal(calls[0][4], calls[1][4])
        self.assertTrue(all(r["gain_percentage_points"] == 0.0 for r in records))
        self.assertTrue(all(r["physical_validity"] == "not_checked" for r in records))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            comparison.write_results(output, {"source": source, "config": self.cfg, "results": records})
            saved = json.loads((output / "comparison.json").read_text())
            self.assertEqual(saved["results"][0]["baseline"]["posterior"][0], [0.5, 0.5])
            with (output / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["active_constraints"], '["test"]')
            with (output / "posterior_curves.csv").open(newline="") as stream:
                curves = list(csv.DictReader(stream))
            self.assertEqual(len(curves), sum(2 * len(r["baseline"]["times"]) for r in records))


if __name__ == "__main__":
    unittest.main()
