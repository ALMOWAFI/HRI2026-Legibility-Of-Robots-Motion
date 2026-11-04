#!/usr/bin/env python3
"""Offline tests for tools/analyze.py on synthetic sessions: python3 -m unittest discover -s test"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import analyze as A  # noqa: E402
from hri_legible_motion import experiment as E  # noqa: E402

T = 5.0
BLOCKS = [{"name": "B%d" % (i + 1)} for i in range(4)]
COLORS = ["green", "red", "orange", "yellow"]
TRAINING = [("predictable", 1), ("legible", 2)]


def session(root, pid, first="legible", pick_at=None, failed=(), training=TRAINING):
    """A participant folder as experiment_node writes it. pick_at(motion, target) -> fraction
    of the reach at which the right cube is pressed (None = never)."""
    pick_at = pick_at or (lambda motion, target: 0.2 if motion == "legible" else 0.6)
    folder = os.path.join(root, pid)
    log, practice = E.SessionLog(folder), E.SessionLog(folder, "practice_")
    t = 1000.0
    for trial in E.make_game_schedule(pid, [0, 1, 2, 3], 2026, first, training):
        for attempt in (("failed", "done") if (trial["phase"], trial["trial_id"]) in failed else ("done",)):
            r = E.TrialRecord(trial, t, 1000, 2.0)
            r.t_start, r.status, r.t_done = t + 0.2, attempt, t + 0.2 + T
            u = pick_at(trial["motion"], trial["target_id"])
            if u is not None:
                r.add_choice(r.t_start + u * T, trial["target_id"])
            (practice if trial["phase"] == "training" else log).write(r, pid, T, BLOCKS, COLORS)
            t += 20.0
    with open(os.path.join(folder, "session.yaml"), "w") as f:
        yaml.safe_dump({"participant_id": pid, "first_motion": first, "duration_T": T,
                        "trajectory_hash": "abc"}, f)
    return folder


class TestLoad(unittest.TestCase):
    def test_game_trials_only_last_attempt_and_practice_marked(self):
        with tempfile.TemporaryDirectory() as d:
            session(d, "P01", failed={("game", 3)})
            session(d, "dry01")
            trials, choices, notes = A.load(d)
            self.assertEqual(sorted(trials["participant_id"].unique()), ["P01"])
            self.assertEqual(notes, [("dry01", "not a P## id (dry run)")])
            self.assertEqual(len(trials), 8)                      # repeated round counted once
            self.assertEqual(set(trials["phase"]), {"game"})
            self.assertEqual(trials[trials["trial_id"] == 3]["status"].item(), "done")
            seen = trials[trials["seen_in_practice"] == 1]
            self.assertEqual(sorted(zip(seen["motion"], seen["target_id"])), sorted(TRAINING))
            self.assertEqual(set(trials["first_motion"]), {"legible"})

    def test_prepare_columns(self):
        with tempfile.TemporaryDirectory() as d:
            session(d, "P02", first="predictable", pick_at=lambda m, t: None if t == 0 else 0.5)
            t = A.prepare(A.load(d)[0])
            self.assertEqual(t["second"].sum(), 4)
            self.assertEqual(set(t[t["second"] == 0]["motion"]), {"predictable"})
            self.assertEqual(sorted(t[t["edge"] == 1]["target_id"].unique()), [0, 3])
            never = t[t["target_id"] == 0]
            self.assertTrue((never["never_correct"] == 1).all() and (never["t_first_correct"] == 1.0).all())


class TestExclusions(unittest.TestCase):
    def test_unfinished_trials_and_short_sessions(self):
        rows = [{"participant_id": p, "trial_id": i, "status": "done" if (p, i) != ("P01", 2) else "failed"}
                for p in ("P01", "P02") for i in range(8 if p == "P01" else 4)]
        kept, why = A.exclude(pd.DataFrame(rows), exclude_ids=(), min_trials=6)
        self.assertEqual(sorted(kept["participant_id"].unique()), ["P01"])
        self.assertEqual(len(kept), 7)
        self.assertEqual(set(why["reason"].str.split().str[0]), {"status", "only"})
        kept, _ = A.exclude(pd.DataFrame(rows), exclude_ids=("P01",), min_trials=1)
        self.assertEqual(sorted(kept["participant_id"].unique()), ["P02"])


class TestStatistics(unittest.TestCase):
    def test_legible_advantage_is_found(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(1)
            for k in range(1, 9):
                session(d, "P%02d" % k, first=E.default_first_motion("P%02d" % k),
                        pick_at=lambda m, t: float(np.clip((0.25 if m == "legible" else 0.55) + rng.normal(0, 0.05),
                                                           0, 1)))
            trials = A.prepare(A.load(d)[0])
            tests = A.paired_tests(A.participant_means(trials))
            row = tests[(tests["outcome"] == "f") & (tests["comparison"] == "legible - predictable (all cubes)")]
            self.assertEqual(row["n"].item(), 8)
            self.assertAlmostEqual(row["mean_diff"].item(), 0.3, delta=0.05)
            self.assertGreater(row["ci_low"].item(), 0)
            self.assertLess(row["p_wilcoxon"].item(), 0.05)

    def test_recognition_curve_steps_at_the_press(self):
        with tempfile.TemporaryDirectory() as d:
            session(d, "P01")
            trials, choices, _ = A.load(d)
            grid, curves = A.recognition_curve(A.prepare(trials), choices)
            legible = next(c for c in curves if c["motion"] == "legible" and c["cube_type"] == "all")
            self.assertEqual(legible["share"][grid < 0.19].max(), 0.0)
            self.assertEqual(legible["share"][grid > 0.21].min(), 1.0)

    def test_questionnaire_paired_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "q.csv")
            rows = [{"participant_id": "P%02d" % k, "motion": m, "clarity": (6 if m == "legible" else 3) + k % 2}
                    for k in range(1, 9) for m in E.MOTIONS]
            pd.DataFrame(rows).to_csv(path, index=False)
            first = {"P%02d" % k: E.default_first_motion("P%02d" % k) for k in range(1, 9)}
            q = A.questionnaire_tests(path, first)
            self.assertEqual(q["item"].tolist(), ["clarity"])
            self.assertEqual(q["mean_diff"].item(), 3.0)
            self.assertIn("p_order", q)


class TestMain(unittest.TestCase):
    def test_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            for k in range(1, 5):
                session(os.path.join(d, "data"), "P%02d" % k, first=E.default_first_motion("P%02d" % k))
            out = os.path.join(d, "out")
            with contextlib.redirect_stdout(io.StringIO()):
                A.main(["--data", os.path.join(d, "data"), "--out", out])
            for name in ("trials.csv", "exclusions.csv", "descriptives.csv", "participant_means.csv",
                         "tests.csv", "models.txt", "manipulation.csv", "f_by_cube.png", "recognition_curve.png"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            self.assertEqual(len(pd.read_csv(os.path.join(out, "trials.csv"))), 32)


if __name__ == "__main__":
    unittest.main()
