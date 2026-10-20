#!/usr/bin/env python3
"""Offline tests for experiment.py (no ROS needed): python3 -m unittest discover -s test"""
import collections
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hri_legible_motion import experiment as E  # noqa: E402

TARGETS = [0, 1, 2, 3]
BLOCKS = [{"name": "B%d" % (i + 1)} for i in TARGETS]
COLORS = ["orange", "red", "yellow", "green"]


class TestCounterbalancing(unittest.TestCase):
    def test_first_16_fill_every_cell_twice(self):
        cells = [E.assignment("P%02d" % k, seed=2026) for k in range(1, 17)]
        self.assertEqual(set(collections.Counter(cells).values()), {2})
        self.assertEqual(len(set(cells)), 8)
        orders = collections.Counter(order for order, _ in cells)
        self.assertEqual(orders[("predictable", "legible")], 8)

    def test_continues_cyclically(self):
        self.assertEqual(E.assignment("P17", 2026), E.assignment("P01", 2026))

    def test_participant_number(self):
        self.assertEqual(E.participant_number("P07"), 7)
        self.assertEqual(E.participant_number("pilot_12"), 12)
        self.assertEqual(E.participant_number("dry"), 0)

    def test_rotation(self):
        self.assertEqual(E.rotated_colors(COLORS, 0), COLORS)
        self.assertEqual(E.rotated_colors(COLORS, 1), ["red", "yellow", "green", "orange"])


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.trials = E.make_schedule(("legible", "predictable"), TARGETS, 4, seed=3)

    def phase(self, block):
        return [t for t in self.trials if t["block"] == block]

    def test_structure(self):
        self.assertEqual(len(self.trials), 4 + 16 + 16 + 8)
        self.assertEqual([t["trial_id"] for t in self.trials], list(range(44)))
        self.assertEqual({t["motion"] for t in self.phase("A")}, {"legible"})
        self.assertEqual({t["motion"] for t in self.phase("B")}, {"predictable"})

    def test_warmup(self):
        w = self.phase("W")
        self.assertEqual(sorted(t["target_id"] for t in w), TARGETS)
        self.assertEqual(collections.Counter(t["motion"] for t in w),
                         {"predictable": 2, "legible": 2})

    def test_main_blocks_balanced(self):
        for b in "AB":
            counts = collections.Counter(t["target_id"] for t in self.phase(b))
            self.assertEqual(counts, {t: 4 for t in TARGETS})

    def test_announced_each_pair_once(self):
        pairs = {(t["motion"], t["target_id"]) for t in self.phase("C")}
        self.assertEqual(len(pairs), 8)

    def test_no_cube_more_than_twice_in_a_row(self):
        for seed in range(30):
            trials = E.make_schedule(("predictable", "legible"), TARGETS, 4, seed)
            for b in "ABC":
                self.assertLessEqual(E.max_run([t["target_id"] for t in trials if t["block"] == b]), 2)

    def test_same_seed_same_schedule(self):
        self.assertEqual(self.trials, E.make_schedule(("legible", "predictable"), TARGETS, 4, seed=3))


def trial(target=2, phase="main"):
    return {"trial_id": 5, "phase": phase, "block": "A", "index_in_phase": 0,
            "motion": "legible", "target_id": target}


class TestMeasures(unittest.TestCase):
    T = 4.0

    def record(self, presses, t_start=10.0):
        r = E.TrialRecord(trial(), t_trial_start=9.0)
        r.t_start = t_start
        for t, c in presses:
            r.add_choice(t, c)
        return r

    def test_no_choice(self):
        m = self.record([]).measures(self.T)
        self.assertEqual((m["f"], m["score"], m["final_correct"], m["t_commit"]), (0.0, 0, 0, 1.0))
        self.assertIsNone(m["t_first_correct"])

    def test_correct_from_the_start(self):
        m = self.record([(9.5, 2)]).measures(self.T)  # pressed before onset
        self.assertAlmostEqual(m["f"], 1.0)
        self.assertEqual((m["score"], m["t_first_correct"], m["t_commit"]), (10, 0.0, 0.0))

    def test_switch_to_correct_halfway(self):
        m = self.record([(10.4, 1), (12.0, 2)]).measures(self.T)
        self.assertAlmostEqual(m["f"], 0.5)
        self.assertEqual(m["score"], 5)
        self.assertAlmostEqual(m["t_first_correct"], 0.5)
        self.assertAlmostEqual(m["t_commit"], 0.5)
        self.assertEqual((m["n_switches"], m["final_correct"]), (1, 1))

    def test_correct_then_wrong(self):
        m = self.record([(11.0, 2), (13.0, 3)]).measures(self.T)
        self.assertAlmostEqual(m["f"], 0.5)
        self.assertEqual((m["final_correct"], m["t_commit"]), (0, 1.0))

    def test_back_and_forth(self):
        m = self.record([(11.0, 2), (12.0, 0), (13.0, 2)]).measures(self.T)
        self.assertAlmostEqual(m["f"], 0.5)
        self.assertAlmostEqual(m["t_first_correct"], 0.25)
        self.assertAlmostEqual(m["t_commit"], 0.75)
        self.assertEqual(m["n_switches"], 2)

    def test_presses_after_reach_end_ignored(self):
        m = self.record([(11.0, 1), (14.5, 2)]).measures(self.T)
        self.assertEqual((m["f"], m["final_correct"], m["n_switches"]), (0.0, 0, 0))

    def test_same_button_twice_keeps_choice(self):
        r = self.record([])
        self.assertTrue(r.add_choice(11.0, 2))
        self.assertFalse(r.add_choice(11.5, 2))
        self.assertEqual(len(r.choices), 1)

    def test_no_onset_no_measures(self):
        self.assertEqual(self.record([], t_start=None).measures(self.T), {})


class TestLog(unittest.TestCase):
    def test_writes_rows(self):
        with tempfile.TemporaryDirectory() as d:
            log = E.SessionLog(os.path.join(d, "P01"))
            for i in range(2):
                r = E.TrialRecord(trial(), 9.0)
                r.t_start, r.status, r.confidence = 10.0, "done", 6
                r.add_choice(10.5, 2)
                log.write(r, "P01", 4.0, BLOCKS, COLORS)
            with open(log.trials_path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual((rows[0]["target_color"], rows[0]["score"], rows[0]["confidence"]),
                             ("yellow", "9", "6"))
            with open(log.choices_path) as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)

    def test_announced_has_no_score(self):
        r = E.TrialRecord(trial(phase="announced"), 9.0)
        r.t_start, r.naturalness = 10.0, 5
        row = r.row("P01", 4.0, BLOCKS, COLORS)
        self.assertNotIn("score", row)
        self.assertEqual(row["naturalness"], 5)


if __name__ == "__main__":
    unittest.main()
