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


def max_run(targets):
    """Longest run of the same target in a row."""
    best = run = 1 if targets else 0
    for a, b in zip(targets, targets[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


class TestParticipant(unittest.TestCase):
    def test_participant_number(self):
        self.assertEqual(E.participant_number("P07"), 7)
        self.assertEqual(E.participant_number("pilot_12"), 12)
        self.assertEqual(E.participant_number("dry"), 0)


def trial(target=2, phase="game"):
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
        self.assertEqual((m["f"], m["final_correct"], m["t_commit"], m["points"]), (0.0, 0, 1.0, 0))
        self.assertIsNone(m["t_first_correct"])

    def test_correct_from_the_start(self):
        m = self.record([(9.5, 2)]).measures(self.T)  # pressed before onset
        self.assertAlmostEqual(m["f"], 1.0)
        self.assertEqual((m["t_first_correct"], m["t_commit"]), (0.0, 0.0))

    def test_switch_to_correct_halfway(self):
        m = self.record([(10.4, 1), (12.0, 2)]).measures(self.T)
        self.assertAlmostEqual(m["f"], 0.5)
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
            self.assertEqual((rows[0]["target_color"], rows[0]["points"], rows[0]["confidence"]),
                             ("yellow", "88", "6"))
            with open(log.choices_path) as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)

    def test_choice_sequence_columns(self):
        with tempfile.TemporaryDirectory() as d:
            log = E.SessionLog(d)
            r = E.TrialRecord(trial(), 9.0)
            r.t_start, r.status = 10.0, "done"
            for t, c in [(10.5, 2), (11.25, 0), (12.0, 3)]:
                r.add_choice(t, c)
            row = log.write(r, "P01", 4.0, BLOCKS, COLORS)
            self.assertEqual((row["choice_names"], row["choice_ids"], row["choice_times"]),
                             ("B3;B1;B4", "2;0;3", "0.500;1.250;2.000"))
            self.assertEqual(row["n_switches"], 2)


class TestGameSchedule(unittest.TestCase):
    PARTICIPANTS = ["P%02d" % k for k in range(1, 41)]

    def order(self, p, targets=TARGETS):
        return E.game_order(p, targets, 2026, E.default_first_motion(p))

    def test_every_cube_once_per_motion(self):
        for p in self.PARTICIPANTS:
            for first in E.MOTIONS:
                trials = E.make_game_schedule(p, TARGETS, 2026, first)
                self.assertEqual(len(trials), 8)
                self.assertEqual(sorted((t["motion"], t["target_id"]) for t in trials),
                                 sorted((m, c) for m in E.MOTIONS for c in TARGETS))
                self.assertEqual([t["trial_id"] for t in trials], list(range(8)))
                self.assertEqual([t["block"] for t in trials], list("AAAABBBB"))

    def test_blocks_one_motion_each(self):
        for p in self.PARTICIPANTS:
            for first, second in (E.MOTIONS, E.MOTIONS[::-1]):
                order = E.game_order(p, TARGETS, 2026, first)
                self.assertEqual([m for m, _ in order], [first] * 4 + [second] * 4)
                for half in (order[:4], order[4:]):
                    self.assertEqual(sorted(t for _, t in half), TARGETS)
                self.assertNotEqual([t for _, t in order[:4]], [t for _, t in order[4:]])

    def test_no_cube_twice_in_a_row(self):
        for p in self.PARTICIPANTS:
            for first in E.MOTIONS:
                self.assertEqual(max_run([t for _, t in E.game_order(p, TARGETS, 2026, first)]), 1)

    def test_counterbalanced_over_8_participants(self):
        orders = [self.order(p) for p in self.PARTICIPANTS[:8]]
        self.assertEqual(collections.Counter(o[0][0] for o in orders), {"predictable": 4, "legible": 4})
        for motion in E.MOTIONS:  # Williams square: each cube in each block-A position once per motion
            for position in range(4):
                self.assertEqual(collections.Counter(o[position][1] for o in orders if o[0][0] == motion),
                                 {t: 1 for t in TARGETS})

    def test_reproducible(self):
        self.assertEqual(self.order("P05"), self.order("P05"))
        self.assertEqual(self.order("P05", [10, 11, 12, 13])[0][1] - 10, self.order("P05")[0][1])
        # The chosen first motion changes only the motions, not the cube order.
        self.assertEqual([t for _, t in E.game_order("P05", TARGETS, 2026, "legible")],
                         [t for _, t in E.game_order("P05", TARGETS, 2026, "predictable")])

    def test_training_first(self):
        training = [("predictable", 1), ("legible", 2), ("predictable", 3), ("legible", 0), ("predictable", 2)]
        trials = E.make_game_schedule("P03", TARGETS, 2026, "legible", training)
        self.assertEqual(len(trials), 13)
        self.assertEqual([t["block"] for t in trials], list("TTTTTAAAABBBB"))
        self.assertEqual([t["phase"] for t in trials], ["training"] * 5 + ["game"] * 8)
        self.assertEqual([(t["motion"], t["target_id"]) for t in trials[:5]], training)
        self.assertEqual([t["trial_id"] for t in trials], list(range(5)) + list(range(8)))
        # The game part is the same with or without practice.
        self.assertEqual(trials[5:], E.make_game_schedule("P03", TARGETS, 2026, "legible"))

    def test_practice_log_apart(self):
        with tempfile.TemporaryDirectory() as d:
            log, practice = E.SessionLog(d), E.SessionLog(d, "practice_")
            self.assertEqual(os.path.basename(practice.trials_path), "practice_trials.csv")
            self.assertEqual(os.path.basename(practice.choices_path), "practice_choices.csv")
            self.assertEqual(os.path.dirname(practice.trials_path), os.path.dirname(log.trials_path))
            self.assertNotEqual(practice.trials_path, log.trials_path)

    def test_bad_first_motion(self):
        with self.assertRaises(ValueError):
            E.game_order("P05", TARGETS, 2026, "fast")

    def test_points_left(self):
        self.assertAlmostEqual(E.points_left(0.2, 1000, 2.0), 640)
        self.assertAlmostEqual(E.points_left(0.4, 1000, 2.0), 360)
        self.assertEqual((E.points_left(0.0, 1000, 2.0), E.points_left(1.2, 1000, 2.0)), (1000, 0))

    def test_game_points_quadratic(self):
        def game(presses, T=5.0):
            r = E.TrialRecord(trial(target=2), 9.0, points_max=1000, exponent=2.0)
            r.t_start = 10.0
            for t, c in presses:
                r.add_choice(t, c)
            return r.measures(T)["points"]
        self.assertEqual(game([(11.0, 2)]), 640)              # right at 1 s, kept
        self.assertEqual(game([(10.0, 2)]), 1000)             # right at onset
        self.assertEqual(game([(11.0, 1), (12.0, 2)]), 360)   # wrong first, right at 2 s
        self.assertEqual(game([(11.0, 2), (12.0, 1)]), 280)   # right 1-2 s only: 640 - 360
        self.assertEqual(game([]), 0)

    def test_default_points_stay_linear(self):
        r = E.TrialRecord(trial(target=2), 9.0)
        r.t_start = 10.0
        r.add_choice(11.0, 2)
        self.assertEqual(r.measures(4.0)["points"], 75)


class TestLeaderboard(unittest.TestCase):
    def test_rank(self):
        with tempfile.TemporaryDirectory() as d:
            board = E.Leaderboard(os.path.join(d, "leaderboard.csv"))
            self.assertEqual(board.add("P01", "2026-10-28", 300), (1, 1, None))
            self.assertEqual(board.add("P02", "2026-10-28", 500), (1, 2, 300))
            self.assertEqual(board.add("P03", "2026-10-28", 400), (2, 3, 500))


if __name__ == "__main__":
    unittest.main()
