"""Experiment logic without ROS: the game's trial schedule (practice, block A, block B),
per-trial choice record + points, leaderboard, output files.

Times are seconds (floats); experiment_node.py feeds in rospy times.
"""
import csv
import itertools
import os
import random
import re

MOTIONS = ("predictable", "legible")

TRIAL_FIELDS = ["participant_id", "trial_id", "phase", "block", "index_in_phase", "motion",
                "target_id", "target_name", "target_color", "status",
                "t_trial_start", "t_start", "t_end", "t_done", "f",
                "t_first_correct", "t_commit", "n_switches", "final_choice", "final_correct",
                "confidence", "points", "choice_names", "choice_ids", "choice_times"]
# choice_*: every selection of the trial in order, ';'-separated (first = first press, each next
# one a switch); choice_times in s from reach onset. The same as the trial's rows in choices.csv.
CHOICE_FIELDS = ["participant_id", "trial_id", "stamp", "t_rel", "choice_id"]


# --- schedule ------------------------------------------------------------------------

def participant_number(participant_id):
    """'P07' -> 7. Ids without digits (e.g. a dry run) -> 0."""
    digits = re.findall(r"\d+", str(participant_id))
    return int(digits[-1]) if digits else 0


# Balanced Latin square (Williams design) for 4 cubes: every cube once in every position
# and every ordered pair of cubes once as neighbours, over 4 rows.
WILLIAMS_4 = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]


def default_first_motion(participant):
    """Suggested first block: odd participant numbers predictable, even ones legible, which
    balances game_order() over groups of 8 participants."""
    return MOTIONS[(participant_number(participant) - 1) % 2]


def game_order(participant, targets, seed, first):
    """8 (motion, target) pairs for one participant, reproducible from (participant, seed, first).

    - block A, trials 1-4: every cube once with motion `first`; block B, trials 5-8: every
      cube once with the other motion (questionnaire on paper in between)
    - block A cube order: Williams-square row, one row per pair of participants (P01/P02,
      P03/P04, ...); with `first` alternating as default_first_motion(), groups of 8
      participants see every cube in every position equally often with each motion
    - block B cube order: from the seed, never starting with the last cube of block A and
      never block A's order again
    """
    if len(targets) != 4:
        raise ValueError("The game design needs exactly 4 cubes.")
    if first not in MOTIONS:
        raise ValueError("first must be one of %s, not %r" % (", ".join(MOTIONS), first))
    j = (participant_number(participant) - 1) % 8
    cubes = [targets[i] for i in WILLIAMS_4[j // 2]]
    rng = random.Random("%s/%d" % (seed, participant_number(participant)))
    second = MOTIONS[1 - MOTIONS.index(first)]
    orders = [perm for perm in itertools.permutations(cubes) if perm[0] != cubes[-1] and list(perm) != cubes]
    return [(first, t) for t in cubes] + [(second, t) for t in rng.choice(orders)]


def make_game_schedule(participant, targets, seed, first, training=()):
    """Trials {trial_id, phase, block, index_in_phase, motion, target_id}: first the `training`
    (motion, target) pairs (phase 'training', block 'T', practice, logged separately), then
    phase 'game', block 'A' (rounds 1-4, motion `first`) and 'B' (rounds 5-8, the other
    motion). trial_id counts within each phase, so the game trials are 0-7 with or without
    practice."""
    practice = [{"trial_id": i, "phase": "training", "block": "T", "index_in_phase": i,
                 "motion": motion, "target_id": target} for i, (motion, target) in enumerate(training)]
    return practice + [{"trial_id": i, "phase": "game", "block": "AB"[i // 4], "index_in_phase": i,
                        "motion": motion, "target_id": target}
                       for i, (motion, target) in enumerate(game_order(participant, targets, seed, first))]


def points_left(u, points_max=100, exponent=1.0):
    """Points still possible at reach fraction u (0..1): what a right pick made now and kept
    to the end earns. exponent > 1 drains faster, most of all at the start."""
    return points_max * (1.0 - min(max(u, 0.0), 1.0)) ** exponent


class Leaderboard:
    """<output_dir>/leaderboard.csv: one line per finished game. Only ranks are shown."""
    FIELDS = ["participant_id", "date", "points"]

    def __init__(self, path):
        self.path = path

    def scores(self, exclude=None):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [int(r["points"]) for r in csv.DictReader(f) if r["participant_id"] != exclude]

    def add(self, participant_id, date, total):
        """Store the total; returns (rank, number of players, best before this game)."""
        others = self.scores(exclude=participant_id)
        SessionLog._append(self.path, self.FIELDS, [{"participant_id": participant_id, "date": date,
                                                     "points": total}])
        rank = 1 + sum(1 for s in others if s > total)
        return rank, len(others) + 1, max(others) if others else None


# --- one trial -----------------------------------------------------------------------

class TrialRecord:
    """Choices of one trial and the reach timing; computes the trial measures."""

    def __init__(self, trial, t_trial_start, points_max=100, exponent=1.0):
        self.trial = trial
        self.points_max, self.exponent = points_max, exponent  # game scoring, see points_left()
        self.t_trial_start = t_trial_start
        self.t_start = None     # reach onset (motion node 'executing')
        self.t_done = None      # motion node 'done' / 'failed'
        self.status = "pending"
        self.choices = []       # [(stamp, choice_id)], changes only
        self.confidence = None

    @property
    def current_choice(self):
        return self.choices[-1][1] if self.choices else -1

    def add_choice(self, stamp, choice_id):
        """Record a press. Pressing the selected button again keeps it. True if it changed."""
        if choice_id == self.current_choice:
            return False
        self.choices.append((stamp, choice_id))
        return True

    def choice_at(self, t):
        c = -1
        for stamp, choice in self.choices:
            if stamp > t:
                break
            c = choice
        return c

    def measures(self, duration):
        """f, t_first_correct, t_commit, n_switches, final_choice, final_correct, points.
        Times are fractions of the reach duration T, measured from reach onset."""
        if self.t_start is None:
            return {}
        target, t0 = self.trial["target_id"], self.t_start
        t1 = t0 + duration
        # Selection as a step function on [t0, t1]: the choice held at t0, then the changes.
        steps = [(t0, self.choice_at(t0))] + [(t, c) for t, c in self.choices if t0 < t <= t1]
        correct_time = sum(min(t1, nxt) - t for (t, c), (nxt, _) in zip(steps, steps[1:] + [(t1, None)])
                           if c == target)
        f = correct_time / duration
        # Points: each correct stretch [a, b] earns points_left(a) - points_left(b), so a right
        # pick at u kept to the end earns points_left(u) (exponent 1, max 100: 100 f).
        u = lambda t: (t - t0) / duration
        earned = sum(points_left(u(t), self.points_max, self.exponent)
                     - points_left(u(min(t1, nxt)), self.points_max, self.exponent)
                     for (t, c), (nxt, _) in zip(steps, steps[1:] + [(t1, None)]) if c == target)
        first = next((t for t, c in steps if c == target), None)
        final = steps[-1][1]
        # Commit = start of the final correct stretch (1.0 if the reach did not end on it).
        commit = steps[-1][0] if final == target else t1
        rel = lambda t: None if t is None else min(max((t - t0) / duration, 0.0), 1.0)
        return {"f": f, "t_first_correct": rel(first), "t_commit": rel(commit),
                "n_switches": max(0, sum(1 for t, _ in self.choices if t <= t1) - 1),
                "final_choice": final, "final_correct": int(final == target),
                "points": int(round(earned))}

    def row(self, participant_id, duration, blocks, colors):
        target = self.trial["target_id"]
        row = {k: self.trial[k] for k in ("trial_id", "phase", "block", "index_in_phase",
                                          "motion", "target_id")}
        row.update(participant_id=participant_id, target_name=blocks[target]["name"],
                   target_color=colors[target], status=self.status,
                   t_trial_start=self.t_trial_start, t_start=self.t_start,
                   t_end=None if self.t_start is None else self.t_start + duration,
                   t_done=self.t_done, confidence=self.confidence)
        row.update(self.measures(duration))
        t0 = self.t_start if self.t_start is not None else self.t_trial_start
        row.update(choice_names=";".join(blocks[c]["name"] if 0 <= c < len(blocks) else str(c)
                                         for _, c in self.choices),
                   choice_ids=";".join(str(c) for _, c in self.choices),
                   choice_times=";".join("%.3f" % (t - t0) for t, _ in self.choices))
        return row

    def choice_rows(self, participant_id):
        t0 = self.t_start if self.t_start is not None else self.t_trial_start
        return [{"participant_id": participant_id, "trial_id": self.trial["trial_id"],
                 "stamp": t, "t_rel": t - t0, "choice_id": c} for t, c in self.choices]


class SessionLog:
    """<dir>/<prefix>trials.csv and <dir>/<prefix>choices.csv, appended and flushed after every
    trial, so a crash loses at most the running trial. prefix 'practice_': the practice rounds."""

    def __init__(self, directory, prefix=""):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.trials_path = os.path.join(directory, prefix + "trials.csv")
        self.choices_path = os.path.join(directory, prefix + "choices.csv")

    @staticmethod
    def _append(path, fields, rows):
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerows(rows)

    def write(self, record, participant_id, duration, blocks, colors):
        row = record.row(participant_id, duration, blocks, colors)
        self._append(self.trials_path, TRIAL_FIELDS, [row])
        self._append(self.choices_path, CHOICE_FIELDS, record.choice_rows(participant_id))
        return row
