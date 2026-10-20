"""Experiment logic without ROS (docs/experiment_plan.md §3.4, §4): counterbalancing,
trial schedule, per-trial choice record + scores, output files.

Times are seconds (floats); experiment_node.py feeds in rospy times.
"""
import csv
import os
import random
import re

MOTIONS = ("predictable", "legible")

TRIAL_FIELDS = ["participant_id", "trial_id", "phase", "block", "index_in_phase", "motion",
                "target_id", "target_name", "target_color", "status",
                "t_trial_start", "t_start", "t_end", "t_done", "f", "score",
                "t_first_correct", "t_commit", "n_switches", "final_choice", "final_correct",
                "confidence", "naturalness"]
CHOICE_FIELDS = ["participant_id", "trial_id", "stamp", "t_rel", "choice_id"]


# --- counterbalancing (§4.3) ---------------------------------------------------------

def participant_number(participant_id):
    """'P07' -> 7. Ids without digits (e.g. a dry run) -> 0."""
    digits = re.findall(r"\d+", str(participant_id))
    return int(digits[-1]) if digits else 0


def assignment_list(seed, n_rotations=4):
    """8 cells (block order x colour rotation), each twice, shuffled once with a fixed seed.
    Entry k-1 is participant k's cell; participants 17+ continue cyclically."""
    cells = [(order, rot) for order in (MOTIONS, MOTIONS[::-1]) for rot in range(n_rotations)]
    cells = cells * 2
    random.Random(seed).shuffle(cells)
    return cells


def assignment(participant_id, seed):
    """(block order as a (motion, motion) tuple, colour rotation 0..3) for a participant."""
    cells = assignment_list(seed)
    return cells[(participant_number(participant_id) - 1) % len(cells)]


def rotated_colors(colors, rotation):
    """Cube colour at each position (block id) for a colour rotation."""
    return [colors[(i + rotation) % len(colors)] for i in range(len(colors))]


# --- schedule (§4.1, §4.2) -----------------------------------------------------------

def max_run(targets):
    """Longest run of the same target in a row."""
    best = run = 1 if targets else 0
    for a, b in zip(targets, targets[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def shuffled(items, rng, limit, max_tries=10000):
    """Shuffle (motion, target) pairs so no target appears more than `limit` times in a row."""
    items = list(items)
    for _ in range(max_tries):
        rng.shuffle(items)
        if max_run([t for _, t in items]) <= limit:
            return items
    raise RuntimeError("No order with at most %d same cubes in a row" % limit)


def make_schedule(block_order, targets, reps, seed, limit=2):
    """List of trials: {trial_id, phase, block, index_in_phase, motion, target_id}.

    warmup     2 predictable + 2 legible, four different cubes
    main       block A (block_order[0]) and block B, each cube `reps` times per block
    announced  every cube x every motion once, target announced before the reach
    """
    rng = random.Random(seed)
    cubes = list(targets)
    rng.shuffle(cubes)
    motions = [MOTIONS[0], MOTIONS[0], MOTIONS[1], MOTIONS[1]]
    rng.shuffle(motions)
    phases = [("warmup", "W", list(zip(motions, cubes[:4])))]
    for name, motion in zip("AB", block_order):
        phases.append(("main", name, shuffled([(motion, t) for t in targets for _ in range(reps)],
                                              rng, limit)))
    phases.append(("announced", "C", shuffled([(m, t) for m in MOTIONS for t in targets], rng, limit)))

    trials = []
    for phase, block, items in phases:
        for i, (motion, target) in enumerate(items):
            trials.append({"trial_id": len(trials), "phase": phase, "block": block,
                           "index_in_phase": i, "motion": motion, "target_id": target})
    return trials


# --- one trial (§3.4) ----------------------------------------------------------------

class TrialRecord:
    """Choices of one trial and the reach timing; computes the trial measures."""

    def __init__(self, trial, t_trial_start):
        self.trial = trial
        self.t_trial_start = t_trial_start
        self.t_start = None     # reach onset (motion node 'executing')
        self.t_done = None      # motion node 'done' / 'failed'
        self.status = "pending"
        self.choices = []       # [(stamp, choice_id)], changes only
        self.confidence = None
        self.naturalness = None

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
        """f, score, t_first_correct, t_commit, n_switches, final_choice, final_correct.
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
        first = next((t for t, c in steps if c == target), None)
        final = steps[-1][1]
        # Commit = start of the final correct stretch (1.0 if the reach did not end on it).
        commit = steps[-1][0] if final == target else t1
        rel = lambda t: None if t is None else min(max((t - t0) / duration, 0.0), 1.0)
        return {"f": f, "score": int(round(10 * f)), "t_first_correct": rel(first),
                "t_commit": rel(commit),
                "n_switches": max(0, sum(1 for t, _ in self.choices if t <= t1) - 1),
                "final_choice": final, "final_correct": int(final == target)}

    def row(self, participant_id, duration, blocks, colors):
        target = self.trial["target_id"]
        row = {k: self.trial[k] for k in ("trial_id", "phase", "block", "index_in_phase",
                                          "motion", "target_id")}
        row.update(participant_id=participant_id, target_name=blocks[target]["name"],
                   target_color=colors[target], status=self.status,
                   t_trial_start=self.t_trial_start, t_start=self.t_start,
                   t_end=None if self.t_start is None else self.t_start + duration,
                   t_done=self.t_done, confidence=self.confidence, naturalness=self.naturalness)
        if self.trial["phase"] != "announced":
            row.update(self.measures(duration))
        return row

    def choice_rows(self, participant_id):
        t0 = self.t_start if self.t_start is not None else self.t_trial_start
        return [{"participant_id": participant_id, "trial_id": self.trial["trial_id"],
                 "stamp": t, "t_rel": t - t0, "choice_id": c} for t, c in self.choices]


class SessionLog:
    """<dir>/trials.csv and <dir>/choices.csv, appended and flushed after every trial,
    so a crash loses at most the running trial."""

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.trials_path = os.path.join(directory, "trials.csv")
        self.choices_path = os.path.join(directory, "choices.csv")

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
