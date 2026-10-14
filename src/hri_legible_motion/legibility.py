"""Planner + legibility optimizer.

baseline_path()     straight reach start -> target
legible_path()      bent / optimized path
legibility_score()  Dragan et al. 2013, for logging per trial

Paths are functions s -> xyz with s in [0, 1] proportional to arc length, so the
same time law gives both motion types the same duration and speed profile.
"""
import numpy as np


def baseline_path(start, goal):
    """Straight line start -> goal."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    return lambda s: start + np.asarray(s, float)[..., None] * (goal - start)


def min_jerk(tau):
    """Minimum-jerk time law: tau in [0, 1] -> s in [0, 1], zero vel/acc at both ends."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def sample(path, duration, rate=100.0):
    """Sample a path under the min-jerk time law. Returns (times, xyz waypoints)."""
    n = int(round(duration * rate)) + 1
    t = np.linspace(0.0, duration, n)
    return t, path(min_jerk(t / duration))


def stop_and_go(points, speed, rate=100.0, min_duration=1.0):
    """Straight segments through `points`, each min-jerk with a stop at every corner
    (setup moves, not trials). `speed` is the average speed per segment (m/s).
    Returns (times, xyz); zero-length segments are skipped."""
    points = [np.asarray(p, float) for p in points]
    times, xyz, t0 = [0.0], [points[0]], 0.0
    for a, b in zip(points[:-1], points[1:]):
        length = np.linalg.norm(b - a)
        if length < 1e-4:
            continue
        t, p = sample(baseline_path(a, b), max(min_duration, length / speed), rate)
        times.extend(t0 + t[1:])
        xyz.extend(p[1:])
        t0 += t[-1]
    return np.array(times), np.array(xyz)


def legible_path(start, goal, bend):
    # TODO
    raise NotImplementedError


def legibility_score(path, goals, target):
    # TODO
    raise NotImplementedError
