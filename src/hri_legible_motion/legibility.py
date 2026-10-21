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


def legible_path(start, goal, bend, n=2001):
    """Straight line plus a lateral offset bend * sin(pi u): 0 at both ends, |bend| at
    mid-reach (plan §2.1). `bend` is horizontal (trajectories.bend_vector), so with start
    and goal at the same height the whole path stays in that plane. Re-parametrised so
    s is proportional to arc length, like baseline_path()."""
    start, goal, bend = (np.asarray(v, float) for v in (start, goal, bend))
    u = np.linspace(0.0, 1.0, n)
    xyz = start + u[:, None] * (goal - start) + np.sin(np.pi * u)[:, None] * bend
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    arc /= arc[-1]
    return lambda s: np.stack([np.interp(s, arc, xyz[:, i]) for i in range(3)], axis=-1)


def legibility_score(xyz, goals, target, beta=10.0):
    """Dragan et al. 2013 legibility of a sampled path (rows equally spaced in time),
    in the horizontal plane: cost = x-y path length (m), uniform prior over the goals.
    P(G | path so far) ~ exp(-beta (C(S->Q) + |Q G|)) / exp(-beta |S G|), averaged over
    time with weight (T - t). `beta` (1/m) sets how sharply the observer's guess follows
    the cost; only meaningful for comparing paths scored with the same beta. In [0, 1]."""
    p = np.asarray(xyz, float)[:, :2]
    g = np.asarray(goals, float)[:, :2]
    so_far = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    to_goal = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2)            # (N, goals)
    direct = np.linalg.norm(g - p[0], axis=1)                                   # (goals,)
    logit = -beta * (so_far[:, None] + to_goal - direct[None, :])
    prob = np.exp(logit - logit.max(axis=1, keepdims=True))
    prob = prob[:, target] / prob.sum(axis=1)
    weight = np.linspace(1.0, 0.0, len(p))
    return float((prob * weight).sum() / weight.sum())
