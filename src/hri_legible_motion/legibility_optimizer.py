"""Planner + legibility optimizer for the UR5e Legible Motion HRI study.

baseline_path()     straight reach start -> target
legible_path()      bent / legible path (start -> target + bend * sin(pi * s))
legibility_score()  Dragan et al. 2013 goal-inference score along trajectory

Paths are functions s -> xyz with s in [0, 1] proportional to arc length, so the
same time law gives both motion types the same duration and speed profile.

Also provides analytical UR5e forward kinematics and continuous joint-space SLSQP
optimization for optional joint-level legibility synthesis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, minimize

Array = np.ndarray

# ==============================================================================
# 1. Cartesian Path Generators & Sampling (Experiment Plan §2.1)
# ==============================================================================

def baseline_path(start: Sequence[float], goal: Sequence[float]):
    """Straight line start -> goal."""
    start_arr = np.asarray(start, dtype=float)
    goal_arr = np.asarray(goal, dtype=float)
    return lambda s: start_arr + np.asarray(s, dtype=float)[..., None] * (goal_arr - start_arr)


def legible_path(start: Sequence[float], goal: Sequence[float], bend: Sequence[float]):
    """Legible path: start -> goal bent laterally by offset vector `bend`.

    p(s) = start + s * (goal - start) + bend * sin(pi * s)
    where s in [0, 1] parameterizes the motion from start to goal.
    At s = 0: p(0) = start
    At s = 1: p(1) = goal
    At s = 0.5: midpoint + bend
    """
    start_arr = np.asarray(start, dtype=float)
    goal_arr = np.asarray(goal, dtype=float)
    bend_arr = np.asarray(bend, dtype=float)

    def p(s):
        s_arr = np.asarray(s, dtype=float)
        if s_arr.ndim == 0:
            return start_arr + s_arr * (goal_arr - start_arr) + bend_arr * np.sin(np.pi * s_arr)
        return start_arr + s_arr[..., None] * (goal_arr - start_arr) + bend_arr * np.sin(np.pi * s_arr[..., None])

    return p


def min_jerk(tau: Union[float, Array]) -> Union[float, Array]:
    """Minimum-jerk time law: tau in [0, 1] -> s in [0, 1], zero vel/acc at both ends."""
    tau_clipped = np.clip(tau, 0.0, 1.0)
    return 10.0 * tau_clipped ** 3 - 15.0 * tau_clipped ** 4 + 6.0 * tau_clipped ** 5


def sample(path: Callable[[Array], Array], duration: float, rate: float = 100.0) -> Tuple[Array, Array]:
    """Sample a path under the min-jerk time law. Returns (times, xyz waypoints)."""
    n = int(round(duration * rate)) + 1
    t = np.linspace(0.0, duration, n)
    return t, path(min_jerk(t / duration))


def stop_and_go(points: Sequence[Sequence[float]], speed: float, rate: float = 100.0, min_duration: float = 1.0) -> Tuple[Array, Array]:
    """Straight segments through `points`, each min-jerk with a stop at every corner
    (setup moves, not trials). `speed` is the average speed per segment (m/s).
    Returns (times, xyz); zero-length segments are skipped."""
    pts = [np.asarray(p, dtype=float) for p in points]
    times, xyz, t0 = [0.0], [pts[0]], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1e-4:
            continue
        t, p = sample(baseline_path(a, b), max(min_duration, length / speed), rate)
        times.extend(t0 + t[1:])
        xyz.extend(p[1:])
        t0 += t[-1]
    return np.array(times), np.array(xyz)


# ==============================================================================
# 2. Goal Posterior & Legibility Scoring (Dragan et al. 2013)
# ==============================================================================

def goal_posterior(ee_pos: Array, start_pos: Array, goals: Array, beta: float = 4.0) -> Array:
    """Softmax probability distribution over candidate goals based on end-effector motion.
    
    Handles both single 3D position (3,) and array of positions (N, 3).
    """
    ee_pos = np.asarray(ee_pos, dtype=float)
    goals = np.asarray(goals, dtype=float)
    start_pos = np.asarray(start_pos, dtype=float)

    d_start = np.linalg.norm(goals - start_pos, axis=-1)

    if ee_pos.ndim == 1:
        d_curr = np.linalg.norm(goals - ee_pos, axis=-1)
        logits = beta * (d_start - d_curr)
        logits -= np.max(logits)
        exp_vals = np.exp(logits)
        return exp_vals / np.sum(exp_vals)
    else:
        d_curr = np.linalg.norm(goals[None, :, :] - ee_pos[:, None, :], axis=-1)
        logits = beta * (d_start[None, :] - d_curr)
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(logits)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def legibility_score(
    path: Union[Callable[[Array], Array], Tuple[Array, Array]],
    goals: Sequence[Sequence[float]],
    target: Union[int, Sequence[float]],
    duration: float = 4.0,
    beta: float = 4.0,
) -> float:
    """Compute overall legibility score along trajectory (Dragan et al. 2013).
    
    path: function s -> xyz or tuple of (times, xyz)
    goals: candidate block positions (M, 3)
    target: target index (0..M-1) or 3D target coordinates
    """
    goals_arr = np.asarray(goals, dtype=float)
    if isinstance(target, (int, np.integer)):
        target_idx = int(target)
    else:
        target_pos = np.asarray(target, dtype=float)
        dists = np.linalg.norm(goals_arr - target_pos, axis=1)
        target_idx = int(np.argmin(dists))

    if callable(path):
        times, xyz = sample(path, duration)
    else:
        times, xyz = path

    xyz = np.asarray(xyz, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(xyz) < 2:
        return 0.0

    start_pos = xyz[0]
    total_time = times[-1] - times[0]
    if total_time <= 0:
        return 0.0

    dt = np.diff(times)
    weights = (times[-1] - times[:-1]) / total_time
    total_weight = float(np.sum(weights * dt))
    if total_weight <= 0:
        return 0.0

    probs = goal_posterior(xyz[:-1], start_pos, goals_arr, beta=beta)
    target_probs = probs[:, target_idx]

    score = float(np.sum(target_probs * weights * dt) / total_weight)
    return score


# ==============================================================================
# 3. UR5e Analytical Forward Kinematics & Joint Optimization
# ==============================================================================

UR5E_JOINTS = [
    "arm_shoulder_pan_joint",
    "arm_shoulder_lift_joint",
    "arm_elbow_joint",
    "arm_wrist_1_joint",
    "arm_wrist_2_joint",
    "arm_wrist_3_joint",
]

DH_PARAMS = [
    {"d": 0.1625, "a": 0.0, "alpha": math.pi / 2.0},
    {"d": 0.0, "a": -0.425, "alpha": 0.0},
    {"d": 0.0, "a": -0.3922, "alpha": 0.0},
    {"d": 0.1333, "a": 0.0, "alpha": math.pi / 2.0},
    {"d": 0.0997, "a": 0.0, "alpha": -math.pi / 2.0},
    {"d": 0.0996, "a": 0.0, "alpha": 0.0},
]


def ur5e_fk_position(q: Sequence[float]) -> Array:
    """Compute (x, y, z) position of arm_tool0 from joint angles q in standard order."""
    T = np.eye(4)
    for i, dh in enumerate(DH_PARAMS):
        theta = float(q[i])
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(dh["alpha"]), math.sin(dh["alpha"])
        a, d = dh["a"], dh["d"]

        Ti = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0, sa,      ca,       d],
            [0.0, 0.0,     0.0,      1.0],
        ])
        T = T @ Ti

    return T[:3, 3]


@dataclass(frozen=True)
class LegibilityConfig:
    beta: float = 4.0
    cost_ratio: float = 1.30
    max_joint_deviation: float = 0.50
    deviation_weight: float = 1e-3
    maxiter: int = 200
    ftol: float = 1e-6


@dataclass(frozen=True)
class OptimizationOutcome:
    positions: Array
    times: Array
    success: bool
    message: str
    baseline_score: float
    optimized_score: float
    cost_ratio: float


def compute_legibility_score_from_joints(
    q_traj: Array, times: Array, goal_positions: Array, target_idx: int,
    fk_fn: Callable[[Array], Array] = ur5e_fk_position, beta: float = 4.0
) -> float:
    """Compute legibility score along a joint trajectory."""
    ee_traj = np.array([fk_fn(q) for q in q_traj])
    return legibility_score((times, ee_traj), goal_positions, target_idx, beta=beta)


def optimize_legible_trajectory(
    baseline_q: Array,
    times: Array,
    goal_positions: Array,
    target_idx: int,
    joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    config: Optional[LegibilityConfig] = None,
    fk_fn: Callable[[Array], Array] = ur5e_fk_position,
) -> OptimizationOutcome:
    """Optimize baseline joint trajectory to maximize target legibility using SLSQP.

    Preserves exact start and goal configurations (q[0] and q[-1]).
    Optimizes interior waypoints q[1:-1].
    """
    if config is None:
        config = LegibilityConfig()

    q0 = np.array(baseline_q, dtype=float, copy=True)
    t = np.array(times, dtype=float, copy=True)
    N, D = q0.shape

    if N < 3:
        return OptimizationOutcome(
            positions=q0, times=t, success=False,
            message="Trajectory must have at least 3 points to optimize interior",
            baseline_score=0.0, optimized_score=0.0, cost_ratio=1.0,
        )

    dt = np.diff(t)
    dt_mat = dt[:, None]

    baseline_cost = float(np.sum(np.diff(q0, axis=0) ** 2 / dt_mat))
    max_cost = config.cost_ratio * baseline_cost

    base_score = compute_legibility_score_from_joints(q0, t, goal_positions, target_idx, fk_fn, config.beta)

    x0 = q0[1:-1].ravel()
    start_ee = fk_fn(q0[0])

    def unpack(x: Array) -> Array:
        full_q = np.empty_like(q0)
        full_q[0] = q0[0]
        full_q[1:-1] = x.reshape((N - 2, D))
        full_q[-1] = q0[-1]
        return full_q

    total_time = t[-1] - t[0]

    def objective(x: Array) -> float:
        q_curr = unpack(x)
        ee_curr = np.array([fk_fn(q) for q in q_curr[1:-1]])
        probs = goal_posterior(ee_curr, start_ee, goal_positions, beta=config.beta)
        target_probs = probs[:, target_idx]
        weights = (t[-1] - t[1:-1]) / total_time
        score = float(np.sum(target_probs * weights * dt[:-1]))
        dev_cost = config.deviation_weight * float(np.sum((x - x0) ** 2))
        return float(-score + dev_cost)

    def cost_constraint(x: Array) -> float:
        q_curr = unpack(x)
        cost = float(np.sum(np.diff(q_curr, axis=0) ** 2 / dt_mat))
        return max_cost - cost

    constraints = [{"type": "ineq", "fun": cost_constraint}]

    lower_bounds = np.empty((N - 2, D))
    upper_bounds = np.empty((N - 2, D))

    for d, joint_name in enumerate(UR5E_JOINTS):
        j_min, j_max = (-6.28, 6.28)
        if joint_limits and joint_name in joint_limits:
            j_min, j_max = joint_limits[joint_name]
        lower_bounds[:, d] = np.maximum(j_min, q0[1:-1, d] - config.max_joint_deviation)
        upper_bounds[:, d] = np.minimum(j_max, q0[1:-1, d] + config.max_joint_deviation)

    bounds = Bounds(lower_bounds.ravel(), upper_bounds.ravel())

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": config.maxiter, "ftol": config.ftol},
    )

    opt_q = unpack(res.x)
    opt_cost = float(np.sum(np.diff(opt_q, axis=0) ** 2 / dt_mat))
    realized_ratio = opt_cost / baseline_cost if baseline_cost > 0 else 1.0
    opt_score = compute_legibility_score_from_joints(opt_q, t, goal_positions, target_idx, fk_fn, config.beta)

    return OptimizationOutcome(
        positions=opt_q,
        times=t,
        success=bool(res.success),
        message=str(res.message),
        baseline_score=base_score,
        optimized_score=opt_score,
        cost_ratio=realized_ratio,
    )
