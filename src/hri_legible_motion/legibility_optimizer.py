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


# ==============================================================================
# 4. Fixed-Height (Planar) Legibility Optimizer: the legible reaches of the study
# ==============================================================================
# The reaches are 2D: start and hover point at the same tool height, so only x-y may
# change. The optimizer takes the predictable reach (times, xyz) and returns a legible
# reach with the same start, end, height, duration and time law (min-jerk on arc
# length), which then goes through Robot.plan() (IK + safety checks) like any reach.

def polyline_path(xyz: Array) -> Callable[[Array], Array]:
    """Path through the rows of `xyz` as a function s -> xyz, s proportional to arc
    length (linear between rows, so sample densely)."""
    xyz = np.asarray(xyz, dtype=float)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    arc /= arc[-1]
    return lambda s: np.stack([np.interp(s, arc, xyz[:, i]) for i in range(xyz.shape[1])], axis=-1)


def planar_legibility(xyz: Array, goals: Sequence[Sequence[float]], target: int, beta: float = 10.0) -> float:
    """Dragan et al. 2013 legibility of a path sampled uniformly in time, in the x-y plane.

    Cost C = x-y path length (m), uniform prior over the goals:
    P(G | S->Q) ~ exp(-beta (C(S->Q) + |Q G|)) / exp(-beta |S G|), averaged over time
    with weight (T - t). `beta` (1/m): how sharply the observer's guess follows the
    cost; compare only scores computed with the same beta. Returns a value in [0, 1].
    """
    p = np.asarray(xyz, dtype=float)[:, :2]
    g = np.asarray(goals, dtype=float)[:, :2]
    so_far = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    to_goal = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2)
    direct = np.linalg.norm(g - p[0], axis=1)
    logit = -beta * (so_far[:, None] + to_goal - direct[None, :])
    prob = np.exp(logit - logit.max(axis=1, keepdims=True))
    prob = prob[:, target] / prob.sum(axis=1)
    weight = np.linspace(1.0, 0.0, len(p))
    return float((prob * weight).sum() / weight.sum())


@dataclass(frozen=True)
class PlanarConfig:
    beta: float = 10.0            # 1/m, observer model of planar_legibility()
    length_ratio: float = 1.10    # legible path at most this much longer than the straight reach
    efficiency_weight: float = 0.0  # Dragan's lambda: maximize legibility - lambda * (L / L_straight - 1)
    n_terms: int = 6              # sine terms per axis: shape freedom (more = wigglier)
    smooth_weight: float = 1e-3   # penalty on higher sine terms
    min_radius: float = 0.27      # m from the base axis: closer, the arm cannot point straight down
    max_radius: float = 0.90      # m: reach limit with the pointer down (widened to the reach's ends)
    extension_radius: float = 0.85  # m: beyond this, no farther out than the straight reach (the
                                    # elbow spins fast moving sideways near full extension)
    xy_min: Tuple[float, float] = (-10.0, -10.0)   # workspace box (safety.workspace_min/max)
    xy_max: Tuple[float, float] = (10.0, 10.0)
    max_coeff: float = 0.30       # m, bound per sine coefficient
    maxiter: int = 200

    @classmethod
    def from_cfg(cls, cfg: dict) -> "PlanarConfig":
        """From experiment.yaml: the `optimizer` section plus the safety workspace box."""
        o, s = dict(cfg.get("optimizer", {})), cfg["safety"]
        o.pop("timeout", None)
        return cls(xy_min=tuple(s["workspace_min"][:2]), xy_max=tuple(s["workspace_max"][:2]), **o)


@dataclass(frozen=True)
class PlanarOutcome:
    times: Array
    xyz: Array
    success: bool
    message: str
    baseline_score: float
    optimized_score: float
    length_ratio: float
    max_lateral: float


def optimize_planar_reach(times: Array, xyz: Array, goals: Sequence[Sequence[float]], target: int,
                          config: Optional[PlanarConfig] = None, n_geom: int = 201) -> PlanarOutcome:
    """Legible version of the predictable reach (times, xyz): same start, end, height and
    duration. Shape = straight line + x-y offset sum_k c_k sin(k pi u), k = 1..n_terms, so
    the ends stay fixed and the path smooth; the result is re-timed with min-jerk on arc
    length at the original sample times. `goals`: hover points (x, y[, z]) of all cubes."""
    config = config or PlanarConfig()
    t = np.asarray(times, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    p0, p1 = xyz[0], xyz[-1]
    if np.abs(xyz[:, 2] - p0[2]).max() > 1e-3 or abs(p1[2] - p0[2]) > 1e-3:
        raise ValueError("optimize_planar_reach needs a planar reach (constant z)")
    duration, K = t[-1] - t[0], config.n_terms
    u = np.linspace(0.0, 1.0, n_geom)
    basis = np.sin(np.pi * np.outer(u, np.arange(1, K + 1)))            # (n_geom, K)
    straight_len = float(np.linalg.norm((p1 - p0)[:2]))
    lo, hi = np.asarray(config.xy_min), np.asarray(config.xy_max)
    # The radius limits never exclude the reach's own ends (which the planner already accepted).
    r_ends = [float(np.hypot(*p0[:2])), float(np.hypot(*p1[:2]))]
    r_min, r_max = min([config.min_radius] + r_ends) - 1e-6, max([config.max_radius] + r_ends) + 1e-6

    def geometry(c):
        g = p0 + u[:, None] * (p1 - p0)
        g[:, :2] += basis @ c.reshape(2, K).T
        return g

    def timed(c):
        path = polyline_path(geometry(c))
        return path(min_jerk((t - t[0]) / duration))

    def length(g):
        return float(np.linalg.norm(np.diff(g[:, :2], axis=0), axis=1).sum())

    r_straight = np.hypot(*(p0[:2] + u[:, None] * (p1 - p0)[:2]).T)
    r_cap = np.minimum(r_max, np.maximum(config.extension_radius, r_straight) + 1e-6)
    freq2 = np.tile(np.arange(1, K + 1) ** 2, 2)

    def objective(c):
        extra = length(geometry(c)) / straight_len - 1.0
        return (-planar_legibility(timed(c), goals, target, config.beta) + config.efficiency_weight * extra
                + config.smooth_weight * float(freq2 @ c ** 2))

    def constraints(c):
        g = geometry(c)
        r = np.hypot(g[:, 0], g[:, 1])
        return np.concatenate([[config.length_ratio * straight_len - length(g)],
                               r - r_min, r_cap - r,
                               (g[:, :2] - lo).ravel(), (hi - g[:, :2]).ravel()])

    baseline = planar_legibility(xyz, goals, target, config.beta)
    # Start from small bends to either side of the reach (the straight line is a saddle
    # for the symmetric terms) and keep the better feasible result.
    side = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / max(straight_len, 1e-9)
    best = None
    for sign in (1.0, -1.0):
        c0 = np.zeros(2 * K)
        c0[0], c0[K] = 0.02 * sign * side
        res = minimize(objective, c0, method="SLSQP", bounds=Bounds(-config.max_coeff, config.max_coeff),
                       constraints=[{"type": "ineq", "fun": constraints}],
                       options={"maxiter": config.maxiter, "ftol": 1e-9})
        feasible = constraints(res.x).min() >= -1e-6
        if feasible and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return PlanarOutcome(t, xyz, False, "no feasible legible path within the limits",
                             baseline, baseline, 1.0, 0.0)
    out = timed(best.x)
    out[0], out[-1] = p0, p1
    d = (p1 - p0)[:2] / max(straight_len, 1e-9)
    rel = out[:, :2] - p0[:2]
    return PlanarOutcome(t, out, bool(best.success), str(best.message), baseline,
                         planar_legibility(out, goals, target, config.beta),
                         length(geometry(best.x)) / straight_len,
                         float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max()))
