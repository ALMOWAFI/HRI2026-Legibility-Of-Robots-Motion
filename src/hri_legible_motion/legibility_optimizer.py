"""Planner + legibility optimizer for the UR5e Legible Motion HRI study.

baseline_path()     straight reach start -> target
legible_path()      bent / legible path (start -> target + bend * sin(pi * s))
legibility_score()  Dragan et al. 2013 goal-inference score along trajectory

The active planar optimizer uses minimum-jerk progress along a polyline's arc
length. Equal duration does not imply equal speed: a longer path moves faster.

Also provides analytical UR5e forward kinematics and continuous joint-space SLSQP
optimization for optional joint-level legibility synthesis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    if xyz.ndim != 2 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError("polyline needs at least two finite positions")
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    keep = np.r_[True, np.diff(arc) > 1e-12]
    xyz, arc = xyz[keep], arc[keep]
    if len(arc) < 2:
        raise ValueError("polyline must have positive length")
    arc /= arc[-1]
    return lambda s: np.stack([np.interp(s, arc, xyz[:, i]) for i in range(xyz.shape[1])], axis=-1)


def _planar_inputs(xyz, goals, beta):
    p, g = np.asarray(xyz, float), np.asarray(goals, float)
    if p.ndim != 2 or len(p) < 2 or p.shape[1] not in (2, 3) or not np.isfinite(p).all():
        raise ValueError("xyz must contain at least two finite XY or XYZ positions")
    if g.ndim != 2 or len(g) < 2 or g.shape[1] not in (2, 3) or not np.isfinite(g).all():
        raise ValueError("goals must contain at least two finite XY or XYZ positions")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if len(np.unique(g[:, :2], axis=0)) != len(g):
        raise ValueError("candidate goals must have distinct XY positions")
    return p[:, :2], g[:, :2]


def _time_weights(times, count):
    t = np.asarray(times, float)
    if t.shape != (count,) or not np.isfinite(t).all() or count < 2:
        raise ValueError("times must have one finite timestamp per position")
    dt = np.diff(t)
    if t[0] < 0 or np.any(dt <= 0):
        raise ValueError("timestamps must be nonnegative and strictly increasing")
    # Trapezoidal quadrature of P(t) * (T-t); supports nonuniform sampling.
    quadrature = np.r_[dt[0], dt[:-1] + dt[1:], dt[-1]] / 2.0
    weights = quadrature * (t[-1] - t)
    return t, weights / weights.sum()


def _posterior(p, g, start, beta):
    # The observed prefix cost is common to every goal and cancels in softmax.
    logits = beta * (np.linalg.norm(g - start, axis=1)[None, :]
                     - np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2))
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def planar_posterior(xyz: Array, goals, beta: float = 10.0) -> Array:
    """Uniform-prior distance-cost observer; rows are distributions over goals.

    The start is xyz[0]. These are model probabilities, not calibrated human
    recognition probabilities. Keep beta and goals fixed in comparisons.
    """
    p, g = _planar_inputs(xyz, goals, beta)
    return _posterior(p, g, p[0], beta)


def _check_target(target, count):
    if isinstance(target, bool) or not isinstance(target, (int, np.integer)) or not 0 <= target < count:
        raise ValueError("target must be a valid goal row index")


def planar_legibility(xyz: Array, goals, target: int, beta: float = 10.0, times=None) -> float:
    """Time-weighted target posterior, using planar distance as observer cost.

    This is an adaptation of Dragan's goal-inference formulation. Optional times
    are actual seconds; omission assumes uniform sampling. Version 2 uses
    trapezoidal time integration, so re-score old cached paths for comparisons.
    """
    probabilities = planar_posterior(xyz, goals, beta)
    _check_target(target, probabilities.shape[1])
    _, weights = _time_weights(np.linspace(0, 1, len(xyz)) if times is None else times, len(xyz))
    return float(weights @ probabilities[:, target])


def planar_metrics(times, xyz, goals, target, beta=10.0):
    """Recognition diagnostics under the SAME observer, separate from path bending."""
    p, g = _planar_inputs(xyz, goals, beta)
    _check_target(target, len(g))
    t, weights = _time_weights(times, len(p))
    probabilities = _posterior(p, g, p[0], beta)
    intended = probabilities[:, target]
    other = np.max(np.delete(probabilities, target, axis=1), axis=1)
    lead = intended > other + 1e-12  # uniform-prior ties are not correct guesses

    def sustained_time(mask):
        indices = np.flatnonzero(np.logical_and.accumulate(mask[::-1])[::-1])
        return float(t[indices[0]] - t[0]) if len(indices) else None

    # Triangle inequality proves P(target|Q) <= P(target|target). This is an
    # instantaneous ceiling, NOT an achievable time-averaged optimum.
    ceiling = float(_posterior(g[target:target + 1], g, p[0], beta)[0, target])
    return {
        "score": float(weights @ intended),
        "posterior_ceiling": ceiling,
        "posterior_at_quarters": np.interp(t[0] + np.array([.25, .5, .75]) * (t[-1] - t[0]), t, intended).tolist(),
        "first_confident_time": sustained_time(intended >= .5),
        "first_unique_lead_time": sustained_time(lead),
        "weighted_correct_fraction": float(weights @ lead),
        "minimum_goal_margin": float((intended - other).min()),
        "final_posterior": float(intended[-1]),
    }


@dataclass(frozen=True)
class PlanarConfig:
    beta: float = 10.0            # 1/m, observer model of planar_legibility()
    length_ratio: float = 1.10    # legible path at most this much longer than the straight reach
    efficiency_weight: float = 0.0  # soft efficiency penalty, not the paper's trust-region multiplier
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
    n_starts: int = 5             # deterministic searches: straight, left/right, early left/right
    min_score_gain: float = 1e-4  # reject negligible gains (0.01 percentage points)
    max_tcp_speed: Optional[float] = None  # populated from safety for the real configuration
    protect_goal_ranking: bool = True  # prevent new preferences for competing goals

    def __post_init__(self):
        if not isinstance(self.protect_goal_ranking, bool):
            raise ValueError("protect_goal_ranking must be a boolean")
        for name in ("beta", "length_ratio", "max_coeff"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.length_ratio < 1:
            raise ValueError("length_ratio cannot be below 1")
        for name in ("efficiency_weight", "smooth_weight", "min_radius", "max_radius", "extension_radius", "min_score_gain"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.min_radius >= self.max_radius or self.extension_radius > self.max_radius:
            raise ValueError("inconsistent radial limits")
        for name in ("n_terms", "maxiter", "n_starts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.n_starts > 5:
            raise ValueError("n_starts must be between 1 and 5")
        lo, hi = np.asarray(self.xy_min), np.asarray(self.xy_max)
        if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
            raise ValueError("workspace must contain finite increasing XY bounds")
        if self.max_tcp_speed is not None and (not np.isfinite(self.max_tcp_speed) or self.max_tcp_speed <= 0):
            raise ValueError("max_tcp_speed must be finite and positive")

    @classmethod
    def from_cfg(cls, cfg: dict) -> "PlanarConfig":
        """From experiment.yaml: the `optimizer` section plus the safety workspace box."""
        o, s = dict(cfg.get("optimizer", {})), cfg["safety"]
        o.pop("timeout", None)
        o.setdefault("max_tcp_speed", s.get("max_tcp_speed"))
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
    numerical_success: bool = False
    geometric_valid: bool = False
    physical_validity: str = "not_checked"
    diagnostics: dict = field(default_factory=dict)


def optimize_planar_reach(times: Array, xyz: Array, goals: Sequence[Sequence[float]], target: int,
                          config: Optional[PlanarConfig] = None, n_geom: int = 201) -> PlanarOutcome:
    """Legible version of the predictable reach (times, xyz): same start, end, height and
    duration. Shape = straight line + x-y offset sum_k c_k sin(k pi u), k = 1..n_terms, so
    the ends stay fixed and the path smooth; the result is re-timed with min-jerk on arc
    length at the original sample times. `goals`: hover points (x, y[, z]) of all cubes."""
    config = config or PlanarConfig()
    xyz = np.array(xyz, dtype=float, copy=True)
    _, goal_xy = _planar_inputs(xyz, goals, config.beta)
    if xyz.shape[1] != 3 or len(xyz) < 3:
        raise ValueError("optimization needs at least three XYZ samples")
    _check_target(target, len(goal_xy))
    t, weights = _time_weights(times, len(xyz))
    t = t.copy()
    if isinstance(n_geom, bool) or not isinstance(n_geom, (int, np.integer)) or n_geom < 20:
        raise ValueError("n_geom must be an integer of at least 20")
    p0, p1 = xyz[0], xyz[-1]
    if np.abs(xyz[:, 2] - p0[2]).max() > 1e-3 or abs(p1[2] - p0[2]) > 1e-3:
        raise ValueError("optimize_planar_reach needs a planar reach (constant z)")
    duration, K = t[-1] - t[0], config.n_terms
    u = np.linspace(0.0, 1.0, n_geom)
    basis = np.sin(np.pi * np.outer(u, np.arange(1, K + 1)))            # (n_geom, K)
    straight_len = float(np.linalg.norm((p1 - p0)[:2]))
    if straight_len < 1e-6:
        raise ValueError("reach needs distinct start and end XY positions")
    if np.linalg.norm(p1[:2] - goal_xy[target]) > .01:
        raise ValueError("reach does not end at the intended goal")
    lo, hi = np.asarray(config.xy_min), np.asarray(config.xy_max)
    if np.any(xyz[:, :2] < lo) or np.any(xyz[:, :2] > hi):
        raise ValueError("baseline leaves the configured workspace")
    # Preserve the existing endpoint inclusion rule, but report effective bounds.
    # This is a geometric heuristic, NOT evidence that IK or collisions passed.
    r_ends = [float(np.hypot(*p0[:2])), float(np.hypot(*p1[:2]))]
    r_min, r_max = min([config.min_radius] + r_ends) - 1e-6, max([config.max_radius] + r_ends) + 1e-6

    def geometry(c):
        g = p0 + u[:, None] * (p1 - p0)
        g[:, :2] += basis @ c.reshape(2, K).T
        g[0], g[-1] = p0, p1
        return g

    def timed(c):
        path = polyline_path(geometry(c))
        result = path(min_jerk((t - t[0]) / duration))
        result[0], result[-1] = p0, p1
        return result

    def length(g):
        return float(np.linalg.norm(np.diff(g[:, :2], axis=0), axis=1).sum())

    r_straight = np.hypot(*(p0[:2] + u[:, None] * (p1 - p0)[:2]).T)
    r_cap = np.minimum(r_max, np.maximum(config.extension_radius, r_straight) + 1e-6)
    freq2 = np.tile(np.arange(1, K + 1) ** 2, 2)

    def score(path):
        return float(weights @ _posterior(path[:, :2], goal_xy, p0[:2], config.beta)[:, target])

    def speeds(path):
        return np.linalg.norm(np.diff(path, axis=0), axis=1) / np.diff(t)

    competitors = np.arange(len(goal_xy)) != target

    def margins(path):
        probabilities = _posterior(path[:, :2], goal_xy, p0[:2], config.beta)
        return probabilities[:, target, None] - probabilities[:, competitors]

    # Absolute P(target) alone can improve while a neighbour becomes MORE likely.
    # Preserve correct ordering; if the input is already misleading, do not make
    # its pairwise margin worse. This is an additional experiment-specific rule.
    margin_floor = np.minimum(margins(xyz), 0.0)

    if config.max_tcp_speed is not None and speeds(xyz).max() > config.max_tcp_speed + 1e-6:
        raise ValueError("baseline exceeds max_tcp_speed")

    def terms(c):
        return (score(timed(c)), config.efficiency_weight * (length(geometry(c)) / straight_len - 1.0),
                config.smooth_weight * float(freq2 @ c ** 2))

    def objective(c):
        value, efficiency, smoothness = terms(c)
        return -value + efficiency + smoothness

    def minimum_segment_radii(g):
        a, d = g[:-1, :2], np.diff(g[:, :2], axis=0)
        projection = np.clip(-np.sum(a * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-30), 0, 1)
        return np.linalg.norm(a + projection[:, None] * d, axis=1)

    def constraints(c):
        g = geometry(c)
        r = np.hypot(g[:, 0], g[:, 1])
        values = [[config.length_ratio * straight_len - length(g)],
                  minimum_segment_radii(g) - r_min, r_cap - r,
                  (g[:, :2] - lo).ravel(), (hi - g[:, :2]).ravel()]
        if config.max_tcp_speed is not None:
            values.append(config.max_tcp_speed - speeds(timed(c)))
        if config.protect_goal_ranking:
            values.append((margins(timed(c)) - margin_floor).ravel())
        return np.concatenate(values)

    baseline = score(xyz)
    baseline_objective = -baseline + config.efficiency_weight * (length(xyz) / straight_len - 1.0)
    # Keep both old seeds and add a straight seed and two early-bend seeds.
    # Multiple starts reduce dependence on one local basin; no global claim.
    side = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / max(straight_len, 1e-9)
    seeds = []
    for sign in (1.0, -1.0):
        c0 = np.zeros(2 * K)
        c0[0], c0[K] = 0.02 * sign * side
        seeds.append(c0)
    seeds.append(np.zeros(2 * K))
    for sign in (1.0, -1.0):
        c0 = np.zeros(2 * K)
        c0[0], c0[K] = .04 * sign * side
        if K > 1:
            c0[1], c0[K + 1] = .02 * sign * side
        seeds.append(c0)

    best, best_value, attempts = None, float("inf"), []
    for c0 in seeds[:config.n_starts]:
        c0 = np.clip(c0, -config.max_coeff, config.max_coeff)
        res = minimize(objective, c0, method="SLSQP", bounds=Bounds(-config.max_coeff, config.max_coeff),
                       constraints=[{"type": "ineq", "fun": constraints}],
                       options={"maxiter": config.maxiter, "ftol": 1e-9})
        candidate = np.asarray(res.x, float)
        record = {"numerical_success": bool(res.success), "message": str(res.message),
                  "iterations": int(getattr(res, "nit", 0)), "geometric_valid": False, "accepted": False}
        attempts.append(record)
        if candidate.shape != (2 * K,) or not np.isfinite(candidate).all():
            record["rejection"] = "nonfinite or malformed coefficients"
            continue
        if not res.success:
            record["rejection"] = "solver did not converge"
            continue
        # Recompute from returned coefficients; never trust res.fun alone.
        margin = float(min(constraints(candidate).min(), (config.max_coeff - np.abs(candidate)).min()))
        value, penalty, smoothness = terms(candidate)
        evaluated = -value + penalty + smoothness
        g = geometry(candidate)
        # Independently inspect interpolated geometry at 4x optimization density.
        dense_u = np.linspace(0, 1, 4 * (n_geom - 1) + 1)
        dense = np.column_stack([np.interp(dense_u, u, g[:, i]) for i in range(3)])
        dense_cap = np.minimum(r_max, np.maximum(config.extension_radius,
                              np.linalg.norm(p0[:2] + dense_u[:, None] * (p1 - p0)[:2], axis=1)) + 1e-6)
        dense_margin = float((dense_cap - np.linalg.norm(dense[:, :2], axis=1)).min())
        ranking_margin = float((margins(timed(candidate)) - margin_floor).min())
        feasible = (np.isfinite(evaluated) and min(margin, dense_margin) >= -1e-6
                    and (not config.protect_goal_ranking or ranking_margin >= -1e-8))
        improved = value - baseline >= config.min_score_gain and evaluated < baseline_objective - 1e-9
        record.update(geometric_valid=bool(feasible), score=float(value), objective=float(evaluated),
                      min_constraint_margin=min(margin, dense_margin), min_ranking_margin=ranking_margin,
                      accepted=bool(feasible and improved))
        if not feasible:
            record["rejection"] = "independent geometric check failed"
        elif not improved:
            record["rejection"] = "no sufficient score and objective improvement over input"
        elif evaluated < best_value:
            best, best_value = candidate.copy(), evaluated

    diagnostics = {"score_version": "time_trapezoid_v2", "beta": config.beta,
                   "efficiency_weight": config.efficiency_weight, "baseline_objective": baseline_objective,
                   "protect_goal_ranking": config.protect_goal_ranking,
                   "attempts": attempts, "effective_min_radius": r_min, "effective_max_radius": r_max,
                   "baseline_metrics": planar_metrics(t, xyz, goals, target, config.beta)}
    if best is None:
        direction = (p1 - p0)[:2] / straight_len
        relative = xyz[:, :2] - p0[:2]
        baseline_lateral = float(np.abs(relative[:, 0] * direction[1] - relative[:, 1] * direction[0]).max())
        return PlanarOutcome(t, xyz, False, "no converged, feasible candidate with sufficient improvement; see diagnostics",
                             baseline, baseline, length(xyz) / straight_len, baseline_lateral,
                             numerical_success=any(a["numerical_success"] for a in attempts),
                             geometric_valid=any(a["geometric_valid"] for a in attempts), diagnostics=diagnostics)
    out = timed(best)
    out[0], out[-1] = p0, p1
    d = (p1 - p0)[:2] / max(straight_len, 1e-9)
    rel = out[:, :2] - p0[:2]
    value, efficiency, smoothness = terms(best)
    ratio = length(geometry(best)) / straight_len
    g = geometry(best)
    active = []
    if config.length_ratio - ratio <= 1e-5:
        active.append("length_ratio")
    if (r_cap - np.linalg.norm(g[:, :2], axis=1)).min() <= 1e-5:
        active.append("radial_envelope")
    if min((g[:, :2] - lo).min(), (hi - g[:, :2]).min()) <= 1e-5:
        active.append("workspace")
    if config.max_tcp_speed is not None and config.max_tcp_speed - speeds(out).max() <= 1e-5:
        active.append("sampled_tcp_speed")
    if config.protect_goal_ranking and (margins(out)[1:-1] - margin_floor[1:-1]).min() <= 1e-6:
        active.append("goal_ranking")
    diagnostics.update(optimized_objective=best_value, efficiency_penalty=efficiency,
                       smoothness_penalty=smoothness, score_gain_pp=100 * (value - baseline),
                       active_constraints=active, peak_tcp_speed=float(speeds(out).max()),
                       optimized_metrics=planar_metrics(t, out, goals, target, config.beta))
    return PlanarOutcome(t, out, True, "accepted Cartesian candidate; robot validation still required",
                         baseline, value, ratio,
                         float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max()),
                         numerical_success=True, geometric_valid=True, diagnostics=diagnostics)
