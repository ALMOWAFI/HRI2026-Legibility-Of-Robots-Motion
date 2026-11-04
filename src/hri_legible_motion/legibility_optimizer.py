"""Reach paths and the planar legibility optimizer for the UR5e Legible Motion HRI study.

baseline_path()          straight reach start -> target
sample(), stop_and_go()  min-jerk timing of a path
planar_legibility()      Dragan et al. 2013 goal-inference score in the reach plane
optimize_planar_reach()  predictable reach -> legible reach (same start, end, height, duration)

The planar optimizer uses minimum-jerk progress along a polyline's arc length. Equal
duration does not imply equal speed: a longer path moves faster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

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
# 2. Fixed-Height (Planar) Legibility Optimizer: the legible reaches of the study
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
