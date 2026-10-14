"""Offline FK-based optimization, adapted from the supplied UD5 research report.

Only interior joint positions are decision variables. Numerical acceptance is
not robot validation: timing, collisions, task constraints and execution belong
to a separate integration. No ROS imports, publishers or controller clients.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

Array = np.ndarray
FKFn = Callable[[Array], Array]


@dataclass(frozen=True)
class OptimizerConfig:
    beta: float = 4.0
    cost_ratio: float = 1.20
    early_weight_power: float = 1.0
    deviation_weight: float = 1e-3
    min_legibility_improvement: float = 1e-5
    maxiter: int = 250
    ftol: float = 1e-8
    constraint_tolerance: float = 1e-7

    def __post_init__(self):
        for name in ("beta", "ftol", "constraint_tolerance"):
            _finite_scalar(getattr(self, name), name, positive=True)
        for name in ("early_weight_power", "deviation_weight",
                     "min_legibility_improvement"):
            _finite_scalar(getattr(self, name), name, nonnegative=True)
        _finite_scalar(self.cost_ratio, "cost_ratio", positive=True)
        if self.cost_ratio < 1.0:
            raise ValueError("cost_ratio must be >= 1")
        if isinstance(self.maxiter, bool) or not isinstance(self.maxiter, Integral):
            raise ValueError("maxiter must be a positive integer")
        if self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")


@dataclass(frozen=True)
class OptimizationResult:
    positions: Array
    times: Array
    solver_success: bool
    solver_output_valid: bool
    numerical_accepted: bool
    message: str
    baseline_legibility: float
    optimized_legibility: float
    baseline_joint_cost: float
    optimized_joint_cost: float
    realized_cost_ratio: float
    joint_limit_ok: bool
    cost_budget_ok: bool
    local_deviation_ok: bool
    velocity_ok: Optional[bool]
    acceleration_ok: Optional[bool]
    improvement_ok: bool

    @property
    def candidate_ok(self) -> bool:
        """Report-compatible alias for numerical acceptance only."""
        return self.numerical_accepted

    @property
    def physical_validity(self) -> None:
        """Always unassessed here; no robot validation occurs in this module."""
        return None

    @property
    def retime_required(self) -> bool:
        return True

    @property
    def execution_ready(self) -> bool:
        return False


def _finite_scalar(value, name, *, positive=False, nonnegative=False):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _vector(value, size, name, *, positive=False) -> Array:
    x = np.array(value, dtype=float, copy=True)
    if x.shape != (size,) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    if positive and np.any(x <= 0):
        raise ValueError(f"{name} must be positive")
    return x


def validate_trajectory_arrays(joint_positions, times, *, minimum_points=2):
    """Copy arrays and check nominal seconds-from-start, without retiming."""
    q = np.array(joint_positions, dtype=float, copy=True)
    t = np.array(times, dtype=float, copy=True)
    if q.ndim != 2 or q.shape[0] < minimum_points or q.shape[1] < 1:
        raise ValueError(f"joint_positions must have shape (N,D), N >= {minimum_points}")
    if not np.all(np.isfinite(q)):
        raise ValueError("joint_positions contains non-finite values")
    if t.shape != (len(q),) or not np.all(np.isfinite(t)):
        raise ValueError("times must be a finite vector matching the waypoints")
    if t[0] < 0:
        raise ValueError("timestamps must be nonnegative seconds from start")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return q, t


def _goals(value) -> Array:
    goals = np.asarray(value, dtype=float)
    if goals.ndim != 2 or goals.shape[0] < 2 or goals.shape[1] != 3:
        raise ValueError("goal_positions must have shape (M,3), M >= 2")
    if not np.all(np.isfinite(goals)):
        raise ValueError("goal_positions contains non-finite values")
    return goals


def joint_motion_cost(joint_positions, times) -> float:
    """Sum ||q[i+1]-q[i]||^2 / dt[i], on the supplied nominal grid."""
    q, t = validate_trajectory_arrays(joint_positions, times)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        return float(np.sum(np.diff(q, axis=0) ** 2 / np.diff(t)[:, None]))


def segment_velocities(joint_positions, times) -> Array:
    q, t = validate_trajectory_arrays(joint_positions, times)
    return np.diff(q, axis=0) / np.diff(t)[:, None]


def interior_accelerations(joint_positions, times) -> Array:
    q, t = validate_trajectory_arrays(joint_positions, times)
    dt = np.diff(t)
    velocity = np.diff(q, axis=0) / dt[:, None]
    return np.diff(velocity, axis=0) / (0.5 * (dt[:-1] + dt[1:]))[:, None]


def fk_path(joint_positions, fk_fn: FKFn) -> Array:
    """FK must be deterministic and return xyz in the candidate goals' frame."""
    return np.vstack([
        _vector(fk_fn(np.array(q, dtype=float, copy=True)), 3, "FK output")
        for q in joint_positions
    ])


def goal_posterior(ee_position, start_ee_position, goal_positions, *,
                   beta: float, priors: Optional[Sequence[float]] = None) -> Array:
    """Stable softmax of beta * (distance(start,g)-distance(current,g)) + log P(g)."""
    _finite_scalar(beta, "beta", positive=True)
    goals = _goals(goal_positions)
    current = _vector(ee_position, 3, "ee_position")
    start = _vector(start_ee_position, 3, "start_ee_position")
    prior = (np.full(len(goals), 1.0 / len(goals)) if priors is None
             else _vector(priors, len(goals), "priors", positive=True))
    if not np.isclose(prior.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("priors must sum to 1")
    with np.errstate(over="raise", invalid="raise"):
        # The report smooths Euclidean distances with epsilon = 1e-12.
        d_start = np.sqrt(np.sum((goals - start) ** 2, axis=1) + 1e-12)
        d_current = np.sqrt(np.sum((goals - current) ** 2, axis=1) + 1e-12)
        logits = beta * (d_start - d_current) + np.log(prior)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def true_goal_probability_trace(joint_positions, times, goal_positions,
                                true_goal_index, fk_fn: FKFn, *, beta=4.0,
                                priors=None):
    q, t = validate_trajectory_arrays(joint_positions, times)
    goals = _goals(goal_positions)
    if (isinstance(true_goal_index, bool)
            or not isinstance(true_goal_index, Integral)
            or not 0 <= true_goal_index < len(goals)):
        raise ValueError("true_goal_index must be an integer indexing goal_positions")
    ee = fk_path(q, fk_fn)
    probabilities = np.array([
        goal_posterior(p, ee[0], goals, beta=beta, priors=priors)[true_goal_index]
        for p in ee
    ])
    return (t - t[0]) / (t[-1] - t[0]), probabilities


def legibility_score(joint_positions, times, goal_positions, true_goal_index,
                     fk_fn: FKFn, *, beta=4.0, early_weight_power=1.0,
                     priors=None) -> float:
    """Report's waypoint-weighted, pre-arrival score; not a time integral."""
    q, t = validate_trajectory_arrays(joint_positions, times, minimum_points=3)
    _finite_scalar(early_weight_power, "early_weight_power", nonnegative=True)
    tau, probabilities = true_goal_probability_trace(
        q, t, goal_positions, true_goal_index, fk_fn, beta=beta, priors=priors)
    # Log weights avoid underflow for large, but finite, early-time powers.
    log_weights = early_weight_power * np.log(np.maximum(1 - tau[1:-1], 1e-12))
    weights = np.exp(log_weights - np.max(log_weights))
    return float(np.dot(weights, probabilities[1:-1]) / weights.sum())


def densify_joint_path(joint_positions, times, samples_per_segment):
    """Linear samples for inspection, NOT continuous collision certification."""
    q, t = validate_trajectory_arrays(joint_positions, times)
    if (isinstance(samples_per_segment, bool)
            or not isinstance(samples_per_segment, Integral)
            or samples_per_segment < 1):
        raise ValueError("samples_per_segment must be a positive integer")
    dense_q, dense_t = [q[0]], [t[0]]
    for i in range(len(q) - 1):
        for k in range(1, samples_per_segment + 1):
            if k == samples_per_segment:
                dense_q.append(q[i + 1].copy())
                dense_t.append(t[i + 1])
            else:
                alpha = k / samples_per_segment
                dense_q.append((1 - alpha) * q[i] + alpha * q[i + 1])
                dense_t.append((1 - alpha) * t[i] + alpha * t[i + 1])
    return np.asarray(dense_q), np.asarray(dense_t)


def optimize_joint_path(joint_positions, times, goal_positions, true_goal_index,
                        fk_fn: FKFn, *, joint_lower, joint_upper,
                        max_joint_deviation, velocity_limits=None,
                        acceleration_limits=None, priors=None,
                        config: Optional[OptimizerConfig] = None) -> OptimizationResult:
    """Return a diagnostic candidate, never an executable robot trajectory.

    Bounds and local per-joint deviations are mandatory finite vectors in the
    input joint order and units. Continuous joints require a chosen unwrapped
    interval; no wrapping or resampling is inferred. Optional dynamic limits
    constrain finite differences on the nominal grid only. FK errors propagate.
    Rejected results may retain a diagnostic path; callers must inspect status.
    """
    config = OptimizerConfig() if config is None else config
    q0, t = validate_trajectory_arrays(joint_positions, times, minimum_points=3)
    dof = q0.shape[1]
    lower = _vector(joint_lower, dof, "joint_lower")
    upper = _vector(joint_upper, dof, "joint_upper")
    local = _vector(max_joint_deviation, dof, "max_joint_deviation", positive=True)
    if np.any(lower >= upper):
        raise ValueError("every lower joint limit must be less than its upper limit")
    vmax = None if velocity_limits is None else _vector(
        velocity_limits, dof, "velocity_limits", positive=True)
    amax = None if acceleration_limits is None else _vector(
        acceleration_limits, dof, "acceleration_limits", positive=True)
    if np.any(q0 < lower) or np.any(q0 > upper):
        raise ValueError("baseline violates supplied joint limits")
    if vmax is not None and np.any(np.abs(segment_velocities(q0, t)) > vmax):
        raise ValueError("baseline violates supplied velocity limits")
    if amax is not None and np.any(np.abs(interior_accelerations(q0, t)) > amax):
        raise ValueError("baseline violates supplied acceleration limits")
    baseline_cost = joint_motion_cost(q0, t)
    if baseline_cost <= 1e-14:
        raise ValueError("baseline trajectory has near-zero motion cost")

    def score(q):
        return legibility_score(q, t, goal_positions, true_goal_index, fk_fn,
                                beta=config.beta, priors=priors,
                                early_weight_power=config.early_weight_power)

    baseline_score = score(q0)  # Validate goals, index, priors and FK before SLSQP.
    maximum_cost = config.cost_ratio * baseline_cost
    _finite_scalar(maximum_cost, "maximum motion cost", positive=True)
    initial = q0[1:-1].reshape(-1).copy()

    def unpack(z):
        candidate = q0.copy()
        candidate[1:-1] = np.asarray(z).reshape(q0[1:-1].shape)
        return candidate  # Endpoints are copied, not numerically constrained.

    def objective(z):
        q = unpack(z)
        deviation = np.mean((q[1:-1] - q0[1:-1]) ** 2)
        return -score(q) + config.deviation_weight * deviation

    bounds = Bounds(np.maximum(lower, q0[1:-1] - local).ravel(),
                    np.minimum(upper, q0[1:-1] + local).ravel())
    constraints = [{"type": "ineq", "fun": lambda z:
                    maximum_cost - joint_motion_cost(unpack(z), t)}]
    if vmax is not None:
        constraints.append({"type": "ineq", "fun": lambda z:
                            (vmax - np.abs(segment_velocities(unpack(z), t))).ravel()})
    if amax is not None:
        constraints.append({"type": "ineq", "fun": lambda z:
                            (amax - np.abs(interior_accelerations(unpack(z), t))).ravel()})
    result = minimize(objective, initial, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": config.maxiter, "ftol": config.ftol,
                               "disp": False})
    z = np.asarray(result.x, dtype=float)
    output_valid = z.shape == initial.shape and bool(np.all(np.isfinite(z)))
    # A malformed solver result is rejected even if the fallback baseline meets
    # the constraints and a zero improvement threshold was explicitly requested.
    candidate = unpack(z) if output_valid else q0.copy()
    candidate_cost = joint_motion_cost(candidate, t)
    candidate_score = score(candidate)
    tol = config.constraint_tolerance
    joint_ok = bool(np.all(candidate >= lower - tol)
                    and np.all(candidate <= upper + tol))
    local_ok = bool(np.all(np.abs(candidate - q0) <= local + tol))
    cost_ok = candidate_cost <= maximum_cost + tol * max(1.0, maximum_cost)
    velocity_ok = None if vmax is None else bool(
        np.all(np.abs(segment_velocities(candidate, t)) <= vmax + tol))
    acceleration_ok = None if amax is None else bool(
        np.all(np.abs(interior_accelerations(candidate, t)) <= amax + tol))
    improvement_ok = candidate_score >= baseline_score + config.min_legibility_improvement
    checks = {
        "solver failed: " + str(result.message): bool(result.success),
        "solver output is malformed or non-finite": output_valid,
        "joint limit violation": joint_ok,
        "local deviation limit violated": local_ok,
        "motion cost budget violated": cost_ok,
        "nominal velocity limit violated": velocity_ok is not False,
        "nominal acceleration limit violated": acceleration_ok is not False,
        "legibility did not improve enough": improvement_ok,
    }
    accepted = all(checks.values())
    messages = [name for name, ok in checks.items() if not ok]
    if accepted:
        messages.append("candidate accepted numerically")
    messages.append("physical validity unassessed; retiming and robot validation required")
    return OptimizationResult(
        positions=candidate, times=t.copy(), solver_success=bool(result.success),
        solver_output_valid=output_valid, numerical_accepted=accepted,
        message="; ".join(messages), baseline_legibility=baseline_score,
        optimized_legibility=candidate_score, baseline_joint_cost=baseline_cost,
        optimized_joint_cost=candidate_cost, realized_cost_ratio=candidate_cost / baseline_cost,
        joint_limit_ok=joint_ok, cost_budget_ok=cost_ok, local_deviation_ok=local_ok,
        velocity_ok=velocity_ok, acceleration_ok=acceleration_ok,
        improvement_ok=improvement_ok)
