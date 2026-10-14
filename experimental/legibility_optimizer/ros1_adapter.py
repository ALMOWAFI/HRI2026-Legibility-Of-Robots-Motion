"""Read-only ROS 1 RobotTrajectory adapter; no ROS node or execution publisher.

Uses the ROS 1 message attribute protocol so offline tests need no ROS install.
The report's asynchronous target/candidate topics are deliberately replaced by
one explicit call carrying the trajectory, target and goals for the same trial.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .optimizer_core import OptimizationResult, optimize_joint_path, validate_trajectory_arrays


@dataclass(frozen=True)
class TrajectoryCandidate:
    joint_names: Tuple[str, ...]
    goal_frame: str
    result: OptimizationResult


def extract_path(message):
    """Extract named positions and nominal time; reject unsupported multi-DOF paths."""
    multi = message.multi_dof_joint_trajectory
    if multi.points or multi.joint_names:
        raise ValueError("multi-DOF trajectories are unsupported by this prototype")
    trajectory = message.joint_trajectory
    names = tuple(trajectory.joint_names)
    if (not names or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("joint_names must be nonempty, unique strings")
    positions, times = validate_trajectory_arrays(
        [point.positions for point in trajectory.points],
        [point.time_from_start.to_sec() for point in trajectory.points],
        minimum_points=3)
    if positions.shape[1] != len(names):
        raise ValueError("joint_names and position columns do not match")
    return names, positions, times


def _ordered(mapping, names, label):
    if set(mapping) != set(names):
        raise ValueError(f"{label} must contain exactly the trajectory's joint names")
    return [mapping[name] for name in names]


def optimize_robot_trajectory(message, goal_positions, true_goal_index, *,
                              fk_position, goal_frame, fk_frame,
                              joint_limits, max_joint_deviation,
                              velocity_limits=None, acceleration_limits=None,
                              config=None, priors=None) -> TrajectoryCandidate:
    """Prepare an offline candidate with explicit joint ordering and frame names.

    fk_position(q, joint_names) must use the confirmed robot model, full state
    and tool transform. Matching labels are required but cannot verify that the
    provider actually computes in that frame. No UR5/UR5e/UD5 model is inferred.
    Original velocities/accelerations/effort are neither copied nor reused.
    """
    if (not isinstance(goal_frame, str) or not goal_frame.strip()
            or goal_frame != fk_frame):
        raise ValueError("goal_frame and fk_frame must be explicit and equal")
    names, positions, times = extract_path(message)
    limits = _ordered(joint_limits, names, "joint_limits")
    if any(len(pair) != 2 for pair in limits):
        raise ValueError("each joint limit must contain (lower, upper)")
    result = optimize_joint_path(
        positions, times, goal_positions, true_goal_index,
        lambda q: fk_position(q, names),
        joint_lower=[pair[0] for pair in limits],
        joint_upper=[pair[1] for pair in limits],
        max_joint_deviation=_ordered(max_joint_deviation, names, "max_joint_deviation"),
        velocity_limits=None if velocity_limits is None else
        _ordered(velocity_limits, names, "velocity_limits"),
        acceleration_limits=None if acceleration_limits is None else
        _ordered(acceleration_limits, names, "acceleration_limits"),
        config=config, priors=priors)
    return TrajectoryCandidate(names, goal_frame, result)
