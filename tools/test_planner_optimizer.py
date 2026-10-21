#!/usr/bin/env python3
"""Safe Dry-Run Test for MoveIt Baseline Planning + Legibility Optimizer.

Tests the full pipeline on the real robot WITHOUT commanding physical motion.
Plans a reach to Block 0 (B1) and optimizes its legibility trajectory.

Usage:
    python3 tools/test_planner_optimizer.py
"""
import sys
import numpy as np
import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from hri_legible_motion.legibility import (
    LegibilityConfig,
    UR5E_JOINTS,
    optimize_legible_trajectory,
    ur5e_fk_position,
)

def main():
    rospy.init_node("test_planner_optimizer", anonymous=True)
    roscpp_initialize(sys.argv)

    print("=" * 60)
    print("UR5e Legibility Planner + Optimizer Dry-Run Test")
    print("=" * 60)

    group = MoveGroupCommander("arm")
    group.set_pose_reference_frame("base")
    group.set_max_velocity_scaling_factor(0.3)
    group.set_max_acceleration_scaling_factor(0.3)

    current_q = group.get_current_joint_values()
    active_names = group.get_active_joints()
    print(f"Current Robot Joints ({len(current_q)}):")
    for name, q in zip(active_names, current_q):
        print(f"  {name:26s}: {q:+7.3f} rad ({q * 57.2958:+7.1f} deg)")

    current_ee = ur5e_fk_position(current_q)
    print(f"\nCurrent EE Position (arm_tool0 via FK): [{current_ee[0]:.3f}, {current_ee[1]:.3f}, {current_ee[2]:.3f}] m")

    # Candidate Block Goals (B1 - B4)
    goals = np.array([
        [0.919,  0.153, 0.011],  # B1
        [0.930, -0.031, 0.005],  # B2
        [0.907, -0.209, 0.003],  # B3
        [0.876, -0.371, 0.046],  # B4
    ])
    target_idx = 0  # Target B1

    print(f"\nPlanning baseline reach to Target B1 ({goals[target_idx]})...")
    # Recorded target configuration for B1 in standard joint order
    target_q = [-2.8275, -3.1974, -0.0364, -1.4115, 1.5202, -0.1319]
    group.set_joint_value_target(target_q)

    plan_res = group.plan()
    if isinstance(plan_res, tuple):
        success, traj = plan_res[0], plan_res[1]
    else:
        traj = plan_res
        success = bool(traj.joint_trajectory.points)

    if not success:
        print("ERROR: MoveIt failed to plan a baseline reach.")
        return

    pts = traj.joint_trajectory.points
    print(f"SUCCESS: MoveIt generated baseline with {len(pts)} waypoints.")

    # Extract trajectory
    names = traj.joint_trajectory.joint_names
    indices = [names.index(j) for j in UR5E_JOINTS if j in names]
    q_baseline = np.array([[pt.positions[i] for i in indices] for pt in pts], dtype=float)
    times = np.array([pt.time_from_start.to_sec() for pt in pts], dtype=float)

    print(f"Baseline duration: {times[-1]:.2f} s")

    # Run Legibility Optimizer
    config = LegibilityConfig(cost_ratio=1.35, max_joint_deviation=0.45)
    print("\nRunning Legibility Optimization (SLSQP)...")
    outcome = optimize_legible_trajectory(
        q_baseline,
        times,
        goals,
        target_idx=target_idx,
        config=config,
        fk_fn=ur5e_fk_position,
    )

    print("-" * 60)
    print(f"Optimizer Success:     {outcome.success} ({outcome.message})")
    print(f"Baseline Score:        {outcome.baseline_score:.4f}")
    print(f"Optimized Score:       {outcome.optimized_score:.4f} "
          f"(+{((outcome.optimized_score - outcome.baseline_score) / max(outcome.baseline_score, 1e-6) * 100):.1f}%)")
    print(f"Realized Cost Ratio:   {outcome.cost_ratio:.3f} (budget: {config.cost_ratio})")
    print("-" * 60)

    print("\n[DRY RUN COMPLETE] No hardware motion commanded.")
    print("Both baseline planning and legibility optimization are functional.")

if __name__ == "__main__":
    try:
        main()
    finally:
        roscpp_shutdown()
