# Experimental integration boundary

The intended chain is **MoveIt/RRT-Connect baseline → standalone legibility
optimizer → retiming → robot-specific trajectory validation → explicit execution**.
An optional collision precheck may precede retiming. No part of this change
connects the prototype to the existing controller tool or ROS launch files.

## What the repository actually contains

Inspected revision: `18a1b668db2d7c521900cdee2a70034bd6dc10ec`.

| Evidence | Finding and implication |
|---|---|
| [`README.md`](../../README.md), [`package.xml`](../../package.xml), [`launch/hri.launch`](../../launch/hri.launch) | ROS 1/catkin, documented ROS Noetic and UR5e; dependencies name `moveit_commander`, `ur_robot_driver`, `ur5e_moveit_config`, NumPy and ROS messages. No installed MoveIt version is pinned or established by this checkout. |
| [`scripts/motion_node.py`](../../scripts/motion_node.py) | Docstring and TODO only. No planner call, baseline publisher, retimer or execution implementation. |
| [`src/hri_legible_motion/legibility.py`](../../src/hri_legible_motion/legibility.py) | TODO scaffold describing a Cartesian straight/bent path. No working RRT-Connect implementation exists to intercept. |
| [`tools/goto_pose.py`](../../tools/goto_pose.py) | Separate working recorded-pose action client. It discovers a controller and sends a single target waypoint. It is not a MoveIt baseline generator and must not become the experimental optimizer's input/execution pipeline. Unchanged. |
| [`config/experiment.yaml`](../../config/experiment.yaml) | Empty blocks and motion settings; `frame_id: base_link` is a placeholder. No selected group, limits, EE link or measured goal set. |
| [`tools/goto_pose.py`](../../tools/goto_pose.py) | Recorded names are `arm_elbow_joint`, `arm_shoulder_lift_joint`, `arm_shoulder_pan_joint`, `arm_wrist_1_joint`, `arm_wrist_2_joint`, `arm_wrist_3_joint`; TF uses `arm_base_link` → `arm_tool0`. This differs from the README/config frame names and requires confirmation. Do not reuse the recorded ordering without matching each trajectory's names. |
| [`msg/TrialInfo.msg`](../../msg/TrialInfo.msg) | `target_id` is a block ID. It is not necessarily a zero-based row in the goal matrix. Preserve a trial-local ID-to-row mapping. |
| [`setup.py`](../../setup.py), [`CMakeLists.txt`](../../CMakeLists.txt) | Install only the original catkin package and nodes. The new `experimental` package is deliberately outside that install/launch path. |

There is no URDF/SRDF, collision geometry, joint-limits YAML, OMPL planner
configuration, controller interpolation specification or saved baseline in this
repository. “UD5” remains the supplied report's designation; repository references
to UR5e do not establish equivalence to UD5 or UR5.

## Where a future baseline connects

Once the actual lab planner exists, capture its successful, nonempty
`moveit_msgs/RobotTrajectory` **after planning and before any execution**. Record
the full start state, scene version, joint names, goal frame, tool frame, planner
configuration and trial ID alongside it. Select RRT-Connect using the actual
installed OMPL configuration, not a guessed planner ID.

The Noetic upstream `MoveGroupCommander.plan()` source returns a success flag,
trajectory, planning time and error code. Verify the installed wrapper before
unpacking; do not treat the entire tuple as a trajectory. The integration point
would be after obtaining that successful trajectory in the future motion node,
with the optimizer's result sent to a separate review/retiming/validation stage.
The existing TODO nodes provide no exact executable call site today.
[Upstream Noetic API source](https://github.com/moveit/moveit/blob/noetic-devel/moveit_commander/src/moveit_commander/move_group.py)

The implemented offline adapter is
[`optimize_robot_trajectory`](../../experimental/legibility_optimizer/ros1_adapter.py).
It takes one message, one goal set and one intended index in the same call,
preventing the report sketch's independent target and trajectory subscriptions
from silently pairing different trials. It uses the ROS 1 message attribute
protocol without importing ROS. ROS runtime integration will need the actual
`moveit_msgs` package; none is needed to run these offline tests.

```python
from experimental.legibility_optimizer.ros1_adapter import optimize_robot_trajectory

# All variables below must come from the confirmed lab setup and the same trial.
candidate = optimize_robot_trajectory(
    baseline_robot_trajectory,
    measured_goal_positions,
    goal_row_for_trial_target,
    fk_position=verified_fk_position,  # callable(q, joint_names) -> xyz
    goal_frame=measured_goal_frame,
    fk_frame=verified_fk_frame,
    joint_limits=limits_by_name,       # {name: (lower, upper)}
    max_joint_deviation=local_limits_by_name,
    velocity_limits=velocity_limits_by_name,
    acceleration_limits=acceleration_limits_by_name,
)
if not candidate.result.numerical_accepted:
    raise RuntimeError(candidate.result.message)
# Stop here in this prototype. Candidate is NOT a controller message.
```

The adapter rejects duplicate/missing joint names, malformed positions, fewer
than three points, unsupported multi-DOF motion, and nonfinite, negative or
nonincreasing seconds-from-start. All-zero planner timestamps must be assigned a
documented feasible nominal timing by the lab integration first; the adapter
does not invent durations. Nonzero initial offsets and nonuniform intervals are
preserved. A later executor must separately check that the first waypoint matches
the current full robot state and whether a pre-start hold is intended.

It orders named limit dictionaries by the message's joint names. Matching FK and
goal frame labels are necessary, but not proof of a correct transform. The result
contains only candidate arrays, their nominal timestamps and diagnostics;
original derivatives, effort, timestamps in message headers and multi-DOF data
are never repackaged as an execution command.

## Missing robot integrations and required validation

[`robot_interfaces.py`](../../experimental/legibility_optimizer/robot_interfaces.py)
contains explicit `MissingRobotIntegration` (`NotImplementedError`) failures for
each missing operation. Even its execution placeholder fails if passed a true
validation flag or explicit authorization.

| Operation | Required information and acceptance criteria |
|---|---|
| FK | Actual robot URDF/SRDF, group, joint-name mapping, fixed/passive/mimic joint state, calibrated tool transform, common goal frame. Compare multiple configurations with MoveIt/RViz. No borrowed UR5 model. |
| Nominal constraints | Confirm position, velocity and acceleration limits, continuous-joint unwrapping and per-joint local deviations. Bounds are mandatory; optional dynamic checks return `None` if absent. Finite-difference checks do not establish continuous-time feasibility. |
| Retiming | Use the installed, verified MoveIt retimer, real start state and controller limits. Rebuild velocities/accelerations and valid timestamps; never copy derivatives from the baseline after changing positions. Failure or an empty result must reject the candidate. |
| Final trajectory validation | Check start/end states, intended target pose/orientation, intermediate task constraints, joint and dynamic limits, self/environment collisions with full robot geometry and current scene, attached bodies, interpolation between points and controller compatibility. Densified samples alone are not continuous collision guarantees. |
| Resampling | Retiming can move/resample waypoints. Recheck the full resulting geometry, endpoints and local deviations. Re-evaluate motion budget using a specified matching grid/timing convention and report post-retiming scores separately; a nominal cost ratio does not automatically certify a retimed one. |
| Simulation | Review trajectory and EE trace in RViz and execute in a simulator/fake controller after validation. Test every relevant target and alternative-goal set. Record solver failures and validator rejections. |
| Execution | A future separate operator-controlled integration must execute only the exact validated retimed trajectory against a fresh scene/start state, after simulation and explicit authorization. Do not call a new planner or `go()` to reach the final pose; that can replace the optimized path. |

MoveIt's Noetic documentation explicitly notes that TOTG may alter waypoint
geometry and require additional collision checking. This is why validation of the
retimed path is mandatory, regardless of an earlier geometric precheck.
[MoveIt time parameterization](https://moveit.github.io/moveit_tutorials/doc/time_parameterization/time_parameterization_tutorial.html)

The core's `solver_success` reports SLSQP termination. `numerical_accepted` (also
available as the report-compatible `candidate_ok`) requires independent numerical
constraint and improvement checks. `physical_validity` remains `None`,
`retime_required` remains `True`, and `execution_ready` remains `False` for every
result. Rejected results retain diagnostic arrays; do not interpret their presence
as acceptance or silently execute a baseline fallback.

Before a participant study, separately establish that changes in the model's
posterior predict human goal inference. Identity-FK tests and higher objective
scores provide neither robot validation nor human-subject evidence.
