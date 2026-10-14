# hri_legible_motion

Legible robot motion HRI experiment on a **UR5e** (ROS Noetic).
Tests whether bending the robot's reach makes its target easier to infer.

## Architecture
```
config (fixed block coordinates) ─┐
Experiment node ── /trial_start ──► Motion node: baseline → legibility optimizer
                                    → MoveIt IK → joint trajectory → UR5e
Participant → Buttons → /button_press → Experiment node → choices + timestamps + trial data (CSV)
```
No vision node: blocks don't move, so their coordinates live in `config/experiment.yaml`.

## Layout
| Path | Purpose |
|---|---|
| `scripts/motion_node.py` | Motion/planning node |
| `scripts/experiment_node.py` | Experiment/game node |
| `src/hri_legible_motion/legibility.py` | Planner + legibility optimizer |
| `msg/TrialInfo.msg` | Experiment → motion: trial id, target, condition, bend |
| `msg/ChoiceEvent.msg` | Participant choice change |
| `config/experiment.yaml` | Block coordinates, robot, trial settings |
| `launch/hri.launch` | Starts both nodes |

## Topics
| Topic | Type | From → To |
|---|---|---|
| `/trial_start` | `TrialInfo` | experiment → motion |
| `/motion_status` | `std_msgs/String` (executing / done / failed / home) | motion → experiment |
| `/button_press` | `std_msgs/Int32` (block id 0–3) | buttons → experiment |
| `/choice` | `ChoiceEvent` | experiment → logging |

UR5e side (driver + MoveIt): `/joint_states`, `/tf` (`base_link` → `tool0`),
`/scaled_pos_joint_traj_controller/follow_joint_trajectory`, `/move_group`.

## Run (once implemented)
```
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=<IP>
roslaunch ur5e_moveit_config moveit_planning_execution.launch
roslaunch hri_legible_motion hri.launch participant:=P01
```

## Experimental FK-based optimizer

The supplied UD5 research report, repaired LaTeX manuscript, standalone optimizer,
offline ROS 1 trajectory adapter and unit tests are in
[`docs/research`](docs/research/README.md). The prototype optimizes interior joint
positions while evaluating goal inference through injected forward kinematics.
It is separate from the existing nodes, launch files and recorded-pose controller
tool. Numerical acceptance is not physical validation, and it never executes a
trajectory. UD5 is not assumed to be the UR5e described by this scaffold.

See the [integration notes](docs/research/integration.md) for the missing baseline
planner, actual robot FK, retiming, collision validation and explicit execution
requirements, and the [research README](docs/research/README.md) for dependencies,
test commands and manuscript build instructions.
