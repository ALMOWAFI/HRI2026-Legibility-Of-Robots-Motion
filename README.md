# hri_legible_motion

Legible robot motion HRI experiment on a **UR5e** (ROS Noetic).
Tests whether bending the robot's reach makes its target easier to infer.

## Architecture
```
config (fixed block coordinates) ─┐
Experiment node ── /trial_start ──► Motion node: baseline → legibility optimizer
                                    → MoveIt IK → joint trajectory → UR5e
Participant → Joystick → Experiment node → choices + timestamps + trial data (CSV)
```
No vision node: blocks don't move, so their coordinates live in `config/experiment.yaml`.

## Layout
| Path | Purpose |
|---|---|
| `scripts/motion_node.py` | Motion/planning node |
| `scripts/experiment_node.py` | Experiment/game node |
| `src/hri_legible_motion/legibility.py` | Planner + legibility optimizer (no ROS) |
| `msg/TrialInfo.msg` | Experiment → motion: trial id, target, condition, bend |
| `msg/ChoiceEvent.msg` | Participant choice change |
| `config/experiment.yaml` | Block coordinates, robot, trial settings |
| `launch/hri.launch` | Starts joy + both nodes |

## Run (once implemented)
```
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=<IP>
roslaunch ur5e_moveit_config moveit_planning_execution.launch
roslaunch hri_legible_motion hri.launch participant:=P01
```
