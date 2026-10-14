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
No vision node: block coordinates live in `config/experiment.yaml`, in the robot frame.
The table moves between lab sessions, so the robot places the blocks (see below).

## Before every lab session (calibration)
The cube spots are fixed relative to the robot, not the table. Each session:
1. Put the table under the arc of spots (75–83 cm in front of the robot base,
   from 11 cm to its left to 38 cm to its right) and clear it.
2. Bring up the robot (one terminal each, keep them running):
   ```
   ~/scripts/launch_hardware.sh
   ~/scripts/resend_robot_program.sh      # pendant in Remote Control
   ```
3. Preview, then run the placement at the robot, hand on the e-stop:
   ```
   python3 tools/place_blocks.py              # plan + RViz preview, no motion
   python3 tools/place_blocks.py --execute    # robot points at each spot
   ```
   For each colour the robot stops 3 cm above where the cube's top will be; put the
   cube under the tool tip (long side pointing back toward the start pose), press Enter. The cube
   appears in RViz. If the tip is clearly not ~3 cm above the table top, the table
   height differs: stop and adjust `motion.touch_z`.
4. Optional check: `python3 tools/test_motion.py yellow` (preview) / `--execute`.

After restarting MoveIt or RViz, `python3 tools/spawn_cubes.py` shows the cubes again
(no motion). Robot motion always needs a person at the terminal typing `go`;
Claude Code is blocked from running it (`.claude/hooks/block_robot_motion.py`).

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
| `src/hri_legible_motion/robot.py` | FK, Cartesian planning with our timing, execution |
| `src/hri_legible_motion/safety.py` | Workspace / speed / joint-jump checks, operator confirmation |
| `src/hri_legible_motion/scene.py` | Cubes in the MoveIt scene and as RViz markers |
| `tools/place_blocks.py` | Session calibration: robot points, you place the cubes |
| `tools/spawn_cubes.py` | Show the cubes in RViz (no motion) |
| `tools/test_motion.py` | Straight test reach to one cube |
| `docs/experiment_plan.md` | Study implementation plan |

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
