# hri_legible_motion

Legible robot motion HRI experiment on a **UR5e**, using ROS 1 and MoveIt.
The active lab integration compares an RRT-Connect baseline with FK-based
legibility optimization. The exact installed ROS/MoveIt version must be checked
on the lab computer.

**Start here:** [connected pipeline lab guide](docs/connected_pipeline.md).
Use this repository's `integration/rrt-optimizer-session` branch with the optimizer
repository's `integration/preview-planner-optimizer` branch. The changes are in
draft PRs. The companion package now includes a separate validation/execution
node, disabled for motion by default. Live ROS and hardware verification remain.

## Connected lab architecture
```
Shared config → block layout publisher → RRT planner
Experiment node --pipeline → trial request ─┘
RRT baseline → FK optimizer / baseline bypass → trajectory proposal
  → downstream retiming + sampled model/scene checks → confirmed execution
  → correlated measured motion feedback → Experiment node
Participant → Buttons → /button_press → Experiment node → choices + timestamps + trial data (CSV)
```
No vision node: block coordinates live in `config/experiment.yaml`, in the robot frame.
The table moves between lab sessions, so the placement tool guides human block
placement at fixed positions (see below).

Use `roslaunch hri_legibility_preview group_pipeline.launch` with the current
`experiment_config`, then the guide's planning-only trial checks. The legacy
`hri.launch`, replay motion node and trajectory cache are not used in this flow.
They remain available for the original mode described at the end of this README.

## Before every lab session (calibration)
The cube spots are fixed relative to the robot, not the table. Each session:
1. Put the table under the arc of spots (80–88 cm in front of the robot base,
   from 12 cm to its left to 39 cm to its right) and clear it.
   To change the layout: `python3 tools/make_layout.py --back 0.05 --radius 0.45` (preview,
   no motion), add `--write` to update the config, then redo the placement.
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
   For each colour the robot stops 5 cm above where the cube's top will be (`setup.point_height`);
   put the cube under the tool tip (long side pointing back toward the start pose), press Enter.
   The cube appears in RViz. If the tip is clearly not ~6 cm above the table top (5 cm + cube
   height), the table height differs: stop and adjust `motion.touch_z`.
4. Optional check: `python3 tools/test_motion.py yellow` (preview) / `--execute`.

After restarting MoveIt or RViz, `python3 tools/spawn_cubes.py` shows the cubes again
(no motion). Robot motion always needs a person at the terminal typing `go`;
Claude Code is blocked from running it (`.claude/hooks/block_robot_motion.py`).

## Layout
| Path | Purpose |
|---|---|
| `scripts/motion_node.py` | Legacy replay mode; not used by the connected pipeline |
| `scripts/experiment_node.py` | Session state machine, buttons, scoring, ratings, CSV + rosbag |
| `scripts/display_node.py` | Participant monitor (choice, score, rating prompt) + 1-7 keypad |
| `src/hri_legible_motion/experiment.py` | Counterbalancing, trial schedule, measures (f, score, ...), CSV (no ROS) |
| `src/hri_legible_motion/trajectories.py` | Legacy trajectory-cache save/load (no ROS) |
| `tools/precompute_trajectories.py` | Legacy cache-generation tool: 2D reaches in the hover plane, both motion types |
| `test/` | Offline tests: `python3 -m unittest discover -s test` |
| `src/hri_legible_motion/legibility.py` | Legacy straight / sine-bend paths and 2D legibility score; the FK optimizer lives in the separate package |
| `msg/TrialInfo.msg` | Experiment → motion: trial id, target, condition, bend |
| `msg/ChoiceEvent.msg` | Participant choice change |
| `config/experiment.yaml` | Block coordinates, robot, trial settings |
| `launch/hri.launch` | Starts legacy replay motion and display; not the connected pipeline |
| `src/hri_legible_motion/robot.py` | FK, Cartesian planning with our timing, execution |
| `src/hri_legible_motion/safety.py` | Workspace / speed / joint-jump checks, operator confirmation |
| `src/hri_legible_motion/scene.py` | Cubes in the MoveIt scene and as RViz markers |
| `tools/place_blocks.py` | Session calibration: robot points, you place the cubes |
| `tools/spawn_cubes.py` | Show the cubes in RViz (no motion) |
| `tools/test_motion.py` | Straight test reach to one cube |
| `docs/experiment_plan.md` | Study implementation plan |

## Existing group topics

The `/trial_start`, `/motion_status` and `/motion_home` rows below apply only to
the legacy mode. The connected mode uses the typed `/legibility/*` interfaces
listed in the [current guide](docs/connected_pipeline.md). Button, choice, display
and rating topics are shared by both modes.
| Topic | Type | From → To |
|---|---|---|
| `/trial_start` | `TrialInfo` | experiment → motion |
| `/motion_status` | `std_msgs/String` (executing / done / failed / home) | motion → experiment |
| `/motion_home` | `std_msgs/Empty` | experiment → motion (go to the start pose) |
| `/button_press` | `std_msgs/Int32` (button 0–3, left to right as the participant sees them) | buttons → experiment |
| `/choice` | `ChoiceEvent` | experiment → logging |
| `/display` | `std_msgs/String` (JSON) | experiment → display |
| `/rating` | `std_msgs/Int32` (1–7) | display keypad → experiment |

Confirmed lab identifiers: planning group `arm`, planning frame `base`, goal frame
`arm_base_link`, and end-effector `arm_tool0`. Frames must be transformed explicitly,
not relabeled. The config names the controller action
`/scaled_pos_joint_traj_controller/follow_joint_trajectory`; verify it on the robot.

## Legacy mode: precompute the reaches
With MoveIt running (robot stack, or `roslaunch ur5e_moveit_config demo.launch`):
```
python3 tools/precompute_trajectories.py        # plans, checks, RViz preview, saves config/trajectories.npz
```
No robot motion. Saves `predictable_0..3` and `legible_0..3` with a hash that goes into every
session log. Until `legibility.legible_path()` exists, only the predictable reaches are saved and
legible trials fail with a clear message. Rerun after changing cubes, start pose or `motion.duration`.

## Legacy mode: run a cached session

An opt-in [RRT/FK optimizer integration](docs/connected_pipeline.md) is available
with `experiment_node.py --pipeline --config /absolute/path/to/experiment.yaml`.
It uses the separate optimizer package and its opt-in downstream node. Complete
the connected guide's lab checks before enabling execution. The cache/replay
instructions below remain the original mode.

After the calibration above, one terminal each:
```
roslaunch hri_legible_motion hri.launch                        # motion node + participant display
rosrun hri_legible_motion experiment_node.py P07               # experimenter terminal
```
The experiment node prints the cube colours for this participant (colour rotation), starts the
rosbag, asks you to type `go` to send the robot to the start pose, then waits for Enter before
every trial. `q` stops; resume with `--start-trial N`. Output: `~/hri_data/P07/`
(`session.yaml`, `trials.csv`, `choices.csv`, `session.bag`).

Dry run without the robot: `roslaunch ur5e_moveit_config demo.launch`, then
`roslaunch hri_legible_motion hri.launch sim:=true`, and set `display.keyboard_buttons: true`
to use keys a s d f as the four buttons. Ratings are keys 1-7 on the display window.

Offline tests (no ROS): `python3 -m unittest discover -s test`
