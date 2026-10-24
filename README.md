# UR5e legible-motion experiment

ROS 1 / MoveIt, with fixed block positions and horizontal reaches above the table.

The current planner is [tools/test_motion.py](tools/test_motion.py). The optimizer
node is [scripts/legibility_optimizer_node.py](scripts/legibility_optimizer_node.py),
and its mathematics is in [src/hri_legible_motion/legibility_optimizer.py](src/hri_legible_motion/legibility_optimizer.py).

```text
config/experiment.yaml
  -> test_motion.py: straight Cartesian baseline
  -> /hri/predictable_reach
  -> legibility_optimizer_node.py: constrained 2D optimization
  -> /hri/legible_reach
  -> test_motion.py: MoveIt Cartesian IK + trajectory checks + RViz
  -> explicit operator execution
```

Both reach topics use `moveit_msgs/CartesianTrajectory`. The optimizer publishes
Cartesian proposals and never executes the robot. This flow uses Cartesian
planning; it does not use RRT-Connect. The earlier separate RRT/FK integration is
historical context in [docs/connected_pipeline.md](docs/connected_pipeline.md).

## Preview in the lab

Start the lab's existing robot/MoveIt stack, then from this package directory:

```bash
# Terminal 1: optimizer, no motion
roslaunch hri_legible_motion optimizer.launch
# Alternatively: python3 scripts/legibility_optimizer_node.py

# Terminal 2: baseline preview, then optimized preview
python3 tools/test_motion.py B2
python3 tools/test_motion.py B2 --legible
```

Default configuration: `config/experiment.yaml`. Restart the optimizer after
editing it. To select another file, use `config:=/path/experiment.yaml` with
roslaunch and `--config /path/experiment.yaml` with the Python tools.

The configured group is `arm`, tool link `arm_tool0`, and coordinate frame
`arm_base_link`. The current tool0 reach plane is 0.113 m in that frame; this is
not the height above the table. Block placement remains `tools/place_blocks.py`;
`tools/spawn_cubes.py` displays the configured scene. Review the current lab
geometry and pointer model before motion.

Execution remains explicit through `--execute` and an interactive operator
confirmation. Offline tests do not establish robot readiness. The lab must
validate IK, joint limits, timing/acceleration, controller interpolation and the
current collision scene for new trajectories.

## Compare the optimizer without the robot

```bash
python -m pip install -r requirements-offline.txt
python -m unittest discover -s test -v
python tools/compare_optimizer.py --lambdas 0 0.2 2 --bounds
```

Open the PNG/SVG figures and `summary.csv` in `build/optimizer-comparison/`.
They compare the straight baseline, previous cached legible path, and new
optimization under the same observer and motion limits. They show path shape,
model probability, and whether a competing goal becomes more likely.

A visible bend and a model score gain are different measurements. Increasing
`efficiency_weight` (lambda) discourages extra length. `length_ratio` is the hard
length budget. The new `protect_goal_ranking` constraint prevents introducing a
wrong-goal preference. Read [the optimizer guide](docs/planar_optimizer.md) for
its equations, acceptance rules, model limitations and paper comparison.

## Study and participant interface

`launch/hri.launch` starts the motion replay node and participant display.
`scripts/experiment_node.py` handles trials, buttons, ratings and recording.
These use the saved trajectory cache, not the live optimizer topic.

The existing `config/trajectories.npz` is retained as the previous study cache.
Offline comparisons do not replace it. With MoveIt available, generate and
review a separate new cache before selecting it for a study:

```bash
python3 tools/precompute_trajectories.py --out build/trajectories-reviewed.npz
```

This command plans and previews without executing. After lab validation, select
the reviewed cache via `trajectories` in the session configuration, restart the
motion node, and use the existing study commands:

```bash
roslaunch hri_legible_motion hri.launch
rosrun hri_legible_motion experiment_node.py P07
```

Buttons publish `/button_press`; the experiment publishes `/trial_start`,
`/display` and `/choice`, and consumes `/motion_status` and `/rating`. Study details
are in [docs/experiment_plan.md](docs/experiment_plan.md).
