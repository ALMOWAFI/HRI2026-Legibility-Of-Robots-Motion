# Optional RRT / FK optimizer session

`experiment_node.py --pipeline` connects this experiment controller to the
separate `hri_legibility_preview` package. It reuses scheduling, counterbalancing,
buttons, display, ratings and logging. The default cache/replay mode is unchanged.
The placeholder motion node, bend generator and trajectory bank are not used
by the new mode.

Use the optimizer repository's `integration/preview-planner-optimizer` branch.
Full setup, exact interfaces and validation requirements:
[connected-session guide](https://github.com/ALMOWAFI/The-legibilityOptimizer-HRI2026/blob/integration/preview-planner-optimizer/docs/integration/connected-session.md).

Build both packages in a separate ROS 1 workspace, sourced over the actual lab
ROS/MoveIt environment. Then, with MoveIt already running:

```bash
export HRI_CONFIG="$(rospack find hri_legible_motion)/config/experiment.yaml"
roslaunch hri_legibility_preview group_pipeline.launch \
  experiment_config:="$HRI_CONFIG"
```

This starts the layout publisher, RRT planner, FK optimizer and existing display.
It starts no motion executor. First test non-executing requests with:

```bash
# Another sourced terminal; run one check at a time:
rosrun hri_legibility_preview check_pipeline.py --target-id 0 --condition baseline
rosrun hri_legibility_preview check_pipeline.py --target-id 0 --condition legible
```

The check verifies subscriptions, types and actual matching trial messages before
printing `PASS`. Layout data is latched; trajectory messages arrive per trial,
so these topics are normally quiet while idle. The check requests no motion.

Both accepted conditions publish a `TrajectoryProposal`. The baseline is
RRT-Connect, not necessarily a straight Cartesian reach. Legible trials require
numerical acceptance; rejection does not silently fall back. Every proposal
still requires retiming and physical validation.

For a future hardware session, the teammate must implement and validate
`/legibility/validate_trajectory`, `/legibility/request_motion`, and correlated
`/legibility/motion_feedback`. Missing providers fail explicitly. After that,
run in a separate interactive terminal:

```bash
export HRI_CONFIG="$(rospack find hri_legible_motion)/config/experiment.yaml"
rosrun hri_legible_motion experiment_node.py P07 --pipeline --config "$HRI_CONFIG"
```

Do not run `hri.launch`/`motion_node.py` alongside this mode. It publishes no
legacy `/trial_start` or `/motion_home` triggers. `predictable` maps to the
pipeline's `baseline`; `bend`/`bend_side` are unused. All processes must load the
same YAML, and must restart after edits.

Planning and validation finish before the get-ready phase. The operator must
type `go` before each reach and explicitly authorize a return home. Scoring uses
measured motion onset/completion, never solver completion. Failure or missing
onset stops the session without automatically issuing another motion.

Session output adds `pipeline_attempts.jsonl`, serialized `.proposal` files, and
pipeline topics in the bag. Each attempt has a unique ID even for retries or a
resumed session; the attempt log binds these IDs to existing scheduled trial
numbers and timestamps. Existing CSV column schemas remain unchanged.

Offline tests:

```bash
PYTHONPATH=src python -m unittest discover -s test -v
```

The session tests use fake ROS and downstream providers. Catkin generation,
real MoveIt behavior and hardware operation still require lab validation.
