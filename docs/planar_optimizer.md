# Planar optimizer: what is optimized and what the results mean

The active lab path is `test_motion.py` -> `/hri/predictable_reach` ->
`legibility_optimizer_node.py` -> `/hri/legible_reach` -> `Robot.plan()`.
The baseline is a Cartesian straight reach. MoveIt solves Cartesian IK after
optimization. This is not the earlier RRT/FK joint-trajectory pipeline.
The optimizer never connects to the trajectory controller.

## Objective and constraints

For start S, candidate goals G, and observed tool position Q in the XY plane:

```
P(g | Q) = softmax_g(beta * (distance(S, g) - distance(Q, g)))
score = integral(P(target | Q(t)) * (T-t) dt) / integral((T-t) dt)
minimize -score + efficiency_weight * (path_length / straight_length - 1)
                + smooth_weight * sum(k^2 * coefficient_k^2)
```

The observed prefix length cancels during goal normalization. Scores use
trapezoidal integration at the supplied timestamps, including nonuniform samples.
This corrects the previous equal-sample weighting. Re-evaluate both paths with
the same scoring version, beta, goal set, and timing when comparing old results.

SLSQP searches six sine coefficients per axis, starting from five deterministic
seeds. Sines are a path representation; their coefficients are decision variables.
Endpoints and timestamps are preserved exactly. The measured start and endpoint
heights must agree within 1 mm; the existing cache differs by about 7 micrometres.
Both retain their exact measured values. Path length, radial envelope, workspace,
coefficient bounds, and configured sampled TCP speed constrain the search.
The radial heuristic includes the endpoint radii; effective bounds are reported.
These heuristics do not establish IK reachability, joint limits, or collision freedom.

`efficiency_weight` is a soft penalty: increasing it encourages shorter paths.
`length_ratio` is a separate hard budget (currently 1.04). `beta` controls the
observer's sensitivity and must be held fixed for a before/after comparison.

## Avoiding a misleading goal preference

Increasing P(target) can also increase P(neighbour) even more. This happened in
the previous cached middle-block reaches. With `protect_goal_ranking: true`,
each sampled pairwise target-versus-competitor probability margin must be at least
the smaller of its baseline margin and zero. Thus optimization cannot introduce
a new competing-goal preference; an already negative margin cannot become worse.
Independent acceptance allows at most 1e-8 probability-margin numerical error.
Ties can remain and are not confident recognition. This task-specific constraint
can reduce the optimized average score; the report exposes that tradeoff.

The solver's convergence flag alone never authorizes publication. Returned
coefficients are independently checked, including interpolated geometry, bounds,
sampled speed and goal ranking. A candidate must improve both the regularized
objective and the model score (default minimum score gain 0.0001). Failure returns
an unmodified baseline with `success=False`; the ROS node publishes an empty
failure response. Numerical success, geometric acceptance and physical validity
are distinct fields. Physical validity remains `not_checked` here.

## Reproduce the comparison without ROS

From the package root:

```bash
python -m pip install -r requirements-offline.txt
python -m unittest discover -s test -v
python tools/compare_optimizer.py --lambdas 0 0.2 2 --bounds
```

The default input is the hash-verified predictable Cartesian paths in
`config/trajectories.npz`, with the old cached legible paths shown for comparison.
These are saved planning inputs, not measured robot motion. `--start X Y Z`
explicitly selects a synthetic baseline. Outputs go to ignored
`build/optimizer-comparison/`: JSON, summary CSV, posterior CSV, and PNG/SVG figures.

Read the target-versus-strongest-competitor margin alongside the average score.
A negative margin means the model predicts the wrong goal; a large visible bend
does not establish greater recognition. The report also includes quarter-time
probabilities and sustained model confidence, with `null` if never achieved.
No score or model winner percentage is a measured human success rate.

The optional bound relaxes workspace, radial and robot constraints. At each time,
the fixed arc-length timing and total length budget restrict the point to two
intersecting disks. A grid enlarged by its cell covering radius, plus a posterior
Lipschitz allowance beta/2 times that radius, bounds the pointwise probability.
The same time weights produce a conservative score upper bound. It is not an
attainable trajectory or a guarantee of global solver optimality.

## Lab preview and study replay

Start MoveIt using the lab's existing instructions, then:

```bash
roslaunch hri_legible_motion optimizer.launch
python3 tools/test_motion.py B2 --legible
```

Both commands default to `config/experiment.yaml`. Pass the same explicit path
with launch argument `config:=/path/experiment.yaml` and tool option
`--config /path/experiment.yaml` when needed. Restart the optimizer after editing
the YAML; it logs the absolute path and loaded settings. `test_motion.py` without
`--legible` previews the baseline. `all` previews only the final reach and return.

The live optimizer publishes Cartesian proposals; subsequent MoveIt IK, current
planning scene, robot-specific trajectory validation and explicit operator
execution remain required. Cartesian timing is not joint-space retiming.
The existing robot helper checks sampled joint speeds and jumps; acceleration,
controller interpolation and collision coverage between samples still need lab
verification. No hardware-readiness claim follows from offline tests.

`hri.launch` runs the separate cached study/replay path. It does not subscribe to
the live optimizer. The committed cache was deliberately not overwritten by
offline comparisons. To use new optimized reaches in the study, generate a NEW
cache with MoveIt, inspect it, then select that file and restart the replay node:

```bash
python3 tools/precompute_trajectories.py --out build/trajectories-reviewed.npz
```

This plans and previews; it does not execute. Keep the previous study cache until
the new trajectories have passed the lab's validation.

## Relation to the paper

[Dragan and Srinivasa, Generating Legible Motion (RSS 2013)](https://www.roboticsproceedings.org/rss09/p24.pdf)
provides the goal-inference and time-weighted legibility formulation. Its examples
use squared-velocity cost and functional gradient updates with a trust region.
Here the cost model is Cartesian distance and the search uses a finite sine basis
with [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).
Our fixed efficiency penalty is not the paper's derived KKT multiplier lambda;
our observer beta is not its trust-region parameter beta. The goal-ranking rule
is an additional experimental constraint. This is an adaptation, not a numerical
reproduction of the paper. No third-party implementation was copied.
