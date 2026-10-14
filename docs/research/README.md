# FK-based legibility optimizer research

This is an offline experimental integration of the supplied UD5 report. It does
not change the existing ROS nodes, launch files, package dependencies or recorded
pose controller tool. The inspected checkout documents ROS 1 Noetic and a UR5e;
the report's UD5 name is not treated as a verified robot model.

- [Manuscript source](legibility_optimizer_ud5.tex)
- [Repository findings, future connection point and required lab validation](integration.md)
- [Source, adaptations, citations and reuse/license record](provenance.md)
- [Standalone optimizer](../../experimental/legibility_optimizer/optimizer_core.py)
- [Tests](../../tests/experimental)

## Environment and commands

Run from the repository root in an isolated Python environment. The prototype
uses Python 3.8-compatible syntax and NumPy/SciPy APIs; ROS Noetic compatibility
is a source-level target, not a claim of a tested ROS installation.

```sh
python -m venv .venv-legibility
# Linux/macOS: source .venv-legibility/bin/activate
# PowerShell: .venv-legibility\Scripts\Activate.ps1
python -m pip install -r experimental/requirements-legibility.txt
python -m unittest discover -s tests/experimental -v
python -m experimental.legibility_optimizer.demo --output build/legibility-demo
```

No `setup.py` invocation, catkin build, ROS master, robot driver or hardware is
needed. Tests use `unittest`. Optional plotting requires Matplotlib:

```sh
python -m pip install "matplotlib>=3.3"
python -m experimental.legibility_optimizer.demo --output build/legibility-demo --plot
```

The demo writes `summary.json`, `probabilities.csv` and `candidate.npz`; `--plot`
adds `comparison.png` and `comparison.pdf`. The identity FK and bounds in this
fixture are synthetic and must not be used as robot parameters.

## Build the manuscript

Install a LaTeX distribution providing `pdflatex`, `geometry`, `amsmath`,
`amssymb`, `hyperref` and `listings`, then run:

```sh
python docs/research/build_manuscript.py
```

The script checks every `\lstinputlisting` path, runs `pdflatex` twice from
`docs/research` with shell escape disabled, and checks the final log for unresolved
references and layout warnings. Output and both build transcripts go to
`build/research/`. No external bibliography processor, Mermaid renderer, generated
image or file outside the checkout is needed. The code listings include the
actual prototype, adapter, explicit robot stubs, tests and demo.

Equivalent manual commands, from `docs/research` after creating `../../build/research`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -output-directory=../../build/research legibility_optimizer_ud5.tex
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape -output-directory=../../build/research legibility_optimizer_ud5.tex
```

## Numerical contract

`optimize_joint_path` accepts `(N,D)` baseline joint positions, `N` nonnegative
strictly increasing timestamps, `(M,3)` Cartesian candidate goals (`M >= 2`), an
intended goal index, and an injected `FK(q) -> xyz`. FK and goals must share a
verified frame. Joint lower/upper limits and positive per-joint local deviations
are mandatory. The first and last configurations and timestamps are copied
exactly. Only interior positions are optimized.

The motion budget is `sum(||dq||^2 / dt) <= cost_ratio * baseline_cost`; hard
local bounds intersect each waypoint's baseline neighborhood with the joint
limits. Optional velocity/acceleration limits apply to nominal finite differences.
All numerical bounds use the supplied joint units. Continuous joints need an
explicitly chosen unwrapped interval. No robot-specific constants are supplied.

`solver_success` is the solver's termination status. `numerical_accepted` (alias
`candidate_ok`) additionally checks output finiteness/shape, constraints and minimum
model-score improvement. The default constraint tolerance is `1e-7`; the cost
allowance is `1e-7 * max(1, maximum_cost)`. These are numerical tolerances, not
physical clearance margins. Configured limits must include any required physical
margin. Callers must reject `numerical_accepted=False`, even though diagnostic
arrays are returned. Invalid solver output uses a rejected baseline fallback.

Every result has `physical_validity=None`, `retime_required=True` and
`execution_ready=False`. Missing FK, retiming, collision validation and execution
implementations raise explicitly in `robot_interfaces.py`. Actual FK is injected
by the caller; the prototype does not select a robot or contact a controller.

The score is the report's early weighted average across interior waypoints. It
supports nonuniform timestamps but is sensitive to waypoint density. Keep a common
sampling convention when comparing trials. Retiming may change the score and
motion cost; evaluate those separately and validate the complete retimed path.

## Recorded verification

The integration run on 2026-10-14 used Windows, CPython 3.11.9, NumPy 2.4.4 and
SciPy 1.15.3. The complete test run passed **34 tests** in 15.167 seconds. It includes
actual SLSQP solves, independent constraint calculations, injected solver failures,
joint-order/frame checks, explicit missing-integration failures and an offline
demo with ROS imports and socket creation blocked.

All nine prototype/test Python files also passed Python 3.8 grammar parsing;
this does not substitute for a runtime test under Python 3.8 or ROS Noetic.

The synthetic demo was accepted numerically: score **0.4982769760 → 0.6877432078**,
realized cost ratio **1.450000000263** against a configured 1.45 limit (within the
documented numerical tolerance). `physical_validity` was `null` and
`execution_ready` was `false`. PNG/PDF diagnostic plots were generated and viewed.

The manuscript compiled with **MiKTeX 26.2 / pdfTeX 1.40.29**, in two passes,
producing **25 pages**. All seven listing companions resolved. The final log had
no unresolved references, overfull/underfull boxes or missing-character warnings.
The title, middle-section posterior/objective/constraint equations, code listings
and bibliography were visually inspected. Build artifacts are intentionally
ignored by Git and can be reproduced with the commands above.

No ROS, MoveIt, simulator, robot, continuous collision, retiming or human-subject
test has been performed by this offline integration.
