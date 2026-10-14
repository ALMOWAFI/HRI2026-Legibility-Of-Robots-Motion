# Source and adaptation record

The user supplied **Legibility Optimizer for the UD5 Robot: Research Basis,
Mathematical Formulation, Code, and Validation Plan** as a PDF on 2026-10-14.
SHA-256 of the supplied PDF:

```text
57cd2236932a16da712573b483d23237c06cd855bbd52fc310fd1fe70f028845
```

The original PDF is not vendored. Its manuscript source block (printed pages
58–73) was extracted, stripped of PDF page numbers/form-feed artifacts, and
repaired. Equations, cited papers, reference repositories and their listed paths,
experimental parameter ranges, validation discussion and proposed schedule are
retained. The schedule is explicitly a proposal, not a claim of completed lab work.

| Supplied material | Integrated destination and changes |
|---|---|
| `legibility_optimizer_ud5.tex` | [`legibility_optimizer_ud5.tex`](legibility_optimizer_ud5.tex): repaired paragraph breaks inside equations, wrapped architecture equations, added readable code listings and correct repository-relative paths. Added actual checkout findings and clearly described numerical versus physical status. Removed unused LaTeX packages, including `caption`, which was missing locally and is unnecessary for `listings` captions. |
| `optimizer_core.py`, printed pp. 12–34 | [`optimizer_core.py`](../../experimental/legibility_optimizer/optimizer_core.py): preserves FK-based progress posterior, early waypoint-weighted score, joint velocity cost, interior-only SLSQP, soft deviation penalty and optional nominal dynamic constraints. Adds mandatory position bounds and hard local deviation bounds; validates finite inputs/configuration; rejects malformed solver output without accepting a fallback. |
| ROS 1 node sketch, pp. 36–42 | [`ros1_adapter.py`](../../experimental/legibility_optimizer/ros1_adapter.py): preserves message extraction and joint/FK binding as a read-only adapter. Removes guessed topics, asynchronous target pairing, dynamic module loading and publishing. The repository has no baseline topic to connect to. No ROS runtime or controller imported. |
| FK provider sketch, p. 43 | [`robot_interfaces.py`](../../experimental/legibility_optimizer/robot_interfaces.py): explicit failing FK, retiming, validation and execution contracts. No robot model inferred from the report's name. |
| Tests, pp. 48–52 | [`tests/experimental`](../../tests/experimental): adapts the synthetic fixture to standard-library `unittest`, strengthens endpoint checks to exact equality, and adds independent solver rejection, input, dynamic-constraint, message-ordering and execution-isolation coverage. |
| Plotting script, pp. 54–55 | [`demo.py`](../../experimental/legibility_optimizer/demo.py): runnable synthetic demo, probability CSV, candidate NPZ, JSON status and optional standalone PNG/PDF plots. Labels the baseline as synthetic. |

The report's combined `candidate_ok`/optional path callback has been split:
numerical acceptance is computed in the core, while physical validation is a
separate, unimplemented stage after retiming. Unprovided dynamic constraints are
unassessed (`None`), not reported as passing. A soft deviation penalty is retained
but supplemented by hard per-joint local bounds because the requested integration
requires an enforceable local limit.

The original normalized-time weighted average is retained, not silently changed
to a time integral. The model uses smoothed Euclidean distances and ignores
orientation/viewpoint. Constraints on nominal finite differences and sampled
positions are not collision or controller feasibility certificates.

## Citations and external reuse

The manuscript retains these references; their bibliographic metadata was checked
against the following primary sources:

- Dragan and Srinivasa, *Generating Legible Motion*, RSS 2013,
  [doi:10.15607/RSS.2013.IX.024](https://www.roboticsproceedings.org/rss09/p24.html).
- Bodden, Rakita, Mutlu and Gleicher, *A Flexible Optimization-Based Method for
  Synthesizing Intent-Expressive Robot Arm Motion*, IJRR 37(11), 2018,
  [doi:10.1177/0278364918792295](https://graphics.cs.wisc.edu/Papers/2018/BRMG18/).
- Amirian, Abrini and Chetouani, *Legibot*, 2024,
  [arXiv:2404.05100](https://arxiv.org/abs/2404.05100).

The supplied manuscript's references to `uwgraphics/trajectoryoptimizer-public`,
`khoilsr/hrc_legible_motion_generation`, `shiwenli0908/Legibility-Modulator`,
`isir-socialrobotics/legibot` and `munawarkazmi/legibility-bounds` are retained as
research references. No source code, robot kinematics or assets from those
repositories were copied, vendored or made dependencies. Their exact current
licenses are therefore not asserted by this integration. Before future code
reuse, inspect the exact revision's license and copyright notices for compatibility.

The host repository declares MIT in `package.xml` but contains no standalone
license file at the inspected revision. This change does not invent a new license
or relicense upstream work. The prototype adapts the user-supplied report and adds
original integration/validation code. NumPy and SciPy are ordinary declared
dependencies, not bundled source. The upstream Noetic message/API definitions and
[SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
were consulted to verify interfaces; no substantial code was copied from them.
