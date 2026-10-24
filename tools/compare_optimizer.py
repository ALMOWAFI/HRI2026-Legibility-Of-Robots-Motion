#!/usr/bin/env python3
"""Compare planar optimization offline on identical baselines and observer settings.

Default input: verified predictable reaches in config/trajectories.npz. The cache
contains Cartesian inputs to MoveIt, not measurements of executed robot motion.
Use --start X Y Z explicitly for synthetic straight baselines. No ROS is imported.
Results are observer-model predictions; they do not establish human recognition
or collision, IK, joint-speed, or hardware validity.
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hri_legible_motion import legibility_optimizer as optimizer  # noqa: E402
from hri_legible_motion import trajectories  # noqa: E402


def nonnegative_float(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return value


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiment.yaml")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--cache", type=Path, help="verified trajectory cache; defaults to config's trajectories path")
    source.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="synthesize straight baselines from this position, in config's frame")
    parser.add_argument("--lambdas", nargs="+", type=nonnegative_float,
                        help="efficiency weights; defaults to the configured value")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "optimizer-comparison")
    parser.add_argument("--no-plots", action="store_true", help="write JSON and CSV without Matplotlib")
    parser.add_argument("--bounds", action="store_true",
                        help="also estimate a conservative score upper bound for the fixed timing/length budget")
    return parser


def goal_positions(cfg):
    """Rows follow config order; block IDs need not be consecutive or ordered."""
    blocks = cfg["blocks"]
    ids = [b["id"] for b in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError("block IDs must be unique")
    z = cfg["motion"]["touch_z"] + cfg["motion"]["hover_height"]
    return np.array([[b["position"][0], b["position"][1], z] for b in blocks], dtype=float)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_baselines(cfg, cache=None, start=None):
    """Return source metadata and (times, xyz) per goal row, without invoking ROS."""
    if cache is not None and start is not None:
        raise ValueError("choose either a cache or a synthetic start")
    goals = goal_positions(cfg)
    if start is not None:
        start = np.asarray(start, dtype=float)
        if start.shape != (3,) or not np.all(np.isfinite(start)):
            raise ValueError("--start requires three finite coordinates")
        if abs(start[2] - goals[0, 2]) > 1e-3:
            raise ValueError("synthetic start must be in the configured reach plane (within 1 mm)")
        source = {"kind": "synthetic_straight_baseline", "start": start.tolist(),
                  "frame_id": cfg["frame_id"], "physical_validity": "not_checked"}
        return source, [optimizer.sample(optimizer.baseline_path(start, g), cfg["motion"]["duration"])
                        for g in goals]

    path = Path(cache) if cache is not None else ROOT / cfg.get("trajectories", "config/trajectories.npz")
    _, _, reaches, meta = trajectories.load(str(path))  # verifies stored content hash
    if "frame_id" in meta and meta["frame_id"] != cfg["frame_id"]:
        raise ValueError("cache frame does not match the selected configuration")
    baselines = []
    for block, goal in zip(cfg["blocks"], goals):
        key = trajectories.key("predictable", block["id"])
        if key not in reaches:
            raise ValueError("cache is missing %s" % key)
        times, xyz = reaches[key]["times"], reaches[key]["xyz"]
        if times.ndim != 1 or len(times) < 3 or xyz.shape != (len(times), 3):
            raise ValueError("%s: invalid timestamp/position dimensions" % key)
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(xyz)) or np.any(np.diff(times) <= 0):
            raise ValueError("%s: positions must be finite and timestamps strictly increasing" % key)
        if np.linalg.norm(xyz[-1] - goal) > 1e-3:
            raise ValueError("%s endpoint does not match the configured goal; use a matching cache/config" % key)
        if np.max(np.abs(xyz[:, 2] - goal[2])) > 1e-3:
            raise ValueError("%s leaves the configured reach plane" % key)
        if not np.isclose(times[-1] - times[0], cfg["motion"]["duration"], atol=1e-6, rtol=0):
            raise ValueError("%s duration does not match the selected configuration" % key)
        if baselines and np.linalg.norm(xyz[0] - baselines[0][1][0]) > 1e-3:
            raise ValueError("cached predictable reaches do not share one start position")
        baselines.append((times.copy(), xyz.copy()))
    source = {"kind": "cached_moveit_cartesian_input", "path": str(path.resolve()),
              "file_sha256": sha256(path), "content_hash": meta["hash"], "metadata": meta,
              "frame_id": cfg["frame_id"],
              "frame_provenance": "cache metadata" if "frame_id" in meta else
                  "assumed from selected configuration; cache does not record its frame",
              "description": "Cartesian inputs stored with precomputed MoveIt joint plans, not measured motion",
              "physical_validity": "not_checked"}
    source["cached_legible"] = {}
    beta = optimizer.PlanarConfig.from_cfg(cfg).beta
    for target, (block, baseline) in enumerate(zip(cfg["blocks"], baselines)):
        key = trajectories.key("legible", block["id"])
        if key not in reaches:
            continue
        times, xyz = reaches[key]["times"], reaches[key]["xyz"]
        if xyz.shape != baseline[1].shape or not np.array_equal(times, baseline[0]) \
                or not np.allclose(xyz[[0, -1]], baseline[1][[0, -1]], atol=1e-9, rtol=0):
            raise ValueError("%s is not comparable: cached endpoints/timestamps differ from its baseline" % key)
        source["cached_legible"][str(block["id"])] = {
            "times": times.tolist(), "xyz": xyz.tolist(),
            "posterior": optimizer.planar_posterior(xyz, goals, beta).tolist(),
            "metrics": optimizer.planar_metrics(times, xyz, goals, target, beta),
            "physical_validity": "not_checked",
        }
    return source, baselines


def compare(cfg, baselines, lambdas):
    """Use each unchanged baseline and all candidate goals for every lambda."""
    goals = goal_positions(cfg)
    config = optimizer.PlanarConfig.from_cfg(cfg)
    if len(baselines) != len(goals):
        raise ValueError("one baseline is required per configured goal")
    records = []
    for target, (block, (times, xyz)) in enumerate(zip(cfg["blocks"], baselines)):
        before = optimizer.planar_metrics(times, xyz, goals, target, config.beta)
        base_probs = optimizer.planar_posterior(xyz, goals, config.beta)
        for lam in lambdas:
            if not np.isfinite(lam) or lam < 0:
                raise ValueError("lambda must be finite and nonnegative")
            settings = replace(config, efficiency_weight=float(lam))
            result = optimizer.optimize_planar_reach(times.copy(), xyz.copy(), goals, target, settings)
            after = optimizer.planar_metrics(result.times, result.xyz, goals, target, config.beta)
            probabilities = optimizer.planar_posterior(result.xyz, goals, config.beta)
            records.append({
                "target": block["name"], "target_id": block["id"], "target_row": target,
                "lambda": float(lam), "beta": config.beta, "optimizer_config": asdict(settings),
                "baseline_score": before["score"], "optimized_score": after["score"],
                "gain_percentage_points": 100 * (after["score"] - before["score"]),
                "baseline_weighted_correct_fraction": before["weighted_correct_fraction"],
                "optimized_weighted_correct_fraction": after["weighted_correct_fraction"],
                "baseline_first_unique_lead_s": before["first_unique_lead_time"],
                "optimized_first_unique_lead_s": after["first_unique_lead_time"],
                "max_lateral_cm": 100 * result.max_lateral, "length_ratio": result.length_ratio,
                "accepted": bool(result.success), "numerical_success": bool(result.numerical_success),
                "geometric_valid": bool(result.geometric_valid),
                "physical_validity": result.physical_validity, "message": result.message,
                "diagnostics": result.diagnostics,
                "baseline_metrics": before, "optimized_metrics": after,
                "baseline": {"times": times.tolist(), "xyz": xyz.tolist(), "posterior": base_probs.tolist()},
                "optimized": {"times": result.times.tolist(), "xyz": result.xyz.tolist(),
                              "posterior": probabilities.tolist()},
            })
    return records


def json_value(value):
    """Normalize NumPy diagnostics; undefined metrics are represented by JSON null."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def timing_score_bound(times, xyz, goals, target, config, spacing=0.0005):
    """Upper bound for the sampled score under min-jerk arc timing and a length cap.

    At fraction s of arc length, a path lies within both endpoint disks with
    radii s*Lmax and (1-s)*Lmax. Search that relaxed lens, ignoring workspace,
    radial, IK, and path continuity restrictions. Expand the disks by the grid
    rounding distance, then add a beta/2 Lipschitz error allowance. Thus this is
    a conservative numerical bound, not a proposed attainable trajectory.
    It does not apply to arbitrary timing laws. Floating-point arithmetic is used.
    """
    metrics = optimizer.planar_metrics(times, xyz, goals, target, config.beta)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("bound grid spacing must be finite and positive")
    times, xyz, goals = np.asarray(times), np.asarray(xyz), np.asarray(goals)
    start, end, goals_xy = xyz[0, :2], xyz[-1, :2], goals[:, :2]
    distance = np.linalg.norm(end - start)
    if distance <= 0:
        raise ValueError("bound needs distinct reach endpoints")
    direction = (end - start) / distance
    normal = np.array([-direction[1], direction[0]])
    # Optimizer shapes only XY; its linear Z interpolation is monotone. Including
    # the small endpoint Z difference also bounds the full 3D arc used for timing.
    max_length = config.length_ratio * distance + abs(xyz[-1, 2] - xyz[0, 2])
    rounding = spacing / np.sqrt(2) + 1e-12
    direct = np.linalg.norm(goals_xy - start, axis=1)
    tau = (times - times[0]) / (times[-1] - times[0])
    probabilities = optimizer.planar_posterior(xyz, goals, config.beta)
    bound = []
    for index, fraction in enumerate(optimizer.min_jerk(tau)):
        if index in (0, len(times) - 1):
            bound.append(float(probabilities[index, target]))
            continue
        a, b = fraction * max_length + rounding, (1 - fraction) * max_length + rounding
        low, high = max(-a, distance - b), min(a, distance + b)
        xs = np.arange(np.floor(low / spacing), np.ceil(high / spacing) + 1) * spacing
        ymax = np.sqrt(np.maximum(0, np.minimum(a*a - xs*xs, b*b - (xs-distance)**2))).max()
        ny = int(np.ceil(ymax / spacing))
        if len(xs) * (2 * ny + 1) > 2_000_000:
            raise ValueError("bound grid would exceed two million cells; use a smaller geometry or larger spacing")
        ys = np.arange(-ny, ny + 1) * spacing
        x, y = np.meshgrid(xs, ys, indexing="ij")
        keep = (x*x + y*y <= a*a) & ((x-distance)**2 + y*y <= b*b)
        points = start + x[keep, None] * direction + y[keep, None] * normal
        if not len(points):
            raise ValueError("bound grid unexpectedly contains no feasible points")
        logits = config.beta * (direct - np.linalg.norm(points[:, None, :] - goals_xy, axis=2))
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        upper = np.max(exp[:, target] / exp.sum(axis=1)) + config.beta * 0.5 * rounding + 1e-12
        bound.append(min(metrics["posterior_ceiling"] + 1e-12, float(upper)))
    dt = np.diff(times)
    trapezoid = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    weights = trapezoid * (1 - tau)
    weights /= weights.sum()
    return {"score_upper_bound": float(weights @ bound), "posterior_upper_bound": bound,
            "grid_spacing_m": spacing, "probability_error_allowance": config.beta * 0.5 * rounding,
            "assumptions": "Fixed minimum-jerk arc timing, configured length budget, and the same observer; "
                           "workspace/radial/IK restrictions relaxed. Numerical grid bound, not an attainable path."}


SUMMARY_FIELDS = ["target", "target_id", "lambda", "beta", "baseline_score", "optimized_score",
                  "gain_percentage_points", "max_lateral_cm", "length_ratio", "numerical_success",
                  "accepted", "geometric_valid", "physical_validity", "cached_legible_score",
                  "gain_over_cached_percentage_points", "baseline_weighted_correct_fraction",
                  "optimized_weighted_correct_fraction", "cached_weighted_correct_fraction",
                  "baseline_first_unique_lead_s", "optimized_first_unique_lead_s",
                  "cached_first_unique_lead_s", "score_upper_bound", "active_constraints", "message"]


def goal_margin(probabilities, target):
    probabilities = np.asarray(probabilities)
    return probabilities[:, target] - np.max(np.delete(probabilities, target, axis=1), axis=1)


def write_results(output, report):
    output.mkdir(parents=True, exist_ok=True)
    report = json_value(report)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in report["results"]:
            row = {key: record.get(key) for key in SUMMARY_FIELDS}
            row["active_constraints"] = json.dumps(record["diagnostics"].get("active_constraints", []))
            writer.writerow(row)
    with (output / "posterior_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target", "lambda", "variant", "time_s", "x_m", "y_m", "z_m", "p_target", "goal_margin"])
        for record in report["results"]:
            for variant in ("baseline", "optimized"):
                curve = record[variant]
                for t, xyz, posterior, margin in zip(curve["times"], curve["xyz"], curve["posterior"],
                                                     goal_margin(curve["posterior"], record["target_row"])):
                    writer.writerow([record["target"], record["lambda"], variant, t, *xyz,
                                     posterior[record["target_row"]], margin])


def plot_results(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    goals = goal_positions(report["config"])
    for target in sorted({r["target_row"] for r in report["results"]}):
        variants = [r for r in report["results"] if r["target_row"] == target]
        record = variants[0]
        fig, (xy_ax, p_ax, margin_ax) = plt.subplots(1, 3, figsize=(17, 5))
        baseline = record["baseline"]
        xyz = np.asarray(baseline["xyz"])
        t = np.asarray(baseline["times"])
        xy_ax.plot(xyz[:, 0], xyz[:, 1], "k--", label="Baseline")
        xy_ax.scatter(goals[:, 0], goals[:, 1], marker="s", color="gray")
        for block, goal in zip(report["config"]["blocks"], goals):
            xy_ax.annotate(block["name"], goal[:2], xytext=(5, 5), textcoords="offset points")
        xy_ax.scatter(*xyz[0, :2], marker="o", color="black", s=20)
        p_ax.plot(t - t[0], np.asarray(baseline["posterior"])[:, target], "k--", label="Baseline")
        margin_ax.plot(t - t[0], goal_margin(baseline["posterior"], target), "k--", label="Baseline")
        cached = report["source"].get("cached_legible", {}).get(str(record["target_id"]))
        if cached is not None:
            cached_xyz = np.asarray(cached["xyz"])
            xy_ax.plot(cached_xyz[:, 0], cached_xyz[:, 1], color="orange", linestyle=":", label="Previous cached legible")
            p_ax.plot(np.asarray(cached["times"]) - cached["times"][0],
                      np.asarray(cached["posterior"])[:, target], color="orange", linestyle=":", label="Previous cached legible")
            margin_ax.plot(np.asarray(cached["times"]) - cached["times"][0], goal_margin(cached["posterior"], target),
                           color="orange", linestyle=":", label="Previous cached legible")
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(variants)))
        for variant, color in zip(variants, colors):
            curve = variant["optimized"]
            xyz = np.asarray(curve["xyz"])
            times = np.asarray(curve["times"])
            label = "lambda=%g (%+.2f pp)%s" % (variant["lambda"], variant["gain_percentage_points"],
                                                 "" if variant["accepted"] else " REJECTED")
            xy_ax.plot(xyz[:, 0], xyz[:, 1], color=color, label=label)
            p_ax.plot(times - times[0], np.asarray(curve["posterior"])[:, target], color=color, label=label)
            margin_ax.plot(times - times[0], goal_margin(curve["posterior"], target), color=color, label=label)
        ceiling = record["baseline_metrics"].get("posterior_ceiling")
        if ceiling is not None:
            p_ax.axhline(ceiling, color="gray", linestyle=":", label="Observer upper bound")
        timing_bound = report.get("timing_bounds", {}).get(str(record["target_id"]))
        if timing_bound is not None:
            p_ax.plot(t - t[0], timing_bound["posterior_upper_bound"], color="gray", linestyle="-.",
                      label="Timing/length upper bound")
        xy_ax.set(xlabel="X (m), " + report["config"]["frame_id"], ylabel="Y (m)", title="Cartesian path")
        xy_ax.set_aspect("equal", adjustable="datalim")
        p_ax.set(xlabel="Elapsed time (s)", ylabel="Model P(target)", ylim=(0, 1), title="Target probability over time")
        p_ax.yaxis.set_major_formatter(PercentFormatter(1))
        margin_ax.axhline(0, color="gray", linewidth=0.7)
        margin_ax.set(xlabel="Elapsed time (s)", ylabel="P(target) - largest P(other goal)",
                      title="Negative = model favors a different goal")
        margin_ax.yaxis.set_major_formatter(PercentFormatter(1))
        for ax in (xy_ax, p_ax, margin_ax):
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("%s: identical baseline, beta=%g, length budget=%g" %
                     (record["target"], record["beta"], record["optimizer_config"]["length_ratio"]))
        fig.text(0.5, 0.01, "%s | Observer predictions; robot validity not checked" % report["source"]["kind"],
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        for extension in ("png", "svg"):
            fig.savefig(output / ("target_%s.%s" % (record["target_id"], extension)), dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        source, baselines = load_baselines(cfg, args.cache, args.start)
        settings = optimizer.PlanarConfig.from_cfg(cfg)
        lambdas = args.lambdas if args.lambdas is not None else [settings.efficiency_weight]
        records = compare(cfg, baselines, lambdas)
        bounds = {}
        if args.bounds:
            goals = goal_positions(cfg)
            for target, (block, (times, xyz)) in enumerate(zip(cfg["blocks"], baselines)):
                bounds[str(block["id"])] = timing_score_bound(times, xyz, goals, target, settings)
        for record in records:
            cached = source.get("cached_legible", {}).get(str(record["target_id"]))
            record["cached_legible_score"] = None if cached is None else cached["metrics"]["score"]
            record["gain_over_cached_percentage_points"] = None if cached is None else \
                100 * (record["optimized_score"] - cached["metrics"]["score"])
            record["cached_weighted_correct_fraction"] = None if cached is None else cached["metrics"]["weighted_correct_fraction"]
            record["cached_first_unique_lead_s"] = None if cached is None else cached["metrics"]["first_unique_lead_time"]
            bound = bounds.get(str(record["target_id"]))
            record["score_upper_bound"] = None if bound is None else bound["score_upper_bound"]
        report = {"schema_version": 1, "config_path": str(args.config.resolve()),
                  "config_sha256": sha256(args.config), "config": cfg, "source": source,
                  "interpretation": "Scores and probability gains are observer-model predictions, not human success rates. "
                                    "Cartesian candidates require MoveIt planning and robot-specific validation.",
                  "timing_bounds": bounds, "results": records}
        write_results(args.output, report)
        for record in records:
            print("%s lambda=%g: %.4f -> %.4f (%+.2f pp), lateral=%.2f cm, length=x%.4f, accepted=%s" %
                  (record["target"], record["lambda"], record["baseline_score"], record["optimized_score"],
                   record["gain_percentage_points"], record["max_lateral_cm"], record["length_ratio"], record["accepted"]))
        if not args.no_plots:
            plot_results(args.output, report)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, "Comparison failed: %s\n" % error)
    print("Results: %s" % args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
