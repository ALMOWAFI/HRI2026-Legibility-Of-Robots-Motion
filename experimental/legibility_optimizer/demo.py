"""Synthetic mathematics demo. Identity FK here is NOT a robot model.

Run from the repository root:
    python -m experimental.legibility_optimizer.demo --output build/legibility-demo
Add --plot after installing matplotlib to save a standalone diagnostic figure.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .optimizer_core import OptimizerConfig, optimize_joint_path, true_goal_probability_trace


def synthetic_case():
    times = np.linspace(0.0, 4.0, 9)
    goals = np.array([[1.0, 0.55, 0.0], [1.0, -0.55, 0.0], [1.15, 0.0, 0.0]])
    positions = np.linspace(np.zeros(3), goals[0], len(times))
    return positions, times, goals


def identity_fk(q):
    return np.asarray(q[:3], dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/legibility-demo"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    q, t, goals = synthetic_case()
    result = optimize_joint_path(
        q, t, goals, 0, identity_fk, joint_lower=[-2, -2, -1],
        joint_upper=[2, 2, 1], max_joint_deviation=[0.25, 0.25, 0.1],
        config=OptimizerConfig(cost_ratio=1.45, deviation_weight=5e-4,
                               min_legibility_improvement=1e-7, maxiter=300, ftol=1e-9))
    args.output.mkdir(parents=True, exist_ok=True)
    tau, baseline = true_goal_probability_trace(q, t, goals, 0, identity_fk)
    _, candidate = true_goal_probability_trace(result.positions, t, goals, 0, identity_fk)
    np.savez(args.output / "candidate.npz", baseline=q, positions=result.positions,
             times=t, goals=goals)
    with (args.output / "probabilities.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["normalized_time", "baseline_probability", "candidate_probability"])
        writer.writerows(zip(tau, baseline, candidate))
    summary = {
        "fixture": "synthetic identity FK; not a robot or human-subject validation",
        "solver_success": result.solver_success,
        "numerical_accepted": result.numerical_accepted,
        "physical_validity": result.physical_validity,
        "execution_ready": result.execution_ready,
        "baseline_legibility": result.baseline_legibility,
        "optimized_legibility": result.optimized_legibility,
        "realized_cost_ratio": result.realized_cost_ratio,
        "message": result.message,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].plot(q[:, 0], q[:, 1], "o-", label="Synthetic baseline")
        axes[0].plot(result.positions[:, 0], result.positions[:, 1], "o-", label="Candidate")
        axes[0].scatter(goals[:, 0], goals[:, 1], marker="*", s=90, color="black", label="Goals")
        axes[0].set(xlabel="Synthetic x", ylabel="Synthetic y", title="Identity-FK path")
        axes[0].legend(fontsize=8)
        axes[1].plot(tau, baseline, "o-", label="Baseline")
        axes[1].plot(tau, candidate, "o-", label="Candidate")
        axes[1].set(xlabel="Normalized trajectory time", ylabel="P(true goal)",
                    ylim=(0, 1), title="Model observer only")
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(args.output / "comparison.png", dpi=180)
        figure.savefig(args.output / "comparison.pdf")
        plt.close(figure)
    return 0 if result.numerical_accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
