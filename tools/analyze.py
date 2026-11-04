#!/usr/bin/env python3
"""Analysis of the game study: one table of all participants, exclusions, descriptives,
participant-level paired tests, mixed models, questionnaires, manipulation check, figures.
No ROS, no robot.

usage: analyze.py [--data ~/hri_data] [--out build/analysis] [--questionnaires FILE]
                  [--exclude P03,P05] [--min-trials 6] [--include-non-p]

Reads <data>/<participant>/trials.csv, choices.csv and session.yaml (practice files are
only used to mark the game trials whose reach was already seen in practice). Writes to
<out>/: trials.csv (the analysed trials), exclusions.csv, descriptives.csv,
participant_means.csv, tests.csv, models.txt, manipulation.csv, questionnaire_tests.csv,
f_by_cube.png, recognition_curve.png.

Model (trial level, one trial per motion x cube per participant):
    f ~ legible * edge + second + (1 + legible | participant)
legible = motion legible, edge = B1/B4, second = block B (the second half). The order
check adds legible:second. Needs statsmodels (Debian: sudo apt install python3-statsmodels); without it the
models are skipped and everything else is still written. The same model in R:
    lmer(f ~ legible * edge + second + (1 + legible | participant_id), data = trials)

--questionnaires: CSV with participant_id, motion (predictable | legible) and one numeric
column per scale or item (typed in from the paper forms, one row per participant and block).
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from hri_legible_motion import trajectories  # noqa: E402

OUTCOMES = ["f", "t_first_correct", "t_commit", "final_correct", "n_switches", "points"]


# --- loading -------------------------------------------------------------------------

def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) else pd.DataFrame()


def load_participant(folder):
    """Game trials of one participant folder (DataFrame), its session info and its choices."""
    with open(os.path.join(folder, "session.yaml")) as f:
        session = yaml.safe_load(f) or {}
    trials = read_csv(os.path.join(folder, "trials.csv"))
    choices = read_csv(os.path.join(folder, "choices.csv"))
    practice = read_csv(os.path.join(folder, "practice_trials.csv"))
    if trials.empty:
        return trials, session, choices
    # Early sessions logged the practice rounds in trials.csv; keep them apart here too.
    practice = pd.concat([practice, trials[trials["phase"] == "training"]], ignore_index=True)
    trials = trials[trials["phase"] == "game"].copy()
    if trials.empty:
        return trials, session, choices
    trials["first_motion"] = session.get("first_motion") or trials.sort_values("t_trial_start")["motion"].iloc[0]
    trials["duration_T"] = float(session.get("duration_T", 5.0))
    trials["trajectory_hash"] = session.get("trajectory_hash")
    seen = set() if practice.empty else set(zip(practice["motion"], practice["target_id"]))
    trials["seen_in_practice"] = [int((m, t) in seen) for m, t in zip(trials["motion"], trials["target_id"])]
    return trials, session, choices


def keep_last_attempt(trials):
    """A failed round can be repeated: keep the last attempt of every game trial."""
    return (trials.sort_values("t_trial_start")
            .drop_duplicates(["participant_id", "trial_id"], keep="last")
            .sort_values(["participant_id", "trial_id"]))


def load(data_dir, include_non_p=False):
    """(trials of all participants, choices of all participants, notes on skipped folders)."""
    frames, choice_frames, notes = [], [], []
    for folder in sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "*", ""))):
        pid = os.path.basename(os.path.dirname(folder))
        if not include_non_p and not re.fullmatch(r"P\d+", pid):
            notes.append((pid, "not a P## id (dry run)"))
            continue
        if not os.path.exists(os.path.join(folder, "session.yaml")):
            notes.append((pid, "no session.yaml"))
            continue
        trials, _, choices = load_participant(folder)
        if trials.empty:
            notes.append((pid, "no game trials"))
            continue
        frames.append(trials)
        if not choices.empty:
            choice_frames.append(choices)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), notes
    trials = keep_last_attempt(pd.concat(frames, ignore_index=True))
    choices = pd.concat(choice_frames, ignore_index=True) if choice_frames else pd.DataFrame()
    return trials, choices, notes


def prepare(trials, edge_ids=(0, 3)):
    """Analysis columns: legible, edge, second (block B), never_correct, t_first_correct
    with 'never correct' set to 1.0 (the end of the reach)."""
    t = trials.copy()
    t["participant_id"] = t["participant_id"].astype(str)
    t["legible"] = (t["motion"] == "legible").astype(int)
    t["edge"] = t["target_id"].isin(edge_ids).astype(int)
    t["cube_type"] = np.where(t["edge"] == 1, "edge", "middle")
    t["second"] = (t["block"] == "B").astype(int)
    t["position"] = np.where(t["second"] == 1, "second", "first")
    t["never_correct"] = t["t_first_correct"].isna().astype(int)
    t["t_first_correct"] = t["t_first_correct"].fillna(1.0)
    return t


def exclude(trials, exclude_ids=(), min_trials=6):
    """(kept trials, exclusions table). Rules: trials that did not finish ('done'); then
    participants listed in exclude_ids or with fewer than min_trials finished game trials."""
    rows = [{"participant_id": p, "trial_id": i, "reason": "status %s" % s}
            for p, i, s in zip(trials["participant_id"], trials["trial_id"], trials["status"]) if s != "done"]
    kept = trials[trials["status"] == "done"]
    counts = kept.groupby("participant_id").size()
    for pid in sorted(set(trials["participant_id"])):
        n = int(counts.get(pid, 0))
        if pid in exclude_ids:
            rows.append({"participant_id": pid, "trial_id": "all", "reason": "excluded by the experimenter"})
        elif n < min_trials:
            rows.append({"participant_id": pid, "trial_id": "all",
                         "reason": "only %d finished game trials (< %d)" % (n, min_trials)})
    dropped = {r["participant_id"] for r in rows if r["trial_id"] == "all"}
    kept = kept[~kept["participant_id"].isin(dropped)]
    return kept, pd.DataFrame(rows, columns=["participant_id", "trial_id", "reason"])


# --- statistics ----------------------------------------------------------------------

def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def descriptives(trials):
    rows = []
    for (motion, cube), g in trials.groupby(["motion", "target_name"]):
        row = {"motion": motion, "cube": cube, "n": len(g)}
        for o in OUTCOMES:
            row[o + "_mean"], row[o + "_ci_low"], row[o + "_ci_high"] = mean_ci(g[o])
        row["never_correct_share"] = g["never_correct"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def participant_means(trials):
    """Mean per participant x motion (and per cube type), the unit of the paired tests."""
    by_motion = trials.groupby(["participant_id", "first_motion", "motion"])[OUTCOMES].mean().reset_index()
    by_motion["cube_type"] = "all"
    by_type = trials.groupby(["participant_id", "first_motion", "motion", "cube_type"])[OUTCOMES].mean().reset_index()
    return pd.concat([by_motion, by_type], ignore_index=True)


def paired_test(diff, label, rng):
    """Legible - predictable differences of each participant: mean, bootstrap 95 % CI,
    d_z, paired t and Wilcoxon signed-rank."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    row = {"comparison": label, "n": len(d), "mean_diff": d.mean() if len(d) else np.nan}
    if len(d) >= 2:
        boot = rng.choice(d, (5000, len(d))).mean(axis=1)
        row["ci_low"], row["ci_high"] = np.percentile(boot, [2.5, 97.5])
        sd = d.std(ddof=1)
        row["d_z"] = d.mean() / sd if sd > 0 else np.nan
        row["t"], row["p_t"] = stats.ttest_1samp(d, 0.0) if sd > 0 else (np.nan, np.nan)
        if np.any(d != 0):
            with warnings.catch_warnings():  # small-n / ties notes; n and the CI are reported
                warnings.simplefilter("ignore")
                row["W"], row["p_wilcoxon"] = stats.wilcoxon(d)
    return row


def paired_tests(means, seed=0):
    """Per outcome: legible vs predictable over all cubes, on edge cubes, on middle cubes,
    and the edge - middle difference of the effect (H3)."""
    rng = np.random.default_rng(seed)
    rows = []
    for o in OUTCOMES:
        wide = means.pivot_table(index=["participant_id", "cube_type"], columns="motion", values=o)
        if not {"legible", "predictable"} <= set(wide.columns):
            continue
        effect = (wide["legible"] - wide["predictable"]).unstack("cube_type")
        for cube_type in ("all", "edge", "middle"):
            if cube_type in effect:
                rows.append(dict(paired_test(effect[cube_type], "legible - predictable (%s cubes)" % cube_type, rng),
                                 outcome=o))
        if {"edge", "middle"} <= set(effect.columns):
            rows.append(dict(paired_test(effect["edge"] - effect["middle"], "effect on edge - effect on middle", rng),
                             outcome=o))
    cols = ["outcome", "comparison", "n", "mean_diff", "ci_low", "ci_high", "d_z", "t", "p_t", "W", "p_wilcoxon"]
    return pd.DataFrame(rows).reindex(columns=cols)


def mixed_models(trials):
    """Text with the trial-level models, or why they were skipped."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return "Mixed models skipped: statsmodels is not installed (Debian: sudo apt install python3-statsmodels).\n"
    if trials["participant_id"].nunique() < 3:
        return "Mixed models skipped: fewer than 3 participants.\n"
    out = []

    def lmm(formula, data):
        for re_formula, label in (("~legible", "random intercept + legible slope"), (None, "random intercept")):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    fit = smf.mixedlm(formula, data, groups=data["participant_id"], re_formula=re_formula).fit(
                        reml=True, method="lbfgs")
                    return fit, label
                except Exception:  # convergence or singular fit: simpler random effects
                    continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.mixedlm(formula, data, groups=data["participant_id"]).fit(reml=True)
        return fit, "random intercept; warnings: " + "; ".join(sorted({str(w.message) for w in caught}))

    def gee(formula, data):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.gee(formula, "participant_id", data, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
        note = "; ".join(sorted({str(w.message) for w in caught}))
        return fit, "logistic GEE (exchangeable within participant)" + ("; warnings: " + note if note else "")

    models = [(o, t, lmm, "%s ~ %s" % (o, rhs)) for o in ("f", "t_first_correct", "t_commit")
              for t, rhs in (("main model", "legible * edge + second"),
                             ("order check (legible:second)", "legible * second + edge"))]
    models.append(("final_correct", "main model", gee, "final_correct ~ legible * edge + second"))
    for outcome, title, fit_model, formula in models:
        try:
            fit, label = fit_model(formula, trials)
            with warnings.catch_warnings():  # degenerate fits warn again here; label has the fit's warnings
                warnings.simplefilter("ignore")
                summary = fit.summary()
            out.append("=== %s, %s, %s ===\n%s\n" % (outcome, title, label, summary))
        except Exception as error:  # e.g. no variation at all in a pilot sample
            out.append("=== %s, %s: failed (%s: %s) ===\n" % (outcome, title, type(error).__name__, error))
    return "\n".join(out)


def questionnaire_tests(path, first_motion):
    """Paired Wilcoxon (legible vs predictable) per numeric column, and the order check:
    the legible - predictable difference compared between the two first-motion groups."""
    q = pd.read_csv(path)
    q["participant_id"] = q["participant_id"].astype(str)
    rows = []
    for col in q.columns:
        if col in ("participant_id", "motion", "block") or not pd.api.types.is_numeric_dtype(q[col]):
            continue
        wide = q.pivot_table(index="participant_id", columns="motion", values=col)
        if not {"legible", "predictable"} <= set(wide.columns):
            continue
        d = (wide["legible"] - wide["predictable"]).dropna()
        row = {"item": col, "n": len(d), "legible_median": wide["legible"].median(),
               "predictable_median": wide["predictable"].median(), "mean_diff": d.mean()}
        if np.any(d != 0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row["W"], row["p_wilcoxon"] = stats.wilcoxon(d)
        groups = [d[[first_motion.get(p) == m for p in d.index]] for m in ("predictable", "legible")]
        if all(len(g) >= 2 for g in groups):
            row["U_order"], row["p_order"] = stats.mannwhitneyu(*groups)
        rows.append(row)
    return pd.DataFrame(rows)


def manipulation_check(config_path, hashes):
    """Executed path shape of every reach from the trajectory file: lateral deviation from
    the straight line and the model legibility score, plus whether all sessions used it."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    path = cfg["trajectories"]
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    meta = trajectories.read_meta(path)
    names = {b["id"]: b["name"] for b in cfg["blocks"]}
    rows = []
    for key in sorted(meta.get("lateral", {})):
        motion, target = key.rsplit("_", 1)
        rows.append({"motion": motion, "cube": names.get(int(target), target),
                     "lateral_deviation_m": meta["lateral"][key],
                     "model_legibility": meta.get("legibility", {}).get(key)})
    note = None
    other = sorted(h for h in hashes if h and h != meta["hash"])
    if other:
        note = ("Trajectory file %s has hash %s, but sessions used %s: the table describes the current "
                "file, not those sessions." % (path, meta["hash"], ", ".join(other)))
    return pd.DataFrame(rows), note


# --- figures -------------------------------------------------------------------------

def recognition_curve(trials, choices, grid=np.linspace(0, 1, 101)):
    """Share of trials with the right cube selected at each fraction of the reach, per motion
    and cube type. Each trial: the choices between its onset and its end, as a step function."""
    rows = []
    for _, t in trials.iterrows():
        c = choices[(choices["participant_id"].astype(str) == t["participant_id"])
                    & (choices["trial_id"] == t["trial_id"])
                    & (choices["stamp"] >= t["t_start"] - 1e-6) & (choices["stamp"] <= t["t_end"] + 1e-6)]
        stamps = ((c["stamp"] - t["t_start"]) / t["duration_T"]).to_numpy()
        ids = c["choice_id"].to_numpy()
        held = np.full(len(grid), -1)
        for s, k in sorted(zip(stamps, ids)):
            held[grid >= s] = k
        rows.append((t["motion"], t["cube_type"], held == t["target_id"]))
    out = []
    for motion in ("predictable", "legible"):
        for cube_type in ("all", "edge", "middle"):
            sel = [r[2] for r in rows if r[0] == motion and cube_type in ("all", r[1])]
            if sel:
                out.append({"motion": motion, "cube_type": cube_type, "n": len(sel),
                            "share": np.mean(sel, axis=0)})
    return grid, out


def plot(trials, choices, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"predictable": "#4c72b0", "legible": "#dd8452"}

    cubes = sorted(trials["target_name"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, motion in enumerate(("predictable", "legible")):
        for i, cube in enumerate(cubes):
            g = trials[(trials["motion"] == motion) & (trials["target_name"] == cube)]["f"]
            m, lo, hi = mean_ci(g)
            x = i + (k - 0.5) * 0.3
            ax.errorbar(x, m, yerr=None if np.isnan(lo) else [[m - lo], [hi - m]], fmt="o", capsize=4,
                        color=colors[motion], label=motion if i == 0 else None)
            ax.scatter(np.full(len(g), x) + np.random.default_rng(i + 10 * k).uniform(-0.05, 0.05, len(g)),
                       g, s=10, alpha=0.35, color=colors[motion])
    ax.set_xticks(range(len(cubes)))
    ax.set_xticklabels(cubes)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("f: share of the reach with the right cube selected")
    ax.set_title("Mean and 95 % CI per cube (dots: trials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f_by_cube.png"), dpi=150)
    plt.close(fig)

    if choices.empty:
        return
    grid, curves = recognition_curve(trials, choices)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, cube_type in zip(axes, ("all", "edge", "middle")):
        for c in curves:
            if c["cube_type"] == cube_type:
                ax.plot(grid, c["share"], color=colors[c["motion"]], label="%s (n=%d)" % (c["motion"], c["n"]))
        ax.set_title("%s cubes" % cube_type)
        ax.set_xlabel("fraction of the reach")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("share with the right cube selected")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recognition_curve.png"), dpi=150)
    plt.close(fig)


# --- main ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="~/hri_data")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "analysis"))
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "experiment.yaml"))
    ap.add_argument("--questionnaires", help="CSV: participant_id, motion, numeric scale/item columns")
    ap.add_argument("--exclude", default="", help="comma-separated participant ids to leave out")
    ap.add_argument("--min-trials", type=int, default=6, help="fewest finished game trials to keep a participant")
    ap.add_argument("--include-non-p", action="store_true", help="also analyse ids that are not P## (dry runs)")
    args = ap.parse_args(argv)

    trials, choices, notes = load(args.data, args.include_non_p)
    if trials.empty:
        sys.exit("No game trials found in %s." % os.path.expanduser(args.data))
    trials = prepare(trials)
    kept, exclusions = exclude(trials, [p for p in args.exclude.split(",") if p], args.min_trials)
    os.makedirs(args.out, exist_ok=True)
    exclusions = pd.concat([exclusions, pd.DataFrame(
        [{"participant_id": p, "trial_id": "all", "reason": "skipped: " + r} for p, r in notes])], ignore_index=True)
    exclusions.to_csv(os.path.join(args.out, "exclusions.csv"), index=False)
    if kept.empty:
        sys.exit("No trials left after the exclusions (see %s/exclusions.csv)." % args.out)
    kept.to_csv(os.path.join(args.out, "trials.csv"), index=False)

    desc = descriptives(kept)
    desc.to_csv(os.path.join(args.out, "descriptives.csv"), index=False)
    means = participant_means(kept)
    means.to_csv(os.path.join(args.out, "participant_means.csv"), index=False)
    tests = paired_tests(means)
    tests.to_csv(os.path.join(args.out, "tests.csv"), index=False)
    models = mixed_models(kept)
    with open(os.path.join(args.out, "models.txt"), "w") as f:
        f.write(models)
    manipulation, hash_note = manipulation_check(args.config, set(kept["trajectory_hash"]))
    manipulation.to_csv(os.path.join(args.out, "manipulation.csv"), index=False)
    if args.questionnaires:
        first = kept.groupby("participant_id")["first_motion"].first().to_dict()
        questionnaire_tests(args.questionnaires, first).to_csv(
            os.path.join(args.out, "questionnaire_tests.csv"), index=False)
    plot(kept, choices, args.out)

    n = kept["participant_id"].nunique()
    print("%d participants, %d game trials (first motion: %s)" % (
        n, len(kept), ", ".join("%s %d" % kv for kv in
                                kept.groupby("participant_id")["first_motion"].first().value_counts().items())))
    print("Excluded / skipped: %d (exclusions.csv)" % len(exclusions))
    if hash_note:
        print("NOTE: " + hash_note)
    if n % 8:
        print("NOTE: %d participants is not a multiple of 8; the cube orders are only balanced per 8." % n)
    print("\nf (share of the reach with the right cube), mean [95 % CI]:")
    for motion in ("predictable", "legible"):
        m, lo, hi = mean_ci(kept[kept["motion"] == motion]["f"])
        print("  %-11s %.3f [%.3f, %.3f]" % (motion, m, lo, hi))
    print("\nParticipant-level legible - predictable (tests.csv):")
    for _, r in tests[tests["outcome"] == "f"].iterrows():
        print("  f, %-38s n=%2d  diff %+.3f [%+.3f, %+.3f]  d_z %s  p(Wilcoxon) %s" % (
            r["comparison"], r["n"], r["mean_diff"], r["ci_low"], r["ci_high"],
            "%.2f" % r["d_z"] if pd.notna(r["d_z"]) else "-",
            "%.3f" % r["p_wilcoxon"] if pd.notna(r["p_wilcoxon"]) else "-"))
    print("\n" + (models.splitlines()[0] if models.startswith("Mixed models skipped") else "Mixed models: models.txt"))
    print("Written to %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
