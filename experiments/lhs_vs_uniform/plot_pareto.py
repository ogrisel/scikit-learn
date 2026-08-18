#!/usr/bin/env python3
"""Pareto / anytime plots for problems with large LHS vs uniform gaps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from evaluate import run_paired_search
from problems import (
    build_mixed_pipeline,
    phase1_problems,
    phase2_problems,
)
from run_utils import RESULTS, REPORTS, utc_now

FIGDIR = REPORTS / "figures"
ARTDIR = Path("/opt/cursor/artifacts")
FIGDIR.mkdir(parents=True, exist_ok=True)
ARTDIR.mkdir(parents=True, exist_ok=True)


def mean_anytime(runs, grid_fracs=np.linspace(0, 1, 51)):
    """Interpolate best-so-far vs normalized cum-time; average across seeds."""
    curves = []
    for run in runs:
        t = np.asarray(run.cum_times, dtype=float)
        b = np.asarray(run.best_scores, dtype=float)
        if t.size == 0 or not np.isfinite(t[-1]) or t[-1] <= 0:
            continue
        # Also vs n_eval fraction
        n = np.arange(1, len(b) + 1) / len(b)
        curves.append((t / t[-1], b, n, b))
    if not curves:
        return None

    # Average on normalized time grid
    tg = grid_fracs
    bt = []
    bn = []
    for t_norm, b_t, n_norm, b_n in curves:
        # step function: best at time fraction
        vals_t = np.empty_like(tg)
        vals_n = np.empty_like(tg)
        for i, g in enumerate(tg):
            idx_t = np.searchsorted(t_norm, g, side="right") - 1
            idx_n = np.searchsorted(n_norm, g, side="right") - 1
            vals_t[i] = b_t[max(idx_t, 0)]
            vals_n[i] = b_n[max(idx_n, 0)]
        bt.append(vals_t)
        bn.append(vals_n)
    return {
        "grid": tg,
        "mean_best_time": np.mean(bt, axis=0),
        "std_best_time": np.std(bt, axis=0),
        "mean_best_neval": np.mean(bn, axis=0),
        "std_best_neval": np.std(bn, axis=0),
        "n": len(bt),
    }


def plot_paired_instance(uni, lhs, title, out_stem, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, xkey, xlab in (
        (axes[0], "cum_time", "Cumulative fit+CV time (s)"),
        (axes[1], "n_eval", "Evaluations"),
    ):
        if xkey == "cum_time":
            xu, xl = uni.cum_times, lhs.cum_times
        else:
            xu = np.arange(1, len(uni.trials) + 1)
            xl = np.arange(1, len(lhs.trials) + 1)
        ax.step(xu, uni.best_scores, where="post", label="Uniform", color="#1f4e79", lw=2)
        ax.step(xl, lhs.best_scores, where="post", label="LHS", color="#c45c26", lw=2)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    for d in (FIGDIR, ARTDIR):
        fig.savefig(d / f"{out_stem}.png", dpi=140)
    plt.close(fig)


def plot_mean_curves(stats_u, stats_l, title, out_stem, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, xlab in (
        (axes[0], "time", "Normalized cumulative time"),
        (axes[1], "neval", "Normalized evaluation budget"),
    ):
        g = stats_u["grid"]
        if key == "time":
            mu, su = stats_u["mean_best_time"], stats_u["std_best_time"]
            ml, sl = stats_l["mean_best_time"], stats_l["std_best_time"]
        else:
            mu, su = stats_u["mean_best_neval"], stats_u["std_best_neval"]
            ml, sl = stats_l["mean_best_neval"], stats_l["std_best_neval"]
        ax.plot(g, mu, color="#1f4e79", lw=2, label="Uniform")
        ax.fill_between(g, mu - su, mu + su, color="#1f4e79", alpha=0.15)
        ax.plot(g, ml, color="#c45c26", lw=2, label="LHS")
        ax.fill_between(g, ml - sl, ml + sl, color="#c45c26", alpha=0.15)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(title + f" (n={stats_u['n']} seeds)", fontsize=12)
    fig.tight_layout()
    for d in (FIGDIR, ARTDIR):
        fig.savefig(d / f"{out_stem}.png", dpi=140)
    plt.close(fig)


def score_ylabel(scoring: str) -> str:
    return {
        "neg_log_loss": "Best CV neg_log_loss (higher better)",
        "neg_brier_score": "Best CV neg_brier_score (higher better)",
        "neg_mean_squared_error": "Best CV neg_MSE (higher better)",
    }.get(scoring, f"Best CV {scoring} (higher better)")


def run_problem_all_seeds(problem, seeds):
    X, y = problem.load_xy()
    uni_runs, lhs_runs = [], []
    for seed in seeds:
        paired = run_paired_search(
            problem.make_estimator(),
            X,
            y,
            problem.specs,
            n_iter=problem.n_iter,
            scoring=problem.scoring,
            cv=problem.cv,
            seed=seed,
            problem_id=problem.problem_id,
        )
        uni_runs.append(paired["uniform"])
        lhs_runs.append(paired["lhs"])
    return uni_runs, lhs_runs


def run_pipeline_all_seeds(cfg, seeds):
    X, y = cfg["load_xy"]()
    est = build_mixed_pipeline(X, model=cfg["model"])
    uni_runs, lhs_runs = [], []
    for seed in seeds:
        paired = run_paired_search(
            est,
            X,
            y,
            cfg["specs"],
            n_iter=cfg["n_iter"],
            scoring=cfg["scoring"],
            cv=cfg["cv"],
            seed=seed,
            problem_id=cfg["problem_id"],
        )
        uni_runs.append(paired["uniform"])
        lhs_runs.append(paired["lhs"])
    return uni_runs, lhs_runs


def pick_divergent_seeds(uni_runs, lhs_runs, k=3):
    """Pick seeds with largest |area| or end/mid gap between anytime curves."""
    scored = []
    for u, l in zip(uni_runs, lhs_runs):
        # Compare on evaluation index
        n = min(len(u.best_scores), len(l.best_scores))
        gap = np.asarray(l.best_scores[:n]) - np.asarray(u.best_scores[:n])
        score = float(np.max(np.abs(gap)) + 0.25 * np.mean(np.abs(gap)))
        # Also time-to-similar-quality: final best difference
        scored.append((score, u.seed, u, l, gap))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]


def main():
    print(f"Building Pareto plots — {utc_now()}", flush=True)
    p1 = {p.problem_id: p for p in phase1_problems()}
    p2 = {c["problem_id"]: c for c in phase2_problems()}

    # Problems with historically large gaps
    selections = [
        ("phase1", "diabetes_hgb", list(range(20))),
        ("phase1", "diabetes_ridge", list(range(20))),
        ("phase1", "breast_cancer_hgb", list(range(20))),
        ("phase1", "digits_logreg", list(range(20))),
        ("phase2", "synth_mixed_hgb", list(range(15))),
        ("phase2", "credit_g_hgb", list(range(12))),
        ("phase2", "synth_mixed_logreg", list(range(15))),
    ]

    manifest = []
    for phase, pid, seeds in selections:
        print(f"\n=== {pid} ===", flush=True)
        if phase == "phase1":
            problem = p1[pid]
            uni_runs, lhs_runs = run_problem_all_seeds(problem, seeds)
            scoring = problem.scoring
        else:
            cfg = p2[pid]
            uni_runs, lhs_runs = run_pipeline_all_seeds(cfg, seeds)
            scoring = cfg["scoring"]

        ylabel = score_ylabel(scoring)
        stats_u = mean_anytime(uni_runs)
        stats_l = mean_anytime(lhs_runs)
        mean_stem = f"pareto_mean_{pid}"
        plot_mean_curves(
            stats_u,
            stats_l,
            title=f"Mean anytime Pareto — {pid}",
            out_stem=mean_stem,
            ylabel=ylabel,
        )
        manifest.append({"type": "mean", "problem_id": pid, "file": f"{mean_stem}.png"})

        top = pick_divergent_seeds(uni_runs, lhs_runs, k=2)
        for rank, (score, seed, u, l, gap) in enumerate(top, start=1):
            stem = f"pareto_seed_{pid}_seed{seed}"
            plot_paired_instance(
                u,
                l,
                title=f"{pid} — seed {seed} (large gap example #{rank})",
                out_stem=stem,
                ylabel=ylabel,
            )
            manifest.append(
                {
                    "type": "seed",
                    "problem_id": pid,
                    "seed": seed,
                    "gap_score": score,
                    "max_gap": float(np.max(np.abs(gap))),
                    "file": f"{stem}.png",
                }
            )
            print(f"  seed {seed}: max|gap|={np.max(np.abs(gap)):.5g}", flush=True)

    with (RESULTS / "pareto_plot_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Markdown gallery
    lines = [
        f"# Pareto / anytime plots — large sampler differences",
        f"",
        f"_Generated {utc_now()}_",
        f"",
        f"Each plot shows the **anytime Pareto curve**: best CV score so far vs",
        f"cumulative fit+CV time (left) and vs evaluation count (right).",
        f"Higher is better for all scorers used (`neg_*`).",
        f"",
    ]
    by_problem = {}
    for m in manifest:
        by_problem.setdefault(m["problem_id"], []).append(m)
    for pid, items in by_problem.items():
        lines.append(f"## `{pid}`")
        lines.append("")
        for m in items:
            if m["type"] == "mean":
                lines.append(f"### Mean across seeds")
            else:
                lines.append(
                    f"### Divergent seed {m['seed']} (max |LHS−Uniform| gap ≈ {m['max_gap']:.4g})"
                )
            lines.append("")
            lines.append(f"![{m['file']}](figures/{m['file']})")
            lines.append("")
    lines.extend(
        [
            "This pull request includes code written with the assistance of AI.",
            "The code has **not yet been reviewed** by a human.",
            "",
        ]
    )
    (REPORTS / "PARETO_PLOTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", REPORTS / "PARETO_PLOTS.md", flush=True)


if __name__ == "__main__":
    main()
