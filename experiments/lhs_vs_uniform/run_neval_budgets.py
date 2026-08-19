#!/usr/bin/env python3
"""Redo HPO comparison: best score vs number of evaluations.

For each problem and sampling strategy, run full searches at budgets
n_iter ∈ {3, 10, 30, 100}. For each budget, pack as many seed repeats as
fit in a 5s wall-clock window (always ≥3 repeats when a single run is slow).

Plots mean best CV score ± spread (std) vs n_evals, with search space and
n_hparams annotated.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from evaluate import evaluate_candidates
from problems import (
    build_mixed_pipeline,
    make_hgb_clf,
    make_hgb_reg,
    make_logreg,
    make_ridge,
    _breast_cancer,
    _diabetes,
    _digits_multiclass,
    _openml_credit_g,
    _synthetic_mixed_clf,
)
from run_utils import REPORTS, RESULTS, utc_now
from samplers import ParamSpec, sample_search_space

FIGDIR = REPORTS / "figures_neval"
ARTDIR = Path("/opt/cursor/artifacts")
FIGDIR.mkdir(parents=True, exist_ok=True)
ARTDIR.mkdir(parents=True, exist_ok=True)

N_ITERS = (3, 10, 30, 100)
WALL_BUDGET_S = 5.0
MIN_REPEATS = 3
MAX_REPEATS = 5000  # safety only; 5s wall budget is the real limit
CV = 3

LOGREG_SPECS = [
    ParamSpec("C", "loguniform", low=1e-4, high=1e4),
    ParamSpec("l1_ratio", "uniform", low=0.0, high=1.0),
]
HGB_CLF_SPECS = [
    ParamSpec("learning_rate", "loguniform", low=1e-3, high=1.0),
    ParamSpec("max_depth", "int", low=1, high=16),
    ParamSpec("min_samples_leaf", "int", low=1, high=80),
    ParamSpec("l2_regularization", "loguniform", low=1e-8, high=100.0),
]
RIDGE_SPECS = [ParamSpec("alpha", "loguniform", low=1e-4, high=1e4)]
HGB_REG_SPECS = [
    ParamSpec("learning_rate", "loguniform", low=1e-3, high=1.0),
    ParamSpec("max_depth", "int", low=1, high=16),
    ParamSpec("min_samples_leaf", "int", low=1, high=80),
    ParamSpec("l2_regularization", "loguniform", low=1e-8, high=100.0),
]
PIPE_HGB_SPECS = [
    ParamSpec("model__learning_rate", "loguniform", low=1e-3, high=1.0),
    ParamSpec("model__max_depth", "int", low=1, high=16),
    ParamSpec("model__min_samples_leaf", "int", low=1, high=100),
    ParamSpec("model__l2_regularization", "loguniform", low=1e-8, high=50.0),
    ParamSpec("model__max_leaf_nodes", "int", low=8, high=127),
]
PIPE_LOGREG_SPECS = [
    ParamSpec("model__C", "loguniform", low=1e-4, high=1e4),
    ParamSpec("model__l1_ratio", "uniform", low=0.0, high=1.0),
]


def experiment_problems() -> list[dict[str, Any]]:
    return [
        {
            "problem_id": "diabetes_ridge",
            "scoring": "neg_mean_squared_error",
            "load_xy": _diabetes,
            "make_estimator": make_ridge,
            "specs": RIDGE_SPECS,
            "kind": "estimator",
        },
        {
            "problem_id": "diabetes_hgb",
            "scoring": "neg_mean_squared_error",
            "load_xy": _diabetes,
            "make_estimator": make_hgb_reg,
            "specs": HGB_REG_SPECS,
            "kind": "estimator",
        },
        {
            "problem_id": "breast_cancer_logreg",
            "scoring": "neg_log_loss",
            "load_xy": _breast_cancer,
            "make_estimator": make_logreg,
            "specs": LOGREG_SPECS,
            "kind": "estimator",
        },
        {
            "problem_id": "breast_cancer_hgb",
            "scoring": "neg_log_loss",
            "load_xy": _breast_cancer,
            "make_estimator": make_hgb_clf,
            "specs": HGB_CLF_SPECS,
            "kind": "estimator",
        },
        {
            "problem_id": "digits_logreg",
            "scoring": "neg_log_loss",
            "load_xy": _digits_multiclass,
            "make_estimator": make_logreg,
            "specs": LOGREG_SPECS,
            "kind": "estimator",
        },
        {
            "problem_id": "synth_mixed_hgb",
            "scoring": "neg_log_loss",
            "load_xy": _synthetic_mixed_clf,
            "model": "hgb",
            "specs": PIPE_HGB_SPECS,
            "kind": "pipeline",
        },
        {
            "problem_id": "synth_mixed_logreg",
            "scoring": "neg_log_loss",
            "load_xy": _synthetic_mixed_clf,
            "model": "logreg",
            "specs": PIPE_LOGREG_SPECS,
            "kind": "pipeline",
        },
        {
            "problem_id": "credit_g_hgb",
            "scoring": "neg_log_loss",
            "load_xy": _openml_credit_g,
            "model": "hgb",
            "specs": PIPE_HGB_SPECS,
            "kind": "pipeline",
        },
    ]


def score_ylabel(scoring: str) -> str:
    return {
        "neg_log_loss": "Best CV neg_log_loss (higher better)",
        "neg_mean_squared_error": "Best CV neg_MSE (higher better)",
        "neg_brier_score": "Best CV neg_brier_score (higher better)",
    }.get(scoring, f"Best CV {scoring} (higher better)")


def format_search_space(specs: list[ParamSpec]) -> str:
    return "Search space (" + f"{len(specs)} hparams):\n" + "\n".join(
        f"  • {s.format()}" for s in specs
    )


def make_estimator(cfg, X):
    if cfg["kind"] == "pipeline":
        return build_mixed_pipeline(X, model=cfg["model"])
    return cfg["make_estimator"]()


def one_search(cfg, X, y, specs, method: str, n_iter: int, seed: int) -> float:
    method_seed = seed if method == "uniform" else seed + 10_000
    candidates = sample_search_space(specs, n_iter, method, method_seed)
    run = evaluate_candidates(
        make_estimator(cfg, X),
        X,
        y,
        candidates,
        scoring=cfg["scoring"],
        cv=CV,
        method=method,
        problem_id=cfg["problem_id"],
        seed=seed,
    )
    best = float(run.best_scores[-1]) if run.trials else float("-inf")
    return best if np.isfinite(best) else float("-inf")


def repeats_for_budget(
    cfg, X, y, specs, method: str, n_iter: int
) -> dict[str, Any]:
    """Run as many seeds as fit in WALL_BUDGET_S, always ≥ MIN_REPEATS."""
    scores: list[float] = []
    seed = 0
    t0 = time.perf_counter()
    while True:
        sc = one_search(cfg, X, y, specs, method, n_iter, seed)
        scores.append(sc)
        seed += 1
        elapsed = time.perf_counter() - t0
        if len(scores) < MIN_REPEATS:
            continue
        if elapsed >= WALL_BUDGET_S:
            break
        if len(scores) >= MAX_REPEATS:
            break
    arr = np.asarray(scores, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "n_iter": n_iter,
        "method": method,
        "n_repeats": int(len(scores)),
        "wall_s": float(time.perf_counter() - t0),
        "scores": [float(s) for s in scores],
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "q10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
        "q90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
    }


def plot_problem(cfg, specs, by_method: dict[str, list[dict]], out_stem: str):
    n_hp = len(specs)
    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_axes([0.10, 0.38, 0.86, 0.52])

    for method, color, marker in (
        ("uniform", "#1f4e79", "o"),
        ("lhs", "#c45c26", "s"),
    ):
        rows = sorted(by_method[method], key=lambda r: r["n_iter"])
        xs = np.array([r["n_iter"] for r in rows], dtype=float)
        means = np.array([r["mean"] for r in rows], dtype=float)
        stds = np.array([r["std"] for r in rows], dtype=float)
        ns = [r["n_repeats"] for r in rows]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=color,
            marker=marker,
            lw=2,
            capsize=4,
            label=f"{method.capitalize()} (n_rep={ns})",
        )
        # Also shade q10–q90 if available
        q10 = np.array([r["q10"] for r in rows], dtype=float)
        q90 = np.array([r["q90"] for r in rows], dtype=float)
        ax.fill_between(xs, q10, q90, color=color, alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xticks(list(N_ITERS))
    ax.set_xticklabels([str(n) for n in N_ITERS])
    ax.minorticks_off()
    ax.set_xlabel("Number of evaluations (n_iter)")
    ax.set_ylabel(score_ylabel(cfg["scoring"]))
    ax.set_title(
        f"{cfg['problem_id']}  |  {n_hp} tuned hparams  |  "
        f"budgets {list(N_ITERS)}; ≥{MIN_REPEATS} seeds / ≤{WALL_BUDGET_S:.0f}s per budget",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="lower right")

    # Annotate repeat counts under each point lightly
    for method, color, dy in (("uniform", "#1f4e79", 0.02), ("lhs", "#c45c26", -0.02)):
        rows = sorted(by_method[method], key=lambda r: r["n_iter"])
        for r in rows:
            ax.annotate(
                f"n={r['n_repeats']}",
                (r["n_iter"], r["mean"]),
                textcoords="offset points",
                xytext=(0, 8 if method == "uniform" else -14),
                ha="center",
                fontsize=7,
                color=color,
            )

    space_txt = format_search_space(specs)
    fig.text(
        0.08,
        0.02,
        space_txt,
        ha="left",
        va="bottom",
        family="monospace",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#cccccc"),
    )
    for d in (FIGDIR, ARTDIR):
        fig.savefig(d / f"{out_stem}.png", dpi=150)
    plt.close(fig)


def main():
    print(f"n_eval budget sweeps — {utc_now()}", flush=True)
    summary_path = RESULTS / "neval_budget_summary.json"
    all_summary: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            all_summary = json.loads(summary_path.read_text())
        except Exception:
            all_summary = []
    done = {r["problem_id"] for r in all_summary if (FIGDIR / r["figure"]).exists()}

    md_lines = [
        "# Best score vs number of evaluations",
        "",
        f"_Generated {utc_now()}_",
        "",
        f"Budgets: `n_iter ∈ {list(N_ITERS)}`. For each budget/method, as many "
        f"seed repeats as fit in **{WALL_BUDGET_S:.0f}s** wall time "
        f"(always **≥{MIN_REPEATS}** repeats). Points show mean ± std; bands are "
        f"10th–90th percentiles across seeds.",
        "",
    ]

    for cfg in experiment_problems():
        pid = cfg["problem_id"]
        specs = list(cfg["specs"])
        if pid in done:
            print(f"\n=== {pid}: skip ===", flush=True)
            continue
        print(f"\n=== {pid} ({len(specs)} hparams) ===", flush=True)
        X, y = cfg["load_xy"]()
        by_method: dict[str, list[dict]] = {"uniform": [], "lhs": []}
        for n_iter in N_ITERS:
            for method in ("uniform", "lhs"):
                print(f"  n_iter={n_iter} {method}...", flush=True)
                row = repeats_for_budget(cfg, X, y, specs, method, n_iter)
                by_method[method].append(row)
                print(
                    f"    repeats={row['n_repeats']} wall={row['wall_s']:.2f}s "
                    f"mean={row['mean']:.5g} ± {row['std']:.3g}",
                    flush=True,
                )

        stem = f"neval_{pid}"
        plot_problem(cfg, specs, by_method, stem)
        rec = {
            "problem_id": pid,
            "n_hparams": len(specs),
            "search_space": [s.format() for s in specs],
            "figure": f"{stem}.png",
            "uniform": by_method["uniform"],
            "lhs": by_method["lhs"],
        }
        all_summary.append(rec)
        summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
        print(f"  Wrote {stem}.png", flush=True)

    # Rebuild markdown from full summary
    if summary_path.exists():
        all_summary = json.loads(summary_path.read_text())
    for rec in all_summary:
        md_lines.append(f"## `{rec['problem_id']}` — {rec['n_hparams']} tuned hparams")
        md_lines.append("")
        md_lines.append(f"![{rec['figure']}](figures_neval/{rec['figure']})")
        md_lines.append("")
        md_lines.append("| n_iter | Uniform mean±std (n) | LHS mean±std (n) |")
        md_lines.append("|------:|----------------------:|-----------------:|")
        u_by = {r["n_iter"]: r for r in rec["uniform"]}
        l_by = {r["n_iter"]: r for r in rec["lhs"]}
        for n in N_ITERS:
            u, l = u_by[n], l_by[n]
            md_lines.append(
                f"| {n} | {u['mean']:.5g}±{u['std']:.3g} (n={u['n_repeats']}) | "
                f"{l['mean']:.5g}±{l['std']:.3g} (n={l['n_repeats']}) |"
            )
        md_lines.append("")
        md_lines.append("```")
        for s in rec["search_space"]:
            md_lines.append(s)
        md_lines.append("```")
        md_lines.append("")
    md_lines.extend(
        [
            "This pull request includes code written with the assistance of AI.",
            "The code has **not yet been reviewed** by a human.",
            "",
        ]
    )
    (REPORTS / "NEVAL_BUDGETS.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
