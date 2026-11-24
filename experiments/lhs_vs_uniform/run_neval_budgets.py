#!/usr/bin/env python3
"""Best score vs n_iter ∈ {3, 5, 10, 30} for Uniform vs LHS.

For each budget, pack as many seed repeats as fit in 5s wall time
(always ≥3). Plots include search-space annotations. Winning hyperparameter
combos are recorded for every repeat.
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
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

FIGDIR = REPORTS / "figures_neval"
ARTDIR = Path("/opt/cursor/artifacts")
FIGDIR.mkdir(parents=True, exist_ok=True)
ARTDIR.mkdir(parents=True, exist_ok=True)
WINNERS_DIR = RESULTS / "winners"
WINNERS_DIR.mkdir(parents=True, exist_ok=True)

N_ITERS = (3, 5, 10, 30)
WALL_BUDGET_S = 5.0
MIN_REPEATS = 3
MAX_REPEATS = 5000
CV = 3
MD_WINNER_LIST_CAP = 25  # full list in JSON; markdown lists up to this many

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

# From ogrisel/notebooks poly_reg_array_api.ipynb (ignore Array API / GPU bits):
# SplineTransformer → Nystroem(poly/rbf) → RidgeCV, with the notebook search space.
# n_knots / n_components use integer log-discretization over the same min/max ranges.
SPLINE_NYSTROEM_SPECS = [
    ParamSpec("splinetransformer__n_knots", "logint", low=3, high=29),
    ParamSpec("nystroem__kernel", "choice", choices=("poly", "rbf")),
    ParamSpec("nystroem__degree", "int", low=2, high=5),
    ParamSpec("nystroem__gamma", "loguniform", low=1e-6, high=1e6),
    ParamSpec("nystroem__n_components", "logint", low=50, high=300),
]


def _poly_reg_spline_nystroem_data():
    """Synthetic heteroscedastic data from the notebook, subsampled for speed."""

    def true_mean(X):
        return np.sin(X[:, 0] * 2) * np.cos(X[:, 1]) ** 4

    def true_std(X):
        return 0.3 * np.cos(X[:, 1]) ** 6 + 0.1

    rng = np.random.default_rng(seed=0)
    # Notebook uses 1e5; keep the same process but subsample for HPO wall budgets.
    n_samples = 2500
    X = rng.uniform(low=-3, high=3, size=(n_samples, 2)).astype(np.float64)
    y = rng.normal(loc=true_mean(X), scale=true_std(X)).astype(np.float64)
    return X, y


def make_spline_nystroem_ridge():
    """Pipeline matching the notebook's numpy/CPU poly-reg setup."""
    return make_pipeline(
        SplineTransformer(n_knots=5),
        Nystroem(kernel="poly", degree=2, n_components=100, random_state=0),
        RidgeCV(alphas=np.logspace(-6, 6, 13)),
    )


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
        {
            "problem_id": "poly_reg_spline_nystroem",
            "scoring": "neg_mean_squared_error",
            "load_xy": _poly_reg_spline_nystroem_data,
            "make_estimator": make_spline_nystroem_ridge,
            "specs": SPLINE_NYSTROEM_SPECS,
            "kind": "estimator",
            # ~2x the default 5s packing budget for more seed repeats
            "wall_budget_s": 10.0,
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


def format_params(params: dict[str, Any]) -> str:
    parts = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.5g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def make_estimator(cfg, X):
    if cfg["kind"] == "pipeline":
        return build_mixed_pipeline(X, model=cfg["model"])
    return cfg["make_estimator"]()


def one_search(
    cfg, X, y, specs, method: str, n_iter: int, seed: int
) -> tuple[float, dict[str, Any]]:
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
    if not run.trials:
        return float("-inf"), {}
    # Winning trial = first achieving best score (tie-break earliest)
    best = float(run.best_scores[-1])
    win = next(t for t in run.trials if t.best_score_so_far == best or t.score == best)
    # Prefer the trial with actual best score
    best_idx = int(np.argmax(run.scores))
    win = run.trials[best_idx]
    score = float(win.score) if np.isfinite(win.score) else float("-inf")
    return score, dict(win.params)


def summarize_winner_params(
    specs: list[ParamSpec], winners: list[dict[str, Any]]
) -> dict[str, Any]:
    """Median / IQR of winning param values across repeats."""
    out: dict[str, Any] = {}
    for spec in specs:
        vals = [w["params"][spec.name] for w in winners if spec.name in w["params"]]
        if not vals:
            continue
        if spec.kind in ("uniform", "loguniform"):
            arr = np.asarray(vals, dtype=float)
            out[spec.name] = {
                "median": float(np.median(arr)),
                "q10": float(np.percentile(arr, 10)),
                "q90": float(np.percentile(arr, 90)),
                "kind": spec.kind,
            }
        elif spec.kind in ("int", "logint"):
            arr = np.asarray(vals, dtype=float)
            out[spec.name] = {
                "median": float(np.median(arr)),
                "q10": float(np.percentile(arr, 10)),
                "q90": float(np.percentile(arr, 90)),
                "kind": spec.kind,
            }
        else:
            # choice — mode
            uniq, counts = np.unique(vals, return_counts=True)
            out[spec.name] = {
                "mode": uniq[int(np.argmax(counts))].item()
                if hasattr(uniq[int(np.argmax(counts))], "item")
                else uniq[int(np.argmax(counts))],
                "counts": {str(u): int(c) for u, c in zip(uniq, counts)},
                "kind": "choice",
            }
    return out


def repeats_for_budget(
    cfg, X, y, specs, method: str, n_iter: int, *, wall_budget_s: float = WALL_BUDGET_S
) -> dict[str, Any]:
    winners: list[dict[str, Any]] = []
    seed = 0
    t0 = time.perf_counter()
    while True:
        score, params = one_search(cfg, X, y, specs, method, n_iter, seed)
        winners.append({"seed": seed, "score": float(score), "params": params})
        seed += 1
        elapsed = time.perf_counter() - t0
        if len(winners) < MIN_REPEATS:
            continue
        if elapsed >= wall_budget_s:
            break
        if len(winners) >= MAX_REPEATS:
            break
    scores = np.asarray([w["score"] for w in winners], dtype=float)
    finite = scores[np.isfinite(scores)]
    # Overall winning combo among repeats (best seed)
    if finite.size:
        best_i = int(np.argmax(scores))
        overall = winners[best_i]
    else:
        overall = winners[0] if winners else {"seed": -1, "score": float("nan"), "params": {}}
    return {
        "n_iter": n_iter,
        "method": method,
        "n_repeats": int(len(winners)),
        "wall_s": float(time.perf_counter() - t0),
        "scores": [float(w["score"]) for w in winners],
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "q10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
        "q90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
        "winners": winners,
        "best_repeat": overall,
        "winner_param_summary": summarize_winner_params(specs, winners),
    }


def plot_problem(
    cfg,
    specs,
    by_method: dict[str, list[dict]],
    out_stem: str,
    *,
    wall_budget_s: float = WALL_BUDGET_S,
):
    """Plot mean±std vs n_iter with x ticks fixed to N_ITERS (equally spaced)."""
    n_hp = len(specs)
    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_axes([0.10, 0.38, 0.86, 0.52])

    # Equal spacing so every budget in N_ITERS is clearly visible (not log-cramped).
    x_pos = {n: i for i, n in enumerate(N_ITERS)}

    for method, color, marker in (
        ("uniform", "#1f4e79", "o"),
        ("lhs", "#c45c26", "s"),
    ):
        rows_by_n = {r["n_iter"]: r for r in by_method[method]}
        # Require the full budget grid
        missing = [n for n in N_ITERS if n not in rows_by_n]
        if missing:
            raise ValueError(f"{cfg['problem_id']}/{method} missing n_iter={missing}")
        rows = [rows_by_n[n] for n in N_ITERS]
        xs = np.array([x_pos[n] for n in N_ITERS], dtype=float)
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
        q10 = np.array([r["q10"] for r in rows], dtype=float)
        q90 = np.array([r["q90"] for r in rows], dtype=float)
        ax.fill_between(xs, q10, q90, color=color, alpha=0.15, linewidth=0)

    ax.set_xticks(list(range(len(N_ITERS))))
    ax.set_xticklabels([str(n) for n in N_ITERS])
    ax.set_xlim(-0.3, len(N_ITERS) - 0.7)
    ax.set_xlabel("Number of evaluations (n_iter)")
    ax.set_ylabel(score_ylabel(cfg["scoring"]))
    ax.set_title(
        f"{cfg['problem_id']}  |  {n_hp} tuned hparams  |  "
        f"n_iter ∈ {list(N_ITERS)}; ≥{MIN_REPEATS} seeds / ≤{wall_budget_s:.0f}s per budget",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="lower right")

    for method, color in (("uniform", "#1f4e79"), ("lhs", "#c45c26")):
        rows_by_n = {r["n_iter"]: r for r in by_method[method]}
        for n in N_ITERS:
            r = rows_by_n[n]
            ax.annotate(
                f"n={r['n_repeats']}",
                (x_pos[n], r["mean"]),
                textcoords="offset points",
                xytext=(0, 8 if method == "uniform" else -14),
                ha="center",
                fontsize=7,
                color=color,
            )

    fig.text(
        0.08,
        0.02,
        format_search_space(specs),
        ha="left",
        va="bottom",
        family="monospace",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#cccccc"),
    )
    for d in (FIGDIR, ARTDIR):
        fig.savefig(d / f"{out_stem}.png", dpi=150)
    plt.close(fig)


def winners_markdown(row: dict[str, Any], specs: list[ParamSpec]) -> list[str]:
    lines = [
        f"#### `{row['method']}` @ n_iter={row['n_iter']} "
        f"(n={row['n_repeats']}, wall={row['wall_s']:.2f}s)",
        "",
        f"- Mean±std score: **{row['mean']:.5g} ± {row['std']:.3g}**",
        f"- Best repeat: seed={row['best_repeat']['seed']}, "
        f"score={row['best_repeat']['score']:.5g}, "
        f"`{format_params(row['best_repeat']['params'])}`",
        "",
        "Winning-param summary across repeats:",
        "",
    ]
    for name, stats in row["winner_param_summary"].items():
        if stats["kind"] == "choice":
            lines.append(f"- `{name}`: mode={stats['mode']} counts={stats['counts']}")
        else:
            lines.append(
                f"- `{name}`: median={stats['median']:.5g} "
                f"[q10={stats['q10']:.5g}, q90={stats['q90']:.5g}]"
            )
    lines.append("")
    lines.append("| seed | score | winning params |")
    lines.append("|-----:|------:|----------------|")
    shown = row["winners"]
    truncated = False
    if len(shown) > MD_WINNER_LIST_CAP:
        # show best 10 + worst note
        order = sorted(range(len(shown)), key=lambda i: shown[i]["score"], reverse=True)
        keep = order[:MD_WINNER_LIST_CAP]
        shown = [row["winners"][i] for i in sorted(keep)]
        truncated = True
    for w in shown:
        lines.append(
            f"| {w['seed']} | {w['score']:.5g} | `{format_params(w['params'])}` |"
        )
    if truncated:
        lines.append("")
        lines.append(
            f"_Showing {MD_WINNER_LIST_CAP}/{row['n_repeats']} repeats "
            f"(highest scores). Full list in JSON._"
        )
    lines.append("")
    return lines


def rebuild_markdown(all_summary: list[dict[str, Any]]) -> None:
    """Write NEVAL_BUDGETS.md from summary + winners JSON files."""
    md: list[str] = [
        "# Best score vs number of evaluations",
        "",
        f"_Generated {utc_now()}_",
        "",
        f"Budgets: `n_iter ∈ {list(N_ITERS)}`. For each budget/method, as many "
        f"seed repeats as fit in **{WALL_BUDGET_S:.0f}s** wall time "
        f"(always **≥{MIN_REPEATS}**). Points: mean ± std; bands: 10th–90th "
        f"percentiles. X-axis ticks are exactly n_iter ∈ {list(N_ITERS)} "
        f"(equally spaced). Winning hyperparameter combos are listed per repeat.",
        "",
        "`poly_reg_spline_nystroem` follows the SplineTransformer → Nystroem → "
        "RidgeCV setup from "
        "[poly_reg_array_api.ipynb](https://github.com/ogrisel/notebooks/blob/master/poly_reg_array_api.ipynb) "
        "(Array API / GPU steps omitted; data subsampled to 2500 rows).",
        "",
    ]
    for rec in all_summary:
        md.append(f"## `{rec['problem_id']}` — {rec['n_hparams']} tuned hparams")
        md.append("")
        md.append(f"![{rec['figure']}](figures_neval/{rec['figure']})")
        md.append("")
        md.append("### Search space")
        md.append("")
        md.append("```")
        for s in rec["search_space"]:
            md.append(s)
        md.append("```")
        md.append("")
        md.append("| n_iter | Uniform mean±std (n) | LHS mean±std (n) |")
        md.append("|------:|----------------------:|-----------------:|")
        u_by = {r["n_iter"]: r for r in rec["uniform"]}
        l_by = {r["n_iter"]: r for r in rec["lhs"]}
        for n in N_ITERS:
            u, l = u_by[n], l_by[n]
            md.append(
                f"| {n} | {u['mean']:.5g}±{u['std']:.3g} (n={u['n_repeats']}) | "
                f"{l['mean']:.5g}±{l['std']:.3g} (n={l['n_repeats']}) |"
            )
        md.append("")
        md.append("### Winning hyperparameter combos")
        md.append("")
        for n in N_ITERS:
            md.append(f"### n_iter = {n}")
            md.append("")
            for method in ("uniform", "lhs"):
                row = u_by[n] if method == "uniform" else l_by[n]
                # Prefer full winners file when present
                wfile = RESULTS / row.get(
                    "winners_file", f"winners/{rec['problem_id']}__{method}__n{n}.json"
                )
                if wfile.exists():
                    full = json.loads(wfile.read_text())
                    md.extend(winners_markdown(full, []))
                else:
                    # Fallback compact row (no per-repeat list)
                    md.append(
                        f"#### `{method}` @ n_iter={n} (n={row['n_repeats']})\n"
                    )
                    md.append(
                        f"- Mean±std: **{row['mean']:.5g} ± {row['std']:.3g}**\n"
                    )
                    md.append(
                        f"- Best repeat: `{format_params(row['best_repeat']['params'])}` "
                        f"(score={row['best_repeat']['score']:.5g})\n"
                    )
                    md.append("")
    md.extend(
        [
            "Full per-repeat winner JSON: `experiments/lhs_vs_uniform/results/winners/`.",
            "",
            "This pull request includes code written with the assistance of AI.",
            "The code has **not yet been reviewed** by a human.",
            "",
        ]
    )
    (REPORTS / "NEVAL_BUDGETS.md").write_text("\n".join(md), encoding="utf-8")


def run_one_problem(
    cfg: dict[str, Any], *, wall_budget_s: float | None = None
) -> dict[str, Any]:
    pid = cfg["problem_id"]
    specs = list(cfg["specs"])
    budget = float(cfg.get("wall_budget_s", WALL_BUDGET_S) if wall_budget_s is None else wall_budget_s)
    print(f"\n=== {pid} ({len(specs)} hparams, wall_budget={budget:.0f}s) ===", flush=True)
    X, y = cfg["load_xy"]()
    by_method: dict[str, list[dict]] = {"uniform": [], "lhs": []}
    for n_iter in N_ITERS:
        for method in ("uniform", "lhs"):
            print(f"  n_iter={n_iter} {method}...", flush=True)
            row = repeats_for_budget(
                cfg, X, y, specs, method, n_iter, wall_budget_s=budget
            )
            by_method[method].append(row)
            wpath = WINNERS_DIR / f"{pid}__{method}__n{n_iter}.json"
            wpath.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(
                f"    repeats={row['n_repeats']} wall={row['wall_s']:.2f}s "
                f"mean={row['mean']:.5g} ± {row['std']:.3g} | "
                f"best={format_params(row['best_repeat']['params'])}",
                flush=True,
            )

    stem = f"neval_{pid}"
    plot_problem(cfg, specs, by_method, stem, wall_budget_s=budget)

    def compact(rows):
        out = []
        for r in rows:
            out.append(
                {
                    "n_iter": r["n_iter"],
                    "method": r["method"],
                    "n_repeats": r["n_repeats"],
                    "wall_s": r["wall_s"],
                    "mean": r["mean"],
                    "std": r["std"],
                    "median": r["median"],
                    "q10": r["q10"],
                    "q90": r["q90"],
                    "best_repeat": r["best_repeat"],
                    "winner_param_summary": r["winner_param_summary"],
                    "winners_file": f"winners/{pid}__{r['method']}__n{r['n_iter']}.json",
                }
            )
        return out

    rec = {
        "problem_id": pid,
        "n_hparams": len(specs),
        "search_space": [s.format() for s in specs],
        "figure": f"{stem}.png",
        "uniform": compact(by_method["uniform"]),
        "lhs": compact(by_method["lhs"]),
    }
    print(f"  Wrote {stem}.png", flush=True)
    return rec


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

    for cfg in experiment_problems():
        pid = cfg["problem_id"]
        if pid in done:
            print(f"\n=== {pid}: skip (already complete) ===", flush=True)
            continue
        rec = run_one_problem(cfg)
        all_summary = [r for r in all_summary if r["problem_id"] != pid]
        all_summary.append(rec)
        # Keep registry order
        order = {c["problem_id"]: i for i, c in enumerate(experiment_problems())}
        all_summary.sort(key=lambda r: order.get(r["problem_id"], 999))
        summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

    rebuild_markdown(all_summary)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
