#!/usr/bin/env python3
"""Anytime Pareto plots with multi-seed + evaluation-order shuffle uncertainty bands.

For each problem:
  - sample 30 candidates per method per seed
  - evaluate each candidate once (score + fit time)
  - simulate many cumulative runs by shuffling evaluation order
  - plot mean best-so-far vs cumulative time with percentile bands
  - annotate search space and n_hparams
  - detect boundary optima and widen/re-run when meaningful
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone

from evaluate import evaluate_candidates
from problems import (
    Problem,
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

FIGDIR = REPORTS / "figures_bands"
ARTDIR = Path("/opt/cursor/artifacts")
FIGDIR.mkdir(parents=True, exist_ok=True)
ARTDIR.mkdir(parents=True, exist_ok=True)

N_CANDIDATES = 30
N_SEEDS = 10
N_SHUFFLES = 40
CV = 3
BAND_LO, BAND_HI = 10, 90  # percentile band


# --- Search spaces (log scales for multiplicative hparams) -------------------
# Intentionally wider than the first study pass; boundary check may widen more.

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

RIDGE_SPECS = [
    ParamSpec("alpha", "loguniform", low=1e-4, high=1e4),
]

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


def plot_problems() -> list[dict[str, Any]]:
    """Problems used for banded Pareto figures (n_iter fixed to 30)."""
    return [
        {
            "problem_id": "diabetes_ridge",
            "task": "regression",
            "scoring": "neg_mean_squared_error",
            "load_xy": _diabetes,
            "make_estimator": make_ridge,
            "specs": list(RIDGE_SPECS),
            "kind": "estimator",
        },
        {
            "problem_id": "diabetes_hgb",
            "task": "regression",
            "scoring": "neg_mean_squared_error",
            "load_xy": _diabetes,
            "make_estimator": make_hgb_reg,
            "specs": list(HGB_REG_SPECS),
            "kind": "estimator",
        },
        {
            "problem_id": "breast_cancer_logreg",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _breast_cancer,
            "make_estimator": make_logreg,
            "specs": list(LOGREG_SPECS),
            "kind": "estimator",
        },
        {
            "problem_id": "breast_cancer_hgb",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _breast_cancer,
            "make_estimator": make_hgb_clf,
            "specs": list(HGB_CLF_SPECS),
            "kind": "estimator",
        },
        {
            "problem_id": "digits_logreg",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _digits_multiclass,
            "make_estimator": make_logreg,
            "specs": list(LOGREG_SPECS),
            "kind": "estimator",
        },
        {
            "problem_id": "synth_mixed_hgb",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _synthetic_mixed_clf,
            "model": "hgb",
            "specs": list(PIPE_HGB_SPECS),
            "kind": "pipeline",
        },
        {
            "problem_id": "synth_mixed_logreg",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _synthetic_mixed_clf,
            "model": "logreg",
            "specs": list(PIPE_LOGREG_SPECS),
            "kind": "pipeline",
        },
        {
            "problem_id": "credit_g_hgb",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _openml_credit_g,
            "model": "hgb",
            "specs": list(PIPE_HGB_SPECS),
            "kind": "pipeline",
        },
    ]


def score_ylabel(scoring: str) -> str:
    return {
        "neg_log_loss": "Best CV neg_log_loss (higher better)",
        "neg_brier_score": "Best CV neg_brier_score (higher better)",
        "neg_mean_squared_error": "Best CV neg_MSE (higher better)",
    }.get(scoring, f"Best CV {scoring} (higher better)")


def format_search_space(specs: list[ParamSpec]) -> str:
    return "Search space (" + f"{len(specs)} hparams):\n" + "\n".join(
        f"  • {s.format()}" for s in specs
    )


def evaluate_pool(cfg, X, y, specs, method: str, seed: int):
    """Evaluate a fixed pool of N_CANDIDATES; return scores, times, params."""
    method_seed = seed if method == "uniform" else seed + 10_000
    candidates = sample_search_space(specs, N_CANDIDATES, method, method_seed)
    if cfg["kind"] == "pipeline":
        est = build_mixed_pipeline(X, model=cfg["model"])
    else:
        est = cfg["make_estimator"]()
    run = evaluate_candidates(
        est,
        X,
        y,
        candidates,
        scoring=cfg["scoring"],
        cv=CV,
        method=method,
        problem_id=cfg["problem_id"],
        seed=seed,
    )
    scores = np.asarray(run.scores, dtype=float)
    times = np.asarray([t.fit_time for t in run.trials], dtype=float)
    params = [t.params for t in run.trials]
    # Replace non-finite scores with very poor value for accumulate
    scores = np.where(np.isfinite(scores), scores, -1e300)
    return scores, times, params


def simulate_shuffled_curves(
    scores: np.ndarray,
    times: np.ndarray,
    *,
    n_shuffles: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (cum_time, best_so_far) for shuffled evaluation orders."""
    n = len(scores)
    out = []
    for _ in range(n_shuffles):
        order = rng.permutation(n)
        cum = np.cumsum(times[order])
        best = np.maximum.accumulate(scores[order])
        out.append((cum, best))
    return out


def interpolate_step(cum_t: np.ndarray, best: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    vals = np.empty_like(t_grid)
    for i, t in enumerate(t_grid):
        idx = np.searchsorted(cum_t, t, side="right") - 1
        if idx < 0:
            # before first completion: no finished eval yet
            vals[i] = np.nan
        else:
            vals[i] = best[idx]
    return vals


def aggregate_curves(
    all_curves: list[tuple[np.ndarray, np.ndarray]],
    t_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    mat = []
    for cum, best in all_curves:
        mat.append(interpolate_step(cum, best, t_grid))
    mat = np.asarray(mat, dtype=float)
    # Forward-fill leading nans per row with first finite (or leave nan)
    for row in mat:
        finite = np.where(np.isfinite(row))[0]
        if finite.size:
            row[: finite[0]] = row[finite[0]]
    return {
        "mean": np.nanmean(mat, axis=0),
        "median": np.nanmedian(mat, axis=0),
        "lo": np.nanpercentile(mat, BAND_LO, axis=0),
        "hi": np.nanpercentile(mat, BAND_HI, axis=0),
        "n_curves": mat.shape[0],
    }


def collect_best_params(params_list, scores_list) -> list[dict[str, Any]]:
    bests = []
    for params, scores in zip(params_list, scores_list):
        i = int(np.argmax(scores))
        bests.append(params[i])
    return bests


def detect_boundary_widenings(
    specs: list[ParamSpec],
    best_params: list[dict[str, Any]],
    *,
    frac_hits: float = 0.3,
) -> list[ParamSpec] | None:
    """If many bests hug a bound that can be extended, return widened specs.

    Skips further widening when a log-bound is already extremely wide (saturation),
    e.g. regularization ``C``/``alpha`` ≳ 1e4 where the model is effectively
    unregularized — extending further is not meaningful.
    """
    if not best_params:
        return None
    new_specs = list(specs)
    changed = False
    n = len(best_params)
    for j, spec in enumerate(specs):
        if spec.kind == "choice":
            continue
        # Natural [0, 1] ratio bounds
        if spec.name.endswith("l1_ratio") and spec.low == 0.0 and spec.high == 1.0:
            continue
        vals = [bp[spec.name] for bp in best_params if spec.name in bp]
        if not vals:
            continue
        low_hits = sum(spec.near_low_boundary(v) for v in vals) / n
        high_hits = sum(spec.near_high_boundary(v) for v in vals) / n
        side = None
        if low_hits >= frac_hits and high_hits >= frac_hits:
            side = "both"
        elif low_hits >= frac_hits:
            side = "low"
        elif high_hits >= frac_hits:
            side = "high"
        if side is None:
            continue

        # Saturation guards: do not extend meaningless extremes
        short = spec.name.split("__")[-1]
        if spec.kind == "loguniform":
            if side in ("high", "both") and short in {"C", "alpha"} and spec.high >= 1e4:
                print(
                    f"  BOUNDARY(sat): {spec.name} prefers high end "
                    f"(≥{spec.high:g}); treating as effectively unregularized — not widening",
                    flush=True,
                )
                continue
            if side in ("low", "both") and short in {"C", "alpha"} and spec.low <= 1e-6:
                print(
                    f"  BOUNDARY(sat): {spec.name} prefers low end (≤{spec.low:g}) — not widening",
                    flush=True,
                )
                continue
            if side in ("low", "both") and "learning_rate" in short and spec.low <= 1e-4:
                print(
                    f"  BOUNDARY(sat): {spec.name} low end already ≤{spec.low:g} — not widening",
                    flush=True,
                )
                continue
            if side in ("high", "both") and "learning_rate" in short and spec.high >= 1.0:
                # LR > 1 is rarely meaningful for HGB
                print(
                    f"  BOUNDARY(sat): {spec.name} high end already ≥{spec.high:g} — not widening",
                    flush=True,
                )
                continue
        if spec.kind == "int" and side in ("high", "both"):
            if "max_depth" in short and spec.high >= 32:
                print(f"  BOUNDARY(sat): {spec.name} depth≥{int(spec.high)} — not widening", flush=True)
                continue
            if "max_leaf_nodes" in short and spec.high >= 255:
                print(f"  BOUNDARY(sat): {spec.name} leaves≥{int(spec.high)} — not widening", flush=True)
                continue

        widened = spec.widened(side=side)
        if widened != spec:
            print(
                f"  BOUNDARY: {spec.name} hits {side} "
                f"(low={low_hits:.0%}, high={high_hits:.0%}) → {widened.format()}",
                flush=True,
            )
            new_specs[j] = widened
            changed = True
    return new_specs if changed else None


def run_experiment(cfg: dict[str, Any], specs: list[ParamSpec], seeds: list[int]):
    X, y = cfg["load_xy"]()
    pools = {"uniform": [], "lhs": []}  # list of (scores, times, params) per seed
    for seed in seeds:
        print(f"  seed {seed}...", flush=True)
        for method in ("uniform", "lhs"):
            scores, times, params = evaluate_pool(cfg, X, y, specs, method, seed)
            pools[method].append((scores, times, params))
    return pools


def build_band_stats(pools, method: str, t_grid: np.ndarray, base_seed: int = 0):
    curves = []
    best_params = []
    for i, (scores, times, params) in enumerate(pools[method]):
        rng = np.random.default_rng(base_seed + 1000 * i + (0 if method == "uniform" else 1))
        curves.extend(simulate_shuffled_curves(scores, times, n_shuffles=N_SHUFFLES, rng=rng))
        best_params.append(params[int(np.argmax(scores))])
    stats = aggregate_curves(curves, t_grid)
    return stats, best_params, curves


def choose_time_grid(pools) -> np.ndarray:
    totals = []
    for method in ("uniform", "lhs"):
        for scores, times, params in pools[method]:
            totals.append(float(np.sum(times)))
    t_max = float(np.percentile(totals, 90))
    t_max = max(t_max, 1e-6)
    # denser early grid helps anytime comparison
    return np.unique(
        np.concatenate(
            [
                np.linspace(0, t_max * 0.2, 40),
                np.linspace(t_max * 0.2, t_max, 60),
            ]
        )
    )


def plot_experiment(
    cfg: dict[str, Any],
    specs: list[ParamSpec],
    stats_u,
    stats_l,
    t_grid: np.ndarray,
    *,
    out_stem: str,
    widened_note: str = "",
):
    n_hp = len(specs)
    ylabel = score_ylabel(cfg["scoring"])
    title = (
        f"{cfg['problem_id']}  |  {n_hp} tuned hparams  |  "
        f"{N_CANDIDATES} candidates, {N_SEEDS} seeds × {N_SHUFFLES} order-shuffles"
    )
    space_txt = format_search_space(specs)
    if widened_note:
        space_txt += f"\n{widened_note}"

    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_axes([0.08, 0.38, 0.88, 0.52])

    for stats, label, color in (
        (stats_u, "Uniform", "#1f4e79"),
        (stats_l, "LHS", "#c45c26"),
    ):
        ax.plot(t_grid, stats["mean"], color=color, lw=2.2, label=label)
        ax.fill_between(t_grid, stats["lo"], stats["hi"], color=color, alpha=0.22, linewidth=0)

    ax.set_xlabel("Cumulative fit+CV time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")

    # Log x if span is wide enough
    positive = t_grid[t_grid > 0]
    if positive.size and (positive.max() / max(positive.min(), 1e-6) >= 30):
        ax.set_xscale("log")
        ax.set_xlim(left=max(positive.min(), t_grid[1] if len(t_grid) > 1 else positive.min()))

    fig.text(
        0.08,
        0.02,
        space_txt,
        ha="left",
        va="bottom",
        family="monospace",
        fontsize=8.5,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#cccccc"),
    )
    for d in (FIGDIR, ARTDIR):
        fig.savefig(d / f"{out_stem}.png", dpi=150)
    plt.close(fig)


def main():
    print(f"Banded Pareto plots — {utc_now()}", flush=True)
    seeds = list(range(N_SEEDS))
    summary_path = RESULTS / "banded_pareto_summary.json"
    summary = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = []
    done = {r["problem_id"] for r in summary if (FIGDIR / r["figure"]).exists()}

    for cfg in plot_problems():
        pid = cfg["problem_id"]
        if pid in done:
            print(f"\n=== {pid}: skip (already in summary) ===", flush=True)
            continue
        specs = list(cfg["specs"])
        print(f"\n=== {pid} ({len(specs)} hparams) ===", flush=True)

        # Pass 1
        pools = run_experiment(cfg, specs, seeds)
        # Boundary check using best-of-pool params from both methods
        bests = []
        for method in ("uniform", "lhs"):
            for scores, times, params in pools[method]:
                bests.append(params[int(np.argmax(scores))])
        widened = detect_boundary_widenings(specs, bests)
        note = ""
        if widened is not None:
            print(f"  Re-running {pid} with widened search space...", flush=True)
            specs = widened
            pools = run_experiment(cfg, specs, seeds)
            note = "(search space widened after boundary hits)"
            # Second boundary check — widen once more if still hitting
            bests2 = []
            for method in ("uniform", "lhs"):
                for scores, times, params in pools[method]:
                    bests2.append(params[int(np.argmax(scores))])
            widened2 = detect_boundary_widenings(specs, bests2)
            if widened2 is not None:
                print(f"  Second widening for {pid}...", flush=True)
                specs = widened2
                pools = run_experiment(cfg, specs, seeds)
                note = "(search space widened twice after boundary hits)"

        t_grid = choose_time_grid(pools)
        stats_u, best_u, _ = build_band_stats(pools, "uniform", t_grid)
        stats_l, best_l, _ = build_band_stats(pools, "lhs", t_grid)

        stem = f"bands_{pid}"
        plot_experiment(
            cfg,
            specs,
            stats_u,
            stats_l,
            t_grid,
            out_stem=stem,
            widened_note=note,
        )

        # Save compact JSON for review
        rec = {
            "problem_id": pid,
            "n_hparams": len(specs),
            "search_space": [s.format() for s in specs],
            "widened_note": note,
            "final_mean_uniform": float(stats_u["mean"][-1]),
            "final_mean_lhs": float(stats_l["mean"][-1]),
            "n_curves_each": int(stats_u["n_curves"]),
            "figure": f"{stem}.png",
        }
        summary.append(rec)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"  Wrote {stem}.png | final mean U={rec['final_mean_uniform']:.5g} "
            f"LHS={rec['final_mean_lhs']:.5g}",
            flush=True,
        )

    # Markdown gallery
    lines = [
        "# Banded anytime Pareto plots",
        "",
        f"_Generated {utc_now()}_",
        "",
        f"Protocol: **{N_CANDIDATES} candidates**/method/seed, **{N_SEEDS} seeds**, "
        f"**{N_SHUFFLES} evaluation-order shuffles**/seed → uncertainty bands "
        f"({BAND_LO}–{BAND_HI}th percentile) on cumulative time vs best score so far.",
        "",
        "Search spaces use **log-uniform** sampling for multiplicative hyperparameters "
        "(`C`, `alpha`, `learning_rate`, `l2_regularization`). Spaces were widened when "
        "best configs repeatedly hit extendable boundaries (with saturation guards for "
        "effectively unregularized `C`/`alpha` and extreme learning rates).",
        "",
    ]
    for rec in summary:
        lines.append(f"## `{rec['problem_id']}` — {rec['n_hparams']} tuned hparams")
        lines.append("")
        lines.append(f"![{rec['figure']}](figures_bands/{rec['figure']})")
        lines.append("")
        lines.append("```")
        for s in rec["search_space"]:
            lines.append(s)
        if rec["widened_note"]:
            lines.append(rec["widened_note"])
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "This pull request includes code written with the assistance of AI.",
            "The code has **not yet been reviewed** by a human.",
            "",
        ]
    )
    (REPORTS / "PARETO_BANDS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
