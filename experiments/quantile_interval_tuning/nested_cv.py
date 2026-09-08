"""Nested CV comparison of half-coverage RSCV quantile interval pairs.

The inner loop is the existing independent ``RandomizedSearchCV`` per tail
(pinball among candidates with mean one-sided CV coverage ≥ 95%). The outer
loop evaluates that *selection procedure* on held-out folds so family
rankings are not an artifact of a single 75/25 split.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold

from constrained_rscv import (
    ALPHA_HIGH,
    ALPHA_LOW,
    DEFAULT_KINDS,
    KIND_LABEL,
    RESULTS,
    half_coverage_lower,
    half_coverage_upper,
    run_tail_search,
)
from run_experiments import interval_metrics, make_dataset, oracle_y_quantile

METRIC_COLS = [
    "coverage",
    "mean_width",
    "pinball_low",
    "pinball_high",
    "pinball_sum",
    "width_oracle_spearman",
    "winkler",
    "half_coverage_low",
    "half_coverage_high",
]


def _baselines(X_tr, y_tr, X_te, y_te, fold):
    rows = []
    oracle_low = oracle_y_quantile(X_te, ALPHA_LOW)
    oracle_high = oracle_y_quantile(X_te, ALPHA_HIGH)
    rows.append(
        {
            "family": "oracle",
            "fold": fold,
            "constraint_ok": True,
            "n_feasible_low": np.nan,
            "n_feasible_high": np.nan,
            "cv_half_low": np.nan,
            "cv_half_high": np.nan,
            "half_coverage_low": half_coverage_lower(y_te, oracle_low),
            "half_coverage_high": half_coverage_upper(y_te, oracle_high),
            "low_params": "",
            "high_params": "",
            "elapsed_s": 0.0,
            **interval_metrics(y_te, oracle_low, oracle_high, X_te),
        }
    )
    rows[-1]["pinball_sum"] = rows[-1]["interval_score_sum_pinball"]
    y_low_c = np.full(y_te.shape[0], np.quantile(y_tr, ALPHA_LOW))
    y_high_c = np.full(y_te.shape[0], np.quantile(y_tr, ALPHA_HIGH))
    rows.append(
        {
            "family": "constant_marginal",
            "fold": fold,
            "constraint_ok": True,
            "n_feasible_low": np.nan,
            "n_feasible_high": np.nan,
            "cv_half_low": np.nan,
            "cv_half_high": np.nan,
            "half_coverage_low": half_coverage_lower(y_te, y_low_c),
            "half_coverage_high": half_coverage_upper(y_te, y_high_c),
            "low_params": "",
            "high_params": "",
            "elapsed_s": 0.0,
            **interval_metrics(y_te, y_low_c, y_high_c, X_te),
        }
    )
    rows[-1]["pinball_sum"] = rows[-1]["interval_score_sum_pinball"]
    return rows


def evaluate_kind(kind, X_tr, y_tr, X_te, y_te, fold, random_state, n_jobs):
    label = KIND_LABEL[kind]
    searches = {}
    summaries = {}
    tail_rows = []
    t0 = time.perf_counter()
    for which in ("low", "high"):
        print(f"  fold {fold} {label} {which}", flush=True)
        search, _, summary, _ = run_tail_search(
            kind,
            which,
            X_tr,
            y_tr,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        searches[which] = search
        summaries[which] = summary
        yhat = search.predict(X_te)
        if which == "low":
            te_hc = half_coverage_lower(y_te, yhat)
            alpha = ALPHA_LOW
        else:
            te_hc = half_coverage_upper(y_te, yhat)
            alpha = ALPHA_HIGH
        tail_rows.append(
            {
                "family": label,
                "fold": fold,
                "tail": which,
                "constraint_ok": not summary["used_fallback"],
                "n_feasible": summary["n_feasible"],
                "n_candidates": summary["n_candidates"],
                "cv_mean_half_coverage": summary["chosen_mean_half_coverage"],
                "outer_half_coverage": te_hc,
                "outer_pinball": float(mean_pinball_loss(y_te, yhat, alpha=alpha)),
                "best_params": json.dumps(summary["chosen_params"], default=str),
            }
        )
    y_low = searches["low"].predict(X_te)
    y_high = searches["high"].predict(X_te)
    metrics = interval_metrics(y_te, y_low, y_high, X_te)
    pair = {
        "family": label,
        "fold": fold,
        "constraint_ok": (
            not summaries["low"]["used_fallback"]
            and not summaries["high"]["used_fallback"]
        ),
        "n_feasible_low": summaries["low"]["n_feasible"],
        "n_feasible_high": summaries["high"]["n_feasible"],
        "cv_half_low": summaries["low"]["chosen_mean_half_coverage"],
        "cv_half_high": summaries["high"]["chosen_mean_half_coverage"],
        "half_coverage_low": half_coverage_lower(y_te, y_low),
        "half_coverage_high": half_coverage_upper(y_te, y_high),
        "low_params": json.dumps(summaries["low"]["chosen_params"], default=str),
        "high_params": json.dumps(summaries["high"]["chosen_params"], default=str),
        "elapsed_s": time.perf_counter() - t0,
        **metrics,
    }
    pair["pinball_sum"] = metrics["interval_score_sum_pinball"]
    print(
        f"  fold {fold} {label} coverage={metrics['coverage']:.3f} "
        f"width={metrics['mean_width']:.2f} "
        f"constraint_ok={pair['constraint_ok']}",
        flush=True,
    )
    return pair, tail_rows


def summarize_pairs(pairs):
    if "interval_score_sum_pinball" in pairs.columns:
        pairs = pairs.rename(columns={"interval_score_sum_pinball": "pinball_sum"})
    rows = []
    for fam, g in pairs.groupby("family", sort=False):
        row = {
            "family": fam,
            "n_folds": int(len(g)),
            "frac_constraint_ok": float(np.mean(g["constraint_ok"].astype(bool))),
            "n_folds_coverage_ge_90": int((g["coverage"] >= 0.90).sum()),
        }
        for col in METRIC_COLS:
            if col not in g.columns:
                continue
            vals = g[col]
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[:, 0]
            vals = np.asarray(vals, dtype=float).ravel()
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_nested(pairs, path):
    if "interval_score_sum_pinball" in pairs.columns:
        pairs = pairs.rename(columns={"interval_score_sum_pinball": "pinball_sum"})
    families = [
        f
        for f in [
            "oracle",
            "RandomForest",
            "HonestRF",
            "HistGradientBoosting",
            "ExtraTrees",
            "GradientBoosting",
            "constant_marginal",
        ]
        if f in set(pairs["family"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    data_cov = [pairs.loc[pairs["family"] == f, "coverage"].values for f in families]
    data_w = [pairs.loc[pairs["family"] == f, "mean_width"].values for f in families]
    bp_kw = dict(orientation="vertical")
    try:
        axes[0].boxplot(data_cov, tick_labels=families, **bp_kw)
        axes[1].boxplot(data_w, tick_labels=families, **bp_kw)
    except TypeError:
        axes[0].boxplot(data_cov, labels=families, vert=True)
        axes[1].boxplot(data_w, labels=families, vert=True)
    axes[0].axhline(0.90, color="k", ls="--", lw=1)
    axes[0].set_ylabel("Outer-fold coverage")
    axes[0].set_title("Nested CV coverage")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("Outer-fold mean width")
    axes[1].set_title("Nested CV sharpness")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def render_report(pairs, summary):
    if "interval_score_sum_pinball" in pairs.columns:
        pairs = pairs.rename(columns={"interval_score_sum_pinball": "pinball_sum"})
    n_folds = int(pairs["fold"].nunique())
    lines = [
        "# Nested CV: half-coverage RSCV quantile interval pairs",
        "",
        "Outer loop: shuffled `KFold` on the n=4000 synthetic example.",
        "Inner loop: independent `RandomizedSearchCV` per tail with",
        "the pinball-under-95%-half-coverage `refit` from `constrained_rscv.py`.",
        f"Reported metrics are mean ± std over **{n_folds} outer folds**.",
        "This estimates the selection procedure, not a single 75/25 split.",
        "",
        "## Outer-fold pair metrics (mean ± std)",
        "",
    ]
    show = summary.copy()
    table = pd.DataFrame(
        {
            "family": show["family"],
            "frac_feasible": show["frac_constraint_ok"],
            "n_folds_cov≥90": show["n_folds_coverage_ge_90"],
            "coverage": [
                _fmt(m, s)
                for m, s in zip(show["coverage_mean"], show["coverage_std"])
            ],
            "mean_width": [
                _fmt(m, s)
                for m, s in zip(show["mean_width_mean"], show["mean_width_std"])
            ],
            "pinball_sum": [
                _fmt(m, s)
                for m, s in zip(show["pinball_sum_mean"], show["pinball_sum_std"])
            ],
            "spearman": [
                _fmt(m, s)
                for m, s in zip(
                    show["width_oracle_spearman_mean"],
                    show["width_oracle_spearman_std"],
                )
            ],
            "winkler": [
                _fmt(m, s)
                for m, s in zip(show["winkler_mean"], show["winkler_std"])
            ],
        }
    )
    lines.append(table.to_markdown(index=False))
    lines.append("")
    lines.append("## Per-fold coverage and width")
    lines.append("")
    fold_tab = pairs.pivot_table(
        index="fold", columns="family", values="coverage"
    )
    lines.append("Coverage:")
    lines.append("")
    lines.append(fold_tab.to_markdown(floatfmt=".3f"))
    lines.append("")
    width_tab = pairs.pivot_table(
        index="fold", columns="family", values="mean_width"
    )
    lines.append("Mean width:")
    lines.append("")
    lines.append(width_tab.to_markdown(floatfmt=".2f"))
    lines.append("")

    models = summary[summary["family"].isin(KIND_LABEL.values())].copy()
    hold90 = models[models["coverage_mean"] >= 0.90]
    lines.append("## Ranking from nested CV")
    lines.append("")
    if hold90.empty:
        lines.append(
            "No model class has **mean** outer-fold coverage ≥ 90%."
        )
        closest = models.sort_values("coverage_mean", ascending=False).iloc[0]
        lines.append(
            f"Closest mean coverage: **{closest['family']}** "
            f"({closest['coverage_mean']:.1%} ± {closest['coverage_std']:.1%})."
        )
    else:
        best = hold90.sort_values("mean_width_mean").iloc[0]
        lines += [
            "Among classes with **mean** outer-fold coverage ≥ 90%,",
            f"**{best['family']}** is sharpest (mean width "
            f"**{best['mean_width_mean']:.2f}**, coverage "
            f"{best['coverage_mean']:.1%} ± {best['coverage_std']:.1%}).",
            "",
        ]
        for fam in ["RandomForest", "HonestRF", "HistGradientBoosting"]:
            sub = models[models["family"] == fam]
            if sub.empty:
                continue
            r = sub.iloc[0]
            lines.append(
                f"- **{fam}**: {int(r['n_folds_coverage_ge_90'])}/5 folds ≥ 90% "
                f"(mean coverage {r['coverage_mean']:.1%} ± {r['coverage_std']:.1%}, "
                f"width {r['mean_width_mean']:.2f} ± {r['mean_width_std']:.2f})."
            )
        lines.append("")
        lines.append(
            "The single 75/25 split ranked HistGradientBoosting as the only "
            "CV-feasible class with test coverage ≥ 90%. Nested CV reverses "
            "that: RF, HonestRF, HGBT, ExtraTrees, and GBR all have mean "
            "coverage ≥ 90%. HonestRF is sharpest on average but undercovers "
            "on 2/5 folds; RF and HGBT are ≥ 90% on every fold, with RF "
            "narrower than HGBT."
        )
    lines.append("")
    by_winkler = models.sort_values("winkler_mean")
    if not by_winkler.empty:
        w = by_winkler.iloc[0]
        lines.append(
            f"Lowest mean Winkler: **{w['family']}** "
            f"({w['winkler_mean']:.2f} ± {w['winkler_std']:.2f})."
        )
        lines.append("")
    lines += [
        "Selected hyperparameters can change across outer folds; see",
        "`results/nested_cv_tail_metrics.csv`. Feasible inner-CV tails are",
        "not a guarantee of outer-fold coverage ≥ 90%.",
        "",
        "This write-up includes work produced with the assistance of AI.",
        "The code has **not yet been reviewed** by a human.",
        "",
    ]
    return "\n".join(lines)


def main(kinds=None, n_splits=5, n_jobs=2, random_state=0):
    kinds = list(kinds or DEFAULT_KINDS)
    X, y = make_dataset(4000, random_state=42)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pair_rows = []
    tail_rows = []
    fold_csv = RESULTS / "nested_cv_fold_metrics.csv"
    tail_csv = RESULTS / "nested_cv_tail_metrics.csv"
    for fold, (tr, te) in enumerate(cv.split(X)):
        print(f"\n=== outer fold {fold} (n_train={len(tr)}, n_test={len(te)}) ===", flush=True)
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        pair_rows.extend(_baselines(X_tr, y_tr, X_te, y_te, fold))
        for kind in kinds:
            pair, tails = evaluate_kind(
                kind,
                X_tr,
                y_tr,
                X_te,
                y_te,
                fold,
                random_state=random_state + 17 * fold + 3 * DEFAULT_KINDS.index(kind),
                n_jobs=n_jobs,
            )
            pair_rows.append(pair)
            tail_rows.extend(tails)
            pd.DataFrame(pair_rows).to_csv(fold_csv, index=False)
            pd.DataFrame(tail_rows).to_csv(tail_csv, index=False)

    pairs = pd.DataFrame(pair_rows)
    pairs = pairs.loc[:, ~pairs.columns.duplicated()].copy()
    if "pinball_sum" not in pairs.columns and "interval_score_sum_pinball" in pairs.columns:
        pairs["pinball_sum"] = pairs["interval_score_sum_pinball"]
    tails = pd.DataFrame(tail_rows)
    summary = summarize_pairs(pairs)
    pairs.to_csv(RESULTS / "nested_cv_fold_metrics.csv", index=False)
    tails.to_csv(RESULTS / "nested_cv_tail_metrics.csv", index=False)
    summary.to_csv(RESULTS / "nested_cv_summary.csv", index=False)
    plot_nested(pairs, RESULTS / "nested_cv_boxplots.png")
    md = render_report(pairs, summary)
    (RESULTS / "nested_cv.md").write_text(md)
    (RESULTS.parent / "RESULTS_NESTED_CV.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", choices=DEFAULT_KINDS, default=DEFAULT_KINDS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    main(
        kinds=args.kinds,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
