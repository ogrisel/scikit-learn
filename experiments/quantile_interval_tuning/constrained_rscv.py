"""Independent RSCV per quantile: min pinball s.t. one-sided ("half") coverage.

Pattern: ``RandomizedSearchCV(..., scoring=..., refit=callable)`` as in
``examples/model_selection/plot_grid_search_digits.py``.

Each tail is tuned on its own:

- Lower (α=0.05): keep candidates with mean CV ``P(Y ≥ q̂) ≥ 0.95``, then
  maximize ``neg_pinball`` (α=0.05).
- Upper (α=0.95): keep candidates with mean CV ``P(Y ≤ q̂) ≥ 0.95``, then
  maximize ``neg_pinball`` (α=0.95).

The two selected models are then combined into a 90% interval. If each
one-sided constraint holds, interval coverage is at least 90% in the
Bonferroni / union-bound sense on the CV folds
(``P(Y < q_low) + P(Y > q_high) ≤ 0.05 + 0.05``).
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from run_experiments import (
    ALPHA_HIGH,
    ALPHA_LOW,
    RESULTS,
    interval_metrics,
    make_dataset,
    oracle_y_quantile,
    plot_intervals,
)

RESULTS.mkdir(parents=True, exist_ok=True)
HALF_COVERAGE_FLOOR = 0.95  # one-sided; two tails → 90% interval


def half_coverage_lower(y_true, y_pred):
    """P(Y >= q_low); target >= 0.95 so the lower bound is not too high."""
    return float(np.mean(y_true >= np.ravel(y_pred)))


def half_coverage_upper(y_true, y_pred):
    """P(Y <= q_high); target >= 0.95 so the upper bound is not too low."""
    return float(np.mean(y_true <= np.ravel(y_pred)))


def refit_pinball_under_half_coverage(cv_results, coverage_floor=HALF_COVERAGE_FLOOR):
    """Min pinball among candidates with mean one-sided CV coverage >= floor."""
    df = pd.DataFrame(cv_results)
    fold_cols = [
        c
        for c in df.columns
        if c.startswith("split") and c.endswith("_test_half_coverage")
    ]
    df["_min_fold_half_coverage"] = (
        df[fold_cols].min(axis=1) if fold_cols else df["mean_test_half_coverage"]
    )
    feasible = df["mean_test_half_coverage"] >= coverage_floor
    refit_pinball_under_half_coverage.last_summary = {
        "n_candidates": int(len(df)),
        "n_feasible": int(feasible.sum()),
        "coverage_floor": coverage_floor,
        "used_fallback": bool(not feasible.any()),
    }
    pool = df[feasible] if feasible.any() else df
    best_idx = int(pool["mean_test_neg_pinball"].idxmax())
    refit_pinball_under_half_coverage.last_summary.update(
        {
            "chosen_index": best_idx,
            "chosen_mean_half_coverage": float(df.loc[best_idx, "mean_test_half_coverage"]),
            "chosen_min_fold_half_coverage": float(
                df.loc[best_idx, "_min_fold_half_coverage"]
            ),
            "chosen_mean_neg_pinball": float(df.loc[best_idx, "mean_test_neg_pinball"]),
            "chosen_params": df.loc[best_idx, "params"],
        }
    )
    s = refit_pinball_under_half_coverage.last_summary
    print(
        f"    refit: {s['n_feasible']}/{s['n_candidates']} feasible "
        f"(mean half-coverage >= {coverage_floor:.0%}); "
        f"fallback={s['used_fallback']}; "
        f"chosen half-cov={s['chosen_mean_half_coverage']:.3f} "
        f"params={s['chosen_params']}"
    )
    return best_idx


class HonestQuantileForest(BaseEstimator, RegressorMixin):
    """MSE forest grown on one half; leaf quantiles estimated on the other."""

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=20,
        min_samples_split=2,
        max_features=1.0,
        honest_fraction=0.5,
        quantile=0.5,
        random_state=0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.honest_fraction = honest_fraction
        self.quantile = quantile
        self.random_state = random_state

    def fit(self, X, y):
        X_grow, X_hon, y_grow, y_hon = train_test_split(
            X, y, test_size=self.honest_fraction, random_state=self.random_state
        )
        est = RandomForestRegressor(
            criterion="squared_error",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=1,
        )
        est.fit(X_grow, y_grow)
        leaves = est.apply(X_hon)
        n_trees = leaves.shape[1]
        leaf_q = []
        fallback = float(np.quantile(y_hon, self.quantile))
        for t in range(n_trees):
            ids = leaves[:, t]
            qmap = {}
            for leaf in np.unique(ids):
                qmap[int(leaf)] = float(np.quantile(y_hon[ids == leaf], self.quantile))
            leaf_q.append((qmap, fallback))
        self.estimator_ = est
        self.leaf_quantiles_ = leaf_q
        return self

    def predict(self, X):
        leaves = self.estimator_.apply(X)
        n_samples, n_trees = leaves.shape
        preds = np.empty((n_samples, n_trees), dtype=float)
        for t in range(n_trees):
            qmap, fallback = self.leaf_quantiles_[t]
            preds[:, t] = [qmap.get(int(leaf), fallback) for leaf in leaves[:, t]]
        return preds.mean(axis=1)


def make_estimator(kind, quantile, random_state=0):
    if kind in ("rf", "et"):
        cls = RandomForestRegressor if kind == "rf" else ExtraTreesRegressor
        return cls(
            criterion="quantile",
            quantile=quantile,
            random_state=random_state,
            n_jobs=1,
        )
    if kind == "gbr":
        return GradientBoostingRegressor(
            loss="quantile", alpha=quantile, random_state=random_state
        )
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, random_state=random_state
        )
    if kind == "honest_rf":
        return HonestQuantileForest(quantile=quantile, random_state=random_state)
    raise ValueError(kind)


SEARCH_SPACES = {
    "rf": dict(
        n_estimators=[200],
        min_samples_leaf=[9, 20, 30, 50, 80, 120],
        max_depth=[None, 6, 12],
        min_samples_split=[2, 10],
    ),
    "et": dict(
        n_estimators=[200],
        min_samples_leaf=[9, 20, 40, 80],
        max_depth=[None, 6, 12],
        min_samples_split=[2, 10],
    ),
    "gbr": dict(
        n_estimators=[30, 50, 100, 200, 300],
        learning_rate=[0.05, 0.1],
        max_depth=[2, 3, 5],
        min_samples_leaf=[9, 20, 40],
        subsample=[0.8, 1.0],
    ),
    "hgb": dict(
        max_iter=[50, 100, 200],
        learning_rate=[0.05, 0.1],
        max_depth=[3, 6, None],
        min_samples_leaf=[9, 20, 40],
        l2_regularization=[0.0, 1.0],
    ),
    "honest_rf": dict(
        n_estimators=[200],
        min_samples_leaf=[9, 20, 40, 80],
        max_depth=[None, 6, 12],
        honest_fraction=[0.4, 0.5],
    ),
}

N_ITER = {"rf": 24, "et": 16, "gbr": 40, "hgb": 36, "honest_rf": 16}

KIND_LABEL = {
    "rf": "RandomForest",
    "et": "ExtraTrees",
    "gbr": "GradientBoosting",
    "hgb": "HistGradientBoosting",
    "honest_rf": "HonestRF",
}


def scoring_for_tail(which):
    if which == "low":
        cov = make_scorer(half_coverage_lower)
        pinball = make_scorer(
            mean_pinball_loss, alpha=ALPHA_LOW, greater_is_better=False
        )
    else:
        cov = make_scorer(half_coverage_upper)
        pinball = make_scorer(
            mean_pinball_loss, alpha=ALPHA_HIGH, greater_is_better=False
        )
    return {"half_coverage": cov, "neg_pinball": pinball}


def run_tail_search(kind, which, X_train, y_train, random_state=0):
    quantile = ALPHA_LOW if which == "low" else ALPHA_HIGH
    est = make_estimator(kind, quantile, random_state=random_state)
    n_iter = min(
        N_ITER[kind],
        int(np.prod([len(v) for v in SEARCH_SPACES[kind].values()])),
    )
    search = RandomizedSearchCV(
        est,
        SEARCH_SPACES[kind],
        n_iter=n_iter,
        scoring=scoring_for_tail(which),
        refit=refit_pinball_under_half_coverage,
        cv=3,
        random_state=random_state,
        n_jobs=2,
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    summary = dict(refit_pinball_under_half_coverage.last_summary)
    cv = pd.DataFrame(search.cv_results_)
    cv["family"] = KIND_LABEL[kind]
    cv["tail"] = which
    return search, cv, summary, elapsed


def plot_cv_scatter(all_cv, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, tail, title in [
        (axes[0], "low", r"Lower tail: $P(Y \geq \hat q_{0.05})$ vs pinball"),
        (axes[1], "high", r"Upper tail: $P(Y \leq \hat q_{0.95})$ vs pinball"),
    ]:
        sub = all_cv[all_cv["tail"] == tail]
        for fam, g in sub.groupby("family"):
            ax.scatter(
                -g["mean_test_neg_pinball"],
                g["mean_test_half_coverage"],
                alpha=0.55,
                s=28,
                label=fam,
            )
        ax.axhline(HALF_COVERAGE_FLOOR, color="k", ls="--")
        ax.set_xlabel("CV mean pinball")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("CV mean half-coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    X, y = make_dataset(4000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    x_plot = np.atleast_2d(np.linspace(0, 10, 400)).T

    pair_rows = []
    tail_rows = []
    cv_frames = []

    oracle_low = oracle_y_quantile(X_test, ALPHA_LOW)
    oracle_high = oracle_y_quantile(X_test, ALPHA_HIGH)
    pair_rows.append(
        {
            "family": "oracle",
            "constraint_ok": True,
            **interval_metrics(y_test, oracle_low, oracle_high, X_test),
        }
    )
    y_low_c = np.full_like(y_test, np.quantile(y_train, ALPHA_LOW), dtype=float)
    y_high_c = np.full_like(y_test, np.quantile(y_train, ALPHA_HIGH), dtype=float)
    pair_rows.append(
        {
            "family": "constant_marginal",
            "constraint_ok": True,
            **interval_metrics(y_test, y_low_c, y_high_c, X_test),
        }
    )

    for kind in ["rf", "et", "gbr", "hgb", "honest_rf"]:
        label = KIND_LABEL[kind]
        print(f"\n=== {label} independent half-coverage RSCV ===", flush=True)
        searches = {}
        summaries = {}
        for which in ("low", "high"):
            print(f"  -- {which} --", flush=True)
            search, cv, summary, elapsed = run_tail_search(
                kind, which, X_train, y_train
            )
            searches[which] = search
            summaries[which] = summary
            cv_frames.append(cv)
            yhat_tr = search.predict(X_train)
            yhat_te = search.predict(X_test)
            if which == "low":
                tr_hc = half_coverage_lower(y_train, yhat_tr)
                te_hc = half_coverage_lower(y_test, yhat_te)
                alpha = ALPHA_LOW
            else:
                tr_hc = half_coverage_upper(y_train, yhat_tr)
                te_hc = half_coverage_upper(y_test, yhat_te)
                alpha = ALPHA_HIGH
            tail_rows.append(
                {
                    "family": label,
                    "tail": which,
                    "constraint_ok": not summary["used_fallback"],
                    "n_feasible": summary["n_feasible"],
                    "n_candidates": summary["n_candidates"],
                    "cv_mean_half_coverage": summary["chosen_mean_half_coverage"],
                    "cv_min_fold_half_coverage": summary["chosen_min_fold_half_coverage"],
                    "cv_mean_pinball": float(-summary["chosen_mean_neg_pinball"]),
                    "train_half_coverage": tr_hc,
                    "test_half_coverage": te_hc,
                    "test_pinball": float(
                        mean_pinball_loss(y_test, yhat_te, alpha=alpha)
                    ),
                    "elapsed_s": elapsed,
                    "best_params": json.dumps(summary["chosen_params"], default=str),
                }
            )
            print(
                f"    test half-coverage={te_hc:.3f} "
                f"pinball={mean_pinball_loss(y_test, yhat_te, alpha=alpha):.4f}"
            )

        y_low = searches["low"].predict(X_test)
        y_high = searches["high"].predict(X_test)
        metrics = interval_metrics(y_test, y_low, y_high, X_test)
        pair_rows.append(
            {
                "family": label,
                "constraint_ok": (
                    not summaries["low"]["used_fallback"]
                    and not summaries["high"]["used_fallback"]
                ),
                "n_feasible_low": summaries["low"]["n_feasible"],
                "n_feasible_high": summaries["high"]["n_feasible"],
                "cv_half_low": summaries["low"]["chosen_mean_half_coverage"],
                "cv_half_high": summaries["high"]["chosen_mean_half_coverage"],
                "low_params": json.dumps(summaries["low"]["chosen_params"], default=str),
                "high_params": json.dumps(
                    summaries["high"]["chosen_params"], default=str
                ),
                **metrics,
            }
        )
        print(
            f"  pair test coverage={metrics['coverage']:.3f} "
            f"width={metrics['mean_width']:.2f} "
            f"spearman={metrics['width_oracle_spearman']:.2f}"
        )
        plot_intervals(
            x_plot,
            searches["low"].predict(x_plot),
            searches["high"].predict(x_plot),
            X_test,
            y_test,
            f"{label} independent pinball | half-coverage >= 95%",
            RESULTS / f"intervals_halfcov_{kind}.png",
        )

    all_cv = pd.concat(cv_frames, ignore_index=True)
    all_cv.to_csv(RESULTS / "halfcov_rscv_cv_results.csv", index=False)
    tails = pd.DataFrame(tail_rows)
    pairs = pd.DataFrame(pair_rows)
    tails.to_csv(RESULTS / "halfcov_rscv_tail_metrics.csv", index=False)
    pairs.to_csv(RESULTS / "halfcov_rscv_pair_metrics.csv", index=False)
    plot_cv_scatter(all_cv, RESULTS / "halfcov_rscv_cv_scatter.png")

    md = render_report(tails, pairs)
    (RESULTS / "halfcov_rscv.md").write_text(md)
    (RESULTS.parent / "RESULTS_HALFCOV_RSCV.md").write_text(md)
    print("\n" + md)


def render_report(tails, pairs):
    models = pairs[pairs["family"].isin(KIND_LABEL.values())].copy()
    lines = [
        "# Independent pinball RSCV under a one-sided (half) coverage constraint",
        "",
        "Each quantile regressor is tuned with `RandomizedSearchCV` and a custom",
        "`refit` callable (same mechanism as `plot_grid_search_digits.py`):",
        "",
        "- Lower α=0.05: feasible iff mean CV **P(Y ≥ q̂) ≥ 95%**, then min pinball.",
        "- Upper α=0.95: feasible iff mean CV **P(Y ≤ q̂) ≥ 95%**, then min pinball.",
        "",
        "The two independently selected models form the 90% interval. If both",
        "one-sided constraints hold, fold-level interval miscoverage is at most",
        "10% by a union bound.",
        "",
        "Dataset: n=4000 synthetic example from PR #32903 (train 3000 / test 1000).",
        "",
        "## Per-tail CV selection and test half-coverage",
        "",
        tails.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Combined 90% interval on the test set",
        "",
    ]
    pair_cols = [
        c
        for c in [
            "family",
            "constraint_ok",
            "n_feasible_low",
            "n_feasible_high",
            "cv_half_low",
            "cv_half_high",
            "coverage",
            "mean_width",
            "width_oracle_spearman",
            "winkler",
            "low_params",
            "high_params",
        ]
        if c in pairs.columns
    ]
    lines.append(pairs[pair_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    feasible = models[models["constraint_ok"] == True]  # noqa: E712
    lines.append("## Sharpest calibrated model class")
    lines.append("")
    if feasible.empty:
        lines.append("No class produced two CV-feasible tails.")
    else:
        hold = feasible[feasible["coverage"] >= 0.88]
        pool = hold if not hold.empty else feasible
        best = pool.sort_values("mean_width").iloc[0]
        lines += [
            f"Among classes with both tails CV-feasible, **{best['family']}** is",
            f"sharpest on test (mean width **{best['mean_width']:.2f}**, oracle ~5.52)",
            f"at test coverage **{best['coverage']:.1%}**.",
            "",
            "Ranking by test mean width (CV-feasible only):",
            "",
            feasible[
                [
                    "family",
                    "coverage",
                    "mean_width",
                    "width_oracle_spearman",
                    "winkler",
                ]
            ]
            .sort_values("mean_width")
            .to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    lines += [
        "Independent hparams let the 5% and 95% models differ (the gallery",
        "example already suggested that). The half-coverage floor blocks the",
        "pinball-only choice of an inward-biased tail.",
        "",
        "This write-up includes work produced with the assistance of AI.",
        "The code has **not yet been reviewed** by a human.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
