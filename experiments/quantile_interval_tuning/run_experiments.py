"""Tune GB / RF / ExtraTrees quantile pairs on the PR #32903 synthetic example.

Goals
-----
1. Reproduce the example data (heteroscedastic centered log-normal noise).
2. Grid tree-growth hyperparameters for 5%/95% quantile pairs.
3. Report calibration (coverage), sharpness (mean width), and discriminative
   power (Spearman correlation of predicted width vs oracle width).
4. Compare independent pinball tuning vs a joint interval score, and vs the
   ``min_samples_leaf >= 1 / min(alpha, 1-alpha)`` rule.

This is an experiment script, not a gallery example.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.base import clone
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

ALPHA_LOW = 0.05
ALPHA_HIGH = 0.95
NOMINAL_COVERAGE = ALPHA_HIGH - ALPHA_LOW
MIN_LEAF_RULE = int(np.ceil(1.0 / min(ALPHA_LOW, 1.0 - ALPHA_LOW)))  # 20


def f(x):
    return x * np.sin(x)


def sigma_of_x(x):
    return 0.5 + np.ravel(x) / 10.0


def oracle_noise_quantile(x, q):
    """Quantile of centered log-normal noise used in the example."""
    sigma = sigma_of_x(x)
    return np.exp(sigma * norm.ppf(q)) - np.exp(sigma**2 / 2.0)


def oracle_y_quantile(x, q):
    return f(np.ravel(x)) + oracle_noise_quantile(x, q)


def make_dataset(n_samples, random_state):
    rng = np.random.RandomState(random_state)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=n_samples)).T
    sigma = sigma_of_x(X)
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2.0)
    y = f(X).ravel() + noise
    return X, y


def coverage_fraction(y, y_low, y_high):
    return float(np.mean((y >= y_low) & (y <= y_high)))


def winkler_score(y, y_low, y_high, alpha=1.0 - NOMINAL_COVERAGE):
    """Mean interval / Winkler score (proper scoring rule for central intervals)."""
    width = y_high - y_low
    below = np.maximum(y_low - y, 0.0)
    above = np.maximum(y - y_high, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def interval_metrics(y, y_low, y_high, X=None):
    width = y_high - y_low
    metrics = {
        "coverage": coverage_fraction(y, y_low, y_high),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "width_cv": float(np.std(width) / (np.mean(width) + 1e-12)),
        "crossing_rate": float(np.mean(y_high < y_low)),
        "pinball_low": float(mean_pinball_loss(y, y_low, alpha=ALPHA_LOW)),
        "pinball_high": float(mean_pinball_loss(y, y_high, alpha=ALPHA_HIGH)),
        "winkler": winkler_score(y, y_low, y_high),
        "mae_low_oracle": float(np.mean(np.abs(y_low - oracle_y_quantile(X, ALPHA_LOW)))),
        "mae_high_oracle": float(np.mean(np.abs(y_high - oracle_y_quantile(X, ALPHA_HIGH)))),
    }
    oracle_width = oracle_y_quantile(X, ALPHA_HIGH) - oracle_y_quantile(X, ALPHA_LOW)
    corr = spearmanr(width, oracle_width).correlation
    metrics["width_oracle_spearman"] = float(corr) if np.isfinite(corr) else 0.0
    metrics["interval_score_sum_pinball"] = metrics["pinball_low"] + metrics["pinball_high"]
    return metrics


def coverage_ok(coverage, tol=0.03):
    return abs(coverage - NOMINAL_COVERAGE) <= tol


class HonestQuantileForest:
    """Grow trees on one split, estimate leaf quantiles on a disjoint split."""

    def __init__(self, base_estimator, quantile, honest_fraction=0.5, random_state=0):
        self.base_estimator = base_estimator
        self.quantile = quantile
        self.honest_fraction = honest_fraction
        self.random_state = random_state
        self.estimator_ = None
        self.leaf_quantiles_ = None

    def fit(self, X, y):
        X_grow, X_hon, y_grow, y_hon = train_test_split(
            X, y, test_size=self.honest_fraction, random_state=self.random_state
        )
        est = self.base_estimator
        est.fit(X_grow, y_grow)
        leaves = est.apply(X_hon)
        n_trees = leaves.shape[1]
        leaf_q = []
        fallback = np.quantile(y_hon, self.quantile)
        for t in range(n_trees):
            ids = leaves[:, t]
            qmap = {}
            for leaf in np.unique(ids):
                vals = y_hon[ids == leaf]
                qmap[int(leaf)] = float(np.quantile(vals, self.quantile))
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


def fit_pair(make_low, make_high, X_train, y_train, X_eval, y_eval):
    t0 = time.perf_counter()
    low = make_low()
    high = make_high()
    low.fit(X_train, y_train)
    high.fit(X_train, y_train)
    y_low = low.predict(X_eval)
    y_high = high.predict(X_eval)
    elapsed = time.perf_counter() - t0
    metrics = interval_metrics(y_eval, y_low, y_high, X_eval)
    metrics["fit_predict_s"] = elapsed
    return metrics, y_low, y_high


def rf_factory(cls, quantile, params, random_state):
    def make():
        return cls(
            criterion="quantile",
            quantile=quantile,
            random_state=random_state,
            n_jobs=1,
            **params,
        )

    return make


def gb_factory(quantile, params, random_state):
    def make():
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            random_state=random_state,
            **params,
        )

    return make


def hgb_factory(quantile, params, random_state):
    def make():
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            random_state=random_state,
            **params,
        )

    return make


def run_grid(name, factory_low, factory_high, grid, X_train, y_train, X_test, y_test, extra):
    rows = []
    for i, params in enumerate(ParameterGrid(grid)):
        metrics, _, _ = fit_pair(
            factory_low(params),
            factory_high(params),
            X_train,
            y_train,
            X_test,
            y_test,
        )
        row = {"family": name, **extra, **params, **metrics}
        rows.append(row)
        print(
            f"  [{name} {i + 1}/{len(list(ParameterGrid(grid)))}] "
            f"cov={metrics['coverage']:.3f} width={metrics['mean_width']:.2f} "
            f"spearman={metrics['width_oracle_spearman']:.2f} winkler={metrics['winkler']:.2f} "
            f"params={params}",
            flush=True,
        )
    return rows


def plot_pareto(df, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    families = sorted(df["family"].unique())
    for fam in families:
        sub = df[df["family"] == fam]
        ax.scatter(
            sub["mean_width"],
            sub["coverage"],
            label=fam,
            alpha=0.75,
            s=40,
        )
    ax.axhline(NOMINAL_COVERAGE, color="black", linestyle="--", label="nominal 90%")
    ax.axhspan(NOMINAL_COVERAGE - 0.03, NOMINAL_COVERAGE + 0.03, color="gray", alpha=0.15)
    ax.set_xlabel("Mean interval width (test)")
    ax.set_ylabel("Coverage (test)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_intervals(x_plot, y_low, y_high, X_test, y_test, title, path):
    order = np.argsort(x_plot.ravel())
    xp = x_plot.ravel()[order]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xp, f(xp), "k-", lw=2, label=r"$f(x)=x\sin(x)$")
    ax.plot(xp, oracle_y_quantile(xp, ALPHA_LOW), "k--", lw=1, alpha=0.6, label="oracle 5%/95%")
    ax.plot(xp, oracle_y_quantile(xp, ALPHA_HIGH), "k--", lw=1, alpha=0.6)
    ax.plot(X_test, y_test, "b.", ms=6, alpha=0.4, label="test")
    ax.fill_between(xp, y_low[order], y_high[order], alpha=0.35, label="predicted 90%")
    ax.set_ylim(-10, 25)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_searches(X_train, y_train, X_test, y_test, extra):
    """Independent pinball CV vs constrained leaf size, matching the gallery example."""
    rows = []
    rng = extra["seed"]
    gb_space = dict(
        learning_rate=[0.05, 0.1, 0.2],
        max_depth=[2, 3, 5, 8],
        min_samples_leaf=[1, 5, 9, MIN_LEAF_RULE, 40],
        min_samples_split=[5, 10, 20, 40],
        subsample=[0.8, 1.0],
        n_estimators=[100, 200],
    )
    gb_space_constrained = dict(gb_space)
    gb_space_constrained["min_samples_leaf"] = [MIN_LEAF_RULE, 30, 50, 80]

    rf_space = dict(
        n_estimators=[200],
        min_samples_leaf=[1, 5, 9, MIN_LEAF_RULE, 40, 80],
        max_depth=[None, 6, 12],
        min_samples_split=[2, 10, 20],
    )
    rf_space_constrained = dict(rf_space)
    rf_space_constrained["min_samples_leaf"] = [MIN_LEAF_RULE, 30, 50, 80]

    def fit_search(name, est, space, alpha, X_tr, y_tr):
        scorer = make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)
        search = RandomizedSearchCV(
            est,
            space,
            n_iter=18,
            scoring=scorer,
            cv=3,
            random_state=rng,
            n_jobs=2,
        )
        search.fit(X_tr, y_tr)
        print(f"  [{name} q={alpha}] best={search.best_params_} score={search.best_score_:.4f}")
        return search

    for fam, low_est, high_est, space, space_c in [
        (
            "GB_search_pinball",
            GradientBoostingRegressor(loss="quantile", alpha=ALPHA_LOW, random_state=rng),
            GradientBoostingRegressor(loss="quantile", alpha=ALPHA_HIGH, random_state=rng),
            gb_space,
            gb_space_constrained,
        ),
        (
            "RF_search_pinball",
            RandomForestRegressor(
                criterion="quantile", quantile=ALPHA_LOW, random_state=rng, n_jobs=1
            ),
            RandomForestRegressor(
                criterion="quantile", quantile=ALPHA_HIGH, random_state=rng, n_jobs=1
            ),
            rf_space,
            rf_space_constrained,
        ),
    ]:
        for tag, sp in [("", space), ("_msl_ge20", space_c)]:
            low_s = fit_search(fam + tag, clone(low_est), sp, ALPHA_LOW, X_train, y_train)
            high_s = fit_search(fam + tag, clone(high_est), sp, ALPHA_HIGH, X_train, y_train)
            metrics = interval_metrics(
                y_test, low_s.predict(X_test), high_s.predict(X_test), X_test
            )
            row = {
                "family": fam + tag,
                **extra,
                **{f"low_{k}": v for k, v in low_s.best_params_.items()},
                **{f"high_{k}": v for k, v in high_s.best_params_.items()},
                **metrics,
            }
            rows.append(row)
            print(
                f"  [{fam + tag}] cov={metrics['coverage']:.3f} width={metrics['mean_width']:.2f} "
                f"spearman={metrics['width_oracle_spearman']:.2f} winkler={metrics['winkler']:.2f}"
            )
            x_plot = np.atleast_2d(np.linspace(0, 10, 400)).T
            plot_intervals(
                x_plot,
                low_s.predict(x_plot),
                high_s.predict(x_plot),
                X_test,
                y_test,
                f"{fam + tag} (n4k)",
                RESULTS / f"intervals_{fam + tag}.png",
            )
    return rows


def constant_baseline(y_train, y_test, X_test):
    y_low = np.full_like(y_test, np.quantile(y_train, ALPHA_LOW), dtype=float)
    y_high = np.full_like(y_test, np.quantile(y_train, ALPHA_HIGH), dtype=float)
    return interval_metrics(y_test, y_low, y_high, X_test)


def main():
    t_start = time.perf_counter()
    # Larger than the gallery example so coverage SE is ~1pp instead of ~2pp.
    setups = [
        dict(n_samples=1000, test_size=0.25, seed=0, label="n1k_seed0"),
        dict(n_samples=4000, test_size=0.25, seed=0, label="n4k_seed0"),
    ]

    rf_grid = {
        "n_estimators": [200],
        "min_samples_leaf": [1, 5, 9, MIN_LEAF_RULE, 30, 50],
        "max_depth": [None, 3, 6, 12],
        "max_features": [1.0],
    }
    et_grid = {
        "n_estimators": [200],
        "min_samples_leaf": [1, 9, MIN_LEAF_RULE, 40],
        "max_depth": [None, 6, 12],
        "max_features": [1.0],
    }
    gb_grid = {
        "n_estimators": [200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 5],
        "min_samples_leaf": [5, 9, MIN_LEAF_RULE, 40],
        "subsample": [1.0, 0.8],
    }
    hgb_grid = {
        "max_iter": [200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6, None],
        "min_samples_leaf": [9, MIN_LEAF_RULE, 40],
        "l2_regularization": [0.0, 1.0],
    }

    all_rows = []
    selected_plots_done = set()

    for setup in setups:
        print(f"\n=== setup {setup['label']} ===", flush=True)
        X, y = make_dataset(setup["n_samples"], random_state=42 + setup["seed"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=setup["test_size"], random_state=setup["seed"]
        )
        extra = {
            "setup": setup["label"],
            "n_train": len(y_train),
            "n_test": len(y_test),
            "seed": setup["seed"],
        }

        const = constant_baseline(y_train, y_test, X_test)
        all_rows.append({"family": "constant_marginal", **extra, **const})
        oracle_low = oracle_y_quantile(X_test, ALPHA_LOW)
        oracle_high = oracle_y_quantile(X_test, ALPHA_HIGH)
        oracle = interval_metrics(y_test, oracle_low, oracle_high, X_test)
        all_rows.append({"family": "oracle", **extra, **oracle})
        print(
            f"  constant coverage={const['coverage']:.3f} width={const['mean_width']:.2f} "
            f"spearman={const['width_oracle_spearman']:.2f}"
        )
        print(
            f"  oracle   coverage={oracle['coverage']:.3f} width={oracle['mean_width']:.2f} "
            f"spearman={oracle['width_oracle_spearman']:.2f}"
        )

        all_rows += run_grid(
            "RandomForest",
            lambda p: rf_factory(RandomForestRegressor, ALPHA_LOW, p, setup["seed"]),
            lambda p: rf_factory(RandomForestRegressor, ALPHA_HIGH, p, setup["seed"]),
            rf_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            extra,
        )
        all_rows += run_grid(
            "ExtraTrees",
            lambda p: rf_factory(ExtraTreesRegressor, ALPHA_LOW, p, setup["seed"]),
            lambda p: rf_factory(ExtraTreesRegressor, ALPHA_HIGH, p, setup["seed"]),
            et_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            extra,
        )
        all_rows += run_grid(
            "GradientBoosting",
            lambda p: gb_factory(ALPHA_LOW, p, setup["seed"]),
            lambda p: gb_factory(ALPHA_HIGH, p, setup["seed"]),
            gb_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            extra,
        )
        if setup["label"] == "n4k_seed0":
            all_rows += run_searches(X_train, y_train, X_test, y_test, extra)

        all_rows += run_grid(
            "HistGradientBoosting",
            lambda p: hgb_factory(ALPHA_LOW, p, setup["seed"]),
            lambda p: hgb_factory(ALPHA_HIGH, p, setup["seed"]),
            hgb_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            extra,
        )

        # Honest RF at a few min_samples_leaf values
        for msl in [1, 9, MIN_LEAF_RULE, 40]:
            params = dict(n_estimators=200, min_samples_leaf=msl, max_depth=None, max_features=1.0)
            low = HonestQuantileForest(
                RandomForestRegressor(
                    criterion="squared_error",
                    random_state=setup["seed"],
                    n_jobs=1,
                    **params,
                ),
                quantile=ALPHA_LOW,
                random_state=setup["seed"],
            )
            high = HonestQuantileForest(
                RandomForestRegressor(
                    criterion="squared_error",
                    random_state=setup["seed"],
                    n_jobs=1,
                    **params,
                ),
                quantile=ALPHA_HIGH,
                random_state=setup["seed"],
            )
            t0 = time.perf_counter()
            low.fit(X_train, y_train)
            high.fit(X_train, y_train)
            y_low = low.predict(X_test)
            y_high = high.predict(X_test)
            metrics = interval_metrics(y_test, y_low, y_high, X_test)
            metrics["fit_predict_s"] = time.perf_counter() - t0
            all_rows.append(
                {
                    "family": "HonestRF",
                    **extra,
                    **params,
                    **metrics,
                }
            )
            print(
                f"  [HonestRF] msl={msl} cov={metrics['coverage']:.3f} "
                f"width={metrics['mean_width']:.2f} spearman={metrics['width_oracle_spearman']:.2f}"
            )

        # Example-default GB / RF from the PR gallery notebook
        for fam, make_low, make_high, tag in [
            (
                "GB_example_defaults",
                gb_factory(
                    ALPHA_LOW,
                    dict(
                        learning_rate=0.05,
                        n_estimators=200,
                        max_depth=2,
                        min_samples_leaf=9,
                        min_samples_split=9,
                    ),
                    setup["seed"],
                ),
                gb_factory(
                    ALPHA_HIGH,
                    dict(
                        learning_rate=0.05,
                        n_estimators=200,
                        max_depth=2,
                        min_samples_leaf=9,
                        min_samples_split=9,
                    ),
                    setup["seed"],
                ),
                "gb_defaults",
            ),
            (
                "RF_example_defaults",
                rf_factory(
                    RandomForestRegressor,
                    ALPHA_LOW,
                    dict(n_estimators=200, min_samples_leaf=9, min_samples_split=9),
                    setup["seed"],
                ),
                rf_factory(
                    RandomForestRegressor,
                    ALPHA_HIGH,
                    dict(n_estimators=200, min_samples_leaf=9, min_samples_split=9),
                    setup["seed"],
                ),
                "rf_defaults",
            ),
        ]:
            metrics, y_low, y_high = fit_pair(make_low, make_high, X_train, y_train, X_test, y_test)
            all_rows.append({"family": fam, **extra, **metrics})
            print(f"  [{fam}] cov={metrics['coverage']:.3f} width={metrics['mean_width']:.2f}")
            if setup["label"] == "n4k_seed0" and tag not in selected_plots_done:
                x_plot = np.atleast_2d(np.linspace(0, 10, 400)).T
                # refit already done on train; predict on grid via new pair
                _, y_low_p, y_high_p = fit_pair(
                    make_low, make_high, X_train, y_train, x_plot, oracle_y_quantile(x_plot, 0.5)
                )
                plot_intervals(
                    x_plot,
                    y_low_p,
                    y_high_p,
                    X_test,
                    y_test,
                    f"{fam} on {setup['label']}",
                    RESULTS / f"intervals_{tag}.png",
                )
                selected_plots_done.add(tag)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "metrics.csv", index=False)

    # Pareto plots for the large setup
    df4 = df[df["setup"] == "n4k_seed0"].copy()
    model_df = df4[
        df4["family"].isin(
            [
                "RandomForest",
                "ExtraTrees",
                "GradientBoosting",
                "HistGradientBoosting",
                "HonestRF",
                "constant_marginal",
                "oracle",
                "GB_example_defaults",
                "RF_example_defaults",
                "GB_search_pinball",
                "GB_search_pinball_msl_ge20",
                "RF_search_pinball",
                "RF_search_pinball_msl_ge20",
            ]
        )
    ]
    plot_pareto(model_df, "Coverage vs width (n=4000, seed=0)", RESULTS / "pareto_coverage_width.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for fam, sub in model_df.groupby("family"):
        ax.scatter(sub["width_oracle_spearman"], sub["coverage"], label=fam, alpha=0.75, s=40)
    ax.axhline(NOMINAL_COVERAGE, color="k", ls="--")
    ax.set_xlabel("Spearman(predicted width, oracle width)")
    ax.set_ylabel("Coverage")
    ax.set_title("Discriminative power vs calibration (n=4000)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto_spearman_coverage.png", dpi=120)
    plt.close(fig)

    # Best configs near nominal coverage
    summary = build_summary(df)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "summary.md").write_text(render_markdown(df, summary, time.perf_counter() - t_start))
    print("\nWrote", RESULTS / "summary.md")
    print("Elapsed", round(time.perf_counter() - t_start, 1), "s")


def ranked_near_nominal(df, family, setup, tol=0.03):
    sub = df[(df["family"] == family) & (df["setup"] == setup)].copy()
    if sub.empty:
        return sub
    sub["cov_err"] = (sub["coverage"] - NOMINAL_COVERAGE).abs()
    near = sub[sub["cov_err"] <= tol]
    pool = near if len(near) else sub.nsmallest(5, "cov_err")
    return pool.sort_values(["winkler", "mean_width"])


def build_summary(df):
    summary = {
        "min_samples_leaf_rule": MIN_LEAF_RULE,
        "nominal_coverage": NOMINAL_COVERAGE,
        "coverage_tolerance": 0.03,
    }
    setups = sorted(df["setup"].unique())
    families = [
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "HistGradientBoosting",
        "HonestRF",
        "GB_example_defaults",
        "RF_example_defaults",
        "GB_search_pinball",
        "GB_search_pinball_msl_ge20",
        "RF_search_pinball",
        "RF_search_pinball_msl_ge20",
        "constant_marginal",
        "oracle",
    ]
    picks = {}
    for setup in setups:
        picks[setup] = {}
        for fam in families:
            ranked = ranked_near_nominal(df, fam, setup)
            if ranked.empty:
                continue
            row = ranked.iloc[0]
            picks[setup][fam] = {
                k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                for k, v in row.to_dict().items()
                if k in {
                    "coverage",
                    "mean_width",
                    "width_oracle_spearman",
                    "winkler",
                    "min_samples_leaf",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "n_estimators",
                    "max_iter",
                    "l2_regularization",
                    "pinball_low",
                    "pinball_high",
                    "crossing_rate",
                }
            }
    summary["best_near_nominal"] = picks
    return summary


def render_markdown(df, summary, elapsed_s):
    lines = [
        "# Quantile GB / RF interval tuning",
        "",
        "Synthetic data as in the PR #32903 example: `y = x sin(x) + centered log-normal`",
        "with `sigma = 0.5 + x/10` (heteroscedastic, right-skewed).",
        "",
        f"- Nominal coverage: **{NOMINAL_COVERAGE:.0%}** (5th–95th percentile pair).",
        f"- Leaf-size rule: `min_samples_leaf >= 1/min(alpha, 1-alpha)` = **{MIN_LEAF_RULE}**.",
        f"- Wall time: {elapsed_s:.0f}s.",
        "",
        "Metrics: test coverage, mean width (sharpness), Spearman correlation of predicted",
        "width vs oracle width (discriminative power), and Winkler/interval score",
        "(proper scoring rule combining both).",
        "",
        "## Example-default models from the PR",
        "",
    ]
    defaults = df[df["family"].isin(["GB_example_defaults", "RF_example_defaults", "oracle", "constant_marginal"])]
    cols = [
        "setup",
        "family",
        "coverage",
        "mean_width",
        "width_oracle_spearman",
        "winkler",
    ]
    lines.append(defaults[cols].sort_values(["setup", "family"]).to_markdown(index=False, floatfmt=".3f"))
    lines += ["", "## Best config near nominal coverage (lowest Winkler among |cov-0.90|<=0.03)", ""]
    for setup, fams in summary["best_near_nominal"].items():
        lines.append(f"### {setup}")
        lines.append("")
        rows = []
        for fam, d in fams.items():
            rows.append({"family": fam, **d})
        if rows:
            lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
            lines.append("")

    # Effect of min_samples_leaf for unbounded RF
    lines += ["## RandomForest: min_samples_leaf vs coverage (max_depth=None, n=4000 seed0)", ""]
    rf = df[
        (df["family"] == "RandomForest")
        & (df["setup"] == "n4k_seed0")
        & (df["max_depth"].isna() | (df["max_depth"].astype(str) == "nan"))
    ]
    if not rf.empty:
        # max_depth None stored as NaN
        rf_none = df[(df["family"] == "RandomForest") & (df["setup"] == "n4k_seed0")].copy()
        rf_none = rf_none[rf_none["max_depth"].isna()]
        show = rf_none[
            [
                "min_samples_leaf",
                "coverage",
                "mean_width",
                "width_oracle_spearman",
                "winkler",
            ]
        ].sort_values("min_samples_leaf")
        lines.append(show.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    lines += [
        "## Takeaways (filled after the run from the tables above)",
        "",
        "See `summary.json` and `metrics.csv` for the full grid.",
        "",
        "This pull request includes code written with the assistance of AI.",
        "The code has **not yet been reviewed** by a human.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
