"""Follow-up plots and diagnostics for the winning quantile-interval configs."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split

from run_experiments import (
    ALPHA_HIGH,
    ALPHA_LOW,
    HonestQuantileForest,
    RESULTS,
    coverage_fraction,
    f,
    interval_metrics,
    make_dataset,
    oracle_y_quantile,
    plot_intervals,
    sigma_of_x,
)


def one_sided(y, y_low, y_high):
    return {
        "frac_above_low": float(np.mean(y >= y_low)),  # target 0.95
        "frac_below_high": float(np.mean(y <= y_high)),  # target 0.95
        "frac_below_low": float(np.mean(y < y_low)),
        "frac_above_high": float(np.mean(y > y_high)),
    }


def main():
    X, y = make_dataset(4000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    x_plot = np.atleast_2d(np.linspace(0, 10, 400)).T
    report = {}

    configs = {}

    rf50 = dict(n_estimators=200, min_samples_leaf=50, max_depth=None, random_state=0, n_jobs=1)
    configs["RF_msl50"] = (
        RandomForestRegressor(criterion="quantile", quantile=ALPHA_LOW, **rf50),
        RandomForestRegressor(criterion="quantile", quantile=ALPHA_HIGH, **rf50),
    )

    honest_base = dict(n_estimators=200, min_samples_leaf=40, max_depth=None, random_state=0, n_jobs=1)
    configs["HonestRF_msl40"] = (
        HonestQuantileForest(quantile=ALPHA_LOW, **honest_base),
        HonestQuantileForest(quantile=ALPHA_HIGH, **honest_base),
    )

    gb = dict(loss="quantile", n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=9, subsample=0.8, random_state=0)
    configs["GB_best_grid"] = (
        GradientBoostingRegressor(alpha=ALPHA_LOW, **gb),
        GradientBoostingRegressor(alpha=ALPHA_HIGH, **gb),
    )

    gb_slow = dict(
        loss="quantile",
        n_estimators=800,
        learning_rate=0.01,
        max_depth=2,
        min_samples_leaf=40,
        subsample=0.8,
        random_state=0,
    )
    configs["GB_slow_msl40"] = (
        GradientBoostingRegressor(alpha=ALPHA_LOW, **gb_slow),
        GradientBoostingRegressor(alpha=ALPHA_HIGH, **gb_slow),
    )

    # Slightly conservative nominal levels to buy coverage.
    gb_cons = dict(loss="quantile", n_estimators=400, learning_rate=0.03, max_depth=2, min_samples_leaf=40, subsample=0.8, random_state=0)
    configs["GB_alpha_0.03_0.97"] = (
        GradientBoostingRegressor(alpha=0.03, **gb_cons),
        GradientBoostingRegressor(alpha=0.97, **gb_cons),
    )

    hgb = dict(loss="quantile", max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    configs["HGB_msl40_l2"] = (
        HistGradientBoostingRegressor(quantile=ALPHA_LOW, **hgb),
        HistGradientBoostingRegressor(quantile=ALPHA_HIGH, **hgb),
    )

    # RF with only the 1/alpha leaf rule
    rf20 = dict(n_estimators=200, min_samples_leaf=20, max_depth=None, random_state=0, n_jobs=1)
    configs["RF_msl20_rule"] = (
        RandomForestRegressor(criterion="quantile", quantile=ALPHA_LOW, **rf20),
        RandomForestRegressor(criterion="quantile", quantile=ALPHA_HIGH, **rf20),
    )

    for name, (low, high) in configs.items():
        low.fit(X_train, y_train)
        high.fit(X_train, y_train)
        y_low = low.predict(X_test)
        y_high = high.predict(X_test)
        metrics = interval_metrics(y_test, y_low, y_high, X_test)
        metrics.update(one_sided(y_test, y_low, y_high))
        report[name] = metrics
        print(name, {k: round(metrics[k], 3) if isinstance(metrics[k], float) else metrics[k] for k in [
            "coverage", "mean_width", "width_oracle_spearman", "winkler",
            "frac_above_low", "frac_below_high",
        ]})
        plot_intervals(
            x_plot,
            low.predict(x_plot),
            high.predict(x_plot),
            X_test,
            y_test,
            name,
            RESULTS / f"intervals_{name}.png",
        )

    # Split-conformal on the example GB defaults: wrap to restore coverage.
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
    gb_def = dict(
        loss="quantile",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=9,
        subsample=1.0,
        random_state=0,
    )
    low = GradientBoostingRegressor(alpha=ALPHA_LOW, **gb_def).fit(X_fit, y_fit)
    high = GradientBoostingRegressor(alpha=ALPHA_HIGH, **gb_def).fit(X_fit, y_fit)
    # Symmetric conformal residual on interval: inflate by quantile of max(low-y, y-high, 0)
    scores = np.maximum(low.predict(X_cal) - y_cal, y_cal - high.predict(X_cal))
    qhat = np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * 0.9) / len(scores)))
    y_low = low.predict(X_test) - qhat
    y_high = high.predict(X_test) + qhat
    metrics = interval_metrics(y_test, y_low, y_high, X_test)
    metrics.update(one_sided(y_test, y_low, y_high))
    metrics["conformal_qhat"] = float(qhat)
    report["GB_defaults_split_conformal"] = metrics
    print("GB_defaults_split_conformal", {k: round(metrics[k], 3) for k in [
        "coverage", "mean_width", "width_oracle_spearman", "winkler", "conformal_qhat",
        "frac_above_low", "frac_below_high",
    ]})
    plot_intervals(
        x_plot,
        low.predict(x_plot) - qhat,
        high.predict(x_plot) + qhat,
        X_test,
        y_test,
        "GB defaults + split conformal",
        RESULTS / "intervals_GB_defaults_split_conformal.png",
    )

    # Width vs x for winners vs oracle
    fig, ax = plt.subplots(figsize=(8, 5))
    xp = x_plot.ravel()
    oracle_w = oracle_y_quantile(xp, ALPHA_HIGH) - oracle_y_quantile(xp, ALPHA_LOW)
    ax.plot(xp, oracle_w, "k-", lw=2, label="oracle width")
    # reuse last fitted? refit RF50 and honest for plot
    rf_l, rf_h = configs["RF_msl50"]
    hon_l, hon_h = configs["HonestRF_msl40"]
    gb_l, gb_h = configs["GB_best_grid"]
    ax.plot(xp, rf_h.predict(x_plot) - rf_l.predict(x_plot), label="RF msl=50")
    ax.plot(xp, hon_h.predict(x_plot) - hon_l.predict(x_plot), label="Honest RF msl=40")
    ax.plot(xp, gb_h.predict(x_plot) - gb_l.predict(x_plot), label="GB best grid")
    ax.set_xlabel("x")
    ax.set_ylabel("90% interval width")
    ax.legend()
    ax.set_title("Discriminative power: predicted width vs x")
    fig.tight_layout()
    fig.savefig(RESULTS / "width_vs_x.png", dpi=120)
    plt.close(fig)

    (RESULTS / "followup_metrics.json").write_text(json.dumps(report, indent=2))
    print("wrote followup_metrics.json")


if __name__ == "__main__":
    main()
