"""Diagnostics: GBR quantile undercoverage vs implementation bug vs finite-sample pinball CV.

Checks
------
1. n_estimators=1, learning_rate=1 GBR vs DecisionTreeRegressor(criterion='quantile')
   (same leaf functional; different split criterion).
2. Train vs test one-sided rates and coverage vs boosting stage (overfit curve).
3. Same hparams as n grows: does test coverage approach 90%?
4. Pinball-CV RF vs coverage-selected RF: both undercover under pinball; RF has a
   monotone leaf-size knob that can put coverage on the nominal line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import (  # noqa: E402
    ALPHA_HIGH,
    ALPHA_LOW,
    RESULTS,
    coverage_fraction,
    f,
    make_dataset,
    oracle_y_quantile,
)

RESULTS.mkdir(parents=True, exist_ok=True)


def one_sided(y, y_low, y_high):
    return {
        "coverage": coverage_fraction(y, y_low, y_high),
        "p_below_low": float(np.mean(y < y_low)),
        "p_above_high": float(np.mean(y > y_high)),
        "mean_width": float(np.mean(y_high - y_low)),
        "pinball_low": float(mean_pinball_loss(y, y_low, alpha=ALPHA_LOW)),
        "pinball_high": float(mean_pinball_loss(y, y_high, alpha=ALPHA_HIGH)),
    }


def fit_pair_gbr(X_tr, y_tr, **params):
    low = GradientBoostingRegressor(loss="quantile", alpha=ALPHA_LOW, random_state=0, **params)
    high = GradientBoostingRegressor(loss="quantile", alpha=ALPHA_HIGH, random_state=0, **params)
    low.fit(X_tr, y_tr)
    high.fit(X_tr, y_tr)
    return low, high


def main():
    report = {}
    X, y = make_dataset(4000, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    # --- 1. Single full-step tree: GBR vs pinball-split tree from PR 32903 ---
    gbr_low = GradientBoostingRegressor(
        loss="quantile", alpha=ALPHA_LOW, n_estimators=1, learning_rate=1.0,
        max_depth=3, min_samples_leaf=50, random_state=0,
    ).fit(X_tr, y_tr)
    tree_low = DecisionTreeRegressor(
        criterion="quantile", quantile=ALPHA_LOW, max_depth=3, min_samples_leaf=50, random_state=0,
    ).fit(X_tr, y_tr)
    pred_g = gbr_low.predict(X_te)
    pred_t = tree_low.predict(X_te)
    report["single_tree"] = {
        "mae_gbr_vs_tree": float(np.mean(np.abs(pred_g - pred_t))),
        "corr": float(np.corrcoef(pred_g, pred_t)[0, 1]),
        "mean_gbr": float(pred_g.mean()),
        "mean_tree": float(pred_t.mean()),
        "mean_oracle_low": float(oracle_y_quantile(X_te, ALPHA_LOW).mean()),
        "train_frac_below_gbr": float(np.mean(y_tr < gbr_low.predict(X_tr))),
        "train_frac_below_tree": float(np.mean(y_tr < tree_low.predict(X_tr))),
        "test_frac_below_gbr": float(np.mean(y_te < pred_g)),
        "test_frac_below_tree": float(np.mean(y_te < pred_t)),
    }
    print("single-tree GBR vs quantile DT", json.dumps(report["single_tree"], indent=2))

    # --- 2. Staged coverage for gallery-like GBR ---
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=2, min_samples_leaf=9, subsample=1.0)
    low, high = fit_pair_gbr(X_tr, y_tr, **params)
    train_lows = list(low.staged_predict(X_tr))
    train_highs = list(high.staged_predict(X_tr))
    test_lows = list(low.staged_predict(X_te))
    test_highs = list(high.staged_predict(X_te))
    want = {1, 2, 3, 5, 10, 20, 50, 100, 150, 200, 250, 300}
    stages = []
    for i in range(300):
        m = i + 1
        if m not in want:
            continue
        tr = one_sided(y_tr, train_lows[i], train_highs[i])
        te = one_sided(y_te, test_lows[i], test_highs[i])
        stages.append({"m": m, **{f"train_{k}": v for k, v in tr.items()}, **{f"test_{k}": v for k, v in te.items()}})
    report["staged"] = stages
    print("staged (selected):")
    for s in stages:
        if s["m"] in (1, 11, 51, 101, 201, 301) or s["m"] <= 5:
            print(
                f"  m={s['m']:3d}  train_cov={s['train_coverage']:.3f} "
                f"test_cov={s['test_coverage']:.3f}  "
                f"train_below={s['train_p_below_low']:.3f} test_below={s['test_p_below_low']:.3f}  "
                f"train_above={s['train_p_above_high']:.3f} test_above={s['test_p_above_high']:.3f}"
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([s["m"] for s in stages], [s["train_coverage"] for s in stages], label="GBR train coverage")
    ax.plot([s["m"] for s in stages], [s["test_coverage"] for s in stages], label="GBR test coverage")
    ax.axhline(0.9, color="k", ls="--", label="nominal 90%")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("90% interval coverage")
    ax.set_title("GBR quantile pair: train vs test coverage vs boosting stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "gbr_staged_coverage.png", dpi=120)
    plt.close(fig)

    # --- 3. Coverage vs n with fixed GBR / RF hparams ---
    n_grid = [500, 1000, 2000, 4000, 8000, 16000]
    scaling = []
    gbr_fixed = dict(n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=9)
    rf_msl9 = dict(n_estimators=200, min_samples_leaf=9, random_state=0, n_jobs=1)
    rf_msl50 = dict(n_estimators=200, min_samples_leaf=50, random_state=0, n_jobs=1)
    for n in n_grid:
        Xn, yn = make_dataset(n, random_state=42)
        Xa, Xb, ya, yb = train_test_split(Xn, yn, test_size=0.25, random_state=0)
        row = {"n": n, "n_train": len(ya)}
        gl, gh = fit_pair_gbr(Xa, ya, **gbr_fixed)
        row["gbr"] = one_sided(yb, gl.predict(Xb), gh.predict(Xb))
        row["gbr_train"] = one_sided(ya, gl.predict(Xa), gh.predict(Xa))
        rf_l = RandomForestRegressor(criterion="quantile", quantile=ALPHA_LOW, **rf_msl9).fit(Xa, ya)
        rf_h = RandomForestRegressor(criterion="quantile", quantile=ALPHA_HIGH, **rf_msl9).fit(Xa, ya)
        row["rf_msl9"] = one_sided(yb, rf_l.predict(Xb), rf_h.predict(Xb))
        rf_l = RandomForestRegressor(criterion="quantile", quantile=ALPHA_LOW, **rf_msl50).fit(Xa, ya)
        rf_h = RandomForestRegressor(criterion="quantile", quantile=ALPHA_HIGH, **rf_msl50).fit(Xa, ya)
        row["rf_msl50"] = one_sided(yb, rf_l.predict(Xb), rf_h.predict(Xb))
        scaling.append(row)
        print(
            f"n={n:5d}  GBR test={row['gbr']['coverage']:.3f} train={row['gbr_train']['coverage']:.3f}  "
            f"RF9={row['rf_msl9']['coverage']:.3f}  RF50={row['rf_msl50']['coverage']:.3f}"
        )
    report["scaling"] = scaling

    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [r["n"] for r in scaling]
    ax.plot(ns, [r["gbr"]["coverage"] for r in scaling], "o-", label="GBR defaults (test)")
    ax.plot(ns, [r["gbr_train"]["coverage"] for r in scaling], "o--", label="GBR defaults (train)")
    ax.plot(ns, [r["rf_msl9"]["coverage"] for r in scaling], "s-", label="RF msl=9")
    ax.plot(ns, [r["rf_msl50"]["coverage"] for r in scaling], "^-", label="RF msl=50")
    ax.axhline(0.9, color="k", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("test coverage")
    ax.set_title("Does coverage recover with more data at fixed hparams?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "coverage_vs_n.png", dpi=120)
    plt.close(fig)

    # --- 4. Pinball CV on RF: undercoverage vs jointly large msl ---
    rf_space = dict(
        n_estimators=[200],
        min_samples_leaf=[9, 20, 50, 80],
        max_depth=[None, 6],
    )
    low_est = RandomForestRegressor(criterion="quantile", quantile=ALPHA_LOW, random_state=0, n_jobs=1)
    high_est = RandomForestRegressor(criterion="quantile", quantile=ALPHA_HIGH, random_state=0, n_jobs=1)
    from sklearn.metrics import make_scorer

    search_l = RandomizedSearchCV(
        low_est, rf_space, n_iter=8, cv=3, random_state=0, n_jobs=2,
        scoring=make_scorer(mean_pinball_loss, alpha=ALPHA_LOW, greater_is_better=False),
    ).fit(X_tr, y_tr)
    search_h = RandomizedSearchCV(
        high_est, rf_space, n_iter=8, cv=3, random_state=0, n_jobs=2,
        scoring=make_scorer(mean_pinball_loss, alpha=ALPHA_HIGH, greater_is_better=False),
    ).fit(X_tr, y_tr)
    pinball_pair = one_sided(y_te, search_l.predict(X_te), search_h.predict(X_te))
    report["rf_pinball_cv"] = {
        "low_params": search_l.best_params_,
        "high_params": search_h.best_params_,
        "test": pinball_pair,
        "cv_pinball_low": float(-search_l.best_score_),
        "cv_pinball_high": float(-search_h.best_score_),
    }
    # same msl=50 both sides, pinball on test for comparison
    rf_l = RandomForestRegressor(criterion="quantile", quantile=ALPHA_LOW, n_estimators=200, min_samples_leaf=50, random_state=0).fit(X_tr, y_tr)
    rf_h = RandomForestRegressor(criterion="quantile", quantile=ALPHA_HIGH, n_estimators=200, min_samples_leaf=50, random_state=0).fit(X_tr, y_tr)
    cov_pair = one_sided(y_te, rf_l.predict(X_te), rf_h.predict(X_te))
    report["rf_msl50_both"] = {
        "test": cov_pair,
        "test_pinball_sum": cov_pair["pinball_low"] + cov_pair["pinball_high"],
        "pinball_cv_pinball_sum": pinball_pair["pinball_low"] + pinball_pair["pinball_high"],
    }
    print("RF pinball CV", search_l.best_params_, search_h.best_params_, pinball_pair)
    print("RF msl=50 both", cov_pair)

    # --- 5. Bias vs oracle along x for GBR vs RF ---
    x_plot = np.linspace(0, 10, 200)
    Xp = x_plot.reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    gl, gh = fit_pair_gbr(X_tr, y_tr, **gbr_fixed)
    for ax, name, lo, hi in [
        (axes[0], "GBR defaults", gl.predict(Xp), gh.predict(Xp)),
        (axes[1], "RF msl=50", rf_l.predict(Xp), rf_h.predict(Xp)),
    ]:
        ax.plot(x_plot, lo - oracle_y_quantile(x_plot, ALPHA_LOW), label="q05 − oracle")
        ax.plot(x_plot, hi - oracle_y_quantile(x_plot, ALPHA_HIGH), label="q95 − oracle")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.legend()
    axes[0].set_ylabel("prediction − oracle quantile")
    fig.suptitle("Signed bias of estimated quantiles (positive = too high)")
    fig.tight_layout()
    fig.savefig(RESULTS / "quantile_bias_vs_oracle.png", dpi=120)
    plt.close(fig)

    (RESULTS / "gbr_bugcheck.json").write_text(json.dumps(report, indent=2, default=float))
    print("wrote", RESULTS / "gbr_bugcheck.json")


if __name__ == "__main__":
    main()
