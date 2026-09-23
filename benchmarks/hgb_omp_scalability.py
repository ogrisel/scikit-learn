"""Compare HistGradientBoosting OpenMP scalability vs XGBoost/LightGBM/CatBoost.

Runs the currently importable scikit-learn (so main and PR #34935 can be
compared by pointing PYTHONPATH / an editable install at each build) against
XGBoost, LightGBM and CatBoost with approximately matching hyperparameters.

This machine may have fewer cores than a typical "big" server. Thread counts
above ``os.cpu_count()`` are still included: they mimic the default-OMP
oversubscription / too-many-threads regime that sklearn ``main`` is known to
handle poorly on small-to-medium problems.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info, threadpool_limits

SHAPES = [
    # name, n_samples, n_features — focused on small/medium where default OMP hurts
    ("tiny_1k_x_10", 1_000, 10),
    ("small_5k_x_20", 5_000, 20),
    ("small_wide_2k_x_128", 2_000, 128),
    ("medium_20k_x_20", 20_000, 20),
    ("medium_wide_20k_x_100", 20_000, 100),
    ("medium_50k_x_50", 50_000, 50),
]

N_ESTIMATORS = 40
MAX_LEAF_NODES = 31
LEARNING_RATE = 0.1
MAX_BINS = 255
MAX_DEPTH_CAP = 16  # CatBoost needs a depth cap when using lossguide + max_leaves
RANDOM_STATE = 0


def _sklearn_est():
    return HistGradientBoostingClassifier(
        learning_rate=LEARNING_RATE,
        max_iter=N_ESTIMATORS,
        max_bins=MAX_BINS,
        max_leaf_nodes=MAX_LEAF_NODES,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
        verbose=0,
        loss="log_loss",
    )


def _make_xgb(n_threads: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        tree_method="hist",
        grow_policy="lossguide",
        objective="binary:logistic",
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        max_leaves=MAX_LEAF_NODES,
        max_depth=0,
        reg_lambda=0.0,
        max_bin=MAX_BINS,
        min_child_weight=1e-3,
        n_jobs=n_threads,
        nthread=n_threads,
        verbosity=0,
        eval_metric="logloss",
    )


def _make_lgbm(n_threads: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        num_leaves=MAX_LEAF_NODES,
        max_depth=-1,
        min_data_in_leaf=20,
        reg_lambda=0.0,
        max_bin=MAX_BINS,
        min_data_in_bin=1,
        min_sum_hessian_in_leaf=1e-3,
        min_split_gain=0,
        verbosity=-1,
        n_jobs=n_threads,
        num_threads=n_threads,
        force_row_wise=True,
    )


def _make_cat(n_threads: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
        learning_rate=LEARNING_RATE,
        iterations=N_ESTIMATORS,
        grow_policy="Lossguide",
        max_leaves=MAX_LEAF_NODES,
        depth=MAX_DEPTH_CAP,
        l2_leaf_reg=3.0,  # CatBoost default; 0 is rejected
        max_bin=MAX_BINS,
        thread_count=n_threads,
        verbose=False,
        allow_writing_files=False,
        random_seed=RANDOM_STATE,
        bootstrap_type="No",
        boosting_type="Plain",
    )


def _make_data(n_samples: int, n_features: int):
    n_informative = max(2, n_features // 2)
    # Extra samples are held out so train size stays n_samples while we can
    # report test ROC AUC with the same generator seed.
    X, y = make_classification(
        n_samples=n_samples * 2,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=RANDOM_STATE,
    )
    X = X.astype(np.float32, copy=False)
    return train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y)


def _positive_proba(est, X):
    proba = est.predict_proba(X)
    if getattr(proba, "ndim", 1) == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    return np.ravel(proba)


def _fit_once(lib: str, n_threads: int, X, y):
    if lib == "sklearn":
        est = _sklearn_est()
        with threadpool_limits(limits=n_threads, user_api="openmp"):
            t0 = time.perf_counter()
            est.fit(X, y)
            dt = time.perf_counter() - t0
        return dt, est
    if lib == "xgboost":
        est = _make_xgb(n_threads)
        t0 = time.perf_counter()
        est.fit(X, y)
        return time.perf_counter() - t0, est
    if lib == "lightgbm":
        est = _make_lgbm(n_threads)
        t0 = time.perf_counter()
        est.fit(X, y)
        return time.perf_counter() - t0, est
    if lib == "catboost":
        est = _make_cat(n_threads)
        t0 = time.perf_counter()
        est.fit(X, y)
        return time.perf_counter() - t0, est
    raise ValueError(lib)


def _median(xs):
    return statistics.median(xs)


def run_one(lib, sklearn_label, shape_name, n_samples, n_features, n_threads, repeats, warmup):
    X_train, X_test, y_train, y_test = _make_data(n_samples, n_features)
    for _ in range(warmup):
        _fit_once(lib, n_threads, X_train, y_train)
    times = []
    aucs = []
    for _ in range(repeats):
        dt, est = _fit_once(lib, n_threads, X_train, y_train)
        times.append(dt)
        aucs.append(float(roc_auc_score(y_test, _positive_proba(est, X_test))))
    import sklearn

    row = {
        "sklearn_label": sklearn_label,
        "sklearn_version": sklearn.__version__,
        "lib": lib,
        "shape": shape_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_threads": n_threads,
        "n_estimators": N_ESTIMATORS,
        "max_leaf_nodes": MAX_LEAF_NODES,
        "metric": "roc_auc",
        "fit_seconds_median": _median(times),
        "fit_seconds_min": min(times),
        "fit_seconds_max": max(times),
        "test_roc_auc_median": _median(aucs),
        "test_roc_auc_min": min(aucs),
        "test_roc_auc_max": max(aucs),
        "repeats": repeats,
        "warmup": warmup,
        "times_json": json.dumps(times),
        "aucs_json": json.dumps(aucs),
        "cpu_count": os.cpu_count(),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", ""),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS", ""),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", ""),
    }
    return row, times, aucs


def env_metadata():
    import sklearn

    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "sklearn_file": sklearn.__file__,
        "sklearn_version": sklearn.__version__,
        "threadpool_info": threadpool_info(),
        "env": {
            k: os.environ.get(k, "")
            for k in [
                "OMP_NUM_THREADS",
                "OMP_WAIT_POLICY",
                "OMP_PROC_BIND",
                "OMP_DISPLAY_ENV",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]
        },
    }
    try:
        from sklearn.utils._openmp_helpers import (
            _openmp_effective_n_threads,
            _openmp_parallelism_enabled,
        )

        meta["openmp_enabled"] = bool(_openmp_parallelism_enabled())
        meta["openmp_effective_n_threads"] = int(_openmp_effective_n_threads())
    except Exception as exc:  # pragma: no cover
        meta["openmp_error"] = repr(exc)
    try:
        from sklearn.utils._openmp_helpers import _openmp_uses_active_wait

        meta["openmp_active_wait"] = bool(_openmp_uses_active_wait())
    except Exception:
        meta["openmp_active_wait"] = None
    for pkg in ("xgboost", "lightgbm", "catboost"):
        try:
            mod = __import__(pkg)
            meta[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            meta[f"{pkg}_version"] = f"IMPORT_ERROR:{exc}"
    return meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sklearn-label", required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-meta", type=Path, required=True)
    p.add_argument("--libs", default="sklearn,xgboost,lightgbm,catboost")
    p.add_argument("--threads", default="1,2,4,8,16")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--shapes",
        default=",".join(s[0] for s in SHAPES),
        help="comma-separated shape names",
    )
    return p.parse_args()


def main():
    args = parse_args()
    libs = [x.strip() for x in args.libs.split(",") if x.strip()]
    n_threads_list = [int(x) for x in args.threads.split(",") if x.strip()]
    wanted = {x.strip() for x in args.shapes.split(",") if x.strip()}
    shapes = [s for s in SHAPES if s[0] in wanted]

    meta = env_metadata()
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in meta if k != "threadpool_info"}, indent=2))
    print("threadpool_info:", json.dumps(meta["threadpool_info"], indent=2, default=str))

    fieldnames = None
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out_csv.exists()
    with args.out_csv.open("a", newline="") as f:
        writer = None
        for shape_name, n_samples, n_features in shapes:
            for n_threads in n_threads_list:
                for lib in libs:
                    print(
                        f"=== {args.sklearn_label} {lib} {shape_name} "
                        f"threads={n_threads} ===",
                        flush=True,
                    )
                    try:
                        row, times, aucs = run_one(
                            lib,
                            args.sklearn_label,
                            shape_name,
                            n_samples,
                            n_features,
                            n_threads,
                            args.repeats,
                            args.warmup,
                        )
                    except Exception as exc:
                        print(f"FAILED: {exc!r}", flush=True)
                        row = {
                            "sklearn_label": args.sklearn_label,
                            "lib": lib,
                            "shape": shape_name,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "n_threads": n_threads,
                            "error": repr(exc),
                        }
                        times = []
                        aucs = []
                    print(
                        f"median={row.get('fit_seconds_median')} "
                        f"auc={row.get('test_roc_auc_median')} times={times} aucs={aucs}",
                        flush=True,
                    )
                    if fieldnames is None:
                        fieldnames = list(row.keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        if new_file:
                            writer.writeheader()
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writerow(row)
                    f.flush()


if __name__ == "__main__":
    main()
