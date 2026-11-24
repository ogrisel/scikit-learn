"""Problem registry: datasets, estimators, search spaces, scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    make_classification,
    make_regression,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from samplers import ParamSpec


@dataclass
class Problem:
    problem_id: str
    task: str  # "classification" | "regression"
    scoring: str
    load_xy: Callable[[], tuple[Any, Any]]
    make_estimator: Callable[[], Any]
    specs: list[ParamSpec]
    n_iter: int = 30
    cv: int = 3


def _breast_cancer():
    return load_breast_cancer(return_X_y=True)


def _digits_multiclass():
    # Keep small: 8x8 digits, all 10 classes but subsample for speed.
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(500, len(y)), replace=False)
    return X[idx], y[idx]


def _diabetes():
    return load_diabetes(return_X_y=True)


def _synthetic_clf():
    return make_classification(
        n_samples=600,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        random_state=0,
    )


def _synthetic_reg():
    return make_regression(
        n_samples=600,
        n_features=20,
        n_informative=8,
        noise=15.0,
        random_state=0,
    )


def _synthetic_mixed_clf():
    """Synthetic mixed numeric + categorical classification."""
    rng = np.random.default_rng(1)
    n = 800
    x_num = rng.normal(size=(n, 4))
    cat_a = rng.choice(["a", "b", "c", "d"], size=n)
    cat_b = rng.choice(["x", "y", "z"], size=n)
    # Nonlinear target involving both types.
    logits = (
        1.2 * x_num[:, 0]
        - 0.8 * x_num[:, 1]
        + (cat_a == "a") * 0.9
        + (cat_b == "z") * 1.1
        + 0.3 * x_num[:, 0] * (cat_a == "b")
    )
    y = (logits + rng.normal(scale=0.8, size=n) > 0).astype(int)
    X = pd.DataFrame(
        {
            "n0": x_num[:, 0],
            "n1": x_num[:, 1],
            "n2": x_num[:, 2],
            "n3": x_num[:, 3],
            "c0": cat_a,
            "c1": cat_b,
        }
    )
    return X, y


def _openml_adult_small():
    """Adult income — subsample for speed; mixed types."""
    bunch = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X = bunch.data.copy()
    y = (bunch.target == ">50K").astype(int)
    # Drop rows with missing in target already handled; subsample.
    rng = np.random.default_rng(2)
    idx = rng.choice(len(y), size=min(1500, len(y)), replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].to_numpy()


def _openml_credit_g():
    """German credit — small mixed-type OpenML dataset."""
    bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    X = bunch.data.copy()
    y = (bunch.target == "good").astype(int).to_numpy()
    return X, y


def make_logreg():
    # sklearn>=1.8: prefer l1_ratio over deprecated penalty switches.
    return LogisticRegression(
        max_iter=500,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.5,
        tol=1e-3,
        random_state=0,
    )


def make_hgb_clf():
    return HistGradientBoostingClassifier(random_state=0, max_iter=80)


def make_ridge():
    return Ridge()


def make_hgb_reg():
    return HistGradientBoostingRegressor(random_state=0, max_iter=80)


def make_mixed_hgb_pipeline():
    """Pipeline that accepts DataFrame with numeric + categorical columns."""

    def _factory():
        # Column names are set after peeking at data in run helpers; use a flexible pipe.
        # We detect column types at fit time via a small wrapper pattern: build in problems
        # after loading X. For registry, return a callable that builds given X.
        raise RuntimeError("use build_mixed_pipeline(X)")

    return _factory


def build_mixed_pipeline(X: pd.DataFrame, *, model: str = "hgb"):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    if model == "hgb":
        # HistGradientBoosting supports native categorical via ordinal encoding.
        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "ord",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=60)
        return Pipeline([("pre", pre), ("model", clf)])

    # LogisticRegression + one-hot
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=400, solver="saga", penalty="elasticnet", l1_ratio=0.5, tol=1e-3
    )
    return Pipeline([("pre", pre), ("model", clf)])


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


def phase0_problems() -> list[Problem]:
    return [
        Problem(
            problem_id="synth_clf_logreg",
            task="classification",
            scoring="neg_log_loss",
            load_xy=_synthetic_clf,
            make_estimator=make_logreg,
            specs=LOGREG_SPECS,
            n_iter=20,
            cv=3,
        ),
        Problem(
            problem_id="breast_cancer_logreg",
            task="classification",
            scoring="neg_log_loss",
            load_xy=_breast_cancer,
            make_estimator=make_logreg,
            specs=LOGREG_SPECS,
            n_iter=20,
            cv=3,
        ),
    ]


def phase1_problems() -> list[Problem]:
    return [
        Problem(
            problem_id="breast_cancer_logreg",
            task="classification",
            scoring="neg_log_loss",
            load_xy=_breast_cancer,
            make_estimator=make_logreg,
            specs=LOGREG_SPECS,
            n_iter=40,
            cv=3,
        ),
        Problem(
            problem_id="breast_cancer_hgb",
            task="classification",
            scoring="neg_log_loss",
            load_xy=_breast_cancer,
            make_estimator=make_hgb_clf,
            specs=HGB_CLF_SPECS,
            n_iter=40,
            cv=3,
        ),
        Problem(
            problem_id="digits_logreg",
            task="classification",
            scoring="neg_log_loss",
            load_xy=_digits_multiclass,
            make_estimator=make_logreg,
            specs=LOGREG_SPECS,
            n_iter=25,
            cv=3,
        ),
        Problem(
            problem_id="digits_hgb",
            task="classification",
            scoring="neg_log_loss",
            load_xy=_digits_multiclass,
            make_estimator=make_hgb_clf,
            specs=HGB_CLF_SPECS,
            n_iter=25,
            cv=3,
        ),
        Problem(
            problem_id="diabetes_ridge",
            task="regression",
            scoring="neg_mean_squared_error",
            load_xy=_diabetes,
            make_estimator=make_ridge,
            specs=RIDGE_SPECS,
            n_iter=40,
            cv=3,
        ),
        Problem(
            problem_id="diabetes_hgb",
            task="regression",
            scoring="neg_mean_squared_error",
            load_xy=_diabetes,
            make_estimator=make_hgb_reg,
            specs=HGB_REG_SPECS,
            n_iter=35,
            cv=3,
        ),
        # Secondary proper rule check (binary only)
        Problem(
            problem_id="breast_cancer_logreg_brier",
            task="classification",
            scoring="neg_brier_score",
            load_xy=_breast_cancer,
            make_estimator=make_logreg,
            specs=LOGREG_SPECS,
            n_iter=30,
            cv=3,
        ),
    ]


def phase2_problems() -> list[dict[str, Any]]:
    """Phase-2 problems need DataFrame-aware pipeline construction."""
    return [
        {
            "problem_id": "synth_mixed_hgb",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _synthetic_mixed_clf,
            "model": "hgb",
            "specs": PIPE_HGB_SPECS,
            "n_iter": 30,
            "cv": 3,
        },
        {
            "problem_id": "synth_mixed_logreg",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _synthetic_mixed_clf,
            "model": "logreg",
            "specs": PIPE_LOGREG_SPECS,
            "n_iter": 30,
            "cv": 3,
        },
        {
            "problem_id": "credit_g_hgb",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _openml_credit_g,
            "model": "hgb",
            "specs": PIPE_HGB_SPECS,
            "n_iter": 25,
            "cv": 3,
        },
        {
            "problem_id": "adult_small_hgb",
            "task": "classification",
            "scoring": "neg_log_loss",
            "load_xy": _openml_adult_small,
            "model": "hgb",
            "specs": PIPE_HGB_SPECS,
            "n_iter": 20,
            "cv": 3,
        },
    ]
