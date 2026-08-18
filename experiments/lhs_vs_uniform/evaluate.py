"""Sequential HPO evaluation with anytime curves under proper scoring rules."""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

from samplers import ParamSpec, sample_search_space


@dataclass
class TrialResult:
    params: dict[str, Any]
    score: float
    fit_time: float
    cum_time: float
    best_score_so_far: float
    n_eval: int


@dataclass
class SearchRun:
    method: str
    problem_id: str
    seed: int
    scoring: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def scores(self) -> np.ndarray:
        return np.array([t.score for t in self.trials], dtype=float)

    @property
    def cum_times(self) -> np.ndarray:
        return np.array([t.cum_time for t in self.trials], dtype=float)

    @property
    def best_scores(self) -> np.ndarray:
        return np.array([t.best_score_so_far for t in self.trials], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "problem_id": self.problem_id,
            "seed": self.seed,
            "scoring": self.scoring,
            "trials": [asdict(t) for t in self.trials],
        }


def evaluate_candidates(
    estimator: BaseEstimator,
    X,
    y,
    candidates: Sequence[dict[str, Any]],
    *,
    scoring: str,
    cv: int | Any,
    method: str,
    problem_id: str,
    seed: int,
    n_jobs: int = 1,
    error_score: float = np.nan,
) -> SearchRun:
    """Evaluate candidates sequentially; record score and wall time per trial."""
    from sklearn.model_selection import KFold, StratifiedKFold

    # Fixed CV splitter seeded for fairness across methods.
    if isinstance(cv, int):
        y_arr = np.asarray(y)
        use_strat = (
            y_arr.dtype.kind in "iu"
            and len(np.unique(y_arr)) <= max(20, len(y_arr) // 10)
            and len(np.unique(y_arr)) > 1
        )
        if use_strat:
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        else:
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        cv_obj = cv

    run = SearchRun(
        method=method, problem_id=problem_id, seed=seed, scoring=scoring
    )
    best = -np.inf
    cum = 0.0
    for i, params in enumerate(candidates, start=1):
        est = clone(estimator)
        est.set_params(**params)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    est,
                    X,
                    y,
                    scoring=scoring,
                    cv=cv_obj,
                    n_jobs=n_jobs,
                    error_score=error_score,
                )
            score = float(np.nanmean(scores))
            if np.isnan(score):
                score = -np.inf
        except Exception:
            score = -np.inf
        elapsed = time.perf_counter() - t0
        cum += elapsed
        best = max(best, score)
        run.trials.append(
            TrialResult(
                params=params,
                score=score,
                fit_time=elapsed,
                cum_time=cum,
                best_score_so_far=best,
                n_eval=i,
            )
        )
    return run


def run_paired_search(
    estimator: BaseEstimator,
    X,
    y,
    specs: Sequence[ParamSpec],
    *,
    n_iter: int,
    scoring: str,
    cv: int,
    seed: int,
    problem_id: str,
    n_jobs: int = 1,
) -> dict[str, SearchRun]:
    """Run uniform and LHS with paired seeds (distinct streams, same base seed)."""
    results: dict[str, SearchRun] = {}
    for method, method_seed in (("uniform", seed), ("lhs", seed + 10_000)):
        candidates = sample_search_space(specs, n_iter, method, method_seed)
        results[method] = evaluate_candidates(
            estimator,
            X,
            y,
            candidates,
            scoring=scoring,
            cv=cv,
            method=method,
            problem_id=problem_id,
            seed=seed,
            n_jobs=n_jobs,
        )
    return results
