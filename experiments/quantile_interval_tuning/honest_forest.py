"""Honest quantile forest with per-tree honesty and pinball splits.

For each tree:

1. Draw a bootstrap sample and take its unique in-bag indices.
2. Split those indices into a grow set and a disjoint honest set
   (Athey–Wager style, independently per tree).
3. Grow a ``DecisionTreeRegressor(criterion="quantile")`` on the grow set
   (pinball impurity for split selection and grow-set leaf values).
4. Replace each leaf value by the empirical quantile of the honest samples
   that fall in that leaf (pinball minimizer on the honest set). Leaves with
   no honest sample keep the grow-set quantile.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data


def _overwrite_honest_leaves(tree, X_hon, y_hon, quantile):
    if X_hon.shape[0] == 0:
        return
    leaf_ids = tree.apply(X_hon)
    values = tree.tree_.value
    for leaf in np.flatnonzero(tree.tree_.children_left == TREE_LEAF):
        mask = leaf_ids == leaf
        if np.any(mask):
            values[leaf, 0, 0] = np.quantile(y_hon[mask], quantile)


def _fit_one_honest_tree(
    X,
    y,
    *,
    quantile,
    honest_fraction,
    seed,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    max_features,
    min_grow,
):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    inbag = np.unique(rng.randint(0, n, size=n))
    rng.shuffle(inbag)
    n_hon = int(np.round(honest_fraction * len(inbag)))
    n_hon = min(max(n_hon, 1), len(inbag) - min_grow)
    if n_hon < 1 or len(inbag) - n_hon < min_grow:
        grow_idx = inbag
        hon_idx = np.array([], dtype=int)
    else:
        hon_idx = inbag[:n_hon]
        grow_idx = inbag[n_hon:]

    tree = DecisionTreeRegressor(
        criterion="quantile",
        quantile=quantile,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=seed,
    )
    tree.fit(X[grow_idx], y[grow_idx])
    if hon_idx.size:
        _overwrite_honest_leaves(tree, X[hon_idx], y[hon_idx], quantile)
    return tree


class HonestQuantileForest(BaseEstimator, RegressorMixin):
    """Pinball-split forest with a fresh honesty split on every tree."""

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=20,
        min_samples_split=2,
        max_features=1.0,
        honest_fraction=0.5,
        quantile=0.5,
        n_jobs=1,
        random_state=0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.honest_fraction = honest_fraction
        self.quantile = quantile
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        X = validate_data(self, X, dtype=np.float64, accept_sparse=False)
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        rng = check_random_state(self.random_state)
        seeds = rng.randint(0, np.iinfo(np.int32).max, size=self.n_estimators)
        min_grow = max(2 * int(self.min_samples_leaf), int(self.min_samples_split), 2)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_honest_tree)(
                X,
                y,
                quantile=self.quantile,
                honest_fraction=self.honest_fraction,
                seed=int(seed),
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                min_grow=min_grow,
            )
            for seed in seeds
        )
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, dtype=np.float64, reset=False)
        pred = np.mean([est.predict(X) for est in self.estimators_], axis=0)
        return pred
