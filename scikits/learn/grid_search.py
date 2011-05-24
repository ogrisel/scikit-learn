"""Tune the parameters of an estimator by cross-validation"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD Style.

import copy
import time

import numpy as np
import scipy.sparse as sp

from .externals.joblib import Parallel, delayed, logger
from .cross_val import KFold, StratifiedKFold
from .base import BaseEstimator, is_classifier, clone


try:
    from itertools import product
except ImportError:
    def product(*args, **kwds):
        pools = map(tuple, args) * kwds.get('repeat', 1)
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)


class IterGrid(object):
    """Generators on the combination of the various parameter lists given

    Parameters
    -----------
    kwargs: keyword arguments, lists
        Each keyword argument must be a list of values that should
        be explored.

    Returns
    --------
    params: dictionary
        Dictionnary with the input parameters taking the various
        values succesively.

    Examples
    ---------
    >>> from scikits.learn.grid_search import IterGrid
    >>> param_grid = {'a':[1, 2], 'b':[True, False]}
    >>> list(IterGrid(param_grid)) #doctest: +NORMALIZE_WHITESPACE
    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
     {'a': 2, 'b': True}, {'a': 2, 'b': False}]

    """
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        param_grid = self.param_grid
        if hasattr(param_grid, 'has_key'):
            param_grid = [param_grid]
        for p in param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params


def fit_grid_point(X, y, base_clf, clf_params, train, test, loss_func,
                   score_func, verbose, **fit_params):
    """Run fit on one set of parameters

    Returns the score and the instance of the classifier
    """
    start_time = time.time()
    if verbose > 1:
        msg = '%s' % (', '.join('%s=%s' % (k, v)
                                for k, v in clf_params.iteritems()))
        print "[GridSearchCV] %s %s" % (msg, (64 - len(msg)) * '.')
    # update parameters of the classifier after a copy of its base structure
    clf = copy.deepcopy(base_clf)
    clf._set_params(**clf_params)

    if isinstance(X, list) or isinstance(X, tuple):
        if train.dtype == np.bool:
            # array mask
            X_train = [X[i] for i, cond in enumerate(train) if cond]
            X_test = [X[i] for i, cond in enumerate(test) if cond]
        else:
            # assume indices
            X_train = [X[i] for i in train]
            X_test = [X[i] for i in test]
    else:
        if sp.issparse(X):
            # For sparse matrices, slicing only works with indices
            # (no masked array). Convert to CSR format for efficiency and
            # because some sparse formats don't support row slicing.
            X = sp.csr_matrix(X)
            ind = np.arange(X.shape[0])
            train = ind[train]
            test = ind[test]
        X_train = X[train]
        X_test = X[test]

    if y is not None:
        # supervised learning: score or loss in computed w.r.t. user provided
        # ground truth
        y_test = y[test]
        y_train = y[train]

        clf.fit(X_train, y_train, **fit_params)

        if loss_func is not None:
            y_pred = clf.predict(X_test)
            score = -loss_func(y_test, y_pred)
        elif score_func is not None:
            y_pred = clf.predict(X_test)
            score = score_func(y_test, y_pred)
        else:
            score = clf.score(X_test, y_test)

        n_test_samples = y.shape[0]

    else:
        # unsupervised learning: score or loss is computed w.r.t. ability to
        # find compatible 'predictions' the test set that is concatenated to
        # halves of the splitted training set. This is especially useful to
        # evaluate clustering parameters by measuring the stability of the
        # label assignments using a symmetric measure such as the
        # v_measure_score
        split = X_train.shape[0] / 2
        n_test_samples = X_test.shape[0]

        if isinstance(X, list) or isinstance(X, tuple):
            X_a = X_test + X_train[:split]
            X_b = X_test + X_train[split:]
        elif sp.issparse(X):
            # train and test are integer indices
            X_a = X[np.concatenate((test, train[:split]))]
            X_b = X[np.concatenate((test, train[split:]))]
        else:
            # general array case
            X_a = np.concatenate((X_test, X_train[:split]))
            X_b = np.concatenate((X_test, X_train[split:]))

        # fit models on overlapping subsets and evaluate the stability of the
        # predictions
        if hasattr(clf, 'fit_predict'):
            labels_a = clf.fit_predict(X_a)[:n_test_samples]
            labels_b = copy.deepcopy(clf).fit_predict(X_b)[:n_test_samples]
        else:
            clf_a = clf.fit(X_a)
            clf_b = copy.deepcopy(clf).fit(X_b)
            if hasattr(clf_a, 'labels_'):
                # backward compat with generic clustering API
                labels_a = clf_a.labels_[:n_test_samples]
                labels_b = clf_b.labels_[:n_test_samples]
            else:
                # expect the predict method for clustering models
                labels_a = clf_a.predict(X_test)
                labels_b = clf_b.predict(X_test)

        # loss or score functions are expected to be symmetric
        if loss_func is not None:
            score = -loss_func(labels_a, labels_b)
        elif score_func is not None:
            score = score_func(labels_a, labels_b)
        else:
            # XXX: clf_a.predict is probably redundant in that case...
            score = clf_a.score(X_test, labels_b)

    duration = time.time() - start_time
    if verbose > 1:
        end_msg = "%s - %s" % (msg, logger.short_format_time(duration))
        print "[GridSearchCV] %s %s" % ((64 - len(end_msg)) * '.', end_msg)
    return score, duration, clf, n_test_samples


class GridSearchCV(BaseEstimator):
    """Grid search on the parameters of a classifier

    Important members are fit, predict.

    GridSearchCV implements a "fit" method and a "predict" method like
    any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation

    Parameters
    ----------
    estimator: object type that implements the "fit" and "predict" methods
        A object of that type is instanciated for each grid point

    param_grid: dict
        a dictionary of parameters that are used the generate the grid

    loss_func: callable, optional
        function that takes 2 arguments and compares them in
        order to evaluate the performance of prediciton (small is good)
        if None is passed, the score of the estimator is maximized

    score_func: callable, optional
        function that takes 2 arguments and compares them in
        order to evaluate the performance of prediciton (big is good)
        if None is passed, the score of the estimator is maximized

    fit_params : dict, optional
        parameters to pass to the fit method

    n_jobs: int, optional
        number of jobs to run in parallel (default 1)

    iid: boolean, optional
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : crossvalidation generator
        see scikits.learn.cross_val module

    refit: boolean
        refit the best estimator with the entire dataset

    verbose: integer
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    grid_scores_ : list of (dict(), float) pairs
        Store the (iid) mean score for each parameters dictionary

    scores_ : array with shape [n_grid_points, n_folds]
        Store the row scores of individual fits. In case a loss_func was used
        we have score is defined as `score = -loss`

    durations_ : array with shape [n_grid_points, n_folds]
        Store the recorded durations of the fits in seconds.

    params_ : list of dict
        Parameters matching the first axis of scores_ and durations_

    best_params_: dict
        Combination of parameter values that scored best on average.

    Examples
    --------
    >>> from scikits.learn import svm, grid_search, datasets
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = grid_search.GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(n_jobs=1, verbose=0, fit_params={}, loss_func=None...
           cv=None, iid=True,
           estimator=SVC(kernel='rbf', C=1.0, probability=False,...
           ...

    >>> from pprint import pprint
    >>> pprint(clf.params_)
    [{'C': 1, 'kernel': 'linear'},
     {'C': 1, 'kernel': 'rbf'},
     {'C': 10, 'kernel': 'linear'},
     {'C': 10, 'kernel': 'rbf'}]

    >>> clf.scores_
    array([[ 0.96,  0.98,  1.  ],
           [ 0.92,  0.88,  0.9 ],
           [ 0.96,  0.98,  0.96],
           [ 0.96,  0.94,  0.98]])

    >>> clf.durations_.shape
    (4, 3)

    >>> pprint(clf.best_params_)
    {'C': 1, 'kernel': 'linear'}

    Notes
    ------

    The parameters selected are those that maximize the score of the
    left out data, unless an explicit score_func is passed in which
    case it is used instead. If a loss function loss_func is passed,
    it overrides the score functions and is minimized.

    """

    def __init__(self, estimator, param_grid, loss_func=None, score_func=None,
                 fit_params={}, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0):
        assert hasattr(estimator, 'fit') and (hasattr(estimator, 'predict')
                        or hasattr(estimator, 'score')), (
            "estimator should a be an estimator implementing 'fit' and "
            "'predict' or 'score' methods, %s (type %s) was passed" %
                    (estimator, type(estimator)))

        if loss_func is None and score_func is None:
            assert hasattr(estimator, 'score'), ValueError(
                    "If no loss_func is specified, the estimator passed "
                    "should have a 'score' method. The estimator %s "
                    "does not." % estimator)

        self.estimator = estimator
        self.param_grid = param_grid
        self.loss_func = loss_func
        self.score_func = score_func
        self.n_jobs = n_jobs
        self.fit_params = fit_params
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose

    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters

        Returns the best classifier

        Parameters
        ----------

        X: array, [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y: array, [n_samples] or None
            Target vector relative to X, None for unsupervised problems

        """
        self._set_params(**params)
        estimator = self.estimator
        cv = self.cv
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            # support list of unstructured objects on which feature
            # extraction will be applied later in the tranformer chain
            n_samples = len(X)
        if y is not None and len(y) != n_samples:
            raise ValueError('Target variable (y) has a different number '
                    'of samples (%i) than data (X: %i samples)' %
                        (len(y), n_samples))
        if cv is None:
            if y is not None and is_classifier(estimator):
                cv = StratifiedKFold(y, k=3)
            else:
                cv = KFold(n_samples, k=3)

        grid = IterGrid(self.param_grid)
        base_clf = clone(self.estimator)
        # XXX: Need to make use of Parallel's new pre_dispatch
        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fit_grid_point)(
                X, y, base_clf, clf_params, train, test, self.loss_func,
                self.score_func, self.verbose, **self.fit_params)
                    for clf_params in grid for train, test in cv)

        # out is a list of tuples: score, duration, estimator, n_test_samples
        n_grid_points = len(list(grid))
        n_fits = len(out)
        n_folds = n_fits // n_grid_points

        # Group results for consecutive folds on the same parameters set
        durations = np.empty((n_grid_points, n_folds))
        scores = np.empty((n_grid_points, n_folds))
        n_test_samples = np.empty((n_grid_points, n_folds))
        estimators = list()
        for i in range(0, n_grid_points):
            slice_ = slice(i * n_folds, (i + 1) * n_folds)
            for j, (score, duration, clf, n) in enumerate(out[slice_]):
                scores[i, j] = score
                durations[i, j] = duration
                n_test_samples[i, j] = n
            estimators.append(clf)

        # compute the mean score for each estimator
        if self.iid:
            # XXX: do we really need this special case or should we assume
            # that all folds have approximately the same size?
            n = n_test_samples
            mean_scores = np.sum(scores * n, axis=1) / n.sum(axis=1)
        else:
            mean_scores = scores.mean(axis=1)

        # Note: we do not use max() to make ties deterministic even if
        # comparison on estimator instances is not deterministic
        best_score = None
        for score, estimator, params in zip(mean_scores, estimators, grid):
            if best_score is None:
                best_score = score
                best_estimator = estimator
                best_params = params
            else:
                if score > best_score:
                    best_score = score
                    best_estimator = estimator
                    best_params = params

        if best_score is None:
            raise ValueError('Best score could not be found')
        self.best_score = best_score
        self.best_params_ = best_params

        if self.refit:
            # fit the best estimator using the entire dataset
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)

        self.best_estimator = best_estimator
        if hasattr(best_estimator, 'predict'):
            self.predict = best_estimator.predict
        if hasattr(best_estimator, 'score'):
            self.score = best_estimator.score

        # Store the computed scores
        # XXX: the name is too specific, it shouldn't have 'grid' in it.
        self.grid_scores_ = [
            (params, score) for params, score in zip(grid, mean_scores)]
        self.params_ = list(grid)
        self.scores_ = scores
        self.durations_ = durations

        return self

    def score(self, X, y=None):
        # This method is overridden during the fit if the best estimator
        # found has a score function.
        y_predicted = self.predict(X)
        return self.score_func(y, y_predicted)
