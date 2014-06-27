"""
Testing for Extreme Learning Machines module (sklearn.neural_network)
"""

# Author: Issam H. Laradji
# Licence: BSD 3 clause

import sys

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_equal

from sklearn.datasets import load_digits, load_boston
from sklearn.datasets import make_regression, make_multilabel_classification
from sklearn.externals.six.moves import cStringIO as StringIO
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import ELMClassifier
from sklearn.neural_network import ELMRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2
from scipy.sparse import csr_matrix
from sklearn.utils.testing import assert_raises, assert_greater, assert_equal


np.seterr(all='warn')

random_state = 1

ACTIVATION_TYPES = ["logistic", "tanh"]

digits_dataset_multi = load_digits(n_class=3)

Xdigits_multi = digits_dataset_multi.data[:200]
ydigits_multi = digits_dataset_multi.target[:200]
Xdigits_multi -= Xdigits_multi.min()
Xdigits_multi /= Xdigits_multi.max()

digits_dataset_binary = load_digits(n_class=2)

Xdigits_binary = digits_dataset_binary.data[:200]
ydigits_binary = digits_dataset_binary.target[:200]
Xdigits_binary -= Xdigits_binary.min()
Xdigits_binary /= Xdigits_binary.max()

classification_datasets = [(Xdigits_multi, ydigits_multi),
                           (Xdigits_binary, ydigits_binary)]

boston = load_boston()

Xboston = StandardScaler().fit_transform(boston.data)[: 200]
yboston = boston.target[:200]


def test_classification():
    """
    Tests whether ELMClassifier scores higher than 0.95 for binary-
    and multi-classification digits datasets
    """
    for X, y in classification_datasets:
        X_train = X[:150]
        y_train = y[:150]
        X_test = X[150:]

        expected_shape_dtype = (X_test.shape[0], y_train.dtype.kind)

        for activation in ACTIVATION_TYPES:
            elm = ELMClassifier(n_hidden=50, activation=activation,
                                random_state=random_state)
            elm.fit(X_train, y_train)

            y_predict = elm.predict(X_test)
            assert_greater(elm.score(X_train, y_train), 0.95)
            assert_equal(
                (y_predict.shape[0],
                 y_predict.dtype.kind),
                expected_shape_dtype)


def test_regression():
    """
    Tests whether ELMRegressor achieves score higher than 0.95 for the
    boston dataset
    """
    X = Xboston
    y = yboston
    for activation in ACTIVATION_TYPES:
        elm = ELMRegressor(n_hidden=150, activation=activation)
        elm.fit(X, y)
        assert_greater(elm.score(X, y), 0.95)


def test_multilabel_classification():
    """
    Tests whether multi-label classification works as expected
    """
    # test fit method
    X, y = make_multilabel_classification(
        n_samples=50, random_state=random_state)
    for activation in ACTIVATION_TYPES:
        elm = ELMClassifier(n_hidden=50,
                            activation=activation,
                            random_state=random_state)
        elm.fit(X, y)
        assert_equal(elm.score(X, y), 1)


def test_multioutput_regression():
    """
    Tests whether multi-output regression works as expected
    """
    X, y = make_regression(
        n_samples=200, n_targets=5, random_state=random_state)
    for activation in ACTIVATION_TYPES:
        elm = ELMRegressor(n_hidden=300,
                           activation=activation,
                           random_state=random_state)
        elm.fit(X, y)
        assert_greater(elm.score(X, y), 0.95)


def test_params_errors():
    """Tests whether invalid parameters raise value error"""
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = ELMClassifier

    assert_raises(ValueError, clf(n_hidden=-1).fit, X, y)
    assert_raises(ValueError, clf(activation='ghost').fit, X, y)


def test_predict_proba_binary():
    """
    Tests whether predict_proba works as expected for binary class.
    """
    X = Xdigits_binary[:50]
    y = ydigits_binary[:50]

    clf = ELMClassifier(n_hidden=5)
    clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    (n_samples, n_classes) = y.shape[0], 2

    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    assert_equal(y_proba.shape, (n_samples, n_classes))
    assert_array_equal(proba_max, proba_log_max)
    assert_array_equal(y_log_proba, np.log(y_proba))

    assert_equal(roc_auc_score(y, y_proba[:, 1]), 1.0)


def test_predict_proba_multi():
    """
    Tests whether predict_proba works as expected for multi class.
    """
    X = Xdigits_multi[:10]
    y = ydigits_multi[:10]

    clf = ELMClassifier(n_hidden=5)
    clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    (n_samples, n_classes) = y.shape[0], np.unique(y).size

    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    assert_equal(y_proba.shape, (n_samples, n_classes))
    assert_array_equal(proba_max, proba_log_max)
    assert_array_equal(y_log_proba, np.log(y_proba))


def test_sparse_matrices():
    """
    Tests that sparse and dense input matrices
    yield equal output
    """
    X = Xdigits_binary[:50]
    y = ydigits_binary[:50]
    X_sparse = csr_matrix(X)
    elm = ELMClassifier(random_state=1, n_hidden=15)
    elm.fit(X, y)
    pred1 = elm.decision_function(X)
    elm.fit(X_sparse, y)
    pred2 = elm.decision_function(X_sparse)
    assert_almost_equal(pred1, pred2)
    pred1 = elm.predict(X)
    pred2 = elm.predict(X_sparse)
    assert_array_equal(pred1, pred2)
