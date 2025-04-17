"""
==================================================
Probability Calibration for 3-class classification
==================================================

This example illustrates how sigmoid :ref:`calibration <calibration>` changes
predicted probabilities for a 3-class classification problem. Illustrated is
the standard 2-simplex, where the three corners correspond to the three
classes. Arrows point from the probability vectors predicted by an uncalibrated
classifier to the probability vectors predicted by the same classifier after
sigmoid calibration on a hold-out validation set. Colors indicate the true
class of an instance (red: class 1, green: class 2, blue: class 3).

"""

# %%
# Data
# ----
# Below, we generate a classification dataset with 2000 samples, 2 features
# and 3 target classes. We then split the data as follows:
#
# * train: 600 samples (for training the classifier)
# * valid: 400 samples (for calibrating predicted probabilities)
# * test: 1000 samples

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sklearn.datasets import make_blobs

n_train = 1_000
n_cal = 5_000
n_test = 30_000
X, y = make_blobs(
    n_samples=n_train + n_cal + n_test,
    n_features=2,
    centers=3,  # 3 classes to represent the simplex in 2D
    cluster_std=5.0,
    shuffle=True,
    random_state=42,
)
X_train, y_train = X[:n_train], y[:n_train]
X_valid, y_valid = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
X_test, y_test = X[n_train + n_cal :], y[n_train + n_cal :]

# %%
# Fitting and calibration
# -----------------------
#
# First, we will train a :class:`~sklearn.ensemble.RandomForestClassifier`
# with 25 base estimators (trees) on the concatenated train and validation
# data (1000 samples). This is the uncalibrated classifier.

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

# clf = RandomForestClassifier(n_estimators=25, max_depth=3)
# clf = GaussianNB()
clf = make_pipeline(
    SplineTransformer(),
    PolynomialFeatures(interaction_only=True, include_bias=False),
    LogisticRegression(C=1e6),
)
# clf = make_pipeline(
#     SplineTransformer(),
#     PolynomialFeatures(interaction_only=True, include_bias=False),
#     LogisticRegression(C=1e-1),
# )
clf.fit(X_train, y_train)

# %%
# To train the calibrated classifier, we start with the same
# :class:`~sklearn.ensemble.RandomForestClassifier` but train it using only
# the train data subset (600 samples) then calibrate, with `method='sigmoid'`,
# using the valid data subset (400 samples) in a 2-stage process.

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

cal_clf = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
cal_clf.fit(X_valid, y_valid)

# %%
# Compare probabilities
# ---------------------
# Below we plot a 2-simplex with arrows showing the change in predicted
# probabilities of the test samples.

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
colors = ["r", "g", "b"]

uncal_clf_probs = clf.predict_proba(X_test)
cal_clf_probs = cal_clf.predict_proba(X_test)

# Plot arrows
arrow_alpha = 0.2
grid_alpha = 0.1
for i in range(min(uncal_clf_probs.shape[0], 500)):
    plt.arrow(
        uncal_clf_probs[i, 0],
        uncal_clf_probs[i, 1],
        cal_clf_probs[i, 0] - uncal_clf_probs[i, 0],
        cal_clf_probs[i, 1] - uncal_clf_probs[i, 1],
        color=colors[y_test[i]],
        head_width=1e-2,
        alpha=arrow_alpha,
    )

# Plot perfect predictions, at each vertex
plt.plot([1.0], [0.0], "ro", ms=20, label="Class 1")
plt.plot([0.0], [1.0], "go", ms=20, label="Class 2")
plt.plot([0.0], [0.0], "bo", ms=20, label="Class 3")

# Plot boundaries of unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

# Annotate points 6 points around the simplex, and mid point inside simplex
plt.annotate(
    r"($\frac{1}{3}$, $\frac{1}{3}$, $\frac{1}{3}$)",
    xy=(1.0 / 3, 1.0 / 3),
    xytext=(1.0 / 3, 0.23),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.plot([1.0 / 3], [1.0 / 3], "ko", ms=5)
plt.annotate(
    r"($\frac{1}{2}$, $0$, $\frac{1}{2}$)",
    xy=(0.5, 0.0),
    xytext=(0.5, 0.1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($0$, $\frac{1}{2}$, $\frac{1}{2}$)",
    xy=(0.0, 0.5),
    xytext=(0.1, 0.5),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($\frac{1}{2}$, $\frac{1}{2}$, $0$)",
    xy=(0.5, 0.5),
    xytext=(0.6, 0.6),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($0$, $0$, $1$)",
    xy=(0, 0),
    xytext=(0.1, 0.1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($1$, $0$, $0$)",
    xy=(1, 0),
    xytext=(1, 0.1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
plt.annotate(
    r"($0$, $1$, $0$)",
    xy=(0, 1),
    xytext=(0.1, 1),
    xycoords="data",
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
    verticalalignment="center",
)
# Add grid
plt.grid(False)
for x in np.linspace(0, 1, 11):
    plt.plot([0, x], [x, 0], "k", alpha=grid_alpha)
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=grid_alpha)
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=grid_alpha)

plt.title("Change of predicted probabilities on test samples after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
_ = plt.legend(loc="best")

# %%
# In the figure above, each vertex of the simplex represents
# a perfectly predicted class (e.g., 1, 0, 0). The mid point
# inside the simplex represents predicting the three classes with equal
# probability (i.e., 1/3, 1/3, 1/3). Each arrow starts at the
# uncalibrated probabilities and end with the arrow head at the calibrated
# probability. The color of the arrow represents the true class of that test
# sample.
#
# The uncalibrated classifier is overly confident in its predictions and
# incurs a large :ref:`log loss <log_loss>`. The calibrated classifier incurs
# a lower :ref:`log loss <log_loss>` due to two factors. First, notice in the
# figure above that the arrows generally point away from the edges of the
# simplex, where the probability of one class is 0. Second, a large proportion
# of the arrows point towards the true class, e.g., green arrows (samples where
# the true class is 'green') generally point towards the green vertex. This
# results in fewer over-confident, 0 predicted probabilities and at the same
# time an increase in the predicted probabilities of the correct class.
# Thus, the calibrated classifier produces more accurate predicted probabilities
# that incur a lower :ref:`log loss <log_loss>`
#
# We can show this objectively by comparing the :ref:`log loss <log_loss>` of
# the uncalibrated and calibrated classifiers on the predictions of the 1000
# test samples. Note that an alternative would have been to increase the number
# of base estimators (trees) of the
# :class:`~sklearn.ensemble.RandomForestClassifier` which would have resulted
# in a similar decrease in :ref:`log loss <log_loss>`.

from sklearn.metrics import log_loss

loss = log_loss(y_test, uncal_clf_probs)
cal_loss = log_loss(y_test, cal_clf_probs)

print("Log-loss of:")
print(f" - uncalibrated classifier: {loss:.3f}")
print(f" - calibrated classifier: {cal_loss:.3f}")

# %%
# We can also assess calibration with the Brier score for probabilistic
# predictions (lower is better, possible range is [0, 2]):

from sklearn.metrics import brier_score_loss

loss = brier_score_loss(y_test, uncal_clf_probs)
cal_loss = brier_score_loss(y_test, cal_clf_probs)

print("Brier score of")
print(f" - uncalibrated classifier: {loss:.3f}")
print(f" - calibrated classifier: {cal_loss:.3f}")

# %%
# According to the Brier score, the calibrated classifier is not better than
# the original model.
#
# Finally we generate a grid of possible uncalibrated probabilities over
# the 2-simplex, compute the corresponding calibrated probabilities and
# plot arrows for each. The arrows are colored according the highest
# uncalibrated probability. This illustrates the learned calibration map:

from scipy.stats import gmean

plt.figure(figsize=(10, 10))
# Generate grid of probability values
eps = np.finfo(np.float64).eps
p1d = np.linspace(0, 1, 21)
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]
p = p.clip(0 + eps, 1 - eps)
logits = np.log(p / gmean(p, axis=1)[:, np.newaxis])

# Use the three class-wise calibrators to compute calibrated probabilities.

calibrated_classifier = cal_clf.calibrated_classifiers_[0]
calibrated_predictions = np.vstack(
    [
        calibrator.predict(logit_for_class)
        for calibrator, logit_for_class in zip(
            calibrated_classifier.calibrators, logits.T
        )
    ]
).T

# Re-normalize the calibrated predictions to make sure they stay inside the
# simplex. This same renormalization step is performed internally by the
# predict method of CalibratedClassifierCV on multiclass problems.
calibrated_predictions /= calibrated_predictions.sum(axis=1)[:, None]

# Plot changes in predicted probabilities induced by the calibrators
for i in range(calibrated_predictions.shape[0]):
    plt.arrow(
        p[i, 0],
        p[i, 1],
        calibrated_predictions[i, 0] - p[i, 0],
        calibrated_predictions[i, 1] - p[i, 1],
        head_width=1e-2,
        color=colors[np.argmax(p[i])],
        alpha=arrow_alpha,
    )

# Plot the boundaries of the unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

plt.grid(False)
for x in np.linspace(0, 1, 11):
    plt.plot([0, x], [x, 0], "k", alpha=grid_alpha)
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=grid_alpha)
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=grid_alpha)

plt.title("Learned sigmoid calibration map")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.show()

# %%
# One can observe that, on average, the calibrator is pushing highly confident
# predictions away from the boundaries of the simplex while simultaneously
# moving uncertain predictions towards one of three modes, one for each class.
# We can also observe that the mapping is not symmetric. Furthermore some
# arrows seems to cross class assignment boundaries which is not necessarily
# what one would expect from a calibration map as it means that some predicted
# classes will change after calibration.
#
# All in all, the One-vs-Rest multiclass-calibration strategy implemented in
# `CalibratedClassifierCV` should not be trusted blindly.

# %%
