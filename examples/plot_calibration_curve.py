"""
=============================
Probability Calibration curve
=============================

When performing classfication you often want to predict, not only
the class label, but also the associated probability. This probability
gives you some kind of confidence on the prediction. This example
demonstrates how to transform the decisition function of a generic
classifier into a probability and how to display how well
calibrated is the predicted probability.

"""
print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD Style.

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import train_test_split

data = datasets.fetch_covtype()
X = data.data
y = data.target

# Take only the first 2 classes
X = X[y < 3]
y = y[y < 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,
                                                    random_state=42)

# Logistic regression with no calibration
lr = LogisticRegression(C=1., solver='lbfgs')

# Gaussian Naive-Bayes with no calibration
gnb = GaussianNB()

# Gaussian Naive-Bayes with isotonic calibration
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method='isotonic')

# Gaussian Naive-Bayes with sigmoid calibration
gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method='sigmoid')

###############################################################################
# Plot calibration plots

print("-- Brier scores:")

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (gnb_isotonic, 'Naive Bayes + Isotonic'),
                  (gnb_sigmoid, 'Naive Bayes + Sigmoid')]:
    clf.fit(X_train, y_train)
    prob_pos = clf.predict_proba(X_test)[:, 1]
    clf_score = brier_score_loss(y_test, prob_pos)
    print("%s: %1.3f" % (name, clf_score))
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s (%1.3f)" % (name, clf_score))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="best")

plt.tight_layout()
plt.show()
