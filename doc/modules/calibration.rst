.. _calibration:

=======================
Probability calibration
=======================

.. currentmodule:: sklearn.calibration


When performing classification you often want not only to predict the class
label, but also obtain a probability of the respective label.  This probability
gives you some kind of confidence on the prediction.  Some models can give you
poor estimates of the class probabilities and some even do not not support
probability prediction. The calibration module allows you to better calibrate
the probabilities of a given model, or to add support for probability
prediction.

Well calibrated classifiers are probabilistic classifiers for which the output
of the predict_proba method can be directly interpreted as a confidence level.
For instance, a well calibrated (binary) classifier should classify the samples
such that among the samples to which it gave a predict_proba value close to 0.8,
approx. 80% actually belong to the positive class. The following plot compares
how well the probabilistic predictions of different classifiers are calibrated:

.. figure:: ../auto_examples/images/plot_compare_calibration_001.png
   :target: ../auto_examples/plot_compare_calibration.html
   :align: center

:class:`LogisticRegression` returns well calibrated predictions as it directly
optimizes log-loss. In contrast, the other methods return biased probabilities;
with different biases per method:

 * :class:`GaussianNaiveBayes` tends to push probabilties to 0 or 1 (note the 
   counts in the histograms). This is mainly because it makes the assumption 
   that  features are conditionally independent given the class, which is not 
   the case in  this dataset which contains 2 redundant features.

 * :class:`RandomForestClassifier` shows the opposite behavior: the histograms
   show peaks at approx. 0.2 and 0.9 probability, while probabilities close to
   0 or 1 are very rare. An explanation for this is given by Niculescu-Mizil 
   and Caruana [4]: "Methods such as bagging and random forests that average
   predictions from a base set of models can have difficulty making predictions
   near 0 and 1 because variance in the underlying base models will bias
   predictions that should be near zero or one away from these values. Because
   predictions are restricted to the interval [0,1], errors caused by variance 
   tend to be one-sided near zero and one. For example, if a model should 
   predict p = 0 for a case, the only way bagging can achieve this is if all 
   bagged trees predict zero. If we add noise to the trees that bagging is 
   averaging over, this noise will cause some trees to predict values larger 
   than 0 for this case, thus moving the average prediction of the bagged 
   ensemble away from 0. We observe this effect most strongly with random 
   forests because the base-level trees trained with random forests have 
   relatively high variance due to feature subseting." As a result, the 
   calibration curve shows a characteristic sigmoid shape, indicating that the
   classifier could trust its "intuition" more and return probabilties closer 
   to 0 or 1 typically.

 * Support Vector Classification (:class:`SVC`) shows a similar sigmoid curve
   as  the  RandomForestClassifier, which is typical for maximum-margin methods
   (compare Niculescu-Mizil and Caruana [4]).

Two approaches for performing calibration of probabilistic predictions are
provided: a parametric approach based on Platt's sigmoid model and a  non-
parametric approach based on isotonic regression (:mod:`sklearn.isotonic`).
Probability calibration should be done on new data not used for model fitting.
The class :class:`CalibratedClassifierCV` uses a cross-validation generator and
estimates for each split the model parameter on the train samples and the
calibration of the test samples. The probabilities predicted obtained for each
folds are then averaged. Already fitted classifiers can be calibrated by
:class:`CalibratedClassifierCV` via the paramter cv="prefit". In this case,
the user has to take care manually that data for model fitting and calibration
is disjoint.

The following images demonstrate the benefit of probability calibration.
The first image present a dataset with 2 classes and 3 blobs of
data. The blob in the middle contains random samples of each class.
The probability for the samples in this blob should be 0.5.

.. figure:: ../auto_examples/images/plot_calibration_001.png
   :target: ../auto_examples/plot_calibration.html
   :align: center

The following image shows on the data above the estimated probability
using a Gaussian naive Bayes classifier without calibration,
with a sigmoid calibration and with a non-parametric isotonic
calibration. One can observe that the non-parametric model
provides the most accurate probability estimates for samples
in the middle, i.e., 0.5.

.. figure:: ../auto_examples/images/plot_calibration_002.png
   :target: ../auto_examples/plot_calibration.html
   :align: center

The last image shows on the covertype dataset the estimated probabilities
obtained with :class:`LogisticRegression`, :class:`GaussianNB`, and
:class:`GaussianNB` with both isotonic calibration and sigmoid calibration. The
calibration performance is evaluated with Brier score
:func:`metrics.brier_score_loss`, reported in the legend (the smaller the
better). One can observe here that logistic regression is well
calibrated while raw Gaussian naive Bayes performs very badly. Its  calibration
curve is above the diagonal which indicates that it's classification is
imbalanced and it classifies many positive example as negative (bad precision).
Calibration of the probabilities of Gaussian naive Bayes with isotonic 
regression can fix this issue as can be seen from the nearly diagonal 
calibration curve. Sigmoid calibration also improves the brier score, albeit 
not as strongly as the non-parametric isotonic regression. This can be 
attributed to the fact that we have plenty of calibration data such that the
greater flexibility of the non-parametric model can be exploited.

.. figure:: ../auto_examples/images/plot_calibration_curve_001.png
   :target: ../auto_examples/plot_calibration_curve.html
   :align: center

.. topic:: References:

    .. [1] Obtaining calibrated probability estimates from decision trees
          and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
          Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
          Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning, 
          A. Niculescu-Mizil & R. Caruana, ICML 2005
