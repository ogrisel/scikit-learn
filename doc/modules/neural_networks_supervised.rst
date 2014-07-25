.. _neural_network:

==================================
Neural network models (supervised)
==================================

.. currentmodule:: sklearn.neural_network


.. _elm:

Extreme Learning Machines
=========================
**Extreme Learning Machines (ELM)** is a supervised nonlinear learning 
algorithm that takes the following consecutive steps for training:

  *  it applies random projection on the input space to a possibly larger dimensional space;
  *  the result passes through an element-wise non-linear activation function 
     such as a tanh, or sigmoid function; and
  *  last, it trains a linear one versus all classifier or a multi-output ridge regression model.

Therefore ELM can be considered as a single layer feed forward neural network (see figure 1) where:

  *  it trains only the hidden-to-output connection weights;
  *  assigns random values, as constants, to the input-to-hidden connection weights; and
  *  minimizes the loss function using least squares.

ELM was found to generalize to 1 hidden layer MLPs or RBF kernel support vector
machines on a variety of problems while being significantly faster to train.

The main hyper-parameters of the ELM are:

  *  the variance of the random projection weights;
  *  the number of hidden layer nodes; and
  *  the regularization strength of the output linear model.

.. figure:: ../images/elm_network.png
     :align: center
     :scale: 60%

     Figure 1 : Single-hidden layer feedforward network.

This implements :ref:`classification <elm_classification>` for classification 
and :ref:`regression <elm_regression>` for regression. They support both dense
(``numpy.ndarray`` and convertible to that by ``numpy.asarray``) and
sparse (any ``scipy.sparse``) sample vectors as input. They also support
real-time and batch-based training.


.. _elm_classification:

Classification
==============

:class:`ELMClassifier` supports binary-class, multi-class, and
multi-label classification on datasets.

Like other classifiers, :class:`ELMClassifier` trains on a dataset using the
fit method accepting two input parameters: an array X of size 
``(n_samples, n_features)`` representing the training samples, and an array y 
of integer values, size ``(n_samples)``, holding the class labels respective
to the training samples::


    >>> from sklearn.neural_network import ELMClassifier
    >>> X = [[0, 0], [1, 1]]
    >>> y = [0, 1]
    >>> clf = ELMClassifier()
    >>> clf.fit(X, y)
    ELMClassifier(C=1, activation='tanh', algorithm='standard', batch_size=200,
    class_weight=None, coef0=0.0, degree=3, gamma=None, kernel='random',
    n_hidden=500, random_state=None, verbose=False, weight_scale='auto')

After training, the model can predict the class of a new sample::

    >>> clf.predict([[2., 2.]])
    array([1])


.. _elm_bin_multi_class:

Binary- and multi-class classification
--------------------------------------

In binary- and multi-class classification, the network has one output neuron.
For binary-class classification - or when samples belong to one of two classes - 
the output passes through the logistic function. If the result is higher than
0.5 than the corresponding samples is assigned to class 1, and class 0 
otherwise.

In the other hand, multi-class classification problems involve datasets 
with samples that belong to a class from a pool of more than two classes. 

The output for a sample passes through softmax which returns a set of probabilities 
corresponding to each class. The class representing the highest probability
becomes the class assigned to the sample.


Weighted classification
-----------------------

For datasets containing an imbalanced ratio of samples between classes, it is
important to consider adding more weight to classes having less samples than
others. 

:class:`ELMClassifier` has the parameter ``class_weight`` in
its ``fit`` method that can be set either to ``None`` - where all samples
are treated the same; ``auto`` - where weight is given to a sample based
on its class distribution in the dataset; and a dictionary of the form
``{class_label : value}`` - where value is a floating point number > 0
that sets the parameter ``C`` of class ``class_label`` to ``C * value``.

Figure [2] shows the result of adding weight to the underrepresented 
class - the orange samples, where the decision boundary sorrounding that 
class becomes larger in radius.

.. figure:: ../images/plot_weighted_classification_elm.png
   :align: center
   :scale: 75

   Figure 2 : Effect of sample weighting.


.. _elm_regression:

Regression
==========

:class:`ELMRegressor` can solve regression problems consisting of one or more
output values assigned to each sample. It differs from :class:`ELMClassifier`, 
in that the final output is the value returned by the method ``decision_function``.

The fit method in :class:`ELMRegressor` accepts arguments X and y
where y is expected to be a matrix of floating point values::

    >>> from sklearn.neural_network import ELMRegressor
    >>> X = [[0, 0], [2, 2]]
    >>> y = [0.5, 2.5]
    >>> clf = ELMRegressor()
    >>> clf.fit(X, y)
    ELMRegressor(C=1000000.0, activation='tanh', algorithm='standard',
    batch_size=200, coef0=0.0, degree=3, gamma=None, kernel='random',
    n_hidden=100, random_state=None, verbose=False, weight_scale='auto')
    >>> clf.predict([[1, 1]])
    array([[ 1.58556094]])


Tips on Practical Use
=====================

  * **Setting C**: if the dataset contains noisy samples, decrease ``C`` 
    to reduce overfitting. In fact, it is best to use grid-search to find
    the best value of ``C`` from a set of values returned by for example 
    ``np.logspace(1, 5, 5)``.

  * **Setting class_weight**: for classification, if the dataset 
    contains an imbalanced ratio for classes, it is desirable to give 
    more weight to the class having less number of samples. For example,
    if class 1 is the minority class, you can give the more weight by
    setting class_weight={1:100}.

  * **Setting the algorithm**: if the dataset is small enough to fit into 
    memory then setting algorithm='standard' would be desirable. Otherwise,
    in cases where memory is limited or there is a need for real-time learning,
    it is desirable to set the algorithm='sequential'.

.. _elm_kernels:

Kernel functions
================

The *kernel function* can be any of the following:

  * linear: :math:`\langle x, x'\rangle`.

  * polynomial: :math:`(\gamma \langle x, x'\rangle + r)^d`. `d` is specified by
    keyword ``degree``, `r` by ``coef0``.

  * rbf: :math:`\exp(-\gamma |x-x'|^2)`. :math:`\gamma` is
    specified by keyword ``gamma``, must be greater than 0.

  * sigmoid (:math:`\tanh(\gamma \langle x,x'\rangle + r)`), where `r` is specified by
    ``coef0``.

Different kernels are specified by keyword kernel at initialization::

    >>> elm_linear = ELMClassifier(kernel='linear')
    >>> elm_linear.kernel
    'linear'
    >>> elm_rbf = ELMClassifier(kernel='rbf')
    >>> elm_rbf.kernel
    'rbf'

Mathematical formulation
========================

A standard ELM with ``kernel='random'`` trains a single-hidden layer
feedforward network using the following function,

:math:`f(X) = \beta (W^TX + b)`.

``X`` is the matrix representing the samples; ``W`` and ``b`` are matrices of 
random values assigned based on a uniform distribution ranging from ``-a`` and ``a``,
where ``a`` is a user-defined value; and :math:`\beta` is the matrix that 
ELM solves using least-square.

This implementation uses ridge regression which solves for :math:`\beta`
in the following formulation,

:math:`inv(X^T X + (1/C)*Id) * X^T y`.

C is a regularization term which controls the linearity of the decision
function. Smaller value of C makes the decision boundary more linear.


.. topic:: References:

 *  Zong, Weiwei, Guang-Bin Huang, and Yiqiang Chen.
      "Weighted extreme learning machine for imbalance learning."
      Neurocomputing 101 (2013): 229-242.

 *  Liang, Nan-Ying, et al.
      "A fast and accurate online sequential learning algorithm for
      feedforward networks." Neural Networks, IEEE Transactions on
      17.6 (2006): 1411-1423.
      http://www.ntu.edu.sg/home/egbhuang/pdf/OS-ELM-TNN.pdf

