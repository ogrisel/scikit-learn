How to select the right algorithm for the task
==============================================

To conclude this session here are some practical hints for selecting
the right algorithm when facing a practical problem.

- If the data is high dimensional and sparse (text data), most of the time
  linear classifiers with a bit of regularization will work well.

- If the data is dense, low to medium dimensional: try to further reduce the
  dimensionality with PCA for instance and try both linear and non linear
  models (e.g. SVC with RBF kernel).

- ``SVC`` with gaussian RBF kernel and ``KMeans`` clustering can
  benefit a lot from data normalization either with feature feature with
  scaling using ``Scaler`` or doing whitening with ``RandomizedPCA``
  with ``whiten=True``. Try various values for ``n_components`` with
  grid search to be sure no to truncate the data too hard.

- There is no free lunch: the best algorithm is data-dependent. If
  you try many different models, reserve a held out evaluation set
  that is not used during the model selection process.

A comprehensive practical guide / FAQ / Howto is under work. Stay tuned!

  http://scikit-learn.org
