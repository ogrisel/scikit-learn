.. _loading_data:

============
Loading data
============

All models from scikit-learn require input data for training and making
predictions to be loaded as NumPy_ arrays (or sometimes as sparse matrices
from SciPy_).

For instance, supervised model have are trained by calling ``model.fit(X_train,
y_train)`` and output predictions by calling ``model.predict(X_new)``. Most of
the time  ``X_train`` and ``X_new`` are expected to be NumPy arrays with
``dtype=numpy.float`` and a two dimensional shape ``(n_samples, n_features)``.
``y_train`` is expected to be one dimensional with ``shape=(n_samples,)`` and
``dtype=np.float`` for regression tasks or ``dtype=np.int`` for classification
tasks. See the docstring of your specific model class for more details.

- The rows of ``X_train`` are called the "samples" in scikit-learn and in the
Machine Learning community in general. They are also sometimes called "data
instances", "records" or "observations" in other communities.

- The columns of ``X_train`` are called the "features" are numerical
descriptors of each sample and are sometimes also called "attributes" or
"fields".

- ``y_train`` store the values of the target variable.

If you are new to Python and NumPy_ and don't understand what is are the
``shape`` and ``dtype`` of a NumPy array you should follow first follow a NumPy
tutorial such as the one from `scipy-lectures.github.io <http://scipy-
lectures.github.io/>`.


Loading data from CSV or TSV files
==================================

Comma

To illustrate

    >>> import tempfile
    >>> temp_folder = tempfile.mkdtemp(prefix='data_folder_')

Assuming we have collected some training data in a ``train.csv`` file with the
following lines:

    >>> train_csv_file = temp_folder + '/train.csv'
    >>> open(train_csv_file, 'wb').write("""\
    ... target,feature_1,feature_2,feature_2
    ... 0,4.5,1.0,.)


    >>> iris_lines = open(iris_csv_filename, 'rb').readlines()
    >>> for line in iris_lines[:5]:
    ...     print(line)
    150,4,setosa,versicolor,virginica
    5.1,3.5,1.4,0.2,0
    4.9,3.0,1.4,0.2,0
    4.7,3.2,1.3,0.2,0
    4.6,3.1,1.5,0.2,0

The first lines i


.. note:: Some CSV files can sometimes contain


Loading data from numerical files formats
=========================================

The NumPy / SciPy ecosystem provides many tools





Loading data from a database
============================



.. note::

  There also exists databases that are specialized in the storage of numerical
  data such as PyTables_ or SciDB_. Loading data from such specialized
  databases is often much more efficient than


Loading data from unstructured or semi-structured representations
=================================================================

If the natural data cannot be expressed naturally as 2D numerical arrays.


Combining arrays
================

It is possible to combine.

Scipy's sparse matrices can also be combined in a similar manner with the
``scipy.sparse.vstack`` and ``scipy.sparse.hstack`` functions.


What makes for good numerical features
======================================

Extracting good numerical representations from the raw data highly depends on
the task being achieved and the class of Machine Learning models used to
achieve that .

Many machine learning algorithm will make that assumption that 2 samples that
should be considered as "naturally close" to one another should have numerical
descriptors. That is the `Euclidean distance`_ of the matching row in ``X_train``
or ``X_new` should be smaller than the distance to other rows picked at random
in ``X_train``.


.. _NumPy: http://numpy.org
.. _SciPy: http://scipy.org
.. _PyTables: http://pytables.org
.. _`Euclidean distance`: http://en.wikipedia.org/wiki/Euclidean_distance

