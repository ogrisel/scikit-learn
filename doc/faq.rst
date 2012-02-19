.. title:: Frequently Asked Questions and Howtos

This section will try to organize the most frequent questions asked on
the :ref:`mailing lists <mailing_lists>` or Q&A forums.

Machine Learning is very large field and more often than not there is no
generic answer to a generic question. This FAQ is thus built in a manner
to help the newcomers ask themselves the right questions to narrow down
on the right tools and methods for their specific issues and discover
the relevant sections of the documentation along with worked examples
that can be re-used or adapted.



Which machine learning algorithm to use?
========================================

What is the nature of the problem I am trying to solve?

- Predict a category assignment based on passed observed assignments?

  Example: categorize emails as spam or not spam.

  See :ref:`which_algorithm_for_classification`

- Predict one or more continous variable(s) based on passed obsevered values?

  Example: predict the some gene expression level depending on the activity of
  other genes.

  See :ref:`which_algorithm_for_regression`

- Gather similar "things" into a finite number of groups or clusters?

  Example: identify main customer profiles for a marketing campaign

  See :ref:`which_algorithm_for_clustering`

- Decompose data into a finite number of prototypes?

  Example: find main topics of a corpus of document or protype faces for face
  recognition.

  See :ref:`which_algorithm_for_decomposition`

- Identify the structure or graph of relationships beetween agents
  of a system?

  Example: identify the structure of correlated tickers in a stock market

  See :ref:`which_algorithm_for_structure_identification`

- Build a recommender system to suggest interesting "things" to the
  users based on their history

  Examples: suggest products a user would be more likely to buy

  See :ref:`how_to_recommender_system`



Which algorithm for classification?
===================================

Is the data linearly separable? See

Is the natural representation of the input sparse?

  -

Are there many samples in the training set?


Which algorithm for multi-label classification?
===============================================



How to increase the classification performance?
===============================================

- ensure you have a sound way to measure the performance
  :ref:`howto_measure_performance_classification`

- check that the data is preprocessed correctly
  :ref:`howto_preprocess`

- perform a grid search for the best parameters, see
  :ref:`howto_choose_parameters_classification`

- eliminate noisy features

- use unsupervied data to improve the model: semi supervised and active
  learning


How to measure the performance of a classification algorithm?
=============================================================


How to choose the parameters of a classification algorithm?
===========================================================

Explain grid search / model selection:


How to pre-process the data?
============================

Many machine learning algorithms expect that input variables are
approximately centered aroun zeros or at least have roughly the same
scale: the feature wise variances be close to 1.0.

Regression models can further expect the target variable to be scaled to
the [-1, 1] range as well.

The documentation section on :ref:`preprocessing` introduces the various
tools that can be used to preprocess data in scikit-learn.

See also :ref:`howto_whitening`. # TODO: document whitening in the
preprocessing chapter.

Text datasets generally have pre-processing steps that can be computed
more efficiently in cooperation with the code that extracts the numerical
features from the raw text strings. See :ref:`howto_text_data`.

.. topic:: Impact of preprocessing on performance evaluation

   When measuring performance of a model by using train / test splits,
   the preprocessing step should be included as part of the model.

   In practice that means that the normalization parameters (such as
   the the center position or the scales of the raw features) should be
   "learned" on the training set only and then reused to transform
   the test dataset in a consistent manner.

   The use of the Transformer API and the sklearn.pipeline.Pipeline class


What is whitening, how and when to use it?
==========================================


How to deal with text data?
===========================


How to deal with cagegorical data?
==================================


How to deal with time-based data?
=================================


How to deal with geo-location data?
===================================


How to deal with image data?
============================


How to deal with samples with very few non zero features?
=========================================================


Is my data linearly separable?
==============================

For classification
------------------

For any pair of classes, can the two subsets be reasonably correctly
separated by an hyperplane?

If yes then a (regularized or not) linear classifier will be able to
achieve maximum performance and non linear models will likely bring only
overfitting issues.

If not, linear model will fail due to their lack of degrees of freedom and more
complex models such as non linear kernel support vector machines or random
forests of decision trees.


For regression
--------------

Can the target variable iso-surface be reasonably be approximated by
hyperplanes on the whole domain of interest? if so (regularized or not)
linear regression model will work.

If the iso-surface have "hills" or some other kind of local structures,
linear regression will not have enough degrees of freedom to model
them. Non linear kernel regression models or forest of trees will be
required.


For clustering
--------------

Can any two clusters be separated by an hyperplane?

If so algorithms that assume that the clusters can be represented by
the position of their centers will work as expected (e.g. k-means).

If not they will likely fail and models based on the spectral structure
of some affinity matrix will likely be able to capture this non-regular
structure.

Example of half moons dataset.


Which algorithm for regression?
===============================


- Very large number of dimensions / input variables (more than 10k):

  - Non linear regression would likely to bring overfitting and be
    non-tractable).

  - Even linear regression can over-fit, hence regularization is likely
    necessary.

  - If I can make the assumption that only a few (unidentified
    variables) are relevant to determine the value of the target variable:

    LassoLars (dense input only), Lasso, ElasticNet,
    SGDRregressor with penalty='l1' or 'elasticnet'.

  - Otherwise: RidgeRegression (small to medium number of samples)
    or SGDRegressor with penalty = 'l2' for large number of samples.


- Small to medium number of dimensions:

  - linear model: RidgeRegression

  - non-linear models: SVR, NuSVR or ExtraTreesRegressor.


- How many target variables?

  If more than one, only RidgeRegression is able it fit and predict for
  several targets at once. Other models need to consider each variable
  separately for now.


How to measure the performance of a regression model?
=====================================================




How to find the parameters of a regression model?
=================================================

- lambda path, AIC and BIC

- grid search for SVR and regression trees



Which algorithm for clustering?
===============================

- Is the input data sparse and or ?

  - KMeans

  - MiniBatchKMeans

  - SpetralClustering using a sparse kernel as affinity matrix.

- Is the number of samples large (e.g. more than 50k)

  - MiniBatchKMeans

  - SpetralClustering with a truncated k-NN

- Is the data known to fail to have cluster with non regular shapes
  (e.g. clusters folder around one another)

  - if the number of cluster is small (less than 10)

    - SpetralClustering

  - else:

    - Ward clustering with locality constraints (dense data only for now).


See also the documentation section on :ref:`clustering`.


How to filter non-important features (observered variables)?
============================================================



How to measure the performance of clustering algorithm?
=======================================================



How to choose the number of clusters?
=====================================

- if for exploratory purpose

- let the algorithm decide: mean shift

- if for feature extraction: pickup an arbitrary large yet tracktable
  number of cluster (e.g. 100 for MiniBatchKMeans)


What is sparse data and what is it good for?
============================================

Sparse data is a dataset where the majority of the feature values are
zeros.

Examples:

 - features extracted from text documents: most documents use a very small
   fraction of all the existing words

 - transactional data: most customers of a e-commerce shop have only
   bought a tiny fraction of all the available products for sale on the
   website (the same remark applies for advertisement clicks historical
   data).

 - graph data: the structure of a graph (with edges and vertices) can
   be represented by an squared adjancency matrix where non-zero components
   encode the weights of the edges connecting two vertices. Most social network
   graph data is very sparse: a profile is typically connected to a few
   hundreds of other profiles out of millions.

The main consequence is that is often not possible to represent the all the
features (zeros and non-zeros) explicitly in memory with a traditional numpy
array:

Suppose we have of corpus of  50000 documents with each of them having
1000 distinct words on average out of a total vocabulary of 100000
possible words. Using a numpy array to store the word frequencies as
double precision would require::

  50000. * 100000 * 8 / (1024 ** 3) = 37 GB

This cannot be allocated in the main memory of today's laptops and would
be completely wasteful to do so even if it was possible.

Instead it is possible to only represent the non-zero values and their position
in the virtual 2D matrix.

The ``scipy.sparse`` package features various representations. The following
tutorial explains how to build and manipulate such datastructures.

  http://scipy-lectures.github.com/advanced/scipy_sparse/index.html

If a scipy.sparse matrix is used for representing the word frequencies of our
previous examples we will have approximately to allocate::

  50000 * 1000 * 2 * 8 / (1024 ** 2) == 762 MB

Which is much more reasonable and can be handled and processed on today
consumer laptops.

In scikit-learn we mostly use:

- the COO representation as simple and flexible way to build sparse
  matrices before conversion to either CSR or CSC.

- the CSR representation for algorithms that scan the data along the
  samples axis (most of the algorithm). Notable examples include
  the :ref:`sgd` linear models SGDClassifier and
  SGDRegressor and the MiniBatchKMeans clustering algorithm.

- the CSC representation for the few algorithms that scan data along the
  feature axis. Notable examples include Coordinate Descent linear
  regression models Lasso & ElasticNet.

Read the docstring of the model you plan to use so as to limit the
number of memory allocations by choosing the optimal representation from
the start.

Also note that only the CSR format can be efficiently sliced / indexed
along the samples axis so as to form cross-validation folds for instance.


How to get help efficiently on the mailing list?
================================================

- which platform (Linux, Max, Windows?), which version of scikit-learn, numpy,
  scipy, was scikit-learn build from source?

- what is the goal (binary, multiclass, multilabel classification, regression,
  clustering, other?)

- if error: full error message / traceback

- what kind of data (text features, is so which?, numerical ranges, categorical
  features).

- how many samples, how many

- which preprocessing was applied

- minimalistic reproduction script (10 / 20 lines) + sample data files on
  gist.github.com for instance
