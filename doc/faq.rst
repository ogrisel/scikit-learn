.. title:: Frequently Asked Questions and Howtos

This section will try to organize the most frequent questions asked on
the :ref:`mailing lists <mailing_lists>` or Q&A forums.

Machine Learning is very large field and more often than not there is no
generic answer to a generic question. This FAQ is thus built in a manner
to help the newcomers ask themselves the right questions to narrow down
on the right tools and methods for their specific issues and discover
the relevant sections of the documentation along with worked examples
that can be re-used or adapted.


I am a noob, where should I start?
==================================

1- Learn about Python and the Scientific Python ecosystem (numpy, scipy,
   matplotlib...) by the first chapters of this `tutorial
   <http://scipy-lectures.github.com/>`_

2- Follow the `scikit-learn tutorial <getting_started>` to get a quick
   introduction to the main machine learning concepts and how they map to
   the scikit-learn project conventions.

3- Depending on what you are trying to achieve with machine learning,
   start by asking yourself the :ref:`right questions <which_algorithm>`.

4- Learn more about machine learning:

   - `Good Freely Available Textbooks on Machine Learning
     <http://metaoptimize.com/qa/questions/186/>`_

   - `Good Freely Available Videos on Machine Learning
     <http://metaoptimize.com/qa/questions/258/>`_

   - `Good Machine Learning Blogs
     <http://metaoptimize.com/qa/questions/3163>`_

   You can also enroll in the free online Machine Learning class
   at http://coursera.org .


.. _which_algorithm:

Which machine learning algorithm to use?
========================================

What is the nature of the problem I am trying to solve?

- Predict a category assignment based on passed observed assignments?

  Example: categorize emails as spam or not spam.

  See :ref:`which_algorithm_for_classification`

- Predict one or more continous variable(s) based on passed obsevered
  values?

  Example: predict the some gene expression level depending on the
  activity of other genes.

  See :ref:`which_algorithm_for_regression`

- Gather similar "things" into a finite number of groups or clusters?

  Example: identify main customer profiles for a marketing campaign

  See :ref:`which_algorithm_for_clustering`

- Decompose data into a finite number of topics / components / atoms /
  factors / prototypes?

  Example: find the main topics of a corpus of document or prototype
  faces for face recognition.

  See :ref:`which_algorithm_for_decomposition`

- Identify the structure or graph of relationships beetween agents
  of a system?

  Example: identify the structure of correlated tickers in a stock market

  See :ref:`which_algorithm_for_structure_identification`

- Build a recommender system to suggest interesting "things" to the
  users based on their history

  Examples: suggest products a user would be more likely to buy based
  on past purchasing history and similarity with other user purchasing
  habits.

  See :ref:`howto_recommender_system`

- Build a model to detect novelty or anomaly in data?

  Examples: fraud detection in financial transactions or computer network
  monitoring.

  See :ref:`howto_novelty_and_anomaly_detection`


.. _what_is_the_best_algorithm:

What is the best machine learning algorithm?
============================================

Machine Learning is a wide field and different algorithms are
used to solve different kinds of machine learning problems, see
:ref:`which_algorithm`.

The bad news is that **even for a specific task, there is no free lunch**:
there is no algorithm that will be best (in terms of some predictive
accuracy metrics) than others on all possible datasets. Hence the usual
need to make assumptions on the data that reflects in the structure of
the algorithm (e.g. linear model assumes that the data is :ref:`linearly
separable <is_linearly_separable>`).

Finally there are several ways to define what is a "good" algorithm. In
particular the following properties are often desirable:

- **good predictive accuracy** evaluation both for supervised and for
  unsupervised models (see :ref:`howto_measure_performance_classification`
  and :ref:`howto_measure_performance_clustering`),

- **short training time** and good scalability w.r.t. number of samples and
  features,

- **short prediction time**,

- **low memory usage** both during during and after the training process,

- ability work in a distributed with few synchronization barriers on
  **mutiple cores** or even nodes in a HPC cluster,

- ability to increment an existing model when new data is available
  without restarting from scratch: **warm restarts** and **online
  learning**.

- ability to build **interpretable models** so that a human expert can
  gain new insight on the data fitted by the model: the absolute value and
  sign coefficients of a linear model can be interpreted as how important
  the matching feature is and whether it contributing positively or
  negatively to the likelyhood of the target class.

To demonstrate the tradeoffs at play, here are a some typical examples:

- :ref:`neighbors` models models have very short training times but
  potentially much higher prediction times and high memory usage.

- Some linear models such as :ref:`perceptron` or :ref:`sgd` are usually
  much faster to train, have a low memory usage (compared to the
  original training set) and very fast prediction times. However they
  make the string assumption that the data is :ref:`linearly separable
  <is_linearly_separable>`, hence will get poor predictive accuracy if
  it is not the case.

- More complex models such as :ref:`extra_trees` will be potentially
  much slower to train yet reasonably fast to make predictions. They
  will also be able to perform reasonably well on a much wider scope of
  datasets than linear models.


.. topic:: References:

  TODO: find a good reference that can serve as an intuitive intro to
  the No Free Lunch Theorem (or improve the wikipedia abstract to make
  it clearer).


Which algorithm for classification?
===================================

The first question to ask is
:ref:`Is the data linearly separable? <is_linearly_separable>`.

If you don't know, you can make the assumption first (the linear model
are simpler hence often faster to train and evaluate) and relax it
if it fails.


Classifiers that assume linearly separable data
-----------------------------------------------

- Is the natural representation of the input sparse?

  The following linear classifiers are able to deal with sparse input:

  - TODO list them and don't forget naive bayes

- Are there many samples in the training set?

  The following linear classifiers are known to be able to scale to a
  large number of samples efficiently (more than 100k samples):

  - :class:`sklearn.linear_model.Perceptron`
  - :class:`sklearn.linear_model.SGDClassifier`
  - :class:`sklearn.linear_model.LogisticRegression`
  - :class:`sklearn.linear_model.LinearSVC`

- Are there few samples in the training set?

  The following models are known to be able deal efficiently with datasets
  where `n_samples << n_features`

  - :class:`sklearn.linear_model.RidgeClassifier`
  - :class:`sklearn.linear_model.RidgeClassifierCV` (dense data only)
  - :class:`sklearn.naive_bayes.GaussianNB`
  - :class:`sklearn.naive_bayes.MultinomialNB`
  - :class:`sklearn.naive_bayes.BernoulliNB`


TODO: finalize the time vs f1 score plot on the text classification
models and link to it here.

Classifiers that don't assume linearly separable data
-----------------------------------------------------

TODO: kernel models and forests

TODO: Warning on scalability of SVC + link to RBF kernel approximation.


.. topic:: A good baseline: NearestNeighborClassifier

  NearestNeighborClassifier is not a very practical classifier as it
  keeps a copy of the training set in memory and has comparatively slow
  prediction times (but fast training times, especially in bruteforce
  mode as no training is done in that case).

  However it is a very simple model with very few hyper-parameters (the
  default should work well in many cases). Iti s thus a good practice
  to use it as a sanity check to evaluate the predictive performance of
  smarter classifiers against. See
  :ref:`howto_measure_performance_classification`.

  If the other classifier is performing significantly worst than
  NearestNeighborClassifier there might be a problem in its configuration,
  see :ref:`howto_improve_classification_perf`.


Which algorithm for multi-label classification?
===============================================

TODO



.. _howto_improve_classification_perf:

How to improve the classification performance?
==============================================

- Ensure you have a sound way to measure the performance:
  :ref:`howto_measure_performance_classification`

- Ensure that the quality of the dataset is at least as good as you expect
  it to be: hide the labels and make predictions manually by looking
  at the input data on 30 or more random samples and compare those
  predictions to the data set labels to have a rough idea of the precision
  of the labels. If your data has noisy labels don't expect the machine
  learning algorithm to do any better: **garbage in, garbage out**.

- Count the number of times a class is occurring in the training
  set. Ensure that you have enough training samples for each class: in
  general you should not expect to learn anything interesting with less
  than 10 samples per class. Better have at least hundreds or thousands
  samples per target classes.

- Ensure that the data has been preprocessed correctly:
  :ref:`howto_preprocess`

- Perform a grid search for the best parameters, see
  :ref:`howto_choose_parameters_classification`

- If one of the class is over/under-represented in this can break
  assumptions made by the classifier and make it perform poorly, see
  :ref:`how_imbalanced_data`.

- Analyze the error to understand whether you are most suffering from
  excessive bias or variance, see
  :ref:`howto_analyze_classification_error`

  If you have little bias (low training error) but high variance
  (significantly larger test error than training error) despite
  optimal selection of the regularization parameters through grid
  search, try to increase the bias with simpler models (such as
  ref:`linear_model` or :ref:`naive_bayes`), :ref:`dimensionality
  reduction <howto_dimensionality_eduction>` or :ref:`feature selection
  <howto_feature_selection>` to get rid of noisy features.

  If you have a high bias but low variance error profile, try to fit
  more complex models (such as :ref:`SVM with non linear kernels <svm>`
  or :ref:`forest`).

- Manually inspect samples that were badly classified and see what
  kind of additional features would help the model resolve such
  ambiguities.

  TODO: point on an example that does that.

If you are still unhappy you can:

- try to use unsupervied data to improve the model
  with semi-supervised and / or active learning.

- check whether you need to deal with :ref:`covariate shift correction
  <howto_covariate_shift_correction>`.


How to measure the performance of a classification algorithm?
=============================================================

TODO Explain re-balancing a dataset.


.. _howto_analyze_classification_error:

How to analyze the classification error?
========================================

TODO: explain bias and variance concepts

TODO: simple case: measure train and test error

TODO: more CPU intensive analysis: learning curves

Wrap https://gist.github.com/1540431 as an sklearn example and include some
sample plots.

Link to:

  http://digitheadslabnotebook.blogspot.com/2011/12/practical-advice-for-applying-machine.html

.. topic:: What if I get both high bias and variance?

  If you do a grid search for the regularization parameter and the
  optimal value still has both significant bias and variance it means
  that assumptions made by the algorithm do not hold: the optimal does
  not lie on this regularization path or that the data quality to too
  bad to get anything useful out of it.

  If after manual inspection you are pretty confident that the
  data is good.  Try more complex models such as for instance
  :ref:`extra_trees`. Some people say that they always work if you have
  powerful enough hardware :)


How to choose the parameters of a classification algorithm?
===========================================================

Explain grid search / model selection.

TODO: link to narrative doc

TODO: link to examples


.. _howto_preprocess:

How to pre-process the data?
============================

Many machine learning algorithms expect that input variables are
approximately centered aroun zeros or at least have roughly the same
scale: the feature wise variances be close to 1.0.

.. topic:: A special case: decision trees and random forests

  The recommendation in this section do not necessarily apply to all
  algorithms. In particular decision tree-based models are known to be
  robust to unscaled features.

  See the sections on :ref:`tree` and :ref:`forest`.

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

TODO


How to deal with text data?
===========================

TODO


How to deal with cagegorical data?
==================================

TODO


How to deal with time-based data?
=================================

TODO


How to deal with geo-location data?
===================================

TODO


How to deal with image data?
============================

TODO


.. _howto_imbalanced_data:

How to deal with imbalanced data (in a classification problem)?
===============================================================

List classifiers that support the `class_weight='auto'` parameter.


.. _howto_covariate_shift_correction:

How to deal with biased label distributions (covariate shift correction)?
=========================================================================

TODO explain howto detect, write an example in scikit learn, point to
it along with a link to:

http://blog.smola.org/post/4110255196/real-simple-covariate-shift-correction


.. _howto_very_sparse_samples:

How to deal with very sparse samples?
=====================================

Very sparse datasets (e.g. samples with around 20 non zeros out of
a vocabulary of hundred thousands such as short social network text
messages) can be hard to classify because of the sparsity it-self.

This is especially true when the number of labeled samples is low but
that we have a much larger amount of unlabeled data at our disposal:
some feature might not be seen in the labeled training set while being
highly correlated to another feature that is discriminant. Unsupervised
methods such as PCA or k-means clustering can be used to exploit such
correlated features in an unsupervised way to as to build new topical
features with a broader coverage to feed them the classifier.

TODO: write and example demonstrating how to do semi-supervised learning
or transductive learning on short text classification with PCA or k-means
and link to it.


How to use PCA for classification?
==================================

Principal Component Analysis is a fundamentally unsupervised algorithm:
it completely ignores any provided supervised labels to find the principal
component directions.

However PCA is often used as a dimensionality reduction preprocessing
step for training supervised classifiers.

Preprocessing with PCA can also be useful when classifying when working
very sparse high dimensional data: each sample can have very few non
zero features causing the classifier to perform badly by being unable
to detect synonymic features: see :ref:`howto_very_sparse_samples`.

TODO: add links to the doc and examples.

TODO: add RandomizedPCA to the preprocessing documentation along with
a simple Pipeline example to demonstrate the transformer pattern.



.. _is_linearly_separable:

Is my data linearly separable?
==============================

TODO merge in the examples from the tutorials into the main doc so as
to be able reference it with an internal link here.

In practice very high dimensional datasets (for instance features
extracted from text or transactional data) are more likely to be
approximately linearly separable.

Lower dimensional datasets such as the ones used in computer vision,
audio or neuro imagery related tasks are likely to be non linearly
separable as the measured data generally a low dimensional projection
of hidden higher dimensional manifolds.


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

In practice on can quickly tests whether a problem as a chance
to be linearly separable by trying to fit a simple linear model
such as :class:`sklearn.linear_model.Perceptron`. If the results
is good (f1-score more than 0.8) then the linear separability
assumption might be reasonable. A more complete :ref:`error analysis
<howto_analyze_classification_error>` will help further decide whether
the linearly separable assumption is a good bias or not.


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

TODO


How to find the parameters of a regression model?
=================================================

TODO

- lambda path, AIC and BIC

- grid search for SVR and regression trees



Which algorithm for clustering?
===============================

TODO: here is a skeleton

- Is the input data sparse (see :ref:`what_is_sparse_data`)?

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


.. _howto_feature_selection:

How to filter non-important features (observered variables)?
============================================================

TODO


.. _howto_clustering_perf:

How to measure the performance of clustering algorithm?
=======================================================

TODO


How to choose the number of clusters?
=====================================

TODO

- if for exploratory purpose

- let the algorithm decide: mean shift

- if for feature extraction: pickup an arbitrary large yet tracktable
  number of cluster (e.g. 100 for MiniBatchKMeans)

- stability selection : write an example that works first :)


.. _what_is_sparse_data:

What is sparse data and what is it good for?
============================================

Sparse data is a dataset where the majority of the feature values are
zeros.

Examples:

 - **Features extracted from text documents**: most documents use a
   very small fraction of all the existing words.

 - **Transactional data**: most customers of a e-commerce shop have only
   bought a tiny fraction of all the available products for sale on the
   website (the same remark applies for advertisement clicks historical
   data).

 - **Graph data**: the structure of a graph (with edges and vertices) can
   be represented by an squared adjancency matrix where non-zero
   components encode the weights of the edges connecting two
   vertices. Most social network graph data is very sparse: a profile
   is typically connected to a few hundreds of other profiles out of
   millions.

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

If a ``scipy.sparse`` matrix is used for representing the word frequencies
of our previous examples we will have approximately to allocate::

  50000 * 1000 * 2 * 8 / (1024 ** 2) == 762 MB

Which is much more reasonable and can be handled and processed on today
consumer laptops. Note that this is not just a problem of memory: if
the algorithms skips the zeros (for instance when doing a dot product
of two samples) much less data has to be processed and the processing
can be much faster too.

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


.. _how_to_get_help:

How to get help efficiently on the mailing list?
================================================

If this FAQ, the documentation and the examples do not answer your
questions, please feel free to subscribe to the :ref:`project mailing
list <mailing_lists>` to ask them there.

In order to maximize the chances to get useful replies please make sure
give details on the following:

- Which platform (Linux, Max, Windows?), which version of scikit-learn,
  numpy, scipy, was scikit-learn build from source?

- What is the primary type of task your are trying to achieve: binary
  classification, multiclass classification, multilabel classification,
  regression, clustering, other? (See :ref:`which_algorithm`)

- If you get an error when trying to use scikit-learn, please include the
  full error message (including the traceback) and a minimalistic script
  to reproduce it.

- What kind of data are you dealing with: text features, is so which?,
  numerical ranges, categorical features...

- Which preprocessing was applied (centering, variance scaling, TF-IDF)?

- How many samples? How many features where extracted? Are you using
  a :ref:`sparse reprensation <what_is_sparse_data>`?

Whenever possible, **please include a minimalistic reproduction script**
(e.g. 10-20 lines) along with sample data files on http://gist.github.com
for instance (note that gists are regular git repositories that you can
clone and hence upload small to medium data files there as well).

The mailing list system will refuse emails with large attachements so
please use gists to upload the datafile and reproduction scripts and
just send the URL to the gist in the email to the mailing list.
