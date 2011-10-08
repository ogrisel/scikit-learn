"""
=================================================================
Clustering text documents using MiniBatchKmeans - Model selection
=================================================================

"""
print __doc__

# Author: Olivier Grisel <olivier.grisel@ensta.org>

# License: Simplified BSD

from time import time
import numpy as np
import pylab as pl
import scipy.sparse as sp

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import Vectorizer
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.utils import gen_even_slices
from sklearn.cluster import MiniBatchKMeans

###############################################################################
# Experiment parameters

rng = np.random.RandomState(1)
n_clusters_range = np.arange(2, 21)
n_runs = 10
categories = [
    'rec.autos',
    'rec.motorcycles',

    'rec.sport.baseball',
    'rec.sport.hockey',

    'alt.atheism',
    'talk.religion.misc',

    'talk.politics.misc',

    'sci.space',

    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
]

# parameters for the clustering model
parameters = {
    "init": "random",
    "max_iter": 10,
    "chunk_size": 1000,
    "verbose": 0,
    "random_state": rng,
    "tol": 1e-4,
}

###############################################################################
# Load some categories from the training set

print "Loading 20 newsgroups dataset"
dataset = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=rng)

documents = dataset.data
labels = dataset.target
target_names = dataset.target_names
n_samples = len(documents)

print "%d documents" % n_samples
print "%d categories" % len(target_names)
print

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_df=0.95, max_features=10000)
X_orig = vectorizer.fit_transform(documents)
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X_orig.shape
print

###############################################################################
# Perform a grid search for the optimal number of clusters

print "Estimating the consensus for various values of n_clusters"

scores = np.zeros((n_clusters_range.shape[0], n_runs))

for j in range(n_runs):
    t0 = time()

    # sample two overlapping sub-datasets
    X = shuffle(X_orig, random_state=rng)
    X_a, X_b, X_c = [X[s] for s in gen_even_slices(n_samples, 3)]

    X_ab = sp.vstack((X_a, X_b)).tocsr()
    X_ac = sp.vstack((X_a, X_c)).tocsr()

    n_common = X_a.shape[0]

    for i, k in enumerate(n_clusters_range):
        # find a clustering for each sub-sample
        model_ab = MiniBatchKMeans(k, **parameters).fit(X_ab)
        model_ac = MiniBatchKMeans(k, **parameters).fit(X_ac)

        # extract the label assignment on the overlap
        labels_ab = model_ab.labels_[:n_common]
        labels_ac = model_ac.labels_[:n_common]

        # measure the stability of k as the agreement of the clusterings on the
        # overlap
        scores[i, j] = metrics.adjusted_rand_score(labels_ab, labels_ac)

    print "Run %d/%d done in %0.2fs" % (j + 1, n_runs, time() - t0)

print

n_top = 10
best_mean_scores = scores.mean(axis=1).argsort()[:-(n_top + 1):-1]
print "Top %d optimal number of clusters in range:" % (n_top)
for idx in best_mean_scores:
    print "n_clusters=%d\tconsensus=%0.3f (%0.3f)" % (
        n_clusters_range[idx], scores.mean(axis=1)[idx],
        scores.std(axis=1)[idx])
print

pl.errorbar(n_clusters_range, scores.mean(axis=1), scores.std(axis=1))
pl.plot(n_clusters_range, scores.max(axis=1), 'k--', label='max')
pl.plot(n_clusters_range, scores.min(axis=1), 'k--', label='min')
pl.title("Consensus model selection for Document Clustering\n"
         "on the 20 newsgroups dataset")
pl.xlabel("Number of centers")
pl.ylabel("Mean Consensus ARI score on %d runs" % n_runs)
pl.show()

###############################################################################
# Displaying the learned centers in feature space

n_clusters = n_clusters_range[best_mean_scores[0]]
print "Fitting a model on the complete data with n_clusters=%d" % n_clusters
t0 = time()
model = MiniBatchKMeans(n_clusters, **parameters).fit(X_orig)
inverse_vocabulary = dict((v, k) for k, v in vectorizer.vocabulary.iteritems())
print "done in %fs" % (time() - t0)
print

n_top_words = 10
for center_idx, center in enumerate(model.cluster_centers_):
    print "Cluster #%d:" % center_idx
    print " ".join([inverse_vocabulary[i]
                    for i in center.argsort()[:-(n_top_words + 1):-1]])
    print
