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
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print "Loading 20 newsgroups dataset for categories:"
print categories

dataset = fetch_20newsgroups(subset='train', categories=categories,
                            shuffle=True, random_state=42)

documents = dataset.data
labels = dataset.target
target_names = dataset.target_names
true_n_clusters = np.unique(labels).shape[0]

n_clusters_range = np.arange(2, 21)
n_runs = 10
n_samples = len(documents)

print "%d documents" % n_samples
print "%d categories" % len(target_names)
print


print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_df=0.95, max_features=10000)
X = vectorizer.fit_transform(documents)
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape
print

scores = np.zeros((n_clusters_range.shape[0], n_runs))
rng = np.random.RandomState(42)

for j in range(n_runs):
    t0 = time()

    # sample two overlapping sub-datasets
    X = shuffle(X, random_state=rng)
    X_a, X_b, X_c = [X[s] for s in gen_even_slices(n_samples, 3)]

    X_ab = sp.vstack((X_a, X_b)).tocsr()
    X_ac = sp.vstack((X_a, X_c)).tocsr()

    n_common = X_a.shape[0]

    for i, k in enumerate(n_clusters_range):
        # find a clustering for each sub-sample
        model_ab = MiniBatchKMeans(k, init="random", max_iter=10,
                                   random_state=i, chunk_size=1000,
                                   verbose=0).fit(X_ab)

        model_ac = MiniBatchKMeans(k, init="random", max_iter=10,
                                   random_state=i, chunk_size=1000,
                                   verbose=0).fit(X_ac)

        # extract the label assignment on the overlap
        labels_ab = model_ab.labels_[:n_common]
        labels_ac = model_ac.labels_[:n_common]

        # measure the stability of k as the agreement of the clusterings on the
        # overlap
        scores[i, j] = metrics.adjusted_rand_score(labels_ab, labels_ac)

    print "Run %d/%d done in %0.2fs" % (j + 1, n_runs, time() - t0)

pl.errorbar(n_clusters_range, scores.mean(axis=1), scores.std(axis=1))
pl.title("Consensus model selection for Document Clustering\n"
         "on the 20 newsgroups dataset")
pl.xlabel("Number of centers")
pl.ylabel("Mean Consensus ARI score on %d runs" % n_runs)
pl.show()
