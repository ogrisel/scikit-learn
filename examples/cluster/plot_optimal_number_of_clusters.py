"""
===========================================
Selection of the optimal number of clusters
===========================================

TODO: explain what's happening
"""
print __doc__

import numpy as np
from scikits.learn.cluster import KMeans
from scikits.learn.metrics import v_measure_score
from scikits.learn.datasets.samples_generator import make_blobs

n_samples = 200
n_features = 10
n_centers = 10
cluster_std = 2.0

samples, labels_true = make_blobs(n_samples=n_samples, n_features=n_features,
                                  centers=n_centers, cluster_std=cluster_std)

# split the sample in 3 parts
indice_splits = np.array_split(np.arange(n_samples), 3)

# group 2 of them in a first subset
a = np.concatenate((indice_splits[0], indice_splits[1]))

# group another pair in a second subset
b = np.concatenate((indice_splits[0], indice_splits[2]))

# the intersection is stored in third subset
common = indice_splits[0].shape[0]

scores = []
possible_k = range(2, 20)
for k in possible_k:
    labels_a = KMeans(
        k=k, init='k-means++', random_state=0).fit(samples[a]).labels_
    labels_b = KMeans(
        k=k, init='k-means++', random_state=0).fit(samples[b]).labels_

    # evaluate the aggreement on the intersection
    score = v_measure_score(labels_a[:common], labels_b[:common])
    print "Agreement for k=%d: %0.3f" % (k, score)
    scores.append(score)
    if k == n_centers:
        true_k_score = score

import pylab as pl
pl.plot(possible_k, scores)
pl.ylim(ymin=0.0, ymax=1.0)
pl.vlines(n_centers, 0.0, true_k_score)
pl.title("V-Measure agreement of K-Means on the intersection of two\n"
         "overlapping splits for various of k")
pl.xlabel("Expected number of centers 'k'")
pl.ylabel("V-Measure")
pl.show()
