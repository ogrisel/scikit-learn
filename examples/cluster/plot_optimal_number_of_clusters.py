"""
===========================================================================
Selection of the optimal number of clusters based on bootstrapped agreement
===========================================================================

V-Measure is a score that grows when two cluster labeling assignments
agree on how to split a dataset, independently of the number of splits. This
measure is furthermore symmetric::

    v_measure_score(label_a, label_b) == v_measure_score(label_b, label_a)

The following demonstrates how it is possible to randomly split a dataset
into two overlapping subsets, perform clustering on each split (here
using k-means) and then measure their agreement on the intersection of
the two subsets.

This process can be repeated for various number of the parameter k
of k-means. The best V-Measure score can hence give us a hint on the
optimal value for the parameter k in a completely unsupervied fashion.

In order to find a robust estimate of the optimal agreement score, the
process can be repeated and averaged several times by boostrapping the
overlapping sets used for clustering.
"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD
print __doc__

import numpy as np
from scikits.learn.utils import check_random_state
from scikits.learn.utils import shuffle
from scikits.learn.cluster import KMeans
from scikits.learn.metrics import v_measure_score
from scikits.learn.datasets.samples_generator import make_blobs

n_samples = 500
n_features = 2
n_centers = 4
cluster_std = 0.5
n_bootstraps = 5
possible_k = range(2, 20)

random_state = check_random_state(42)

samples_1, _= make_blobs(n_samples=n_samples / 4, n_features=n_features,
                         centers=n_centers, cluster_std=cluster_std,
                         random_state=0)
samples_2, _= make_blobs(n_samples=n_samples / 4, n_features=n_features,
                         centers=n_centers, cluster_std=cluster_std,
                         random_state=0)
samples_3, _= make_blobs(n_samples=n_samples / 4, n_features=n_features,
                         centers=n_centers, cluster_std=cluster_std,
                         random_state=0)
samples_4, _= make_blobs(n_samples=n_samples / 4, n_features=n_features,
                         centers=n_centers, cluster_std=cluster_std,
                         random_state=0)

samples_1[:, 0] -= 10
samples_2[:, 0] += 10
samples_3[:, 1] -= 10
samples_4[:, 1] += 10

samples = np.concatenate((samples_1, samples_2, samples_3, samples_4))

scores = np.zeros((n_bootstraps, len(possible_k)))

for i in range(n_bootstraps):

    print "Boostrap #%02d/%02d" % (i + 1, n_bootstraps)

    #indices = shuffle(np.arange(n_samples))
    #indices = np.arange(n_samples)
    # resample the data to measure the score on independent splits
    indices = random_state.randint(0, n_samples, size=(n_samples,))

    # split the sample in 3 parts
    indice_splits = np.array_split(indices, 3)

    # group 2 of them in a first subset
    a = np.concatenate((indice_splits[0], indice_splits[1]))

    # group another pair in a second subset
    b = np.concatenate((indice_splits[0], indice_splits[2]))

    # the intersection is stored in third subset
    common = indice_splits[0].shape[0]

    for j, k in enumerate(possible_k):

        # run K-Means independently on each subset
        labels_a = KMeans(k=k, init='k-means++',
                          random_state=random_state).fit(samples[a]).labels_
        labels_b = KMeans(k=k, init='k-means++',
                          random_state=random_state).fit(samples[b]).labels_

        # evaluate the agreement on the intersection
        score = v_measure_score(labels_a[:common], labels_b[:common])
        print "Agreement for k=%d: %0.3f" % (k, score)
        scores[i, j] = score


# compute a bunch of statistics on the results of the bootstrapped runs so as
# to find the optimal value(s) for k
mean_scores = scores.mean(axis=0)
std_scores = scores.std(axis=0)
best_mean_score = mean_scores.max()
admissible_scores = mean_scores >= best_mean_score - std_scores

admissible_k = [k for i, k in enumerate(possible_k) if admissible_scores[i]]

print "Optimal values for k: " + ", ".join(str(k) for k in admissible_k)
print "Ground Truth value for k: %d" % n_centers

# plot the runs
import pylab as pl
pl.subplot(121)
pl.plot(possible_k, mean_scores)
pl.plot(possible_k, mean_scores + std_scores / 2, 'b--')
pl.plot(possible_k, mean_scores - std_scores / 2, 'b--')
pl.ylim(ymin=0.0, ymax=1.0)
pl.vlines(n_centers, 0.0, 1.0, 'g')
pl.title("V-Measure agreement of K-Means on the intersection of two\n"
         "overlapping splits for various values of k")
pl.xlabel("Number of centers 'k' for each run of k-means")
pl.ylabel("V-Measure")

pl.subplot(122)
pl.plot(samples[:, 0], samples[:, 1], '.')
pl.show()
