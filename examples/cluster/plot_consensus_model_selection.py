"""
===============================================================
Finding the best number of clusters using a stability criterion
===============================================================

TODO: write me and references to von Luxburg and Vinh.
"""

import numpy as np
import pylab as pl

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_digits
from sklearn.neighbors import kneighbors_graph

digits = load_digits()

n_samples = digits.data.shape[0]
n_runs = 50
n_clusters_range = np.arange(2, 21)
n_samples_split = n_samples / 3

scores = np.zeros((n_clusters_range.shape[0], n_runs))
X = digits.data.copy()[:100]
rng = np.random.RandomState(42)

for i, k in enumerate(n_clusters_range):
    for j in range(n_runs):
        # sample two overlapping sub-datasets
        rng.shuffle(X)
        X_a = X[:n_samples_split]
        X_b = X[n_samples_split:2 * n_samples_split]
        X_c = X[2 * n_samples_split:3 * n_samples_split]

        X_ab = np.vstack((X_a, X_b))
        S_ab = kneighbors_graph(X_ab, 10)
        X_ac = np.vstack((X_a, X_c))
        S_ac = kneighbors_graph(X_ac, 10)

        # find a clustering for each sub-sample
        model_ab = SpectralClustering(k=k, mode='arpack').fit(S_ab)
        model_ac = SpectralClustering(k=k, mode='arpack').fit(S_ac)

        # extract the label assignment on the overlap
        labels_ab = model_ab.labels_[:n_samples_split]
        labels_ac = model_ac.labels_[:n_samples_split]

        # measure the stability of k as the agreement of the clusterings on the
        # overlap
        scores[i, j] = adjusted_rand_score(labels_ab, labels_ac)

pl.errorbar(n_clusters_range, scores.mean(axis=1), scores.std(axis=1))
pl.title("Consensus model selection for Spectral Clustering\n"
         "on the digits dataset")
pl.xlabel("Number of centers")
pl.ylabel("Mean Consensus ARI score on %d runs" % n_runs)
pl.show()
