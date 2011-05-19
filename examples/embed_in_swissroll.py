import numpy as np

from scikits.learn.datasets.samples_generator import swiss_roll
from scikits.learn.datasets.samples_generator import make_blobs

n_samples = 1000
n_features = 2
n_centers = 4
cluster_std = 0.5

base_clusters, _= make_blobs(n_samples=n_samples / 4, n_features=n_features,
                         centers=n_centers, cluster_std=cluster_std,
                         random_state=0)

samples_1 = base_clusters.copy()
samples_1[:, 0] -= 5
samples_2 = base_clusters.copy()
samples_2[:, 0] += 5
samples_3 = base_clusters.copy()
samples_3[:, 1] -= 10
samples_4 = base_clusters.copy()
samples_4[:, 1] += 10

samples = np.concatenate((samples_1, samples_2, samples_3, samples_4))

t = 0.2 * np.pi * (samples[:, 0] + samples[:, 0].min())
X = np.vstack((t * np.cos(t), t * np.sin(t), samples[:, 1])).T

import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
fig = pl.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
ax.plot3D(X[:, 0], X[:, 1], X[:, 2], 'o')
pl.show()
