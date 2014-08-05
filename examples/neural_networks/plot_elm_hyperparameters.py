"""
===========================================================
Extreme Learning Machines: Effect of tuning hyperparameters
===========================================================

This example first demonstrates how scikit-learn GridsearchCV can
tune 3 of the most impacting hyperparameters of ELM classifiers: n_hidden, C,
and weight_scale.

Then, using color maps, it illustrates the hyperparameter space of the ELM
model varying each 2 parameters while keeping the third parameter as the
optimal value.

"""
print(__doc__)

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_digits
from sklearn.neural_network import ELMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import grid_search

random_state = 0

# Generate sample data
digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)

parameters = {'C': np.logspace(-2, 2, 5),
              'weight_scale': np.logspace(-2, 2, 5),
              'n_hidden': np.array([32, 128, 512, 1024, 2048])}

# List combinations
comb = [('C', 'weight_scale', 'n_hidden'),
        ('weight_scale', 'n_hidden', 'C'),
        ('n_hidden', 'C', 'weight_scale')]

# Compute optimal parameters
optimum_params = {}
for param in parameters:
    elm = ELMClassifier(random_state=random_state)

    clf = grid_search.GridSearchCV(elm, {param: parameters[param]})
    clf.fit(X, y)

    optimum_params[param] = clf.best_params_[param]

scores = np.zeros((3, 5, 5))
for i in range(len(parameters)):
    # Get parameters
    p1, p1_values = comb[i][0], parameters[comb[i][0]]
    p2, p2_values = comb[i][1], parameters[comb[i][1]]
    p3, p3_optimal_value = comb[i][2], optimum_params[comb[i][2]]

    # Loop over the values of the two parameters
    for j, p1_value in enumerate(p1_values):
        for k, p2_value in enumerate(p2_values):
            elm = ELMClassifier(**{p1: p1_value, p2: p2_value,
                                   p3: p3_optimal_value})
            scores[i, k, j] = np.mean(cross_val_score(elm, X, y, cv=2))

# plot a relationship between each two parameters while fixing the third one
min_value, max_value = np.min(scores), np.max(scores)

for i in range(len(parameters)):
    fig = plt.figure(i)
    ax = fig.add_subplot(111)

    # Get parameters
    p1, p1_values = comb[i][0], parameters[comb[i][0]]
    p2, p2_values = comb[i][1], parameters[comb[i][1]]
    p3, p3_optimal_value = comb[i][2], optimum_params[comb[i][2]]

    # Plot results functions
    mat = ax.matshow(scores[i], origin="lower",
                     cmap=plt.cm.Blues, vmin=min_value, vmax=max_value)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mat, ax=ax, cax=cax)

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))

    ax.set_xlabel(p1 + ' parameter')
    ax.set_ylabel(p2 + ' parameter')

    ax.set_xticklabels(p1_values)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticklabels(p2_values)
    ax.set_title('Validation scores when ' + p3 + '=' +
                 str(p3_optimal_value), {'fontsize': 12})

plt.show()
