"""
==========================================
Extreme Learning Machines: Hyperparameters
==========================================

Plots showing the effect of adjusting parameters on the cross-validation
scores against the digits dataset. This involves three parameters: n_hidden, C,
and weight_scale. Each color map shows the relationship between two
of these parameters while the third parameter is optimized using grid search.

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


def plot_color_map(fig, ax, title, scores):
    mat = ax.matshow(scores, origin="lower", cmap=plt.cm.Blues)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(mat, ax=ax, cax=cax)

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))

    ax.set_xlabel(keys[0] + ' parameter')
    ax.set_ylabel(keys[1] + ' parameter')

    ax.set_xticklabels(values[0])
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticklabels(values[1])
    ax.set_title(title, {'fontsize': 12})

np.random.seed(0)

# Generate sample data
digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)

parameters = {}
parameters['C'] = np.around(np.logspace(0, 3, 5), 1)
parameters['weight_scale'] = np.logspace(-2, 2, 5)
parameters['n_hidden'] = np.arange(50, 300, 50)

# get optimal parameters
optimum_params = {}
for param in parameters:
    elm = ELMClassifier()

    clf = grid_search.GridSearchCV(elm, {param: parameters[param]})
    clf.fit(X, y)

    optimum_params[param] = clf.best_params_[param]

# plot a relationship between each two parameters while fixing the third one
fig, plot_axes = plt.subplots(1, 3, figsize=(13, 4))
plot = 0
for param in parameters:
    axes = dict(parameters)
    del axes[param]

    keys = list(axes.keys())
    values = list(axes.values())

    scores = []
    for a in values[0]:
        score_row = []
        for b in values[1]:
            elm = ELMClassifier(**{keys[0]: a, keys[1]: b,
                                   param: optimum_params[param]})
            score = np.mean(cross_val_score(elm, X, y))
            score_row.append(score)
        scores.append(score_row)

    scores = np.array(scores).T

    # Plot results functions
    plot_color_map(fig, plot_axes[plot], 'Validation scores when ' +
                   param + '=' + str(optimum_params[param]), scores)
    plot += 1

fig.suptitle('ELM scores on the digits dataset', fontsize=14)
fig.tight_layout()
plt.show()
