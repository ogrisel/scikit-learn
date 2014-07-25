"""
=====================================================
Extreme Learning Machines: Training vs. Testing Score
=====================================================

Plot how training and testing score change with increasing number of hidden
neurons. The more hidden neurons the less the training error, which eventually
reaches zero. However, testing error does not necessarily decrease as having
more hidden neurons than necessary would cause overfitting on the data.

"""
print(__doc__)

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.neural_network import ELMRegressor
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# Generate sample data
boston = load_boston()
X, y = boston.data, boston.target
X = StandardScaler().fit_transform(X)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Compute train and test errors
n_hidden_list = np.arange(10, 200, 10)
train_errors = list()
test_errors = list()
for n_hidden in n_hidden_list:
    elm = ELMRegressor(n_hidden=n_hidden)
    elm.fit(X_train, y_train)
    train_errors.append(elm.score(X_train, y_train))
    test_errors.append(elm.score(X_test, y_test))

i_n_hidden_optim = np.argmax(test_errors)
n_hidden_optim = n_hidden_list[i_n_hidden_optim]
print("Optimal number of hidden neurons : %s" % n_hidden_optim)

# Plot results functions
plt.plot(n_hidden_list, train_errors, label='Train')
plt.plot(n_hidden_list, test_errors, label='Test')
plt.vlines(n_hidden_optim, plt.ylim()[0], np.max(test_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('Number of hidden neurons')
plt.ylabel('Score')

plt.show()
