"""
==================================================================
Image classification example using convolutional features and SVMs
==================================================================

The dataset used in this example is the CIFAR-10 dataset:

  http://www.cs.toronto.edu/~kriz/cifar.html

This implementation uses an unsupervised feature extraction scheme
to extract a dictionnary of 400 small (6, 6)-shaped filters to be
convolationally applied to the input images as described in:

  An Analysis of Single-Layer Networks in Unsupervised Feature Learning
  Adam Coates, Honglak Lee and Andrew Ng. In NIPS*2010 Workshop on
  Deep Learning and Unsupervised Feature Learning.

Expected results:

  TODO

"""
print __doc__

import os
import math
import cPickle
from time import time
from pprint import pprint

import numpy as np
import pylab as pl

from scikits.learn.grid_search import GridSearchCV
from scikits.learn.metrics import classification_report
from scikits.learn.metrics import confusion_matrix
from scikits.learn.feature_extraction.image import ConvolutionalKMeansEncoder
from scikits.learn.svm import SVC
from scikits.learn.svm import LinearSVC
from scikits.learn.externals.joblib import Memory
from scikits.learn.pca import RandomizedPCA

memory = Memory(cachedir='joblib')

################################################################################
# Download the data, if not already on disk

url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
archive_name = url.rsplit('/', 1)[1]
folder_name = 'cifar-10-batches-py'

if not os.path.exists(folder_name):
    if not os.path.exists(archive_name):
        import urllib
        print "Downloading data, please Wait (163MB)..."
        print url
        opener = urllib.urlopen(url)
        open(archive_name, 'wb').write(opener.read())
        print

    import tarfile
    print "Decompressiong the archive: " + archive_name
    tarfile.open(archive_name, "r:gz").extractall()
    print

################################################################################
# Load dataset in memory

print "Loading CIFAR-10 in memory"

X_train = []
y_train = []
dataset = None

for filename in sorted(os.listdir(folder_name)):
    filepath = os.path.join(folder_name, filename)
    if filename.startswith('data_batch_'):
        dataset = cPickle.load(file(filepath, 'rb'))
        X_train.append(dataset['data'])
        y_train.append(dataset['labels'])
    elif filename == 'test_batch':
        dataset = cPickle.load(file(filepath, 'rb'))
        X_test = np.asarray(dataset['data'], dtype=np.float32)
        y_test = np.asarray(dataset['labels'])
    elif filename == 'batches.meta':
        dataset = cPickle.load(file(filepath, 'rb'))
        label_names = dataset['label_names']

del dataset

X_train = np.asarray(np.concatenate(X_train), dtype=np.float32)
y_train = np.concatenate(y_train)

#n_samples = X_train.shape[0]

# restrict training size for faster runtime as a demo
n_samples = 5000
X_train = X_train[:n_samples]
y_train = y_train[:n_samples]
X_test = X_test[:n_samples]
y_test = y_test[:n_samples]

# reshape pictures to there natural dimension
X_train = X_train.reshape((X_train.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
X_test = X_test.reshape((X_test.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

# bring the int8 color data to the [0.0, 1.0] float range that is expected by
# the matplotlib image viewer when working with float data
X_train /= 255.
X_test /= 255.


################################################################################
# Learn filters from data

parameters = {
    'max_iter': 200, # max number of kmeans EM iterations
    'n_centers': 400, # kmeans centers: convolutional filters
    'n_components': 80, # singular vectors to keep when whitening
    'n_pools': 3, # square root of number of 2D image areas for pooling
    'patch_size': 8 , # size of the side of one filter
    'whiten': True, # perform whitening or not before kmeans
    'n_init': 1,
    'tol': 0.5,
    'local_contrast': True,
    'n_pools': 3,
    'verbose': True,
    'n_drop_components': 0,
}


@memory.cache
def extract_features(X_train, X_test, parameters):
    extractor = ConvolutionalKMeansEncoder(**parameters)

    print "Training convolutional whitened kmeans feature extractor"
    t0 = time()
    extractor.fit(X_train)
    print "done in %0.3fs" % (time() - t0)

    if parameters.get('whiten'):
        vr = extractor.pca.explained_variance_ratio_
        print ("explained variance ratios for %d kept PCA components:" %
               vr.shape[0])
        print vr
    print "kmeans remaining inertia: %0.3fe6" % (extractor.inertia_ / 1e6)

    print "Extracting features on training set"
    t0 = time()
    X_train_features = extractor.transform(X_train)
    print "done in %0.3fs" % (time() - t0)

    print "Extracting features on test set"
    t0 = time()
    X_test_features = extractor.transform(X_test)
    print "done in %0.3fs" % (time() - t0)

    return extractor, X_train_features, X_test_features


# perform the feature extraction while caching the results with joblib
extractor, X_train_features, X_test_features = extract_features(
    X_train, X_test, parameters)

print "Transformed training set in pooled conv feature space has shape:"
print X_train_features.shape


################################################################################
# reduce the dimensionality of the extracted features


#print "Reducing the dimension of the extracted feature using a PCA"
#t0 = time()
#m = np.abs(X_train_features).max()
#pca = RandomizedPCA(n_components=200, whiten=True).fit(X_train_features / m)
#X_train_features_pca = pca.transform(X_train_features / m)
#X_test_features_pca = pca.transform(X_test_features / m)
#print "done in %0.3fs" % (time() - t0)


################################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
#param_grid = {
# 'C': [1, 10, 100],
# #'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#}
#clf = GridSearchCV(SVC(kernel='linear'), param_grid,
#                   fit_params={'class_weight': 'auto'})
clf = SVC(kernel='linear', C=100).fit(X_train_features, y_train)
print "done in %0.3fs" % (time() - t0)
#print "Best estimator found by grid search:"
#print clf.best_estimator


################################################################################
# Quantitative evaluation of the model quality on the test set

y_pred = clf.predict(X_test_features)
print classification_report(y_test, y_pred, labels=range(len(label_names)),
                            class_names=label_names)

print confusion_matrix(y_test, y_pred)


################################################################################
# Qualitative evaluation of the extracted filters

def plot_filters(filters, patch_size=8, n_colors=3, local_scaling=True):
    n_filters = filters.shape[0]
    n_row = int(math.sqrt(n_filters))
    n_col = int(math.sqrt(n_filters))

    filters = filters.copy()
    if local_scaling:
        # local rescaling filters for display with imshow
        filters -= filters.min(axis=1).reshape((filters.shape[0], 1))
        filters /= filters.max(axis=1).reshape((filters.shape[0], 1))
    else:
        # global rescaling
        filters -= filters.min()
        filters /= filters.max()

    pl.figure()
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(filters[i].reshape((patch_size, patch_size, n_colors)),
                  interpolation="nearest")
        pl.xticks(())
        pl.yticks(())

    pl.show()

# matplotlib is slow on large number of filters: restrict to the top 100 by
# default
#plot_filters(extractor.filters_[:100], local_scaling=True)

