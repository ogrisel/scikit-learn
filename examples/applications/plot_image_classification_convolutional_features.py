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

import numpy as np
import pylab as pl

from scikits.learn.metrics import classification_report
from scikits.learn.metrics import confusion_matrix
from scikits.learn.feature_extraction.image import ConvolutionalKMeansEncoder
from scikits.learn.svm import LinearSVC
from scikits.learn.externals.joblib import Memory

memory = Memory(cachedir='.')


################################################################################
# Learn filters from data

parameters = {
    'max_iter': 200, # max number of kmeans EM iterations
    'n_centers': 400, # kmeans centers: convolutional filters
    'n_components': 80, # singular vectors to keep when whitening
    'n_pools': 2, # square root of number of 2D image areas for pooling
    'patch_size': 8, # size of the side of one filter
    'whiten': True, # perform whitening or not before kmeans
    'n_init': 1,
    'tol': 0.5,
    'local_contrast': True,
    'n_pools': 2,
    'verbose': True,
    'n_drop_components': 2,
    'max_patches': 100000,
}

def load_cifar_10(n_samples_train=50000, n_samples_test=10000,
                  keep_color_shift=False):
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
            target_names = dataset['label_names']

    del dataset

    X_train = np.asarray(np.concatenate(X_train), dtype=np.float32)
    y_train = np.concatenate(y_train)

    # restrict training size for faster runtime as a demo
    X_train = X_train[:n_samples_train]
    y_train = y_train[:n_samples_train]
    X_test = X_test[:n_samples_test]
    y_test = y_test[:n_samples_test]

    # reshape pictures to there natural dimension
    X_train = X_train.reshape(
        (X_train.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(
        (X_test.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    if not keep_color_shift:
        # remove average rgb value from each image
        X_train -= X_train.mean(axis=2).mean(axis=1).reshape(
            (n_samples_train, 1, 1, 3))
        X_test -= X_test.mean(axis=2).mean(axis=1).reshape(
            (n_samples_test, 1, 1, 3))
    return X_train, X_test, y_train, y_test, target_names


@memory.cache
def extract_features(parameters, n_samples_train=10000, n_samples_test=5000):
    X_train, X_test, y_train, y_test, target_names = load_cifar_10(
        n_samples_train=n_samples_train, n_samples_test=n_samples_test)
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

    return (extractor, X_train_features, X_test_features, y_train, y_test,
            target_names)


# perform the feature extraction while caching the results with joblib
extractor, X_train, X_test, y_train, y_test, target_names = extract_features(
    parameters)

print "Transformed training set in pooled conv feature space has shape:"
print X_train.shape
print "Transformed test set in pooled conv feature space has shape:"
print X_test.shape


##############################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
clf = LinearSVC(C=0.01, dual=False).fit(X_train, y_train)
print "done in %0.3fs" % (time() - t0)
print clf


##############################################################################
# Quantitative evaluation of the model quality on the test set

y_pred = clf.predict(X_test)

print classification_report(y_test, y_pred, labels=range(len(target_names)),
                            target_names=target_names)
print confusion_matrix(y_test, y_pred)


##############################################################################
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

