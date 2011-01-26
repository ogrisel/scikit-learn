"""
===============================================================
Faces recognition example using convolutional features and SVMs
===============================================================

This implementation uses an unsupervised feature extraction scheme
to extract a dictionnary of 200 small (12, 12)-shaped filters to be
convolutionally applied to the input images as described in:

  An Analysis of Single-Layer Networks in Unsupervised Feature Learning
  Adam Coates, Honglak Lee and Andrew Ng. In NIPS*2010 Workshop on
  Deep Learning and Unsupervised Feature Learning.

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 20 most represented people in the dataset::

                  avg / total       0.87      0.87      0.87       555
                             precision    recall  f1-score   support

              David_Beckham       0.67      0.50      0.57         8
               Roh_Moo-hyun       1.00      1.00      1.00         7
          Silvio_Berlusconi       1.00      0.50      0.67         8
                Vicente_Fox       0.80      0.50      0.62         8
      Megawati_Sukarnoputri       1.00      1.00      1.00         8
                  Tom_Ridge       1.00      1.00      1.00         8
            Nestor_Kirchner       1.00      0.56      0.71         9
               Alvaro_Uribe       0.89      0.89      0.89         9
               Andre_Agassi       0.90      1.00      0.95         9
                  Hans_Blix       1.00      0.90      0.95        10
           Alejandro_Toledo       1.00      0.80      0.89        10
             Lleyton_Hewitt       1.00      0.91      0.95        11
                 Laura_Bush       0.75      0.90      0.82        10
          Jennifer_Capriati       1.00      0.70      0.82        10
      Arnold_Schwarzenegger       0.78      0.70      0.74        10
    Gloria_Macapagal_Arroyo       1.00      1.00      1.00        11
             Vladimir_Putin       0.82      0.82      0.82        11
  Luiz_Inacio_Lula_da_Silva       1.00      1.00      1.00        12
            Serena_Williams       0.92      0.92      0.92        13
             Jacques_Chirac       1.00      1.00      1.00        13
              John_Ashcroft       0.92      0.92      0.92        13
              Jean_Chretien       0.92      0.86      0.89        14
          Junichiro_Koizumi       1.00      0.80      0.89        15
                Hugo_Chavez       0.88      0.88      0.88        17
               Ariel_Sharon       0.95      1.00      0.97        19
          Gerhard_Schroeder       0.93      0.96      0.95        27
            Donald_Rumsfeld       0.88      0.93      0.90        30
                 Tony_Blair       0.92      0.94      0.93        36
               Colin_Powell       0.95      0.92      0.93        59
              George_W_Bush       0.84      0.96      0.90       130

                avg / total       0.91      0.90      0.90       555

"""
print __doc__

import os
from gzip import GzipFile

import numpy as np
import pylab as pl

from scikits.learn.grid_search import GridSearchCV
from scikits.learn.metrics import classification_report
from scikits.learn.metrics import confusion_matrix
from scikits.learn.pca import RandomizedPCA
from scikits.learn.svm import SVC
from scikits.learn.cross_val import StratifiedKFold
from scikits.learn.feature_extraction.image import ConvolutionalKMeansEncoder

################################################################################
# Download the data, if not already on disk

url = "https://downloads.sourceforge.net/project/scikit-learn/data/lfw_preprocessed.tar.gz"
archive_name = "lfw_preprocessed.tar.gz"
folder_name = "lfw_preprocessed"

if not os.path.exists(folder_name):
    if not os.path.exists(archive_name):
        import urllib
        print "Downloading data, please Wait (58.8MB)..."
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

faces_filename = os.path.join(folder_name, "faces.npy.gz")
filenames_filename = os.path.join(folder_name, "face_filenames.txt")

faces = np.load(GzipFile(faces_filename))
face_filenames = [l.strip() for l in file(filenames_filename).readlines()]

# normalize each picture by centering brightness and normalizing contrast
faces -= faces.mean(axis=1)[:, np.newaxis]
faces /= faces.std(axis=1)[:, np.newaxis]


################################################################################
# Index category names into integers suitable for scikit-learn

# Here we do a little dance to convert file names in integer indices
# (class indices in machine learning talk) that are suitable to be used
# as a target for training a classifier. Note the use of an array with
# unique entries to store the relation between class index and name,
# often called a 'Look Up Table' (LUT).
# Also, note the use of 'searchsorted' to convert an array in a set of
# integers given a second array to use as a LUT.
categories = np.array([f.rsplit('_', 1)[0] for f in face_filenames])

# A unique integer per category
category_names = np.unique(categories)

# Turn the categories in their corresponding integer label
target = np.searchsorted(category_names, categories)

# Subsample the dataset to restrict to the most frequent categories
selected_target = np.argsort(np.bincount(target))[-30:]

# If you are using a numpy version >= 1.4, this can be done with 'np.in1d'
mask = np.array([item in selected_target for item in target])

X = faces[mask]
y = target[mask]

n_samples, n_features = X.shape

print "Dataset size:"
print "n_samples: %d" % n_samples
print "n_features: %d" % n_features

# split training / testing with a 3/4, 1/4 ratio while ensuring that each class
# is represented in proportion on each side of the split
train, test = iter(StratifiedKFold(y, 4)).next()

X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

################################################################################
# Compute a learn local edge detectors (similar to Gabor filters) on the face
# dataset (treated as unlabeled dataset) and pool the occurrences of
# convolutions of those filters with the face samples (triangle kmeans):
# unsupervised feature extraction with spatial invariance

print "Extracting the convolutional features using soft kmeans on patches"
parameters = {
    'max_iter': 200, # max number of kmeans EM iterations
    'n_centers': 100, # kmeans centers: convolutional filters
    'n_components': 80, # singular vectors to keep when whitening
    'n_pools': 5, # square root of number of 2D image areas for pooling
    'patch_size': 12, # size of the side of one filter
    'whiten': True, # perform whitening or not before kmeans
    'n_init': 1,
    'tol': 0.5,
    'local_contrast': True,
    'verbose': True,
    'n_drop_components': 0,
    'max_patches': 100000,
}
extractor = ConvolutionalKMeansEncoder(**parameters).fit(X_train)


# project the input data on the eigenfaces orthonormal basis
print "Project data in the pooled convolutional filter space"
X_train_conv = extractor.transform(X_train)
X_test_conv = extractor.transform(X_test)

print "Training set now has shape:"
print X_train_conv.shape

print "Reducing the dimension using PCA"
pca = RandomizedPCA(n_components=300, whiten=True).fit(X_train_conv)
X_train_pca = pca.transform(X_train_conv)
X_test_pca = pca.transform(X_test_conv)

################################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set wiht shape:"
print X_train_pca.shape

#param_grid = {
# 'C': [1, 5, 10, 50, 100],
# 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#}
#clf = GridSearchCV(SVC(kernel='rbf'), param_grid,
#                   fit_params={'class_weight': 'auto'})
#clf = clf.fit(X_train_pca, y_train, class_weight='auto')
#print "Best estimator found by grid search:"
#print clf.best_estimator

# parameters found using the above grid search
clf = SVC(kernel='rbf', C=50, gamma=0.0001)
clf = clf.fit(X_train_pca, y_train, class_weight='auto')


################################################################################
# Quantitative evaluation of the model quality on the test set

print "Evaluation on the test hold out testing set"
y_pred = clf.predict(X_test_pca)
print classification_report(y_test, y_pred, labels=selected_target,
                            class_names=category_names[selected_target])

print confusion_matrix(y_test, y_pred, labels=selected_target)


################################################################################
# Qualitative evaluation of the predictions using matplotlib

n_row = 3
n_col = 4

pl.figure(figsize=(2 * n_col, 2.3 * n_row))
pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.95, hspace=.15)
for i in range(n_row * n_col):
    pl.subplot(n_row, n_col, i + 1)
    pl.imshow(X_test[i].reshape((64, 64)), cmap=pl.cm.gray)
    pl.title('pred: %s\ntrue: %s' % (category_names[y_pred[i]],
                                     category_names[y_test[i]]), size=12)
    pl.xticks(())
    pl.yticks(())

pl.show()

