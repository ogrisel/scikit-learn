"""
==================================================
Clustering and topic extraction of research papers
==================================================

This is an example showing how the scikit-learn can be used to group
documents by topics using a bag-of-words approach followed by a linear
PCA feature extraction followed by a final clustering algorithm on the low
dimensional data.

The dataset used in this example is the PDF archive of the NIPS 2010
conference.

The pdftotext unix command is required to extract the text content of the PDF
files. On debian / ubuntu system install it with:

    $ sudo apt-get install poppler-utils

"""
print __doc__

# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

from time import time
from pprint import pprint
import os

import numpy as np

from scikits.learn.cluster import KMeans
from scikits.learn.cluster import MeanShift
from scikits.learn.feature_extraction.text import WordNGramAnalyzer
from scikits.learn.feature_extraction.text.sparse import Vectorizer
from scikits.learn.pca import RandomizedPCA

url = "http://books.nips.cc/papers/files/nips23/nips2010_pdf.tgz"
directoy_name = "nips2010_pdf"
n_clusters = 10
n_components = 100

################################################################################
# Download the data, if not already on disk

archive_name = url.rsplit('/', 1)[1]

if not os.path.exists(directoy_name):
    if not os.path.exists(archive_name):
        import urllib
        print "Downloading data, please Wait (226MB)..."
        print url
        opener = urllib.urlopen(url)
        open(archive_name, 'wb').write(opener.read())
        print

    import tarfile
    os.mkdir(directoy_name)
    print "Decompressiong the archive: " + archive_name
    try:
        tarfile.open(archive_name, "r:gz").extractall(path=directoy_name)
    except IOError, e:
        # allow for corrupted archive
        print e
    print


################################################################################
# List the pdf files and extract the text version if necessary

pdf_folder = os.path.join(directoy_name, 'files')
text_folder = os.path.join(directoy_name, 'text_files')
if not os.path.exists(text_folder):
    os.mkdir(text_folder)
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            source = os.path.join(pdf_folder, filename)
            target = os.path.join(text_folder, filename + '.txt')
            command = "pdftotext %s %s" % (source, target)
            print command
            os.system(command)

text_filenames = [os.path.join(text_folder, f)
                  for f in os.listdir(text_folder)]


################################################################################
# Extract TF-IDF representation of the documents

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_df=0.85, max_features=100000,
                        analyzer=WordNGramAnalyzer(min_n=1, max_n=2))
X_tfidf = vectorizer.fit_transform((open(f).read() for f in text_filenames))
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X_tfidf.shape
print


################################################################################
# Project on the first 100 singular vectors

print "Extracting top %d singular vectors" % n_components
t0 = time()
pca = RandomizedPCA(n_components=50, copy=False, whiten=True).fit(X_tfidf)
print "done in %fs" % (time() - t0)
print

names = dict((v, k) for k, v in vectorizer.tc.vocabulary.iteritems())

def print_top_features(vectors, vector_type, max_vector, max_features, names):
    for i, v_i in enumerate(vectors[:max_vector]):
        print "Main components %s vector #%d:" % (vector_type, i)
        top_features = list(reversed(v_i.argsort()))[:max_features]
        print '\n'.join([names[j].ljust(30) + "%0.3f" % v_i[j]
                         for j in top_features])

print_top_features(pca.components_, 'singular', 5, 20, names)
print

# transform the data by linear projection in the topic space
X_pca = pca.transform(X_tfidf)


################################################################################
# Cluster the projected data
print "Clustering the projected vectors"
t0 = time()
#clusterer = KMeans(k=n_clusters, init='k-means++').fit(X_pca)
clusterer = MeanShift(17).fit(X_pca)
print "done in %fs" % (time() - t0)
print


# project back the centroids to the original TF-IDF space
centers_tfidf = np.dot(clusterer.cluster_centers_, pca.components_.T)

print_top_features(centers_tfidf, 'centroid', 5, 20, names)
print

# TODO: print the titles for each cluster
