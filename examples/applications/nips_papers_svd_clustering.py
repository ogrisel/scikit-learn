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
from scipy.linalg import svd

from scikits.learn.cluster import KMeans
from scikits.learn.cluster import MeanShift
from scikits.learn.feature_extraction.text import WordNGramAnalyzer
from scikits.learn.feature_extraction.text.sparse import Vectorizer
from scikits.learn.utils.extmath import fast_svd

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
# TODO: make the PCA module support sparse data insted of using the fast_svd API
# directly

print "Extracting top %d singular vectors" % n_components
t0 = time()
u, s, v = fast_svd(X_tfidf, n_components, p=50, q=3)
print "done in %fs" % (time() - t0)
print

names = dict((v, k) for k, v in vectorizer.tc.vocabulary.iteritems())

for i in range(10):
    print "Main components singular vector #%d:" % i
    top_10 = list(reversed(np.abs(v[i]).argsort()))[:10]
    pprint([names[i] for i in top_10])
print

# transform the data by linear projection in the topic space
X_svd = X_tfidf * v.T


################################################################################
# Cluster the projected data
print "Clustering the projected vectors"
t0 = time()
#clusterer = KMeans(k=n_clusters, init='k-means++').fit(X_svd)
clusterer = MeanShift(0.35).fit(X_svd)
print "done in %fs" % (time() - t0)
print


# project back the centroids to the original TF-IDF space
centers_tfidf = np.dot(clusterer.cluster_centers_, v)

for i in range(centers_tfidf.shape[0]):
    print "Main components of centroid #%d:" % i
    top_10 = list(reversed(np.abs(centers_tfidf[i]).argsort()))[:10]
    pprint([names[i] for i in top_10])

print

# TODO: print the titles for each cluster
