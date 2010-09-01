"""Train a predictive model that guesses the language of a text document."""

import numpy as np
import scipy.sparse as ssp
from time import time

from nltk.corpus import movie_reviews
from scikits.learn.features.text import SparseHashingVectorizer
from scikits.learn.features.text import CharNGramAnalyzer
from scikits.learn.sparse.svm import LinearSVC
from scikits.learn.metrics import confusion_matrix

from nltk.corpus import europarl_raw

class BreakNested(Exception): pass

def build_datasets_from_europarl(shuffle=True, seed=42, dim=100000,
                                 max_samples_by_lang=1000):
    """Build the training and test set out of the Europarl corpus from nltk"""

    languages = ["danish", "dutch", "english", "finnish", "french", "german",
                 "greek", "italian", "portuguese", "spanish", "swedish"]

    labels = []
    sentences = []

    for label, language in enumerate(languages):
        language_corpus = getattr(europarl_raw, language)
        fileids = language_corpus.fileids()
        i = 0
        try:
            for fileids in fileids:
                for sent in language_corpus.raw(fileids).split('.'):
                    if len(sent) > 10:
                        labels.append(label)
                        sentences.append((" ".join(sent) 
                                          + '.').encode('utf-8'))
                        i += 1
                        if i >= max_samples_by_lang:
                            raise BreakNested()
        except BreakNested:
            pass

    labels = np.asarray(labels)
    hv = SparseHashingVectorizer(dim=dim,
                                 analyzer=CharNGramAnalyzer(min_n=2, max_n=4),
                                 use_idf=False)
    hv.vectorize(sentences)
    features = hv.get_vectors()

    if shuffle:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(labels))

        features = features[perm]
        labels = labels[perm]

    return features, labels



if __name__ == "__main__":
    import pylab as pl

    t0 = time()
    X, Y = build_datasets_from_europarl()
    print "Feature extraction took %ds" % (time() - t0)
    n_samples = len(Y)

    print "n_samples:", n_samples
    print "avg non zero features per sample: ", X.getnnz() / n_samples

    cutoff = n_samples * 3 / 4
    X_train, X_test = X[:cutoff], X[cutoff:]
    Y_train, Y_test = Y[:cutoff], Y[cutoff:]
    t0 = time()
    clf = LinearSVC(C=10, dual=False, penalty='l1').fit(X_train, Y_train)
    print "Training took %ds" % (time() - t0)

    pred = clf.predict(X_test)
    print "accuracy:", np.mean(pred == Y_test)
    cm = confusion_matrix(Y_test, pred)

    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()

