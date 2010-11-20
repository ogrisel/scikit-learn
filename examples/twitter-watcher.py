# Author: Olivier Grisel <olivier.grisel@ensta.org>
#
# License: BSD Style.
"""Sample CLI app that uses linear model to watch for interesting tweets

You need a twitter account with some history (a couple of hundred of
status updates and retweets should be fine) to be able to build a training set
to train a statistical model that is able to find wether a new tweet is likely
to be of interest or not.

To access your tweeting history you will need a web access to use the twitter
OAuth interface thanks to

Usage:

"""

import os
import sys
import webbrowser
from time import time
from pprint import pprint
from cPickle import dump
from cPickle import load

import numpy as np

from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler

from scikits.learn.feature_extraction.text.sparse import CountVectorizer
from scikits.learn.feature_extraction.text.sparse import TfidfTransformer
from scikits.learn.sgd.sparse import SGD
from scikits.learn.grid_search import GridSearchCV
from scikits.learn.pipeline import Pipeline
from scikits.learn.metrics import f1_score

CONSUMER_KEY = '1Mc9h175eBX8tjt8GOQvQ'
CONSUMER_SECRET = 'zk1tVhgFUcjlFbDx40Ue5KR3x7HxWdrJzRM82OsEQ'

ACCESS_TOKEN_FILENAME = "access_token.txt"

def get_api():
    """Perform OAuth credential checks to get an API instance"""
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

    if os.path.exists(ACCESS_TOKEN_FILENAME):
        lines = open(ACCESS_TOKEN_FILENAME).readlines()
        access_key = lines[0].strip()
        access_secret = lines[1].strip()
        auth.set_access_token(access_key, access_secret)
    else:
        auth_url = auth.get_authorization_url()
        pid = os.fork()
        if not pid:
            # child runs browser then quits.
            webbrowser.open_new_tab(auth_url)
            sys.exit(0)

        print 'Please authorize: ' + auth_url
        verifier = raw_input('PIN: ').strip()
        auth.get_access_token(verifier)

        # cache to disk for later usage
        with open(ACCESS_TOKEN_FILENAME, "w") as f:
            f.write(auth.access_token.key + "\n")
            f.write(auth.access_token.secret + "\n")
    return API(auth)


def collect(cli_args, interesting_filename="interesting.pickle",
            boring_filename="boring.pickle", n_samples=1000):
    """Collect some data to train a model"""
    api = get_api()


    if not os.path.exists(interesting_filename):
        n = n_samples / 2
        print "collecting positive tweets"
        interesting = [s.text for s in Cursor(api.user_timeline).items(n)]
        interesting += [s.text for s in Cursor(api.retweeted_by_me).items(n)]
        dump(interesting, file(interesting_filename, 'w'))

    if not os.path.exists(boring_filename):
        n_pages = n_samples / 20
        print "collecting boring tweets"
        boring = []
        for i in range(n_pages):
            print "page: %d/%d" % (i + 1, n_pages)
            boring += [s.text for s in api.public_timeline()]
            time.sleep(0.500)
            dump(boring, file(boring_filename, 'w'))


def build_model(cli_args, interesting_filename="interesting.pickle",
                boring_filename="boring.pickle",
                model_filename="twitter-model.pickle", seed=42):
    interesting = load(file(interesting_filename))
    boring = load(file(boring_filename))

    # build the dataset as numpy arrays for text input and target
    text = np.asarray(interesting + boring)
    target = np.asarray([1] * len(interesting) + [-1] * len(boring))

    n_samples = text.shape[0]

    # shuffle the dataset
    indices = np.arange(n_samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    text = text[indices]
    target = target[indices]

    # build a grid search for hyperparameters of both the feature extractor and
    # the classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGD()),
    ])

    parameters = {
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__analyzer__max_n': (1, 2), # words or bigrams
        'tfidf__use_idf': (True, False),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        'clf__n_iter': (10, 50, 80),
    }

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters,
                               score_func=f1_score, n_jobs=-1)

    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
    print "parameters:"
    pprint(parameters)
    t0 = time()
    grid_search.fit(text, target)
    print "done in %0.3fs" % (time() - t0)
    print

    print "Best score: %0.3f" % grid_search.best_score
    print "Best parameters set:"
    model = grid_search.best_estimator
    best_parameters = model._get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])

    dump(model, file(model_filename, 'w'))



def predict(cli_args):
    print "Implement me!"


def watch_stream(cli_args):
    print "Implement me!"


command_handlers = {
    'collect': collect,
    'train': build_model,
    'predict': predict,
    'watch': watch_stream,
}

def dispatch(args):
    """Dispatch the execution to the matching sub command"""
    usage = "Usage: python twitter-watcher.py train | predict | watch --help"

    if len(args) == 1:
        print usage
        sys.exit(0)

    cmd = args[1]
    cli_args = args[2:]

    handler = command_handlers.get(cmd)
    if handler is None:
        print "Invalid command " + cmd
        print usage
        sys.exit(1)

    return handler(cli_args)



if __name__ == "__main__":
    dispatch(sys.argv)


