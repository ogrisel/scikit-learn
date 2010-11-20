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
import time
from pprint import pprint

from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler

from scikits.learn.svm.sparse import LinearSVC

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


def build_model(cli_args, n_samples=100):
    """Train a model"""
    api = get_api()

    n = n_samples / 4
    interesting = [s.text for s in Cursor(api.user_timeline).items(n)]
    interesting += [s.text for s in Cursor(api.retweeted_by_me).items(n)]
    print "successfully collected %d interesting tweets" % len(interesting)

    n_pages = n_samples / 20
    boring = []
    for i in range(n_pages):
        boring += [s.text for s in api.public_timeline()]
        time.sleep(0.500)
    print "successfully collected %d boring tweets" % len(boring)

    print "interesting:"
    pprint(interesting)
    print
    print "boring:"
    pprint(boring)
    print


def predict(cli_args):
    print "Implement me!"


def watch_stream(cli_args):
    print "Implement me!"


command_handlers = {
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


