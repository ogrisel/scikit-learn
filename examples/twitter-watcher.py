"""Sample CLI app that uses linear model to watch for interesting tweets"""

import sys
try:
    import tweepy
except ImportError:
    print "This example requires http://github.com/joshthecoder/tweepy"
    sys.exit(-1)

api = tweepy.API()
user_id_to_watch = sys.argv[1]

for status in tweepy.Cursor(api.user_timeline, id=user_id_to_watch).items(50):
    print status.text



