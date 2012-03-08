import pylab as pl
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

digits = load_digits()
n_samples, n_features = digits.data.shape
print "n_samples:", n_samples
print "n_features:", n_features

# The data that we are interested in is made of 8x8 images of digits,
# let's have a look at the first 3 images, stored in the `images`
# attribute of the dataset. If we were working from image files, we
# could load them using pylab.imread. For these images know which
# digit they represent: it is given in the 'target' of the dataset.


def plot_some_digits(data, labels, title, row=0):
    for index, (data, label) in enumerate(zip(data, labels)[:4]):
        pl.subplot(2, 4, (row * 4) + index + 1)
        pl.axis('off')
        pl.imshow(data.reshape(8, 8), cmap=pl.cm.gray_r,
                  interpolation='nearest')
        pl.title(title + ' %i' % label)


data_train, data_test, label_train, label_test = train_test_split(
    digits.data, digits.target, test_fraction=0.5)


# TASK: create a SVC instance with C=10 and gamma=0.001 and train it

# TASK: predict the outcome on the test set and store it in a variable nammed
# `predicted`

# Let's plot some training samples and predicted outcome on testing samples:
plot_some_digits(data_train, label_train, 'Training:', row=0)
plot_some_digits(data_test, predicted, 'Prediction:', row=1)
pl.show()

# TASK: compute the cross validated score of the model on the whole dataset

print "CV score: %0.3f +/- %0.3f" % (scores.mean(), scores.std() / 2)
