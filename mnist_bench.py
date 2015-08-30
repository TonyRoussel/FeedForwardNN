import operator
import numpy as np
from MNIST_db import mnist_numpy
from FeedForwardNeuralNetworkNumpy import FeedForwardNN

from pylab import *

def convert_mnist_2d_1ToNOutput(imgs, labels):
    output_map =\
    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
    X = list()
    y = list()
    for idx, img in enumerate(imgs):
        X.append(list(img.flatten()))
        y.append(output_map[labels[idx][0] - 1])
    return X, y

images_training, labels_training = mnist_numpy.load_mnist('training')

# print "images.shapes"
# print images.shape
# print "labels.shapes"
# print labels.shape
# print "labels_training[0]"
# print labels_training[0]
# print "images[0].shape"
# print images[0].shape
# print "images[0].flatten().shape"
# print images[0].flatten().shape
# imshow(images_training[0], cmap=cm.gray)
# show()

# images_training /= 255.

Xt, yt = convert_mnist_2d_1ToNOutput(images_training, labels_training)

X = np.array(Xt)
y = np.array(yt)

ffnn = FeedForwardNN([len(X[0]), 100, len(y[0])], hidden_layer="sigmoid", output_layer="sigmoid", input_layer="sigmoid")

ffnn.backpropagation_training(X, y, alpha=3, epoch=100, momentum=0., l_regularization=0., plot_error=True)
# ffnn.SGD_training(X, y, alpha=1e-7, epoch=5000, momentum=0.99, mini_batch_size=10, plot_error=True)
# ffnn.adadelta_training(X, y, epoch=100, l_regularization=0, plot_error=True)

images_test, labels_test = mnist_numpy.load_mnist('testing')
# images_test /= 255.
Xp, yp = convert_mnist_2d_1ToNOutput(images_test, labels_test)
Xp = np.array(Xp)
yp = np.array(yp)

prediction = ffnn.run(X)
total = len(y)
count = 0
for i in xrange(total):
    indexp = max(enumerate(prediction[i]), key=operator.itemgetter(1))[0]
    indexe = max(enumerate(y[i]), key=operator.itemgetter(1))[0]
    if indexp == indexe:
        count = count + 1
print "Final rate on training set:", count / float(total)


prediction = ffnn.run(Xp)
total = len(yp)
count = 0
for i in xrange(total):
    indexp = max(enumerate(prediction[i]), key=operator.itemgetter(1))[0]
    indexe = max(enumerate(yp[i]), key=operator.itemgetter(1))[0]
    if indexp == indexe:
        count = count + 1
print "Final rate on validation set:", count / float(total)

raw_input('waiting to close graph')

