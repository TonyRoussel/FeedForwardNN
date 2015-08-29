from MNIST_db import mnist_numpy

from pylab import *
from numpy import *

images, labels = mnist_numpy.load_mnist('training', digits=[2])
imshow(images.mean(axis=0), cmap=cm.gray)
show()
