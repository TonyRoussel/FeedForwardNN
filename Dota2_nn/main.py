import numpy as np
import csv
import operator
from FeedForwardNeuralNetworkNumpy import FeedForwardNN

# from pybrain.datasets import SupervisedDataSet
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.structure import SoftmaxLayer
# from sklearn import preprocessing


def read_training_data():
    with open("trainingdata.txt", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        X = list()
        y = list()
        summon_names = dict()
        for i, row in enumerate(reader):
            if i == 0:
                ncols = len(row)
            for value in row[0:ncols - 1]:
                if value not in summon_names:
                    if len(summon_names) == 0:
                        summon_names[value] = 0
                    else:
                        summon_names[value] = len(summon_names)
            X.append([float(summon_names[value]) for value in row[0:ncols - 1]])
            if int(row[ncols - 1]) == 1:
                y.append([1, 0])
            else:
                y.append([0, 1])
    return np.array(X), np.array(y), summon_names

def normalizer(X):
    return X.sum(axis=1)

def normalize(mtx, norm):
    return mtx / norm[:, np.newaxis]


X, y, d = read_training_data()
norm = normalizer(X)
ffnn = FeedForwardNN([10, 6, 2], hidden_layer="tanh", output_layer="tanh", input_layer="tanh")
# ffnn.backpropagation_training(normalize(X, norm), y, alpha=0.00001, epoch=100, momentum=0.99)
ffnn.adadelta_training(normalize(X, norm), y, epoch=100)

prediction = ffnn.run(normalize(X, norm))
total = len(y)
count = 0
for i in xrange(total):
    indexp = max(enumerate(prediction[i]), key=operator.itemgetter(1))[0]
    indexe = max(enumerate(y[i]), key=operator.itemgetter(1))[0]
    if indexp == indexe:
        count = count + 1
print "Final rate:", count / float(total)

# X = normalize(X, norm)
# norm = preprocessing.Normalizer().fit(X)
# X = norm.transform(X)

# ds = SupervisedDataSet(10, 2)
# for i in range(0, len(X)):
#     ds.addSample(X[i], y[i])

# # nn && trainer construction
# net = buildNetwork(ds.indim, (ds.indim + ds.outdim) / 2, ds.outdim, bias=True) # building the n
# trainer = BackpropTrainer(net, ds, learningrate=0.01, momentum=0., verbose=True)
# trainer.trainUntilConvergence(maxEpochs=1000) # Train, until convergence
