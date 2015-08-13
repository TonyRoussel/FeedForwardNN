import numpy as np
import csv
import operator
import random
from FeedForwardNeuralNetworkNumpy import FeedForwardNN


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
    return X, y, summon_names

def read_training_data_one_of_n():
    with open("trainingdata.txt", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
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
    with open("trainingdata.txt", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        X = list()
        y = list()
        for i, row in enumerate(reader):
            line = list()
            for value in row[0:ncols - 1]:
                oon = [0] * len(summon_names)
                oon[summon_names[value]] = 1
                line = line + oon
            X.append(line)
            if int(row[ncols - 1]) == 1:
                y.append([1, 0])
            else:
                y.append([0, 1])
    return X, y, summon_names


def normalizer(X):
    return X.sum(axis=1)

def normalize(mtx, norm):
    return mtx / norm[:, np.newaxis]


# X, y, d = read_training_data()
Xl, yl, d = read_training_data_one_of_n()

# # crossvalidation data construction RANDOM PICK
Xt = Xl
yt = yl
Xp = list()
yp = list()
for i in range(0, int(0.5 * len(Xl))):
    popi = random.randint(0, len(Xl) - 1)
    Xp.append(Xl[popi])
    yp.append(yl[popi])
    Xt.pop(popi)
    yt.pop(popi)
# # / crossvalidation data construction

X = np.array(Xt)
y = np.array(yt)
Xp = np.array(Xp)
yp = np.array(yp)

# norm = normalizer(X)
ffnn = FeedForwardNN([len(X[0]), (len(X[0]) + len(y[0])) / 2, len(y[0])], hidden_layer="tanh", output_layer="tanh", input_layer="tanh")

# ffnn.backpropagation_training(normalize(X, norm), y, alpha=0.00001, epoch=100, momentum=0.99)
# ffnn.backpropagation_training(X, y, alpha=1e-7, epoch=4000, momentum=0.99)
# ffnn.adadelta_training(normalize(X, norm), y, epoch=100)
ffnn.adadelta_training(X, y, epoch=4000)

# prediction = ffnn.run(normalize(X, norm))
prediction = ffnn.run(Xp)
total = len(yp)
count = 0
for i in xrange(total):
    indexp = max(enumerate(prediction[i]), key=operator.itemgetter(1))[0]
    indexe = max(enumerate(yp[i]), key=operator.itemgetter(1))[0]
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
