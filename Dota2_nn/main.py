import numpy as np
import csv
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
    return np.array(X), np.array(y), summon_names

def normalizer(X):
    return X.sum(axis=1)

def normalize(mtx, norm):
    return mtx / norm[:, np.newaxis]


X, y, d = read_training_data()
norm = normalizer(X)
ffnn = FeedForwardNN([10, 6, 2])
ffnn.backpropagation_training(normalize(X, norm), y, alpha=0.00188, epoch=1000)
