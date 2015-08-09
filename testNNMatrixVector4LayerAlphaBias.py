from MatrixVector import Matrix, Vector

def append_bias(mtx):
    return (Matrix(Matrix.ones(1, mtx.getRowLen())._matrix + mtx.transpose()._matrix).transpose())

def remove_bias(mtx):
    return (Matrix(mtx.transpose()._matrix[1:]).transpose())

##### 4 Layer Neural Net

# input data

X = Matrix([Vector([0.,0.,1.]),
           Vector([0.,1.,1.]),
           Vector([1.,0.,1.]),
            Vector([1.,1.,1.])])

Xones = Matrix(Matrix.ones(1, X.getRowLen())._matrix + X.transpose()._matrix).transpose()
y = Matrix([Vector([0.,1.,1.,0.])]).transpose()

# randomly initialize our weights with mean 0 (entry + 1 for bias unit)
syn0 = 2 * Matrix.random(4, 4) - 1
syn1 = 2 * Matrix.random(5, 4) - 1
syn2 = 2 * Matrix.random(5, 1) - 1

# set the alpha
alpha = 0.07

for j in xrange(10000):

    # Feed forward through layers 0, 1, 2 and 3
    l0 = Xones
    l1 = l0.dotProduct(syn0).nonlin()
    l1tmp = Matrix(Matrix.ones(1, l0.getRowLen())._matrix + l1.transpose()._matrix).transpose()
    l2 = l1tmp.dotProduct(syn1).nonlin()
    l2tmp = Matrix(Matrix.ones(1, l2.getRowLen())._matrix + l2.transpose()._matrix).transpose()
    l3 = l2tmp.dotProduct(syn2).nonlin()

    # how much did we miss the target value?
    l3_error = y - l3

    if (j % 100) == 0:
        print "Error:" + str(l3_error.abs().mean())

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l3_delta = l3_error * l3.nonlin(True)

    # how much did each l2 value contribute to the l3 error (according to the weights)?
    l2_error = l3_delta.dotProduct(syn2.transpose())

    # in what direction is the target l2?
    # were we really sure? if so, don't change too much.
    l2_delta = remove_bias(l2_error) * l2.nonlin(True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dotProduct(syn1.transpose())

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = remove_bias(l1_error) * l1.nonlin(True)

    syn2 += alpha * (l2tmp.transpose().dotProduct(l3_delta))
    syn1 += alpha * (l1tmp.transpose().dotProduct(l2_delta))
    syn0 += alpha * (l0.transpose().dotProduct(l1_delta))

print "output after training"
l1 = (Xones.dotProduct(syn0)).nonlin()
l1tmp = Matrix(Matrix.ones(1, l0.getRowLen())._matrix + l1.transpose()._matrix).transpose()
l2 = l1tmp.dotProduct(syn1).nonlin()
l2tmp = Matrix(Matrix.ones(1, l2.getRowLen())._matrix + l2.transpose()._matrix).transpose()
l3 = l2tmp.dotProduct(syn2).nonlin()
print l3
