from MatrixVector import Matrix, Vector

##### 2 Layer Neural Net

# input data

X = Matrix([Vector([0.,0.,1.]),
           Vector([0.,1.,1.]),
           Vector([1.,0.,1.]),
            Vector([1.,1.,1.])])

Xones = Matrix(Matrix.ones(1, X.getRowLen())._matrix + X.transpose()._matrix).transpose()
y = Matrix([Vector([0.,1.,1.,0.])]).transpose()

# randomly initialize our weights with mean 0 (entry + 1 for bias unit)
syn0 = 2 * Matrix.random(4, 4) - 1
syn1 = 2 * Matrix.random(5, 1) - 1

# set the alpha
alpha = 10

for j in xrange(1000):

    # Feed forward through layers 0, 1, and 2
    l0 = Xones
    l1 = l0.dotProduct(syn0).nonlin()
    l1 = Matrix(Matrix.ones(1, l0.getRowLen())._matrix + l1.transpose()._matrix).transpose()
    l2 = l1.dotProduct(syn1).nonlin()

    # how much did we miss the target value?
    l2_error = y - l2

    if (j % 100) == 0:
        print "Error:" + str(l2_error.abs().mean())

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * l2.nonlin(True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dotProduct(syn1.transpose())

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * l1.nonlin(True)

    syn1 += alpha * (l1.transpose().dotProduct(l2_delta))
    l1_delta = Matrix(l1_delta.transpose()._matrix[1:]).transpose()
    syn0 += alpha * (l0.transpose().dotProduct(l1_delta))

print "output after training"
l1 = (Xones.dotProduct(syn0)).nonlin()
l1 = Matrix(Matrix.ones(1, l1.getRowLen())._matrix + l1.transpose()._matrix).transpose()
print l1.dotProduct(syn1).nonlin()
