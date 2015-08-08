from MatrixVector import Matrix, Vector

##### 2 Layer Neural Net

# input data

X = Matrix([Vector([0.,0.,1.]),
           Vector([0.,1.,1.]),
           Vector([1.,0.,1.]),
            Vector([1.,1.,1.])])

y = Matrix([Vector([0.,1.,1.,0.])]).transpose()

# randomly initialize our weights with mean 0
syn0 = 2 * Matrix.random(3, 4) - 1
syn1 = 2 * Matrix.random(4, 1) - 1

# set the alpha
alpha = 1

for j in xrange(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = l0.dotProduct(syn0).nonlin()
    l2 = l1.dotProduct(syn1).nonlin()

    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
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
    syn0 += alpha * (l0.transpose().dotProduct(l1_delta))
