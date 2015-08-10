from MatrixVector import Matrix, Vector

##### 2 Layer Neural Net

# input data

X = Matrix([Vector([0.,0.,1.]),
           Vector([0.,1.,1.]),
           Vector([1.,0.,1.]),
            Vector([1.,1.,1.])])

y = Matrix([Vector([0.,0.,1.,1.])]).transpose()

syn0 = 2 * Matrix.random(3,1) - 1
print syn0

for iter in xrange(10000):
    # forward propagation
    l0 = X
    l1 = (l0.dotProduct(syn0)).nonlin()

    # error eval
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * l1.nonlin(True)

    # weight update
    syn0 = syn0 + (l0.transpose()).dotProduct(l1_delta)

print "output after training"
print (X.dotProduct(syn0)).nonlin()
