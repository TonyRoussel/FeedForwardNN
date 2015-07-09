import operator
import math
import random

class Vector(object):
    def __init__(self, lst):
        self._vector = lst

    @classmethod
    def fromLen(cls, l, val=0):
        _vector = [val] * l
        return cls(_vector)

    @classmethod
    def random(cls, l):
        _vector = [random.random() for _ in range(0, l)]
        return cls(_vector)

    def len(self):
        return len(self._vector)

    def getContainer(self):
        return self._vector

    def add(self, rhs):
        return Vector(map(lambda x: x + rhs, self._vector))

    def sub(self, rhs):
        return Vector(map(lambda x: x - rhs, self._vector))

    def mul(self, rhs):
        return Vector(map(lambda x: x * rhs, self._vector))

    def div(self, rhs):
        return Vector(map(lambda x: x / rhs, self._vector))

    def pow(self, rhs):
        return Vector(map(lambda x: x ** rhs, self._vector))

    def __add__(self, rhsvec):
        return Vector(map(operator.add, self._vector, rhsvec._vector))

    def __sub__(self, rhs):
        if isinstance(rhs, self.__class__):
            return Vector(map(operator.sub, self._vector, rhs._vector))
        return Vector([x - rhs for x in self._vector])

    def __mul__(self, rhs):
        if isinstance(rhs, self.__class__):
            return Vector(map(operator.mul, self._vector, rhs._vector))
        return Vector([x * rhs for x in self._vector])

    def __div__(self, rhsvec):
        return Vector(map(operator.div, self._vector, rhsvec._vector))

    def __pow__(self, rhs):
        if isinstance(rhs, self.__class__):
            return Vector(map(operator.pow, self._vector, rhsvec._vector))
        return Vector([x ** rhs for x in self._vector])

    def opp(self):
        return Vector(map(lambda x: -x, self._vector))
   
    def sum(self):
        return sum(self._vector)

    def norm(self):
        return math.sqrt(sum(map(lambda x: x * x, self._vector)))

    def normalization(self):
        if (len(self._vector) == 1 or self.norm == 0):
            return Vector(self._vector)
        return Vector(map(lambda x: x / self.norm(), self._vector))

    def abs(self):
        return Vector(map(lambda x: x if x >= 0 else -x, self._vector))

    def mean(self):
        return sum(self._vector) / float(len(self._vector))

    def __str__(self):
        return '[{}]'.format(', '.join(str(i) for i in self._vector))

    def __getitem__(self, idx):
        return self._vector[idx]

    def __setitem__(self, idx, value):
        self._vector[idx] = value

##############################################################

class MatrixError(Exception):
    pass

class Matrix(object):
    def __init__(self, vectorlist):
        self._matrix = vectorlist

    @classmethod
    def fromSize(cls, row, col):
        _matrix = [Vector.fromLen(col)] * row
        return cls(_matrix)

    @classmethod
    def random(cls, row, col):
        _matrix = [Vector.random(col) for _ in range(0, row)]
        return cls(_matrix)

    def isSquare(self):
        return self.getRowLen() == self.getColLen()

    def getRowLen(self):
        return len(self._matrix)

    def getColLen(self):
        return self._matrix[0].len()

    def getSize(self):
        return (self.getRowLen(), self.getColLen())
   
    def transpose(self):
        return Matrix([Vector(list(item)) for item in zip(*self._matrix)])

    def opp(self):
        return Matrix(map(lambda x: x.opp(), self._matrix))

    def dotProduct(self, rhsmat):
        if (self.getColLen() != rhsmat.getRowLen()):
            raise MatrixError, "dotProduct invalid matrix size"
        result = Matrix.fromSize(self.getRowLen(), rhsmat.getColLen())
        for i in range(0, result.getRowLen()):
            for j in range(0, result.getColLen()):
                summ = 0
                for k in range(0, self.getColLen()):
                    summ += self[i][k] * rhsmat[k][j]
                result[i][j] = summ
        return result

    def sum(self, axis=0):
        if (axis == 0):
            return Matrix(map(lambda x: Vector([x.sum()]), self._matrix))
        return Matrix(map(lambda x: Vector([x.sum()]), self.transpose()._matrix))
                      
    def __add__(self, rhsmat):
        if (self.getSize() != rhsmat.getSize()):
            raise MatrixError, "wiseOp on different size matrixs"
        return Matrix(map(lambda l, r: l + r, self._matrix, rhsmat._matrix))

    def __sub__(self, rhs):
        if isinstance(rhs, self.__class__):
            if (self.getSize() != rhs.getSize()):
                raise MatrixError, "wiseOp on different size matrixs"
            return Matrix(map(lambda l, r: l - r, self._matrix, rhs._matrix))
        return Matrix(map(lambda l: l - rhs, self._matrix))

    def __mul__(self, rhs):
        if isinstance(rhs, self.__class__):
            if (self.getSize() != rhs.getSize()):
                raise MatrixError, "wiseOp on different size matrixs"
            return Matrix(map(lambda l, r: l * r, self._matrix, rhs._matrix))
        return Matrix(map(lambda l: l * rhs, self._matrix))

    def __div__(self, rhsmat):
        if (self.getSize() != rhsmat.getSize()):
            raise MatrixError, "wiseOp on different size matrixs"
        return Matrix(map(lambda l, r: l / r, self._matrix, rhsmat._matrix))

    def __pow__(self, rhs):
        if isinstance(rhs, self.__class__):
            if (self.getSize() != rhs.getSize()):
                raise MatrixError, "wiseOp on different size matrixs"
            return Matrix(map(lambda l, r: l ** r, self._matrix, rhs._matrix))
        return Matrix(map(lambda l: l ** rhs, self._matrix))

    def __str__(self):
        return "[" + '\n'.join(str(row) for row in self._matrix) + "]"

    def __getitem__(self, idx):
        return self._matrix[idx]

    def __setitem__(self, idx, value):
        self._matrix[idx] = value

##########################################################################################
import sys

def hx(thetas, X):
    hxdotp = thetas.transpose().dotProduct(X)
    return hxdotp[0].sum()

def costFunction(X, y, thetas):
    m = y.getColLen()
    H = thetas.transpose().dotProduct(X)
    diff = H - y
    diff = diff ** 2
    return diff[0].sum() / (2 * m)

def gradientDescent(X, y, thetas, alpha, numiters):
    m = y.getColLen()
    for i in range(0, numiters):
        H = thetas.transpose().dotProduct(X)
        diff = H - y
        diffM = X * diff[0]
        sdiffm = diffM.sum(axis=0).transpose()
        sigma = sdiffm * (1 / float(m))
        thetas = (thetas.transpose() - (sigma * alpha)).transpose()
    return thetas

contentrequest = [list() for _ in range(0, 2)]
for idx, line in enumerate(sys.stdin):
    spltLine = line.rstrip('\n').split(' ')
    if (idx == 0):
        n = int(spltLine[0])
        m = int(spltLine[1])
        continue
    if (idx == m + 1):
        npredict = int(spltLine[0])
        continue
    spltLine = [float(val) for val in spltLine]
    if (idx <= m):
        contentrequest[0].append(Vector(spltLine))
        continue
    contentrequest[1].append(Vector(spltLine))
    if (idx >= m + npredict + 1):
        break
trainingMatrix = Matrix(contentrequest[0])
predictMatrix = Matrix(contentrequest[1])
trainingMatrix = trainingMatrix.transpose()
predictMatrix = predictMatrix.transpose()
print ("trainingMatrix")
print (trainingMatrix)
print ("predictMatrix")
print (predictMatrix)
X = Matrix([trainingMatrix[i] for i in range(0, n)])
Xones = Matrix([Vector.fromLen(m, val=1)] + X._matrix)
Y = Matrix([trainingMatrix[n]])
thetas = Matrix.random(1, Xones.getRowLen()).transpose()
print ("Xones")
print (Xones)
print ("Y")
print (Y)
print ("thetas")
print (thetas)
print ("hx(thetas, Xones)")
print (hx(thetas, Xones))
print ("costFunction(Xones, Y, thetas)")
print (costFunction(Xones, Y, thetas))
print ("gradientDescent(Xones, Y, thetas, 0.1, 5000)")
print (gradientDescent(Xones, Y, thetas, 0.1, 5000))

# http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables.html
