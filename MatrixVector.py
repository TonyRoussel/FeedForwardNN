import operator
import math
import random

class Vector(object):
    def __init__(self, lst):
        self._vector = lst

    @classmethod
    def fromLen(cls, l):
        _vector = [0] * l
        return cls(_vector)

    @classmethod
    def randomVec(cls, l):
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

    def __sub__(self, rhsvec):
        return Vector(map(operator.sub, self._vector, rhsvec._vector))

    def __mul__(self, rhsvec):
        return Vector(map(operator.mul, self._vector, rhsvec._vector))

    def __div__(self, rhsvec):
        return Vector(map(operator.div, self._vector, rhsvec._vector))

    def __pow__(self, rhsvec):
        return Vector(map(operator.pow, self._vector, rhsvec._vector))

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
    def randomMat(cls, row, col):
        

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
                sum = 0
                for k in range(0, result.getColLen()):
                    sum += self[i][k] * rhsmat[k][j]
                result[i][j] = sum
        return result

    def __add__(self, rhsmat):
        if (self.getSize() != rhsmat.getSize()):
            raise MatrixError, "wiseOp on different size matrixs"
        return Matrix(map(lambda l, r: l + r, self._matrix, rhsmat._matrix))

    def __sub__(self, rhsmat):
        if (self.getSize() != rhsmat.getSize()):
            raise MatrixError, "wiseOp on different size matrixs"
        return Matrix(map(lambda l, r: l - r, self._matrix, rhsmat._matrix))

    def __mul__(self, rhsmat):
        if (self.getSize() != rhsmat.getSize()):
            raise MatrixError, "wiseOp on different size matrixs"
        return Matrix(map(lambda l, r: l * r, self._matrix, rhsmat._matrix))

    def __div__(self, rhsmat):
        if (self.getSize() != rhsmat.getSize()):
            raise MatrixError, "wiseOp on different size matrixs"
        return Matrix(map(lambda l, r: l / r, self._matrix, rhsmat._matrix))

    def __pow__(self, rhsmat):
        if (self.getSize() != rhsmat.getSize()):
            raise MatrixError, "wiseOp on different size matrixs"
        return Matrix(map(lambda l, r: l ** r, self._matrix, rhsmat._matrix))

    def __str__(self):
        return "[" + '\n'.join(str(row) for row in self._matrix) + "]"

    def __getitem__(self, idx):
        return self._matrix[idx]

    def __setitem__(self, idx, value):
        self._matrix[idx] = value




# Test
if (__name__ == "__main__"):
    print ("Vector([1, 2, 3, 4]) -->")
    print (Vector([1, 2, 3, 4]))
    print ("Vector([1, 2, 3, 4]) + Vector([1, 2, 3, 4]).opp() -->")
    print (Vector([1, 2, 3, 4]) + Vector([1, 2, 3, 4]).opp())
    print ("Vector([1, 2, 3, 4]) - Vector([1, 2, 3, 4]) -->")
    print (Vector([1, 2, 3, 4]) - Vector([1, 2, 3, 4]))
    print ("Vector([1, 2, 3, 4]) / Vector([1, 2, 3, 4]) -->")
    print (Vector([1, 2, 3, 4]) / Vector([1, 2, 3, 4]))
    print ("Vector([1, 2, 3, 4]).add(1.5) -->")
    print (Vector([1, 2, 3, 4]).add(1.5))
    print ("(Vector([1, 2, 3, 4]) - Vector([1, 2, 3, 4]).mul(2)).abs() -->")
    print ((Vector([1, 2, 3, 4]) - Vector([1, 2, 3, 4]).mul(2)).abs())
    print ("Vector([1, 2, 3, 4]).normalization() -->")
    print (Vector([1, 2, 3, 4]).normalization())
    print ("Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]) -->")
    print (Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]))
    print ("Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).transpose() -->")
    print (Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).transpose())
    print ("Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]) + Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])])")
    print (Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]) + Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]))
    print ("Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).opp() -->")
    print (Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).opp())
    print ("Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).dotProduct(Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).transpose())")
    print (Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).dotProduct(Matrix([Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4]), Vector([1, 2, 3, 4])]).transpose()))
