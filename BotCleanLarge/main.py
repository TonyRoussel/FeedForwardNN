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

    def min(self):
        idx, val = min(enumerate(self._vector), key=operator.itemgetter(1))
        return idx

    def max(self):
        idx, val = max(enumerate(self._vector), key=operator.itemgetter(1))
        return idx

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

    def sqrt(self):
        return Vector(map(lambda x: math.sqrt(x), self._vector))

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

    def sqrt(self):
        return Matrix(map(lambda x: x.sqrt(), self._matrix))

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


def seekDirt(board):
    dirtVecList = list()
    for irow, row in enumerate(board):
        for icol, c in enumerate(row):
            if c is 'd':
                dirtVecList.append(Vector([irow, icol]))
    return Matrix(dirtVecList)

def takeAction(distDirt):
    euclidianDistDirt = (distDirt ** 2).sum().sqrt().transpose()
    closestDirt = euclidianDistDirt[0].min()
    if distDirt[closestDirt][0] == 0 and distDirt[closestDirt][1] == 0:
        return 0
    elif abs(distDirt[closestDirt][0]) >= abs(distDirt[closestDirt][1]):
        if distDirt[closestDirt][0] < 0:
            return 1
        else:
            return 2
    else:
        if distDirt[closestDirt][1] < 0:
            return 3
        else:
            return 4
    
def next_move(posr, posc, dimh, dimw, board):
    actions = ["CLEAN", "UP", "DOWN", "LEFT", "RIGHT"]
    dirtyAreas = seekDirt(board)
    currPos = Matrix([Vector([posr, posc])])
    distDirt = dirtyAreas - currPos[0]
    action = takeAction(distDirt)
    print actions[action]
    
if __name__ == "__main__":
    pos = [int(i) for i in raw_input().strip().split()]
    dim = [int(i) for i in raw_input().strip().split()]
    board = [[j for j in raw_input().strip()] for i in range(dim[0])]
    next_move(pos[0], pos[1], dim[0], dim[1], board)
                            