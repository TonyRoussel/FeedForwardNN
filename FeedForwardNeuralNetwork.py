from MatrixVector import Matrix, Vector

class FeedForwardNeuralNetworkError(Exception):
    pass

def append_bias(mtx):
    return (Matrix(Matrix.ones(1, mtx.getRowLen())._matrix + mtx.transpose()._matrix).transpose())

def remove_bias(mtx):
    return (Matrix(mtx.transpose()._matrix[1:]).transpose())

class FeedForwardNN(object):
    """ A feed forward neural network"""
    
    def __init__(self, layers_shape, bias_unit=True):
        """ initialisation of the ff neural network"""

        # check minimum layer shape
        if len(layers_shape) < 2:
            raise FeedForwardNeuralNetworkError, "can't init a neural net without at least the in and out size"
        # save the layer shape
        self._layers_shape = layers_shape
        self._layers_count = len(layers_shape)
        self._bias_unit = bias_unit

        # init layer input/output memory
        self._layer_input = list()
        self._layer_output = list()

        # init the neural layer
        self._weights = []
        for s_in, s_out in zip(layers_shape[:-1], layers_shape[1:]):
            # random init with mean 0 of a layer (with bias unit if require)
            self._weights.append(2 * Matrix.random(s_in + bias_unit, s_out) - 1)

    def run(self, X):
        """Run the neural net against data row """
        # reset layer input/output memory
        self._layer_input = list()
        self._layer_output = list()

        self._layer_output.append(X)

        for weight in self._weights:
            layer_input = self._layer_output[-1] if not self._bias_unit else append_bias(self._layer_output[-1])
            layer_input = layer_input.dotProduct(weight)

            self._layer_input.append(layer_input)
            self._layer_output.append(layer_input.nonlin())
        return self._layer_output[-1]



if __name__ == "__main__":
    X = Matrix([Vector([0.,0.,1.]),
                Vector([0.,1.,1.]),
                Vector([1.,0.,1.]),
                Vector([1.,1.,1.])])
    ffnn = FeedForwardNN([3, 4, 1])
    print "ffnn._layers_shape"
    print ffnn._layers_shape
    print "ffnn._layers_count"
    print ffnn._layers_count
    print "ffnn._weights"
    for layer in ffnn._weights:
        print layer
    print "ffnn.run(X)"
    print ffnn.run(X)
