from MatrixVector import Matrix, Vector

class FeedForwardNeuralNetworkError(Exception):
    pass

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

        # init the neural layer
        self._weights = []
        for s_in, s_out in zip(layers_shape[:-1], layers_shape[1:]):
            # random init with mean 0 of a layer (with bias unit if require)
            self._weights.append(2 * Matrix.random(s_in + bias_unit, s_out) - 1)






if __name__ == "__main__":
    ffnn = FeedForwardNN([3, 4, 1])
    print "ffnn._layers_shape"
    print ffnn._layers_shape
    print "ffnn._layers_count"
    print ffnn._layers_count
    print "ffnn._weights"
    print ffnn._weights
