import numpy as np
import sys

class FeedForwardNeuralNetworkError(Exception):
    pass

def append_bias(mtx):
    return np.insert(mtx, 0, 1, axis=1)

def remove_bias(mtx):
    return np.delete(mtx, 0, axis=1)

def nonlin(mtx, deriv=False):
    if deriv is True:
        return mtx * (1 - mtx)
    return 1 / (1 + np.exp(-mtx))

class FeedForwardNN(object):
    """ A feed forward neural network"""
    
    def __init__(self, layers_shape, bias_unit=True, hidden_layer="nonlin", input_layer="nonlin", output_layer="nonlin"):
        """ initialisation of the ff neural network"""

        # check minimum layer shape
        if len(layers_shape) < 2:
            raise FeedForwardNeuralNetworkError, "can't init a neural net without at least the in and out size"

        # init layer types dictionnary
        self._layer_type = {"nonlin" : nonlin}

        # save layers_types
        self._hidden_layer = self._layer_type[hidden_layer]
        self._output_layer = self._layer_type[output_layer]
        self._input_layer = self._layer_type[input_layer]

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
            self._weights.append(2 * np.random.random((s_in + bias_unit, s_out)) - 1)

    def run(self, X):
        """Run the neural net against data row """
        # reset layer input/output memory
        self._layer_input = list()
        self._layer_output = list()

        for idx, weight in enumerate(self._weights):
            if idx == 0:
                layer_input = X if not self._bias_unit else append_bias(X)
            else:
                layer_input = self._layer_output[-1] if not self._bias_unit else append_bias(self._layer_output[-1])
            if idx == 0:
                layer_output = self._input_layer(np.dot(layer_input, weight))
            elif idx < self._layers_count - 1:
                layer_output = self._hidden_layer(np.dot(layer_input, weight))
            else:
                layer_output = self._output_layer(np.dot(layer_input, weight))

            self._layer_input.append(layer_input)
            self._layer_output.append(layer_output)
        return self._layer_output[-1]

    def _measure_deltas(self, y):
        """ Given the last outputs, calculate for each layer weights the distance to target """

        deltas = []
        for idx, output in enumerate(reversed(self._layer_output)):
            if idx == 0:
                l_error = y - output
                glob_error = np.mean(np.abs(l_error))
                delta = l_error * nonlin(output, True)
            else:
                l_error = np.dot(deltas[-idx], self._weights[-idx].T)
                delta = l_error * nonlin(output, True) if not self._bias_unit else remove_bias(l_error) * nonlin(output, True)
            deltas.insert(0, delta)
        return glob_error, deltas

    def _proceed_weights_step(self, alpha, deltas, X):
        """ given an alpha step and the deltas of each layer, move the weights """
        for idx in xrange(self._layers_count - 1):
            self._weights[idx] += alpha * (np.dot(self._layer_input[idx].T, deltas[idx]))

    def backpropagation_training(self, X, y, alpha=0.1, epoch=100, verbose=True):
        """ Train the neural net with the backpropagation algorithm
        return final error """

        verbose_cycle = 0.01 * epoch
        for epk in xrange(0, epoch):
            self.run(X)
            # for each layer output calculate distance to target
            error, deltas = self._measure_deltas(y)

            if verbose and (epk % verbose_cycle) == 0:
                print >> sys.stderr, "Error:", str(error)
            
            # move the weights toward target with alpha step
            self._proceed_weights_step(alpha, deltas, X)

        if verbose:
            print >> sys.stderr, "Final error:", str(error)

        # return last error
        return (np.mean(np.abs(y - self._layer_output[-1])))



if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0, 0],
                  [1, 1],
                  [1, 0],
                  [0, 1]])
    ffnn = FeedForwardNN([3, 4, 2])
    print "ffnn._layers_shape"
    print ffnn._layers_shape
    print "ffnn._layers_count"
    print ffnn._layers_count
    print "ffnn.backpropagation_training(X, y)"
    print ffnn.backpropagation_training(X, y, alpha=0.07, epoch=20000)
    print "ffnn.run(X)"
    print ffnn.run(X)
