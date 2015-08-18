import numpy as np
import sys
import random
import matplotlib.pyplot as plt

class FeedForwardNeuralNetworkError(Exception):
    pass

def append_bias(mtx):
    return np.insert(mtx, 0, 1, axis=1)

def remove_bias(mtx):
    return np.delete(mtx, 0, axis=1)

def sigmoid(mtx, deriv=False):
    if deriv is True:
        sig = sigmoid(mtx)
        return sig * (1 - sig)
    return 1 / (1 + np.exp(-mtx))

def tanh(mtx, deriv=False):
    if deriv is True:
        return 1 - mtx * mtx
    return np.tanh(mtx)

def softmax(mtx, deriv=False):
    if deriv is True:
        soft = softmax(mtx)
        return soft * (1 - soft)
    exp_mtx = np.exp(mtx)
    return exp_mtx / np.sum(exp_mtx)

def update_error_plot(fig, x, y):
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    fig.canvas.draw()
 

class FeedForwardNN(object):
    """ A feed forward neural network """
    
    def __init__(self, layers_shape, bias_unit=True, hidden_layer="sigmoid", input_layer="sigmoid", output_layer="sigmoid"):
        """ initialisation of the ff neural network"""

        # check minimum layer shape
        if len(layers_shape) < 2:
            raise FeedForwardNeuralNetworkError, "can't init a neural net without at least the in and out size"

        # init layer types dictionnary
        self._layer_type = {"sigmoid" : sigmoid,
                            "tanh" : tanh,
                            "softmax" : softmax}

        # save layers_types
        self._hidden_layer_type = hidden_layer
        self._output_layer_type = output_layer
        self._input_layer_type = input_layer
        self._hidden_layer = self._layer_type[hidden_layer]
        self._output_layer = self._layer_type[output_layer]
        self._input_layer = self._layer_type[input_layer]

        # save the layer shape
        self._layers_shape = layers_shape
        self._layers_count = len(layers_shape)
        self._bias_unit = bias_unit

        # init layer input/output memory, deltas memory, gradient && delta accumulator
        self._layer_input = list() # common
        self._layer_output = list() # common
        self._layer_prevdelta = list() # momentum
        self._ms_grad_acc = list() # adadelta
        self._ms_delta_acc = list() # adadelta

        # init the neural layer, prev delta memory, adadelta accumulator
        self._weights = []
        for s_in, s_out in zip(layers_shape[:-1], layers_shape[1:]):
            # random init with mean 0 of a layer (with bias unit if require)
            self._weights.append(2 * np.random.random((s_in + bias_unit, s_out)) - 1)
            self._layer_prevdelta.append(np.zeros((s_in + bias_unit, s_out)))
            self._ms_grad_acc.append(np.zeros((s_in + bias_unit, s_out)))
            self._ms_delta_acc.append(np.zeros((s_in + bias_unit, s_out)))

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

    def _measure_deltas(self, y, lambda_regularization):
        """ Given the last outputs, calculate for each layer weights the distance to target """

        deltas = []
        for idx, output in enumerate(reversed(self._layer_output)):
            if idx == 0:
                l_error = (y - output) + (lambda_regularization / len(y)) * output
                glob_error = np.sum(np.nan_to_num(-y * np.log(output) - (1 - y) * np.log(1 - output))) # np.mean(np.sum(l_error ** 2))
                delta = l_error # * self._output_layer(output, True)
            else:
                l_error = np.dot(deltas[-idx], self._weights[-idx].T)
                if idx < self._layers_count - 1:
                    delta = l_error * self._hidden_layer(output, True) if not self._bias_unit else remove_bias(l_error) * self._hidden_layer(output, True)
                else:
                    delta = l_error * self._input_layer(output, True) if not self._bias_unit else remove_bias(l_error) * self._input_layer(output, True)
            deltas.insert(0, delta)
        return glob_error, deltas

    def _proceed_weights_step(self, alpha, deltas, X, momentum):
        """ given an alpha step and the deltas of each layer, move the weights """
        for idx in xrange(self._layers_count - 1):
            delta = alpha * (np.dot(self._layer_input[idx].T, deltas[idx])) + momentum * self._layer_prevdelta[idx]
            self._weights[idx] += delta
            self._layer_prevdelta[idx] = delta

    def backpropagation_training(self, X, y, alpha=0.1, epoch=100, verbose=True, momentum=0.99, l_regularization=0.1, plot_error=False):
        """ Train the neural net with the backpropagation algorithm
        return final error """

        verbose_cycle = 0.01 * epoch

        if plot_error:
            plot_name = "BPT_shape" + str(self._layers_shape) + "_types|" + self._input_layer_type + "|" + self._hidden_layer_type + "|" + self._output_layer_type + "_epoch" + str(epoch) + "_alpha" + str(alpha) + "_mom" + str(momentum) + "_lregu" + str(l_regularization)
            error_history = []
            epoch_history = []
            fig = plt.figure()
            fig.canvas.set_window_title(plot_name)
            update_error_plot(fig, epoch_history, error_history)
            fig.show()

        for epk in xrange(0, epoch):
            self.run(X)
            # for each layer output calculate distance to target
            error, deltas = self._measure_deltas(y, l_regularization)

            if verbose and (epk % verbose_cycle) == 0:
                print >> sys.stderr, "Error:", str(error)
            
            if plot_error and (epk % verbose_cycle) == 0:
                error_history += [error]
                epoch_history += [epk]
                update_error_plot(fig, epoch_history, error_history)

            # move the weights toward target with alpha step
            self._proceed_weights_step(alpha, deltas, X, momentum)

        if verbose:
            print >> sys.stderr, "Final error:", str(error)

        # return last error
        return error

    def SGD_training(self, X, y, alpha=0.1, epoch=100, verbose=True, momentum=0.99, l_regularization=0.1, mini_batch_size=10, plot_error=False):
        """ Train the neural net with the backpropagation algorithm
        return final error """

        verbose_cycle = 0.01 * epoch

        if plot_error:
            plot_name = "SGD_shape" + str(self._layers_shape) + "_types|" + self._input_layer_type + "|" + self._hidden_layer_type + "|" + self._output_layer_type + "_epoch" + str(epoch) + "_alpha" + str(alpha) + "_mom" + str(momentum) + "_lregu" + str(l_regularization) + "_batchsize" + str(mini_batch_size)
            error_history = []
            epoch_history = []
            fig = plt.figure()
            fig.canvas.set_window_title(plot_name)
            update_error_plot(fig, epoch_history, error_history)
            fig.show()
        n = len(y)
        for epk in xrange(0, epoch):
            # randomize data
            random_data = zip(X, y)
            random.shuffle(random_data)
            random_data = zip(*random_data)
            x_minibatches = list(random_data[0])
            y_minibatches = list(random_data[1])
            # create minibatches
            x_minibatches = [np.array(x_minibatches[k : k + mini_batch_size]) for k in xrange(0, n, mini_batch_size)]
            y_minibatches = [np.array(y_minibatches[k : k + mini_batch_size]) for k in xrange(0, n, mini_batch_size)]
            for idx, x_minibatch in enumerate(x_minibatches):
                y_minibatch = y_minibatches[idx]
                self.run(x_minibatch)
                # for each layer output calculate distance to target
                error, deltas = self._measure_deltas(y_minibatch, l_regularization)
                # move the weights toward target with alpha step
                self._proceed_weights_step(alpha, deltas, x_minibatch, momentum)

            if verbose or plot_error:
                self.run(X)
                error = np.sum(np.nan_to_num(-y * np.log(self._layer_output[-1]) - (1 - y) * np.log(1 - self._layer_output[-1])))
            if verbose and (epk % verbose_cycle) == 0:
                print >> sys.stderr, "Error:", str(error)
            if plot_error and (epk % verbose_cycle) == 0:
                error_history += [error]
                epoch_history += [epk]
                update_error_plot(fig, epoch_history, error_history)

        if verbose:
            print >> sys.stderr, "Final error:", str(error)

        # return last error
        return error

    def _proceed_weights_step_adadelta(self, deltas, X, p=0.95, e=1e-6):
        """ given rho, epsilon parameter and previous gradient of each layer, move the weights """
        for idx in xrange(self._layers_count - 1):
            gradient = np.dot(self._layer_input[idx].T, deltas[idx]) # compute gradient
            self._ms_grad_acc[idx] = p * self._ms_grad_acc[idx] + (1 - p) * (gradient ** 2) # accumulate gradient
            delta = - (np.sqrt(self._ms_delta_acc[idx] + e) / np.sqrt(self._ms_grad_acc[idx] + e)) * gradient # compute update
            self._ms_delta_acc[idx] = p * self._ms_delta_acc[idx] + (1 - p) * (delta ** 2) # accumulate update
            self._weights[idx] -= delta

    def adadelta_training(self, X, y, epoch=100, verbose=True, l_regularization=0.1, plot_error=False):
        """ Train the neural net with the adadelta algorithm
        return final error """

        verbose_cycle = 0.01 * epoch

        if plot_error:
            plot_name = "ADAD_shape" + str(self._layers_shape) + "_types|" + self._input_layer_type + "|" + self._hidden_layer_type + "|" + self._output_layer_type + "_epoch" + str(epoch) + "_lregu" + str(l_regularization)
            error_history = []
            epoch_history = []
            fig = plt.figure()
            fig.canvas.set_window_title(plot_name)
            ax = fig.add_subplot(111)
            update_error_plot(fig, epoch_history, error_history)
            fig.show()

        for epk in xrange(0, epoch):
            self.run(X)
            # for each layer output calculate distance to target
            error, deltas = self._measure_deltas(y, l_regularization)

            if verbose and (epk % verbose_cycle) == 0:
                print >> sys.stderr, "Error:", str(error)
            if plot_error and (epk % verbose_cycle) == 0:
                error_history += [error]
                epoch_history += [epk]
                update_error_plot(fig, epoch_history, error_history)                
            
            # move the weights toward target with alpha step
            self._proceed_weights_step_adadelta(deltas, X) # p stand for rho and e for epsilon

        if verbose:
            print >> sys.stderr, "Final error:", str(error)

        # return last error
        return (np.mean(np.abs(y - self._layer_output[-1])))



if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0, 1],
                  [0, 1],
                  [1, 0],
                  [1, 0]])
    ffnn = FeedForwardNN([3, 2, 2], hidden_layer="sigmoid", input_layer="sigmoid", output_layer="softmax")
    print "ffnn._layers_shape"
    print ffnn._layers_shape
    print "ffnn._layers_count"
    print ffnn._layers_count
    # print "ffnn.backpropagation_training(X, y)"
    # print ffnn.backpropagation_training(X, y, alpha=0.07, epoch=500, plot_error=True)
    # print "ffnn.run(X)"
    # print ffnn.run(X)

    # ffnn = FeedForwardNN([3, 2, 2], hidden_layer="sigmoid", input_layer="sigmoid", output_layer="softmax")
    # print "ffnn.adadelta_training(X, y)"
    # print ffnn.adadelta_training(X, y, epoch=500, plot_error=True)
    # print "ffnn.run(X)"
    # print ffnn.run(X)

    ffnn = FeedForwardNN([3, 2, 2], hidden_layer="sigmoid", input_layer="sigmoid", output_layer="softmax")
    print "ffnn.SGD_training(X, y, alpha=0.07, epoch=500, mini_batch_size=10, plot_error=True)"
    print ffnn.SGD_training(X, y, alpha=0.07, epoch=500, mini_batch_size=10, momentum=0.3, plot_error=True)
    print "ffnn.run(X)"
    print ffnn.run(X)
    raw_input('waiting to close graph')
