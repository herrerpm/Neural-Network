import math
import random
import numpy as np
import Data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1 - x)

random.seed(12)

np.vectorize(sigmoid)
np.vectorize(sigmoid_prime)


class Layer:
    """" The Layer class is responsible for holding the matrix that
    describes the behavior of each neuron. The matrix columns corresponds
    to the layers neurons, while the matrix rows correspond to the
    weights assigned to the previous layer activations.

    The class also holds a vector that corresponds to the bias that are
    assigned to each weighted sum. With these two elements and the previous
    layer output vector, or the current layer input, the layer calculates
    it's activations using the sigmoid function on the weighted sum.
    """


    def __init__(self, expected_inputs: int, neurons: int, position: int):
        self.expected_input = expected_inputs
        self.neurons = neurons
        self.position = position
        self.input = None
        self.weighted_sum = None
        self.output = None
        self.error = None
        self.bias_gradient = None
        self.weight_gradient = None
        self.weights = []
        self.biases = []
        initialization_range = math.sqrt((6/(expected_inputs + neurons)))
        for neuron in range(neurons):
            self.weights.append(np.array([random.uniform(-initialization_range, initialization_range) for _ in range(expected_inputs)]))
        self.biases = np.vstack(np.zeros(neuron+1))
        self.weights = np.array(self.weights)

    def calculate(self, layer_input):
        self.input = layer_input
        self.weighted_sum = (self.weights @ layer_input) + self.biases
        self.output = sigmoid(self.weighted_sum)
        return self.output

    def reset(self):
        self.input = None
        self.weighted_sum = None
        self.output = None
        self.error = None
        self.bias_gradient = None
        self.weight_gradient = None

    def __repr__(self):
        return f'Layer: {self.position} ({self.expected_input} -> {self.neurons})'


class Network:
    """"The Network class is responsible for the communication between
    layers, while the learning process occurs."""


    def __init__(self, layer_composition: list[int]) -> None:
        """" For initializing the network you have to pass a list
        corresponding to the size of the input, hidden layer neurons
        and output layer. For example [784,16,16,10]"""

        self.layers = []
        for index, layer in enumerate(layer_composition[1:]):
            self.layers.append(Layer(layer_composition[index], layer, index))
        self.output_layer = self.layers[-1]

    def foward_propagation(self, input_data):
        """" For the foward propagation we take the initial input
        and run it along the calculations of the first hidden layer,
        later these activations become the input of the next layer and
        so on."""

        layer_input = input_data
        for layer in self.layers:
            layer_input = layer.calculate(layer_input)
        return self.layers[-1].output


    def back_propagation(self, expected_output: int):
        expected_outputs = np.vstack([0] * self.output_layer.neurons)
        expected_outputs[expected_output] = 1
        self.output_layer.error = (2 * (self.output_layer.output - expected_outputs)) * sigmoid_prime(
            self.output_layer.output)
        self.output_layer.weight_gradient = self.output_layer.error @ self.output_layer.input.transpose()
        self.output_layer.bias_gradient = self.output_layer.error
        # Backpropagate
        for layer_index in range(2, len(self.layers)+1):
            layer = self.layers[-layer_index]
            errors = self.layers[-layer_index + 1].error
            weight = self.layers[-layer_index + 1].weights
            layer.error = np.vstack((np.hstack(errors) @ weight)) * sigmoid_prime(layer.output)
            layer.bias_gradient = layer.error
            layer.weight_gradient = layer.error @ layer.input.transpose()

    def asses(self, data: [int], output: int):
        self.foward_propagation(data)
        self.back_propagation(output)

    def reset_layers(self):
        for layer in self.layers:
            layer.reset()
            layer.inputs = None
        self.output = None
        self.cost = None

    def get_gradients(self):
        return np.array([layer.weight_gradient for layer in self.layers], dtype=object)

    def get_bias_gradients(self):
        return np.array([layer.bias_gradient for layer in self.layers], dtype=object)

    def batch_training(self, data, labels):
        gradient = np.array([np.zeros(layer.weights.shape) for layer in self.layers], dtype=object)
        bias_gradient = np.array([np.zeros(layer.biases.shape) for layer in self.layers], dtype=object)
        for index, image in enumerate(data):
            self.asses(np.vstack(image), labels[index])
            gradient += self.get_gradients()
            bias_gradient += self.get_bias_gradients()
            self.reset_layers()
        for index, layer in enumerate(self.layers):
            layer.weights += -(1/len(data) * gradient[index])
            layer.biases += -(1/ len(data) * bias_gradient[index])

    def train(self, dataset, outputs, batch_size):
        data = list(zip(dataset, outputs))
        random.shuffle(data)
        for index in range(0,len(data), batch_size):
            Neural_Network.batch_training(Data.data[index:index+batch_size],
                                          Data.outputs[index:index+batch_size])
            print(f'{index//batch_size} / {len(dataset)}')

Neural_Network = Network([784,16,16,10])
Neural_Network.train(Data.data, Data.outputs, 10)
