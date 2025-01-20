import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        """
        Initialize the neural network.
        :param layer_sizes: List defining the number of neurons in each layer [input_size, hidden1, ..., output_size].
        :param learning_rate: Learning rate for the network.
        """
        self.layers = len(layer_sizes) - 1
        self.weights = [
            np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i + 1]))
            for i in range(self.layers)
        ]
        self.biases = [np.zeros(layer_sizes[i + 1]) for i in range(self.layers)]
        self.learning_rate = learning_rate

    @staticmethod
    def activation_function(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_derivative(x):
        """Derivative of the sigmoid activation function."""
        return x * (1 - x)

    def forward(self, inputs):
        """
        Perform a forward pass through the network.
        :param inputs: Input data.
        :return: Network output.
        """
        self.activations = [inputs]
        for i in range(self.layers):
            inputs = self.activation_function(
                np.dot(inputs, self.weights[i]) + self.biases[i]
            )
            self.activations.append(inputs)
        return inputs

    def backward(self, target_output):
        """
        Perform a backward pass to update weights and biases.
        :param target_output: Target output for the training data.
        """
        deltas = [None] * self.layers
        error = target_output - self.activations[-1]
        deltas[-1] = error * self.activation_derivative(self.activations[-1])

        for i in reversed(range(self.layers - 1)):
            error = np.dot(deltas[i + 1], self.weights[i + 1].T)
            deltas[i] = error * self.activation_derivative(self.activations[i + 1])

        for i in range(self.layers):
            self.weights[i] += np.dot(self.activations[i].T, deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0) * self.learning_rate
