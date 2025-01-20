import numpy as np


class Perceptron:
    def __init__(self, input_size, learning_rate):
        self.weights = 0.1 * (2 * np.random.rand(input_size, 1) - 1)
        self.threshold = 0
        self.learning_rate = learning_rate

    def predict(self, inputs):
        self.weights = self.weights.flatten()
        total_input = np.dot(inputs, self.weights)
        return 1 if total_input >= self.threshold else -1

    def update_weights(self, inputs, target):
        output = self.predict(inputs)
        error = target - output
        if error != 0:
            self.weights = self.weights.flatten()
            self.weights += self.learning_rate * error * inputs
            self.threshold += self.learning_rate * error
