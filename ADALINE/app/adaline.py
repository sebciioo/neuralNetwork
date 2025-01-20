import numpy as np


# Funkcja aktywacji liniowa
def linear_activation(x):
    return x


# Sigmoidalna funkcja aktywacji
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))


class ADALINE:
    def __init__(self, input_size, learning_rate=0.01, activation='sigmoid'):
        self.weights = np.random.randn(input_size + 1) * 0.01  # Wagi + bias
        self.learning_rate = learning_rate
        self.activation = linear_activation if activation == 'linear' else sigmoid_activation
        self.errors = []

    def predict(self, inputs):
        activation_input = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(activation_input)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            total_error = 0
            correct_predictions = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                error = target - output
                total_error += error ** 2
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error
                if (output >= 0 and target == 1) or (output < 0 and target == 0):
                    correct_predictions += 1
            avg_error = total_error / len(X)
            self.errors.append(avg_error)
            print(f"Epoch: {epoch + 1}/{epochs}, Average Error: {avg_error:.4f}")
