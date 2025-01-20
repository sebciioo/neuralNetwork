from .perceptron import Perceptron
import numpy as np
import random


class PerceptronDigitRecognizer:
    def __init__(self, input_size=35, learning_rate=0.1, epochs=100, num_perceptrons=10):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_perceptrons = num_perceptrons
        self.perceptrons = [Perceptron(input_size, learning_rate) for _ in range(self.num_perceptrons)]
        self.accuracies = []

    def get_accuracies(self):
        return self.accuracies

    def train(self, training_data):
        correct_predictions = [0] * 10
        total_predictions = [0] * 10
        for digit, perceptron in enumerate(self.perceptrons):
            for epoch in range(self.epochs):
                random.shuffle(training_data)
                for image, label in training_data:
                    image = image.flatten()
                    target = 1 if digit == label else -1
                    output = perceptron.predict(image)
                    total_predictions[digit] += 1
                    if output == target:
                        correct_predictions[digit] += 1
                    perceptron.update_weights(image, target)
        self.accuracies = [round(100 * (correct / total)) if total > 0 else 0
                           for correct, total in zip(correct_predictions, total_predictions)]

    def predict(self, image):
        image = image.flatten()
        outputs = [perceptron.predict(image) for perceptron in self.perceptrons]
        return outputs
