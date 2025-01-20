import numpy as np
from .neuralNetwork import NeuralNetwork


def train_autoencoder(data, layer_sizes, min_epochs=5000, learning_rate=0.05, error_tolerance=0.01):
    """
    Train the autoencoder until the error falls below a specified tolerance.
    """
    nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=learning_rate)
    epoch = 0
    prev_error = float("inf")
    while True:
        total_loss = 0
        np.random.shuffle(data)
        for inputs, target in data:
            inputs = np.expand_dims(inputs, axis=0)
            target = np.expand_dims(target, axis=0)
            outputs = nn.forward(inputs)
            nn.backward(target)
            total_loss += np.mean((target - outputs) ** 2)
        if epoch % 100 == 0:
            print(f"Epoka {epoch}, Utrata: {total_loss:.6f}")
            if epoch >= min_epochs and abs(prev_error - total_loss) < error_tolerance:
                print(f"Training stopped at epoch {epoch}, Total loss: {total_loss:.6f}")
                break
            prev_error = total_loss
        epoch += 1
    return nn
