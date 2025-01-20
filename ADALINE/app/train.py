import os
import matplotlib.pyplot as plt
from .adaline import ADALINE


def train_adalines(X_train, y_train, input_size, epochs, learning_rate, plots_folder='plots'):
    """
    Trenuje 10 jednostek ADALINE, po jednej dla każdej cyfry.
    :param X_train: Dane wejściowe
    :param y_train: Etykiety
    :param input_size: Liczba wejść dla każdej jednostki ADALINE
    :param epochs: Liczba epok treningu
    :param learning_rate: Szybkość uczenia
    :param plots_folder: Folder do zapisywania wykresów
    :return: Lista wytrenowanych modeli ADALINE
    """
    adalines = []

    # Trenowanie perceptronów
    for digit in range(10):
        print(f"Training ADALINE unit for digit: {digit}")
        y_binary = (y_train == digit).astype(int)  # Binarne etykiety: 1 dla cyfry, 0 dla innych
        adaline = ADALINE(input_size, learning_rate=learning_rate)
        adaline.train(X_train, y_binary, epochs=epochs)
        adalines.append(adaline)

        # Generowanie wykresu błędu
        save_error_plot(adaline, digit, plots_folder)

    return adalines


def save_error_plot(adaline, digit, plots_folder):
    plt.figure(figsize=(8, 6))
    plt.plot(adaline.errors, label=f'Digit {digit}', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(f'Error function during training for digit {digit}')
    plt.legend()
    plt.grid(True)
    os.makedirs(plots_folder, exist_ok=True)  # Tworzenie folderu na wykresy
    plt.savefig(f'{plots_folder}/{digit}.png')
    plt.close()
