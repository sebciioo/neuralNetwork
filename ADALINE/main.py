from app.utils import load_data
from app.train import train_adalines
from app.test import test_adalines

# Globalne ustawienia
EPOCHS = 20
LEARNING_RATE = 0.0001


def main():
    # Wczytanie danych
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    # Trenowanie modeli
    print("Training ADALINE models...")
    input_size = X_train.shape[1]
    adalines = train_adalines(X_train, y_train, input_size, EPOCHS, LEARNING_RATE)

    # Testowanie modeli
    print("Testing ADALINE models...")
    test_adalines(adalines, X_test, y_test)


if __name__ == "__main__":
    main()
