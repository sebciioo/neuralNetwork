import os
import pandas as pd


def prepare_data(df):
    X = df.iloc[:, 1:].values / 255.0  # Normalizacja do zakresu [0, 1]
    y = df.iloc[:, 0].values  # Etykiety
    return X, y


def load_data(data_folder='data'):
    train_data = pd.read_csv(os.path.join(data_folder, 'mnist_train.csv'))
    test_data = pd.read_csv(os.path.join(data_folder, 'mnist_test.csv'))
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)
    return X_train, y_train, X_test, y_test
