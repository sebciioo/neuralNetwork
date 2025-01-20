import numpy as np


def test_adalines(adalines, X_test, y_test):
    correct = 0

    for xi, target in zip(X_test, y_test):
        outputs = [adaline.predict(xi) for adaline in adalines]
        predicted_label = np.argmax(outputs)  # Wybranie cyfry z największym wynikiem
        if predicted_label == target:
            correct += 1

    accuracy = correct / len(y_test)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")
    return accuracy
