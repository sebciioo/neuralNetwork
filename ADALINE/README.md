
# ADALINE Digit Classifier

## Overview

This project implements the ADALINE (Adaptive Linear Neuron) algorithm to classify handwritten digits from the MNIST dataset. Each ADALINE unit is trained to recognize a specific digit (0–9).

### Features
- Train individual ADALINE units to recognize digits.
- Generate error plots for each digit during training.
- Evaluate accuracy on test data.

## Requirements

- Python 3.7 or later
- Libraries:
  - `numpy`
  - `matplotlib`
  - `pandas`

Install missing libraries:

```bash
pip install numpy matplotlib pandas
```
## Running the Project
1. Place mnist_train.csv and mnist_test.csv in the data/ folder. You can get mnist data from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
2. Run the project:
```bash
python main.py
```

## Outputs
Error plots are saved in the plots/ folder.
The console displays the training progress and final accuracy on the test set.