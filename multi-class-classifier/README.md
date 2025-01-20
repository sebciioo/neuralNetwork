# Perceptron Digit Recognizer

## Project Description

The project demonstrates the use of perceptrons for digit recognition. Each of the 10 perceptrons is trained to recognize one digit (from 0 to 9). Digits are represented as binary pixel patterns in a **5x7** matrix.

The program allows:
- Training perceptrons on data from the `data/` folder.
- Testing perceptrons on manually "clicked" patterns.
- Visualizing training results in the form of a bar chart showing the accuracy of each perceptron.
- Intuitive parameter settings, such as the number of perceptrons, learning rate, and number of epochs.

---

## Requirements

To run the project, you need:
- **Python 3.7** or newer.
- Libraries:
  - numpy
  - matplotlib
  - opencv-python
  - tkinter (included by default in Python).

Install the missing libraries using the command:

```bash
pip install numpy matplotlib opencv-python

```

---
## Project Overview
### Running the Program
Run the main.py file in the project root folder:
```bash
python main.py
```
### User Guide
- Number of Perceptrons: Adjusts the number of perceptrons in the system (from 1 to 10).
- Learning Rate: Sets the learning rate (from 0.001 to 1.0).
- Epochs: Defines the number of iterations for training (from 10 to 100).


### Training Perceptrons
Click Train Perceptrons to start training the perceptrons using data from the data/ folder.
![Training Interface](data/image1.png)

### Testing Perceptrons
Click a pattern on the 5x7 matrix on the right and press Test Perceptrons to evaluate the results.
![Training Interface](data/image2.png)
The result can be seen in the text box, -1 means that a digit in the class was not detected, 1 means that it was detected
