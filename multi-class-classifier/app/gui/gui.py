import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from app.loader.data_loader import ImageDataLoader
from app.perceptron.recognizer import PerceptronDigitRecognizer


class PerceptronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron Test")
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.epochs = tk.IntVar(value=100)
        self.num_perceptrons = tk.IntVar(value=10)

        self.matrix = np.zeros((7, 5), dtype=int)
        self.buttons = None
        self.message_box = None
        self.ax = None
        self.canvas = None
        self.recognizer = None

        self.create_widgets()

    def initialize_recognizer(self):
        self.recognizer = PerceptronDigitRecognizer(
            learning_rate=self.learning_rate.get(),
            epochs=self.epochs.get(),
            num_perceptrons=self.num_perceptrons.get()
        )

    def update_parameters(self, event=None):
        if not self.recognizer or self.recognizer.num_perceptrons != self.num_perceptrons.get():
            self.initialize_recognizer()
        else:
            self.recognizer.learning_rate = self.learning_rate.get()
            self.recognizer.epochs = self.epochs.get()

    def create_widgets(self):
        parameter_frame = tk.Frame(self.root)
        parameter_frame.grid(row=0, column=0, padx=5, pady=5, sticky="n")

        tk.Label(parameter_frame, text="Number of Perceptrons:").grid(row=0, column=0, padx=5, pady=5)
        tk.Scale(parameter_frame, from_=1, to=10, orient="horizontal", variable=self.num_perceptrons).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(parameter_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5)
        tk.Scale(parameter_frame, from_=0.001, to=1.0, resolution=0.001, orient="horizontal", variable=self.learning_rate).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(parameter_frame, text="Epochs:").grid(row=2, column=0, padx=5, pady=5)
        tk.Scale(parameter_frame, from_=10, to=100, orient="horizontal", variable=self.epochs).grid(row=2, column=1, padx=5, pady=5)

        tk.Button(parameter_frame, text="Train Perceptrons", command=self.train_perceptrons).grid(row=3, column=0, columnspan=2, pady=10)
        tk.Button(parameter_frame, text="Test Perceptrons", command=self.test_perceptrons).grid(row=4, column=0, columnspan=2, pady=10)

        self.message_box = tk.Text(parameter_frame, height=10, width=30)
        self.message_box.grid(row=5, column=0, columnspan=2, pady=10, sticky="e")

        grid_frame = tk.Frame(self.root)
        grid_frame.grid(row=0, column=1, padx=20, pady=10)

        button_size = 3
        self.buttons = []
        for i in range(7):
            row_buttons = []
            for j in range(5):
                button = tk.Button(grid_frame, width=button_size, height=button_size, bg="white",
                                   command=lambda x=i, y=j: self.toggle_button(x, y))
                button.grid(row=i, column=j, padx=0, pady=0)
                row_buttons.append(button)
            self.buttons.append(row_buttons)

        fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2)

    def toggle_button(self, x, y):
        self.matrix[x, y] = 1 - self.matrix[x, y]
        color = "black" if self.matrix[x, y] == 1 else "white"
        self.buttons[x][y].configure(bg=color)

    def train_perceptrons(self):
        self.message_box.insert(tk.END, f"Training with LR={self.learning_rate.get()} and Epochs={self.epochs.get()}\n")
        training_data = ImageDataLoader(num_folders=self.num_perceptrons.get()).get_data()
        self.update_parameters()
        self.recognizer.train(training_data)
        self.update_accuracy_plot(self.recognizer.get_accuracies())

    def test_perceptrons(self):
        self.message_box.insert(tk.END, "Testing Perceptrons...\n")
        prediction = self.recognizer.predict(self.matrix.flatten())
        for i, pred in enumerate(prediction):
            self.message_box.insert(tk.END, f"Perceptron {i} Prediction: {pred}\n")

    def update_accuracy_plot(self, accuracies):
        self.ax.clear()
        self.ax.bar(range(1, len(accuracies) + 1), accuracies, color="green")
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Accuracy (%)")
        for idx, accuracy in enumerate(accuracies):
            self.ax.text(idx + 1, accuracy + 1, f"{accuracy}%", ha="center")
        self.canvas.draw()
