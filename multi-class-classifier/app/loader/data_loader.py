import cv2
import numpy as np


class ImageDataLoader:
    def __init__(self, num_folders=4, num_images_per_folder=5, threshold_value=127):
        self.num_folders = num_folders
        self.num_images_per_folder = num_images_per_folder
        self.threshold_value = threshold_value
        self.training_data = []
        self.load_data()

    def load_data(self):
        for folder in range(self.num_folders):
            for i in range(self.num_images_per_folder):
                image_path = f'data/{folder}/{i}.png'
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image {image_path}")
                    continue

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, self.threshold_value, 1, cv2.THRESH_BINARY)
                binary_image = np.abs((binary_image - 1) // 255)
                self.training_data.append((binary_image, folder))

    def preprocess_image(self, image):
        resized_image = cv2.resize(image, (5, 7), interpolation=cv2.INTER_AREA)
        _, binary_image = cv2.threshold(resized_image, self.threshold_value, 1, cv2.THRESH_BINARY)
        return binary_image

    def get_data(self):
        return self.training_data
