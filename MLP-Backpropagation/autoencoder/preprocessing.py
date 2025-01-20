import numpy as np
from PIL import Image


def preprocess_coordinates(x, y, width, height, num_harmonics=3):
    """
    Generate trigonometric features for better algorithm performance.
    """
    x_norm = x / width
    y_norm = y / height
    features = [x_norm, y_norm]
    for n in range(1, num_harmonics + 1):
        features.append(np.sin(n * x_norm))
        features.append(np.cos(n * x_norm))
        features.append(np.sin(n * y_norm))
        features.append(np.cos(n * y_norm))
    return np.array(features)


def preprocess_image(image_path, num_harmonics=3):
    """
    Preprocess the input image and extract features for each pixel.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = np.array(img) / 255.0

    data = []
    for y in range(height):
        for x in range(width):
            inputs = preprocess_coordinates(x, y, width, height, num_harmonics)
            target = pixels[y, x]
            data.append((inputs, target))
    return data, width, height
