import numpy as np
from autoencoder.preprocessing import preprocess_coordinates


def generate_image(nn, width, height, num_harmonics=3):
    """
    Generate an image using the trained autoencoder.
    """
    generated_pixels = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            inputs = preprocess_coordinates(x, y, width, height, num_harmonics)
            inputs = np.expand_dims(inputs, axis=0)
            outputs = nn.forward(inputs)
            generated_pixels[y, x] = outputs
    generated_pixels = np.clip(generated_pixels * 255, 0, 255).astype(np.uint8)
    return generated_pixels
