import os
from autoencoder.preprocessing import preprocess_image, preprocess_coordinates
from autoencoder.training import train_autoencoder
from autoencoder.generation import generate_image
from PIL import Image
import matplotlib.pyplot as plt

# Path to the input image
image_path = "data/image.png"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Preprocess the data
print("Preparing data...")
num_harmonics = 3
data, width, height = preprocess_image(image_path, num_harmonics)

# Neural network settings
input_size = len(preprocess_coordinates(0, 0, width, height, num_harmonics))
layer_sizes = [input_size, 256, 128, 3]

# Train the autoencoder
print("Training the autoencoder...")
nn = train_autoencoder(data, layer_sizes, min_epochs=5000, learning_rate=0.01)

# Generate the output image
print("Generating the image...")
generated_pixels = generate_image(nn, width, height, num_harmonics)
output_image = Image.fromarray(generated_pixels)
output_image.save(f"{output_folder}/output_image.png")
print(f"Generated image saved to {output_folder}/output_image.png")

# Display the generated image
plt.imshow(generated_pixels)
plt.axis("off")
plt.show()
