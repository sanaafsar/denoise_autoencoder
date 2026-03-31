"""
Load and use the trained Denoising Autoencoder model
=====================================================
This script demonstrates how to load the saved model and perform inference
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import mnist
from auto_encoder import add_noise

# def add_noise(image):
#     """Add random noise to an image"""
#     p = np.random.random()
#     if p <= 0.25:
#         # Gaussian noise
#         r, c = image.shape
#         mean = 0
#         var = 0.1
#         sigma = var ** 0.5
#         gaussian = np.random.normal(mean, sigma, (r, c))
#         noisy = image + gaussian
#     elif p <= 0.5:
#         # Salt and pepper
#         ratio = 0.9
#         amount = 0.1
#         noisy = np.copy(image)
#         salt_count = np.ceil(amount * image.size * ratio)
#         coords = [np.random.randint(0, i - 1, int(salt_count)) for i in image.shape]
#         noisy[coords] = 1
#         pepper_count = np.ceil(amount * image.size * (1. - ratio))
#         coords = [np.random.randint(0, i - 1, int(pepper_count)) for i in image.shape]
#         noisy[coords] = 0
#     elif p <= 0.75:
#         # Poisson
#         vals = len(np.unique(image))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy = np.random.poisson(image * vals) / float(vals)
#     else:
#         # Speckle
#         r, c = image.shape
#         speckle = np.random.randn(r, c)
#         noisy = image + image * speckle
#     return noisy


# Load test data
(_, _), (testX, _) = mnist.load_data()
test_clean = np.reshape(testX / 255, (10000, 28, 28, 1))
test_noisy = np.reshape([add_noise(img / 255) for img in testX], (10000, 28, 28, 1))

# ─────────────────────────────────────────────
# OPTION 1: Load entire model (Recommended)
# ─────────────────────────────────────────────
print("Loading model...")
autoencoder = tensorflow.keras.models.load_model('autoencoder_model.h5')
print("Model loaded successfully!")

# Test on some examples
print("\nDenoising samples...")
offset = 50
for i in range(9):
    plt.subplot(330 + 1 + i)
    output = autoencoder.predict(np.array([test_noisy[i + offset]]))
    denoised = np.reshape(output[0] * 255, (28, 28))
    plt.imshow(denoised, cmap='gray')
plt.savefig('denoised_results.png')
print("Results saved to 'denoised_results.png'")

# ─────────────────────────────────────────────
# OPTION 2: Load only weights (if rebuilding architecture)
# ─────────────────────────────────────────────
"""
# Rebuild the model architecture
input_data = tensorflow.keras.layers.Input(shape=(28, 28, 1))
encoder = tensorflow.keras.layers.Conv2D(64, (5, 5), activation='relu')(input_data)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)
encoder = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu')(encoder)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)
encoder = tensorflow.keras.layers.Conv2D(256, (3, 3), activation='relu')(encoder)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)

decoder = tensorflow.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu')(encoder)
decoder = tensorflow.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2, 2))(decoder)
decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2, 2))(decoder)
decoded = tensorflow.keras.layers.Conv2DTranspose(1, (5, 5), activation='relu')(decoder)

autoencoder = tensorflow.keras.models.Model(inputs=input_data, outputs=decoded)
autoencoder.load_weights('autoencoder_weights.h5')
print("Weights loaded successfully!")
"""
