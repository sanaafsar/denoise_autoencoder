"""
Load and use the trained Denoising Autoencoder model
=====================================================
This script demonstrates how to load the saved model and perform inference
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from noise_functions import add_noise
from model import load_model


# Load test data
(_, _), (testX, _) = mnist.load_data()
test_clean = np.reshape(testX / 255, (10000, 28, 28, 1))
test_noisy = np.reshape([add_noise(img / 255) for img in testX], (10000, 28, 28, 1))

offset = 92
print("Noisy test images")
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(test_noisy[i + offset, :, :, -1], cmap='gray')
#plt.show()
plt.savefig('noisy_samples_1.png')
plt.clf()


# ─────────────────────────────────────────────
# OPTION 1: Load entire model (Recommended)
# ─────────────────────────────────────────────
print("Loading model...")
autoencoder = load_model('autoencoder_model.h5')
print("Model loaded successfully!")

# Test on some examples
print("\nDenoising samples...")
offset = 92
for i in range(9):
    plt.subplot(330 + 1 + i)
    output = autoencoder.predict(np.array([test_noisy[i + offset]]))
    denoised = np.reshape(output[0] * 255, (28, 28))
    plt.imshow(denoised, cmap='gray')
plt.savefig('denoised_results_1.png')
plt.clf()
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
