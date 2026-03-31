"""
Convolutional Denoising Autoencoder for Image Denoising
========================================================
Source: "Denoising AutoEncoders can reduce noise in images"
Author: Kartik Chaudhary — Game of Bits (Medium)
Link: https://medium.com/game-of-bits/denoising-autoencoders-can-reduce-noise-in-images-5b74753eaf97
GitHub: https://github.com/kartikgill/Autoencoders

Requirements:
    pip install tensorflow matplotlib numpy pandas seaborn
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow
from tensorflow.keras.datasets import mnist


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
(trainX, trainy), (testX, testy) = mnist.load_data()
print('Training data shapes: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Testing data shapes: X=%s, y=%s' % (testX.shape, testy.shape))

# Visualize 5 sample training images
for j in range(5):
    i = np.random.randint(0, 10000)
    plt.subplot(550 + 1 + j)
    plt.imshow(trainX[i], cmap='gray')
    plt.title(trainy[i])
#plt.show()
plt.savefig('sample_images.png')


# ─────────────────────────────────────────────
# 2. NOISE FUNCTIONS
# ─────────────────────────────────────────────
def guassian_noise(image):
    r, c = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (r, c))
    gaussian = gaussian.reshape(r, c)
    noisy = image + gaussian
    return noisy


def salt_and_pepper_noise(image):
    ratio = 0.9
    amount = 0.1
    noisy = np.copy(image)

    salt_count = np.ceil(amount * image.size * ratio)
    coords = [np.random.randint(0, i - 1, int(salt_count)) for i in image.shape]
    noisy[coords] = 1

    pepper_count = np.ceil(amount * image.size * (1. - ratio))
    coords = [np.random.randint(0, i - 1, int(pepper_count)) for i in image.shape]
    noisy[coords] = 0
    return noisy


def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy


def speckle_noise(image):
    r, c = image.shape
    speckle = np.random.randn(r, c)
    speckle = speckle.reshape(r, c)
    noisy = image + image * speckle
    return noisy


def add_noise(image):
    p = np.random.random()
    if p <= 0.25:
        noisy = guassian_noise(image)
    elif p <= 0.5:
        noisy = salt_and_pepper_noise(image)
    elif p <= 0.75:
        noisy = poisson_noise(image)
    else:
        noisy = speckle_noise(image)
    return noisy


# ─────────────────────────────────────────────
# 3. VISUALIZE CORRUPTED SAMPLES
# ─────────────────────────────────────────────
print("Corrupted Example Samples")
for j in range(9):
    i = np.random.randint(0, 10000)
    plt.subplot(330 + 1 + j)
    noisy = add_noise(trainX[i] / 255)
    plt.imshow(noisy, cmap='gray')
#plt.show()
plt.savefig('corrupted_samples.png')


# ─────────────────────────────────────────────
# 4. DATA PREPARATION
# ─────────────────────────────────────────────
train_clean = [image / 255 for image in trainX]
test_clean  = [image / 255 for image in testX]
train_noisy = [add_noise(image / 255) for image in trainX]
test_noisy  = [add_noise(image / 255) for image in testX]

train_clean = np.reshape(train_clean, (60000, 28, 28, 1))
test_clean  = np.reshape(test_clean,  (10000, 28, 28, 1))
train_noisy = np.reshape(train_noisy, (60000, 28, 28, 1))
test_noisy  = np.reshape(test_noisy,  (10000, 28, 28, 1))

print(train_clean.shape, train_noisy.shape, test_clean.shape, test_noisy.shape)


# ─────────────────────────────────────────────
# 5. BUILD MODEL
# ─────────────────────────────────────────────
input_data = tensorflow.keras.layers.Input(shape=(28, 28, 1))

# Encoder part
encoder = tensorflow.keras.layers.Conv2D(64, (5, 5), activation='relu')(input_data)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)
encoder = tensorflow.keras.layers.Conv2D(128, (3, 3), activation='relu')(encoder)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)
encoder = tensorflow.keras.layers.Conv2D(256, (3, 3), activation='relu')(encoder)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)

# Decoder part
decoder = tensorflow.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu')(encoder)
decoder = tensorflow.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2, 2))(decoder)
decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2, 2))(decoder)
decoded = tensorflow.keras.layers.Conv2DTranspose(1, (5, 5), activation='relu')(decoder)


# ─────────────────────────────────────────────
# 6. COMPILE & TRAIN
# ─────────────────────────────────────────────
autoencoder = tensorflow.keras.models.Model(inputs=input_data, outputs=decoded)
autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.summary()

autoencoder.fit(
    train_noisy, train_clean,
    epochs=30,
    batch_size=64,
    validation_data=(test_noisy, test_clean)
)

# Save the trained model and weights
print("Saving model...")
autoencoder.save('autoencoder_model.h5')  # Saves entire model (architecture + weights)
autoencoder.save_weights('autoencoder_weights.h5')  # Saves only weights
print("Model saved as 'autoencoder_model.h5' and weights saved as 'autoencoder_weights.h5'")


# ─────────────────────────────────────────────
# 7. RESULTS
# ─────────────────────────────────────────────
offset = 92

print("Noisy test images")
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(test_noisy[i + offset, :, :, -1], cmap='gray')
#plt.show()
plt.savefig('noisy_samples.png')

print("Cleaned Version (Denoising Autoencoder) :)")
for i in range(9):
    plt.subplot(330 + 1 + i)
    output = autoencoder.predict(np.array([test_noisy[i + offset]]))
    op_image = np.reshape(output[0] * 255, (28, 28))
    plt.imshow(op_image, cmap='gray')
plt.savefig('denoised_samples.png')
#plt.show()


# ─────────────────────────────────────────────
# 8. LOAD & USE SAVED MODEL (FOR FUTURE USE)
# ─────────────────────────────────────────────
"""
To load the model in a separate script or session:

# Option 1: Load entire model (recommended)
loaded_model = tensorflow.keras.models.load_model('autoencoder_model.h5')
noisy_image = np.array([test_noisy[0]])  # Your noisy image
denoised = loaded_model.predict(noisy_image)

# Option 2: Load only weights (if you have model architecture defined)
autoencoder_new = tensorflow.keras.models.Model(...)  # Define architecture
autoencoder_new.load_weights('autoencoder_weights.h5')
denoised = autoencoder_new.predict(noisy_image)
"""
