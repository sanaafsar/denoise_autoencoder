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
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.datasets import mnist
from noise_functions import add_noise
from model import build_autoencoder, save_model


def prepare_data(trainX, testX):
    """
    Prepare and preprocess training and test data
    
    Args:
        trainX: Raw training images
        testX: Raw test images
        
    Returns:
        tuple: (train_clean, test_clean, train_noisy, test_noisy)
    """
    train_clean = [image / 255 for image in trainX]
    test_clean  = [image / 255 for image in testX]
    train_noisy = [add_noise(image / 255) for image in trainX]
    test_noisy  = [add_noise(image / 255) for image in testX]

    train_clean = np.reshape(train_clean, (60000, 28, 28, 1))
    test_clean  = np.reshape(test_clean,  (10000, 28, 28, 1))
    train_noisy = np.reshape(train_noisy, (60000, 28, 28, 1))
    test_noisy  = np.reshape(test_noisy,  (10000, 28, 28, 1))
    
    return train_clean, test_clean, train_noisy, test_noisy


def visualize_samples(images, title, filename):
    """Visualize and save image samples"""
    print(f"{title}")
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(images[i, :, :, -1], cmap='gray')
    plt.savefig(filename)
    plt.clf()


def main():
    """Main training pipeline"""
    
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
    plt.savefig('sample_images.png')
    plt.clf()

    # ─────────────────────────────────────────────
    # 2. VISUALIZE CORRUPTED SAMPLES
    # ─────────────────────────────────────────────
    print("Corrupted Example Samples")
    for j in range(9):
        i = np.random.randint(0, 10000)
        plt.subplot(330 + 1 + j)
        noisy = add_noise(trainX[i] / 255)
        plt.imshow(noisy, cmap='gray')
    plt.savefig('corrupted_samples.png')
    plt.clf()

    # ─────────────────────────────────────────────
    # 3. DATA PREPARATION
    # ─────────────────────────────────────────────
    train_clean, test_clean, train_noisy, test_noisy = prepare_data(trainX, testX)
    print(train_clean.shape, train_noisy.shape, test_clean.shape, test_noisy.shape)

    # ─────────────────────────────────────────────
    # 4. BUILD & COMPILE MODEL
    # ─────────────────────────────────────────────
    autoencoder = build_autoencoder()
    autoencoder.summary()

    # ─────────────────────────────────────────────
    # 5. TRAIN MODEL
    # ─────────────────────────────────────────────
    print("\nStarting training...")
    autoencoder.fit(
        train_noisy, train_clean,
        epochs=30,
        batch_size=64,
        validation_data=(test_noisy, test_clean)
    )

    # ─────────────────────────────────────────────
    # 6. SAVE MODEL & WEIGHTS
    # ─────────────────────────────────────────────
    save_model(autoencoder)

    # ─────────────────────────────────────────────
    # 7. VISUALIZE RESULTS
    # ─────────────────────────────────────────────
    offset = 92

    visualize_samples(test_noisy[offset:offset+9], "Noisy test images", 'noisy_samples.png')
    
    print("Cleaned Version (Denoising Autoencoder) :)")
    for i in range(9):
        plt.subplot(330 + 1 + i)
        output = autoencoder.predict(np.array([test_noisy[i + offset]]))
        op_image = np.reshape(output[0] * 255, (28, 28))
        plt.imshow(op_image, cmap='gray')
    plt.savefig('denoised_samples.png')
    plt.clf()


if __name__ == '__main__':
    main()
