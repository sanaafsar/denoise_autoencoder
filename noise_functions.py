"""
Noise Functions for Image Denoising
====================================
Collection of functions to add different types of noise to images
"""

import numpy as np


def gaussian_noise(image):
    """Add Gaussian noise to an image"""
    r, c = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (r, c))
    gaussian = gaussian.reshape(r, c)
    noisy = image + gaussian
    return noisy


def salt_and_pepper_noise(image):
    """Add salt and pepper noise to an image"""
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
    """Add Poisson noise to an image"""
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy


def speckle_noise(image):
    """Add speckle noise to an image"""
    r, c = image.shape
    speckle = np.random.randn(r, c)
    speckle = speckle.reshape(r, c)
    noisy = image + image * speckle
    return noisy


def add_noise(image):
    """Add random noise to an image (randomly selects between 4 noise types)"""
    p = np.random.random()
    if p <= 0.25:
        noisy = gaussian_noise(image)
    elif p <= 0.5:
        noisy = salt_and_pepper_noise(image)
    elif p <= 0.75:
        noisy = poisson_noise(image)
    else:
        noisy = speckle_noise(image)
    return noisy
