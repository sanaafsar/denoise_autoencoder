"""
Model Architecture for Denoising Autoencoder
=============================================
Functions to build and compile the autoencoder model
"""

import tensorflow


def build_autoencoder():
    """
    Build the convolutional denoising autoencoder model
    
    Returns:
        tensorflow.keras.models.Model: Compiled autoencoder model
    """
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

    # Create and compile model
    autoencoder = tensorflow.keras.models.Model(inputs=input_data, outputs=decoded)
    autoencoder.compile(loss='mse', optimizer='adam')
    
    return autoencoder


def save_model(model, model_path='autoencoder_model.h5', weights_path='autoencoder_weights.h5'):
    """
    Save trained model and weights
    
    Args:
        model: Trained Keras model
        model_path (str): Path to save full model
        weights_path (str): Path to save weights only
    """
    print(f"Saving model to {model_path}...")
    model.save(model_path)
    print(f"Saving weights to {weights_path}...")
    model.save_weights(weights_path)
    print("Model and weights saved successfully!")


def load_model(model_path='autoencoder_model.h5'):
    """
    Load trained model from file
    
    Args:
        model_path (str): Path to model file
        
    Returns:
        tensorflow.keras.models.Model: Loaded model
    """
    return tensorflow.keras.models.load_model(model_path)


def load_weights(model, weights_path='autoencoder_weights.h5'):
    """
    Load weights into an existing model
    
    Args:
        model: Keras model to load weights into
        weights_path (str): Path to weights file
    """
    model.load_weights(weights_path)
    return model
