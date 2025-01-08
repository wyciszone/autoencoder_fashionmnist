from tensorflow.keras import layers, models
import numpy as np


def create_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation="relu")(input_layer)
    encoded = layers.Dense(64, activation="relu")(encoded)
    encoded = layers.Dense(32, activation="relu")(encoded)

    decoded = layers.Dense(64, activation="relu")(encoded)
    decoded = layers.Dense(128, activation="relu")(decoded)
    decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

    encoder = models.Model(input_layer, encoded)
    autoencoder = models.Model(input_layer, decoded)

    return autoencoder, encoder

def train_general_autoencoder(x_train, input_dim, epochs=10, batch_size=256):
    autoencoder, encoder = create_autoencoder(input_dim)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1
    )
    return autoencoder, encoder

def train_class_autoencoders(x_train, y_train, input_dim, epochs=10, batch_size=256):
    class_autoencoders = {}
    for class_label in range(10):
        class_data = x_train[y_train == class_label]
        autoencoder, _ = create_autoencoder(input_dim)
        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.fit(
            class_data,
            class_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1
        )
        class_autoencoders[class_label] = autoencoder
    return class_autoencoders

def classify(x, class_autoencoders):
    errors = []
    for class_label in range(10):
        reconstructed = class_autoencoders[class_label].predict(x)
        error = np.mean(np.square(x - reconstructed), axis=1)
        errors.append(error)
    return np.argmin(errors, axis=0)
