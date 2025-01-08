from data import load_data
from encoder import train_general_autoencoder, train_class_autoencoders, classify
from test import test_single_image, plot_confusion_matrix
import numpy as np

x_train, y_train, x_test, y_test = load_data()

input_dim = x_train.shape[1]
general_autoencoder, general_encoder = train_general_autoencoder(x_train, input_dim)

class_autoencoders = train_class_autoencoders(x_train, y_train, input_dim)

predicted_classes = classify(x_test, class_autoencoders)
accuracy = np.mean(predicted_classes == y_test)
print(f"Dokładność klasyfikatora wieloklasowego: {accuracy}")

test_single_image(x_test, y_test, class_autoencoders, index=0)
plot_confusion_matrix(y_test, predicted_classes)
