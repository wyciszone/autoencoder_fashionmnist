import matplotlib.pyplot as plt
from encoder import classify
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def test_single_image(x_test, y_test, class_autoencoders, index):
    image = x_test[index]
    true_label = y_test[index]

    predicted_label = classify(np.array([image]), class_autoencoders)[0]
    reconstructed_image = class_autoencoders[predicted_label].predict(np.array([image]))[0].reshape(28, 28)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Oryginał (Label: {true_label})")
    plt.imshow(image.reshape(28, 28), cmap="gray")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Rekonstrukcja (Pred: {predicted_label})")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.show()

def plot_confusion_matrix(y_test, predicted_classes):
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
    disp.plot(cmap="viridis", values_format='d')
    plt.title("Macierz konfuzji dla klasyfikatora")
    plt.show()