"""
Utility Functions for TensorFlow Projects
-----------------------------------------
A collection of functions for image processing, model evaluation, and visualization.
"""

import os
import datetime
import itertools
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import List, Tuple, Dict, Union, Optional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


def load_and_prep_image(filename: str, img_size: Tuple[int, int] = 224, scale: bool = True) -> tf.Tensor:
    """
    Reads an image from a file, preprocesses it into a tensor, and resizes it.

    Args:
        filename (str): Path to the target image file.
        img_size (Tuple[int, int]): Size to resize the image to (default: (224, 224)).
        scale (bool): Whether to scale pixel values to [0, 1] (default: True).

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, img_size)
    return img / 255. if scale else img


def make_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: Optional[List[str]] = None,
    figsize: tuple = (10, 10),
    text_size: int = 15,
    norm: bool = False,
    savefig: bool = False
) -> None:
    """
    Creates a labeled confusion matrix plot comparing predictions and true labels.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.
        classes (Optional[List[str]]): List of class names (optional).
        figsize (tuple): Figure size (default: (10, 10)).
        text_size (int): Text size for annotations (default: 15).
        norm (bool): Normalize values or not (default: False).
        savefig (bool): Save figure to file (default: False).

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if norm else cm
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes else np.arange(n_classes)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        text = f"{cm[i, j]}" if not norm else f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)"
        plt.text(j, i, text, ha="center", color="white" if cm[i, j] > threshold else "black", size=text_size)

    if savefig:
        fig.savefig("confusion_matrix.png")


def pred_and_plot(model: tf.keras.Model, filename: str, class_names: List[str]) -> None:
    """
    Predicts and visualizes the prediction on a single image.

    Args:
        model (tf.keras.Model): Trained TensorFlow model.
        filename (str): Path to the image file.
        class_names (List[str]): List of class names.

    Returns:
        None
    """
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    pred_class = class_names[pred.argmax()] if len(pred[0]) > 1 else class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


def create_tensorboard_callback(dir_name: str, experiment_name: str) -> tf.keras.callbacks.TensorBoard:
    """
    Creates a TensorBoard callback instance to log training metrics.

    Args:
        dir_name (str): Directory to save logs.
        experiment_name (str): Experiment name.

    Returns:
        tf.keras.callbacks.TensorBoard: TensorBoard callback instance.
    """
    log_dir = f"{dir_name}/{experiment_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Saving TensorBoard logs to: {log_dir}")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def plot_loss_curves(history: tf.keras.callbacks.History) -> None:
    """
    Plots training and validation loss and accuracy curves.

    Args:
        history (tf.keras.callbacks.History): TensorFlow model History object.

    Returns:
        None
    """
    epochs = range(len(history.history['loss']))

    plt.figure()
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def compare_histories(
    original_history: tf.keras.callbacks.History,
    new_history: tf.keras.callbacks.History,
    initial_epochs: int = 5
) -> None:
    """
    Compares training histories of models.

    Args:
        original_history (tf.keras.callbacks.History): History object from initial training.
        new_history (tf.keras.callbacks.History): History object from continued training.
        initial_epochs (int): Initial epochs for plotting adjustment.

    Returns:
        None
    """
    total_acc = original_history.history["accuracy"] + new_history.history["accuracy"]
    total_loss = original_history.history["loss"] + new_history.history["loss"]
    total_val_acc = original_history.history["val_accuracy"] + new_history.history["val_accuracy"]
    total_val_loss = original_history.history["val_loss"] + new_history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.axvline(x=initial_epochs - 1, linestyle='--', label='Start Fine Tuning')
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.axvline(x=initial_epochs - 1, linestyle='--', label='Start Fine Tuning')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.show()


def unzip_data(filename: str) -> None:
    """
    Unzips a ZIP file to the current working directory.

    Args:
        filename (str): Path to the ZIP file.

    Returns:
        None
    """
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


def walk_through_dir(dir_path: str) -> None:
    """
    Walks through a directory and prints its structure.

    Args:
        dir_path (str): Path to the directory.

    Returns:
        None
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")


def calculate_results(y_true: List[int], y_pred: List[int]) -> Dict[str, Union[float, int]]:
    """
    Calculates accuracy, precision, recall, and F1-score for a classification model.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.

    Returns:
        Dict[str, Union[float, int]]: Dictionary of accuracy, precision, recall, and F1-score.
    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1,
    }
