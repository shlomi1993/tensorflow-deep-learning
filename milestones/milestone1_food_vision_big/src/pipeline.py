import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Tuple
from tensorflow import keras
from keras import layers

from common.utilities import create_tensorboard_callback


class Food101Pipeline:
    """
    A pipeline for training a food classification model using the Food101 dataset.

    Attributes:
        dataset_name (str): Name of the dataset to load.
        img_shape (int): Shape to resize images to.
        batch_size (int): Batch size for training and testing.
        class_names (list[str]): List of class names in the dataset.
        train_data (tf.data.Dataset): Preprocessed training dataset.
        test_data (tf.data.Dataset): Preprocessed testing dataset.
        ds_info (tfds.core.DatasetInfo): Dataset information object.
        checkpoint_path (str): Path to save the model checkpoints.
    """
    def __init__(self, dataset_name: str = "food101", img_size: Tuple[int, int] = (244, 244), batch_size: int = 32):
        """
        Initializes the Food101Pipeline.

        Args:
            dataset_name (str): Name of the dataset to load. Defaults to "food101".
            img_size (int): Image resolution by hight and width. Defaults to (244, 244).
            batch_size (int): Batch size for training and testing. Defaults to 32.
        """
        self.dataset_name: str = dataset_name
        self.img_size: Tuple[int, int] = img_size
        self.img_shape: Tuple[int, int, int] = img_size + (3,)
        self.batch_size: int = batch_size
        self.class_names: list[str] = []
        self.train_data: tf.data.Dataset | None = None
        self.test_data: tf.data.Dataset | None = None
        self.ds_info: tfds.core.DatasetInfo | None = None
        self.checkpoint_path: str = "model_checkpoints/cp.ckpt.weights.h5"

    def load_data(self, local_dataset: str = None) -> None:
        """
        Loads the dataset and splits it into training and validation sets.

        Args:
            local_dataset (str, optional): Use a local dataset dir instead of
                downloading a new one. Defaults to None (means to download).

        Sets the class names from the dataset information.
        """
        print("Loading local dataset..." if local_dataset else "Downloading dataset...")
        (self.train_data, self.test_data), self.ds_info = tfds.load(
            name=self.dataset_name,
            split=["train", "validation"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=local_dataset
        )
        self.class_names = self.ds_info.features["label"].names
        print(f"Loaded class names: {self.class_names}")

    def preprocess_image(self, image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocesses an image by resizing and casting it to float32.

        Args:
            image (tf.Tensor): Input image tensor.
            label (tf.Tensor): Corresponding label tensor.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Preprocessed image and label.
        """
        image = tf.image.resize(image, self.img_size)
        return tf.cast(image, tf.float32), label

    def prepare_data(self) -> None:
        """
        Prepares the training and validation datasets for model training.

        Applies preprocessing, shuffling, batching, and prefetching.
        """
        self.train_data = self.train_data.map(map_func=self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_data = self.train_data.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_data = self.test_data.map(map_func=self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_data = self.test_data.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def build_model(self) -> keras.Model:
        """
        Builds a TensorFlow model using EfficientNetB0 as the base.

        Returns:
            keras.Model: The compiled TensorFlow model.
        """
        base_model = keras.applications.EfficientNetB0(include_top=False)
        base_model.trainable = False

        inputs = layers.Input(shape=self.img_shape, name="input_layer")
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
        x = layers.Dense(len(self.class_names))(x)
        outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"]
        )
        return model

    def train_model(self, model: keras.Model, epochs: int = 3) -> keras.callbacks.History:
        """
        Trains the model using the training dataset.

        Args:
            model (keras.Model): The model to train.
            epochs (int): Number of training epochs. Defaults to 3.

        Returns:
            keras.callbacks.History: Training history object.
        """
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )

        history = model.fit(
            self.train_data,
            epochs=epochs,
            validation_data=self.test_data,
            validation_steps=int(0.15 * len(self.test_data)),
            callbacks=[
                create_tensorboard_callback("training_logs", "efficientnetb0_101_classes_feature_extract"),
                checkpoint_cb
            ]
        )
        return history

    def evaluate_model(self, model: keras.Model) -> list[float]:
        """
        Evaluates the model on the testing dataset.

        Args:
            model (keras.Model): The model to evaluate.

        Returns:
            list[float]: Evaluation metrics.
        """
        return model.evaluate(self.test_data)

    def save_model(self, model: keras.Model, save_dir: str) -> None:
        """
        Saves the trained model to the specified directory.

        Args:
            model (keras.Model): The model to save.
            save_dir (str): Directory path to save the model.
        """
        model.save(save_dir)

    def load_model(self, save_dir: str) -> keras.Model:
        """
        Loads a saved model from the specified directory.

        Args:
            save_dir (str): Directory path to load the model from.

        Returns:
            keras.Model: The loaded TensorFlow model.
        """
        return keras.models.load_model(save_dir)
