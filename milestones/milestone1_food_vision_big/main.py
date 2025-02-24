import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import plot_model

from src.pipeline import Food101Pipeline


def main() -> None:
    """
    Executes the pipeline to train and evaluate a food classification model.

    Steps:
        1. Initialize the pipeline.
        2. Load and preprocess the dataset.
        3. Build and compile the model.
        4. Train the model.
        5. Evaluate and save the model.
        6. Load the saved model and verify its performance.
    """
    device_names = [device.name for device in tf.config.list_physical_devices()]
    print(f"Using TensorFlow version {tf.__version__}\nUsing devices: {device_names}")

    # Create a pipeline
    pipeline = Food101Pipeline()

    # Handle data
    pipeline.load_data("datasets/food-101")
    pipeline.prepare_data()

    # Build model
    model = pipeline.build_model()
    model.summary()
    plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)

    # Train
    pipeline.train_model(model, epochs=5)

    # Evaluate
    loss, accuracy = pipeline.evaluate_model(model)
    print(f"Evaluation loss: {loss}")
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")

    # Save model
    save_dir = "efficientnetb0_feature_extract_model_mixed_precision.keras"
    pipeline.save_model(model, save_dir)

    # Load model and re-evaluate it
    loaded_model = pipeline.load_model(save_dir)
    loaded_loss, loaded_accuracy = pipeline.evaluate_model(loaded_model)
    print(f"Loaded model loss: {loaded_loss}")
    print(f"Loaded model accuracy: {loaded_accuracy * 100:.2f}%")

    # Verify that the loaded model performance is not far from the trained and saved one
    np.testing.assert_allclose([loss, accuracy], [loaded_loss, loaded_accuracy], atol=1e-5)


if __name__ == "__main__":
    main()
