from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "pneumonia_classifier.keras"

# Training settings. Five epochs keeps training quick for learning experiments.
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 4
SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def check_dataset_folders() -> None:
    "Ensure the required dataset folders exist and contain images."
    required_folders = [
        TRAIN_DIR / "NORMAL",
        TRAIN_DIR / "PNEUMONIA",
        VALIDATION_DIR / "NORMAL",
        VALIDATION_DIR / "PNEUMONIA",
        TEST_DIR / "NORMAL",
        TEST_DIR / "PNEUMONIA",
    ]

    missing_folders = [folder for folder in required_folders if not folder.is_dir()]
    if missing_folders:
        formatted_folders = "\n".join(f"- {folder}" for folder in missing_folders)
        raise FileNotFoundError(
            "Missing dataset folders. Please create the following folders and "
            f"add chest X-ray images to them:\n{formatted_folders}"
        )

    empty_folders = [folder for folder in required_folders if not contains_images(folder)]
    if empty_folders:
        formatted_folders = "\n".join(f"- {folder}" for folder in empty_folders)
        raise ValueError(
            "The dataset folder structure exists, but these folders do not contain "
            "supported image files (.bmp, .gif, .jpeg, .jpg, .png):\n"
            f"{formatted_folders}"
        )


def contains_images(folder: Path) -> bool:
    "Return True when a class folder contains at least one supported image."
    return any(
        file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        for file_path in folder.iterdir()
    )


def load_dataset(directory: Path, shuffle: bool) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def prepare_for_training(dataset: tf.data.Dataset, shuffle: bool = False) -> tf.data.Dataset:
    if shuffle:
        # Shuffle batches so the CNN sees images in a different order each epoch.
        dataset = dataset.shuffle(buffer_size=1000, seed=SEED)

    # Prefetch lets TensorFlow prepare the next batch while the current one trains.
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def build_model() -> keras.Sequential:
    """Build a simple CNN from scratch with the Keras Sequential API."""
    model = keras.Sequential(
        [
            # Input layer declares the image size and three RGB color channels.
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

            # Rescaling converts pixel values from 0-255 to 0-1, which helps neural networks train more reliably.
            layers.Rescaling(1.0 / 255),

            # Conv2D layers learn small visual patterns such as edges, textures and shapes. MaxPooling reduces image size while keeping key signals.
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            # Dropout randomly disables some neurons during training. This can reduce overfitting by encouraging the network to learn robust features.
            layers.Dropout(0.3),

            # Flatten converts the feature maps into a vector for Dense layers.
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),

            # Sigmoid outputs one probability: close to 0=NORMAL, close to 1=PNEUMONIA.
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history: keras.callbacks.History) -> None:
    """Show training/validation accuracy and loss graphs with Matplotlib."""
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label="Training Accuracy")
    plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    "Run training, validation, testing, and saving workflow."
    check_dataset_folders()

    print("Loading datasets...")
    train_dataset = load_dataset(TRAIN_DIR, shuffle=True)
    validation_dataset = load_dataset(VALIDATION_DIR, shuffle=False)
    test_dataset = load_dataset(TEST_DIR, shuffle=False)

    train_dataset = prepare_for_training(train_dataset, shuffle=True)
    validation_dataset = prepare_for_training(validation_dataset)
    test_dataset = prepare_for_training(test_dataset)

    print("Building CNN model...")
    model = build_model()
    model.summary()

    print("Training model...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
    )

    print("Evaluating model on the test dataset...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    plot_training_history(history)


if __name__ == "__main__":
    main()