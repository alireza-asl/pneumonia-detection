import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pneumonia_classifier.keras"
IMG_SIZE = (180, 180)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def parse_args() -> argparse.Namespace:
    "Read the image path provided by the user on the command line."
    parser = argparse.ArgumentParser(
        description="Predict NORMAL or PNEUMONIA for a chest X-ray image."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the chest X-ray image to classify.",
    )
    return parser.parse_args()


def load_trained_model() -> keras.Model:
    "Load the saved Keras model and report a clear error if it is missing."
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python main.py` first "
            "to train and save the model."
        )

    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Load, resize, and batch one image for model prediction.

    The model was trained on 180x180 RGB images, so prediction images must be
    resized the same way. Keras converts the image to an array, and expand_dims
    adds a batch dimension because models expect batches of images.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = keras.utils.load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    image_array = keras.utils.img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch


def predict_image(model: keras.Model, image_batch: np.ndarray) -> tuple[str, float]:
    "Return the predicted label and confidence percentage for one image."
    prediction_probability = float(model.predict(image_batch, verbose=0)[0][0])

    if prediction_probability >= 0.5:
        predicted_class = CLASS_NAMES[1]
        confidence = prediction_probability * 100
    else:
        predicted_class = CLASS_NAMES[0]
        confidence = (1 - prediction_probability) * 100

    return predicted_class, confidence


def main() -> None:
    "Run the complete prediction workflow."
    args = parse_args()
    model = load_trained_model()
    image_batch = preprocess_image(args.image_path)
    predicted_class, confidence = predict_image(model, image_batch)
    print(f"Prediction: {predicted_class} ({confidence:.1f}%)")


if __name__ == "__main__":
    main()