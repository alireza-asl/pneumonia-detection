from tempfile import NamedTemporaryFile
from predict import preprocess_image
from io import BytesIO
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from tensorflow import keras
from pathlib import Path
from predict import DEFAULT_MODEL_PATH, load_prediction_model, predict_image
model: keras.Model | None = None

"load the model."
@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = load_prediction_model()
    yield
    model = None


app = FastAPI(
    title="Pneumonia Detection API",
    description="upload a MRI image of your selected lung to classify it.",
    version="1.0.0",
    lifespan=lifespan,
)
    
@app.get("/")
def root() -> dict:
    return {"message": "Pneumonia Detection API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="the prediction model is not loaded.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="please upload a image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="uploaded file is empty.")
    
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        image_batch = preprocess_image(Path(temp_path))

        predicted_class, confidence = predict_image(model, image_batch)

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    

    
    return {
    "filename": file.filename,
    "prediction": predicted_class,
    "confidence": round(confidence, 2)
}


