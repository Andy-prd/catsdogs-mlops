"""M2: FastAPI inference service for cats vs dogs."""
import io
import logging
import time
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference")

app = FastAPI(title="Cats vs Dogs Classifier")
MODEL_PATH = "models/model.h5"
model = None
request_count = 0
prediction_count = 0
total_latency_ms = 0.0

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("model loaded")
    return model

def prepare_image(data):
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def label_from_probability(p, threshold=0.5):
    return "dog" if p >= threshold else "cat"

@app.middleware("http")
async def log_requests(request, call_next):
    global request_count, total_latency_ms
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    request_count += 1
    total_latency_ms += latency
    logger.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, latency)
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global prediction_count
    data = await file.read()
    try:
        batch = prepare_image(data)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")
    prob_dog = float(get_model().predict(batch, verbose=0)[0][0])
    label = label_from_probability(prob_dog)
    prediction_count += 1
    logger.info("prediction=%s prob_dog=%.4f", label, prob_dog)
    return {
        "label": label,
        "probabilities": {"cat": round(1 - prob_dog, 4), "dog": round(prob_dog, 4)},
    }

@app.get("/metrics")
def metrics():
    avg = total_latency_ms / request_count if request_count else 0.0
    return {
        "request_count": request_count,
        "prediction_count": prediction_count,
        "avg_latency_ms": round(avg, 1),
    }
