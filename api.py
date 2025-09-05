# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import joblib
import os

# change if your model is elsewhere
MODEL_PATH = os.environ.get("MODEL_PATH", "/content/url_model.joblib")

app = FastAPI(title="URL Classifier API")

# Request models
class SingleURL(BaseModel):
    url: str

class BatchURLs(BaseModel):
    urls: List[str]

# Load model at startup
model_bundle = None

@app.on_event("startup")
def load_model():
    global model_bundle
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train model first.")
    model_bundle = joblib.load(MODEL_PATH)
    # Expected keys: "model" or "pipeline" and "encoder" or "label_encoder"
    if not (("model" in model_bundle or "pipeline" in model_bundle) and ("encoder" in model_bundle or "label_encoder" in model_bundle)):
        raise RuntimeError("Model bundle doesn't contain expected keys. Found keys: " + ", ".join(model_bundle.keys()))

def _get_model_and_encoder():
    # normalize names
    model = model_bundle.get("model") or model_bundle.get("pipeline")
    encoder = model_bundle.get("encoder") or model_bundle.get("label_encoder")
    return model, encoder

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else "missing"}

@app.post("/predict")
def predict(item: Union[SingleURL, BatchURLs]):
    """
    Accepts either {"url": "https://..."}  OR  {"urls": ["u1", "u2", ...]}
    Returns prediction(s).
    """
    model, encoder = _get_model_and_encoder()
    if model is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model or encoder not loaded")

    # single or batch
    if isinstance(item, SingleURL):
        inputs = [item.url]
        single = True
    else:
        inputs = item.urls
        single = False

    try:
        preds_idx = model.predict(inputs)
        preds = encoder.inverse_transform(preds_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if single:
        return {"prediction": preds[0]}
    else:
        return {"predictions": preds.tolist()}
