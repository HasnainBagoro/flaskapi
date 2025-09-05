from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI()

# Load trained model
model = joblib.load("url_model_tldfreq.pkl")

class URLInput(BaseModel):
    url: str

@app.get("/")
def home():
    return {"message": "Malicious URL Detection API is running"}

@app.post("/predict")
def predict_url(data: URLInput):
    prediction = model.predict([data.url])[0]
    return {"url": data.url, "malicious": bool(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
