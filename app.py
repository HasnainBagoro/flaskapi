from fastapi import FastAPI
import joblib
import uvicorn

app = FastAPI()

# Load trained model
model = joblib.load("url_model_tldfreq.pkl")

@app.get("/")
def home():
    return {"message": "Malicious URL Detection API is running"}

@app.post("/predict")
def predict_url(url: str):
    # Example: you can add preprocessing of URL here
    prediction = model.predict([url])[0]
    return {"url": url, "malicious": bool(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.18.91", port=8000)
