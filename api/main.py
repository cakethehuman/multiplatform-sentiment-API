import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

with open(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\models\Spotify_model.pkl", 'rb') as f:
    model = joblib.load(f)

app = FastAPI()

class ModelInput(BaseModel):
    feature: str

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/sentiment")
async def sentiment(data: ModelInput):
    pred = model.predict([data.feature])[0]
    pred = int(pred) if hasattr(pred, "__int__") else pred
    return {"input": data.feature, "prediction": int(pred)}