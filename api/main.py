import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\models\Spotify_model.pkl")
vec = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\word vectors\Vectorizer.pkl")

app = FastAPI()

class ModelInput(BaseModel):
    feature: str

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/sentiment")
async def sentiment(data: ModelInput):
    pred = model.predict([data.feature])[0]
    return pred
