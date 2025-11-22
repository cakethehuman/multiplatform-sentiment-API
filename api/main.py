import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from procces import Procces
import joblib

model = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\models\Spotify_model.pkl")

app = FastAPI()

class ModelInput(BaseModel):
    feature: list[str]

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/sentiment")
async def sentiment(data: ModelInput):
    word_df = pd.DataFrame(data.feature, columns= ["words"])
    Wordsdf = Procces(word_df, "words")
    hasilwords = Wordsdf.proccesdata()
    pred = model.predict(hasilwords)[0]
    return {"Hasil" : str(pred)}
