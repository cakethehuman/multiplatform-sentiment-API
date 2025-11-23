import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from procces import Procces
import joblib

SpotifyModel = joblib.load(r"models\Spotify_model.pkl")
ThreadsModel = joblib.load(r"models\Threads_model.pkl")

app = FastAPI()

class ModelInput(BaseModel):
    feature: list[str]
    App : str

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/sentiment")
async def sentiment(data: ModelInput):
    word_df = pd.DataFrame(data.feature, columns= ["words"])
    Wordsdf = Procces(word_df, "words", data.App)
    hasilwords = Wordsdf.proccesdata()
    
    if data.app == "Spotify":
        model = SpotifyModel
    elif data.app == "Threads":
        model = ThreadsModel
    else:
        return {"Please pick a valid APP"}
    
    pred = model.predict(hasilwords)[0]
    return {"Hasil": str(pred)}

