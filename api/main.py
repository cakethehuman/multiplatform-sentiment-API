import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from procces import Procces
import joblib

SpotifyModel = joblib.load(r"..\models\Spotify_model.pkl")
ThreadsModel = joblib.load(r"..\models\Threads_model.pkl")

app = FastAPI()

class ModelInput(BaseModel):
    feature: list[str]
    App : str

@app.get("/")
async def root():
    return {"status": "ok"}

def keluarkanhasil(word_df, App, model):
    Wordsdf = Procces(word_df, "words", App)
    hasilwords = Wordsdf.proccesdata()
    pred = model.predict(hasilwords)[0]
    return pred


@app.post("/sentiment")
async def sentiment(data: ModelInput):
    word_df = pd.DataFrame(data.feature, columns= ["words"])

    
    if data.App == "Spotify":
        pred = keluarkanhasil(word_df, data.App, SpotifyModel)
        return {f"hasil : {pred}"}
    elif data.App == "Threads":
        pred = keluarkanhasil(word_df, data.App, ThreadsModel)
        return {f"hasil : {pred}"}
    else:
        return {"Please pick a valid APP"}
        


