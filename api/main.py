import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from procces import Procces
import joblib

SpotifyModel = joblib.load(r"..\models\Spotify_model.pkl")
ThreadsModel = joblib.load(r"..\models\Threads_model.pkl")
InstagramModel = joblib.load(r"..\models\Instagram_model.pkl")

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
    proba = model.predict_proba(hasilwords)[0]
    index = np.argmax(proba)
    score = proba[index]
    return pred,score


@app.post("/sentiment")
async def sentiment(data: ModelInput):
    word_df = pd.DataFrame(data.feature, columns= ["words"])
    
    match data.App.lower():
        case "spotify":
            pred,score = keluarkanhasil(word_df, data.App, SpotifyModel)
            return {
                "App" : data.App,
                "sentiment": pred,
                "confidence": round(float(score),2)
            }
        case "threads":
            pred,score = keluarkanhasil(word_df, data.App, ThreadsModel)
            return {
                "App" : data.App,
                "sentiment": pred,
                "confidence": round(float(score),2)
            }
        case "instagram":
            pred,score = keluarkanhasil(word_df, data.App, InstagramModel)
            return {
                "App" : data.App,
                "sentiment": pred,
                "confidence": round(float(score),2)
            }
        case _:  # Default case, equivalent to 'else'
            return {"Please pick a valid APP"}


    
    # if data.App == "Spotify":
    #     pred = keluarkanhasil(word_df, data.App, SpotifyModel)
    #     return {f"hasil : {pred}"}
    # elif data.App == "Threads":
    #     pred = keluarkanhasil(word_df, data.App, ThreadsModel)
    #     return {f"hasil : {pred}"}
    # else:
    #     return {"Please pick a valid APP"}
        


