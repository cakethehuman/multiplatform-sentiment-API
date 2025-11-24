import pandas as pd
from procces import Procces
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load(r"../word vectors\Spotify_Vectorizer.pkl")


print(model)