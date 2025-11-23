import pandas as pd
from procces import Procces
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load(r"models\Spotify_model.pkl")
vec = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\word vectors\Vectorizer.pkl")

print(model)