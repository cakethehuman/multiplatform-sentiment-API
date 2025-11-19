import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\models\Spotify_model.pkl")
vec = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\word vectors\Vectorizer.pkl")

print(vec)

X = vec.transform(["the ads sucks"])
pred = model.predict(X)[0]
print(pred)
# print(preds)