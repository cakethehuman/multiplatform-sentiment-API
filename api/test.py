import pandas as pd
from procces import Procces
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\models\Spotify_model.pkl")
vec = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\word vectors\Vectorizer.pkl")

print(vec)

words = ["Test words for what ever lmao yea so much apps like cmon man"]
word_df = pd.DataFrame(words, columns= ["words"])
# words.columns = "Words"
Wordsdf = Procces(word_df, "words")
hasilwords = Wordsdf.proccesdata()
# words = vec.transform(words.to_list())
# pred = model.predict(words)[0]
print(hasilwords)
# print(preds)
