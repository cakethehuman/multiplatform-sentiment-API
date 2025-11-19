import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class Vectorize:
    def __init__(self,df ,targetColumn):
        self.df = df
        self.targetColumn = targetColumn
    
    def transform(self):
        tf = TfidfVectorizer(stop_words= 'english')
        hasil = tf.fit_transform(self.df[self.targetColumn])
        return hasil