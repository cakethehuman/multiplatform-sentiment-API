import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class Vectorize:
    def __init__(self,x_train ,x_test, App):
        self.x_train = x_train
        self.x_test = x_test
        self.App = App
    
    def transform(self):
        tf = TfidfVectorizer(stop_words= 'english')
        X_train = tf.fit_transform(self.x_train)
        X_test = tf.transform(self.x_test)
        
        joblib.dump(X_train, f'Data\interim\{self.App}\X_train.pkl')
        joblib.dump(X_test, f'Data\interim\{self.App}\X_test.pkl')
        joblib.dump(tf, f"word vectors\{self.App}_Vectorizer.pkl")
        
        
        return X_train,X_test