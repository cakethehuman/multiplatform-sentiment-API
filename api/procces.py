import pandas as pd

import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

vec = joblib.load(r"C:\Users\wilsen\OneDrive\Desktop\Sentiment api\word vectors\Vectorizer.pkl")

wnl = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


# Apply Stop words removal and lemmatization
def words(txt):
    words = word_tokenize(txt)
    filter = [row for row in words if row.lower() not in stop_words]
    filter = [wnl.lemmatize(row.lower()) for row in filter]
    return " ".join(filter) 

class Procces:
    def __init__(self, words, cols):
        self.words = words
        self.cols = cols

        
    def normalize(self):
        
        # Remove special characters
        
        hasil_normalize = self.words[self.cols].replace(r"[^A-Za-z]", value = " " ,regex = True)
        return hasil_normalize
         
    
    def proccesdata(self):
        self.words[self.cols] = self.normalize()
        self.words[self.cols] = self.words[self.cols].apply(words)
        return self.words
        

# p = ['sdwdw']
# x = pd.DataFrame(p, columns= ["w"])
# hasil = Procces(x,'w')
# hasil.proccesdata()

