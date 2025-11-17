import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

wnl = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))


# Apply Stop words removal and lemmatization
def words(txt):
    words = word_tokenize(txt)
    filter = [row for row in words if row.lower() not in stop_words]
    filter = [wnl.lemmatize(row.lower()) for row in filter]
    return " ".join(filter) 

class Procces:
    def __init__(self,df, targetColumn):
        self.df = df
        self.targetColumn = targetColumn
        
    def normalize(self):
        
        # Remove special characters
        
        self.df[self.targetColumn] = self.df[self.targetColumn].replace(r"[^A-Za-z]", value = " " ,regex = True)   
         
    
    def proccesdata(self):
        self.normalize()
        self.df[self.targetColumn] = self.df[self.targetColumn].apply(words)
        return self.df
        
        
