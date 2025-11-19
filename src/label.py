import joblib

import pandas as pd

def reviewChanger(df, columnName):
    try : 
        if df[columnName] > 3:
            return "Positive"
        else:
            return "Negative"
    except Exception as e:
        print(f"The error : {e}")
        return "Uknown"
    
class Label:
    def __init__(self, df, reviewColumnName,App):
        self.df = df
        self.reviewColumnName = reviewColumnName
        self.App = App
        

    
    def sentiment(self):
        
        """
        Apply reviewchanger funtion to all rows where the it also take the name of the column name so it
        can be used for other dataframes
        
        """
        
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()
        self.df["Sentiment"] = self.df.apply(lambda row : reviewChanger(row, self.reviewColumnName), axis = 1)
        return self.df

