from sklearn.model_selection import train_test_split
import joblib
class Split:
    def __init__(self, df, x, y, App):
        self.df = df
        self.x = x
        self.y = y
        self.App = App
    
    def tts(self):
        X_train,X_test,y_train,y_test = train_test_split(self.x,self.y, random_state= 42, test_size= 0.2)
        joblib.dump(X_train, f'Data\interim\{self.App}\X_train.pkl')
        joblib.dump(X_test, f'Data\interim\{self.App}\X_test.pkl')
        joblib.dump(y_train, f'Data\interim\{self.App}\y_train.pkl')
        joblib.dump(y_test, f'Data\interim\{self.App}\y_test.pkl')
        
        return X_train,X_test,y_train,y_test