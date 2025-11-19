from Label import Label
from Procces import Procces
from Vectorize import Vectorize
from Split import Split
import joblib

import pandas as pd


def main():
    # Insert data
    threads = pd.read_csv(r"Data\raw\threads\threads_reviews.csv")
    spotify = pd.read_csv(r"Data\raw\spotify\reviews.csv")

    # Dropping unwanted columns
    threads = threads.drop(['source','review_date'], axis= 1)
    spotify = spotify.drop(["Time_submitted", 'Total_thumbsup', 'Reply'], axis= 1)

    # Data labeling
    sp_cleaner = Label(spotify, "Rating", "Spotify")
    spotify_clean = sp_cleaner.sentiment()

    joblib.dump(spotify_clean, f'Data\preproccesed\spotify_clean.pkl')

    # 2) Process text (assume review column is called "Review" or similar)
    sp_processor = Procces(spotify_clean, "Review")
    spotify_processed = sp_processor.proccesdata()

    # 3) dump the cleaned data
    joblib.dump(spotify_processed, f'Data\preproccesed\spotify_processed.pkl')
    
    X_spotify = spotify_processed["Review"]
    y_spotify = spotify_processed["Sentiment"]

    # 6) Train-test split + save
    splitter = Split(X_spotify, y_spotify, App = "spotify")
    X_train, X_test, y_train, y_test = splitter.tts()
    
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test: ", y_test.shape)
    
    Vtr = Vectorize(x_train = X_train, x_test = X_test, App = "spotify")
    X_train_vec,X_test_vec = Vtr.transform()
    
    print(f"hasil vectorize train: {X_train_vec}" )
    print(f"hasil vectorize test: {X_test_vec}" )



if __name__ == "__main__":
    main()