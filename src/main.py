from Label import Label
from Procces import Procces
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
    


if __name__ == "__main__":
    main()