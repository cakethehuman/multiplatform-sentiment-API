from Label import Label
from Procces import Procces
from Vectorize import Vectorize
from Split import Split
import joblib

import pandas as pd

def setup(Data, ratingcolumn, App, textColumn):
    
    # 1. Label the data
    label = Label(Data, ratingcolumn, App)
    hasillabel = label.sentiment()
    
    # Dump the data to analize before applying any techique
    joblib.dump(hasillabel, f'Data\preproccesed\{App}\{App}_clean.pkl')
    
    # 2. Preprocess data
    proccesdata = Procces(hasillabel, textColumn)
    hasilprocces = proccesdata.proccesdata()
    
    # 3. dump data
    
    joblib.dump(hasilprocces, f"Data\preproccesed\{App}\{App}_proccesed.pkl")
    
    X = hasilprocces[textColumn]
    y = hasilprocces["Sentiment"]
    
    splitter = Split(X, y, App)
    X_train, X_test, y_train, y_test = splitter.tts()
    
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test: ", y_test.shape)
    
    Vtr = Vectorize(X_train,X_test, App)
    X_train_vec,X_test_vec = Vtr.transform()
    
    print(f"hasil vectorize train: {X_train_vec}" )
    print(f"hasil vectorize test: {X_test_vec}" )


def main():
    
    # Insert data
    threads = pd.read_csv(r"Data\raw\threads\threads_reviews.csv")
    Instagram = pd.read_csv(r"Data\raw\Instagram\instagram.csv")
    # spotify = pd.read_csv(r"Data\raw\spotify\reviews.csv")
    
    # Dropping unwanted columns
    # threads = threads.drop(['source','review_date'], axis= 1)
    Instagram = Instagram.drop(['review_date'], axis = 1)
    # spotify = spotify.drop(["Time_submitted", 'Total_thumbsup', 'Reply'], axis= 1)
    
    # Spotify
    # spotifyRatingColumn = "Rating"
    # spotifyName = "Spotify"
    # spotifyTextColumn = "Review"
    # setup(spotify, spotifyRatingColumn, spotifyName, spotifyTextColumn)
    # print(threads)
    print(Instagram)
    # Instargram
    InstagramRatingColumn = "rating"
    InstagramName = "Instagram"
    InstagramSpotifyTextColumn = 'review_description'
    setup(Instagram, InstagramRatingColumn, InstagramName, InstagramSpotifyTextColumn)
    
    # Threads
    # threadsRatingColumn = 'rating'
    # threadsAppName = "Threads"
    # threadsTextColumn = 'review_description'
    # setup(threads, threadsRatingColumn, threadsAppName, threadsTextColumn)


if __name__ == "__main__":
    main()
    
    
    
    # # Insert data
    # threads = pd.read_csv(r"Data\raw\threads\threads_reviews.csv")
    # spotify = pd.read_csv(r"Data\raw\spotify\reviews.csv")

    # # Dropping unwanted columns
    # threads = threads.drop(['source','review_date'], axis= 1)
    # spotify = spotify.drop(["Time_submitted", 'Total_thumbsup', 'Reply'], axis= 1)

    # # Data labeling
    # sp_label = Label(spotify, "Rating", "Spotify")
    # spotify_label = sp_label.sentiment()

    # joblib.dump(spotify_label, f'Data\preproccesed\spotify_clean.pkl')

    # # 2) Process text (assume review column is called "Review" or similar)
    # sp_processor = Procces(spotify_label, "Review")
    # spotify_processed = sp_processor.proccesdata()

    # # 3) dump the cleaned data
    # joblib.dump(spotify_processed, f'Data\preproccesed\spotify_processed.pkl')
    
    # X_spotify = spotify_processed["Review"]
    # y_spotify = spotify_processed["Sentiment"]

    # # 6) Train-test split + save
    # splitter = Split(X_spotify, y_spotify, App = "spotify")
    # X_train, X_test, y_train, y_test = splitter.tts()
    
    # print("X_train:", X_train.shape)
    # print("X_test: ", X_test.shape)
    # print("y_train:", y_train.shape)
    # print("y_test: ", y_test.shape)
    
    # Vtr = Vectorize(x_train = X_train, x_test = X_test, App = "spotify")
    # X_train_vec,X_test_vec = Vtr.transform()
    
    # print(f"hasil vectorize train: {X_train_vec}" )
    # print(f"hasil vectorize test: {X_test_vec}" )
