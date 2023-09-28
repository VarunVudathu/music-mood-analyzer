from dotenv import load_dotenv
from flask import Flask, render_template

import os
import requests
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

# from nltk.corpus import stopwords
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier

from nrclex import NRCLex

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
scope = '' # add later if needed

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

try:
    token = util.prompt_for_user_token(client_id=client_id,client_secret=client_secret,redirect_uri='http://localhost:3000')
except:
    print("Access Denied")

def get_data():
    types = ['4h2MD8T5fNW2Ss8sO5up68', '6nRNEBYokMy5gzOVgly8TF', '4kSPi8qRKjdtKnT7wYwL6S', '6nxPNnmSE0d5WlplUsa5L3']
    moods = ['calm', 'energetic', 'happy', 'sad']

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    offset = 0

    info = {
        'name': [],
        'danceability': [],
        'energy': [],
        'loudness': [],
        'speechiness': [],
        'acousticness': [],
        'instrumentalness': [],
        'liveness': [],
        'valence': [],
        'tempo': [],
        'key': [],
        'length': [],
        'time_signature': [],
        'mood': []
    }

    num = 0
    df = pd.DataFrame(info)

    for type in types:
        response = sp.playlist_items(type,offset=offset,fields='items.track.id,total',additional_types=['track'])
        all_ids = response['items']

        for id in all_ids:
            info = get_track_info(id["track"]["id"])
            feat = get_music_features(id["track"]["id"])
            df.loc[len(df.index)] = [info["track"], feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7], feat[8], feat[9], feat[10], feat[11], moods[num]]

        num+=1

    df.to_csv('song_moods.csv')

def model_train():
    df = pd.read_csv("song_moods.csv")
    col_feats = df.columns[2:-3]
    x = MinMaxScaler().fit_transform(df[col_feats])
    x2 = np.array(df[col_feats])
    y = df['mood']

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    X_train,X_test,Y_train,Y_test = train_test_split(x, encoded_y,test_size=0.2, random_state=15)

    target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'], ascending=True)
    
    estimator = KerasClassifier(build_fn=model(), epochs=300, batch_size=200, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, x, encoded_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def model():
    model = Sequential()
    model.add(Dense(8, input_dim=10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def user_auth():
    spotify = spotipy.Spotify(auth=token)

    if (spotify):
        results = spotify.current_user_playlists()
        id = results["items"][8]["id"]
        print(results["items"][8]["name"])

        print("------------------------------------------")

        offset = 0


        response = spotify.playlist_items(id,offset=offset,fields='items.track.id,total',additional_types=['track']) # limits to 100

        all_ids = response['items']

        all_tracks = []
        for id in all_ids:
            all_tracks.append(get_track_info(id["track"]["id"]))


        # for track in all_tracks:
        #     lyrics = get_lyrics(track)
        #     sentiment_analysis(lyrics)

        print(all_ids[0]['track']['id'])
        get_music_features(all_ids[0]['track']['id'])

        # lyrics = get_lyrics(all_tracks[len(all_tracks) - 1])
        # print(lyrics)
        # sentiment_analysis(lyrics)

def get_track_info(id):
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    track = sp.track(id)
    return {'artist': track["album"]["artists"][0]["name"], "track" : track["name"]}


def get_music_features(id):
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    feat = sp.audio_features(id)[0]
    all_feats = [feat["danceability"], feat["energy"], feat["loudness"], feat["speechiness"], feat["acousticness"], feat["instrumentalness"], feat["liveness"], feat["valence"], feat["tempo"], feat["key"], feat["duration_ms"], feat["time_signature"]]
    return all_feats

def get_lyrics(info):
    artist_name = info["artist"].lower().replace(" ", "-")

    track_name = info["track"].lower()
    track_name = re.sub("\((.*?)\)", "", track_name)
    track_name = re.sub("(?<=\s\-\s)(.*)", "", track_name)
    track_name = track_name.replace(" - ", "")
    track_name = track_name.strip()
    track_name = track_name.replace(" ","-").replace("’", "").replace("'", "").replace("!", "").replace("?","")

    print(track_name + ": " + artist_name)
    page = requests.get('https://genius.com/'+ artist_name + '-' + track_name + '-' + 'lyrics')
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics2 = html.find_all("div", class_="Lyrics__Container-sc-1ynbvzw-5 Dzxov")

    all_lyrics = ""
    for section in lyrics2:
        all_lyrics = all_lyrics + section.get_text()
    # print("--------------------------")

    # print(all_lyrics)
    lyrics = clean_lyrics(all_lyrics)

    return lyrics

def clean_lyrics(lyrics):
    lyrics = lyrics.strip().replace(",", "").replace("-", " ").replace(" [", "[").replace(" (", "(")
    lyrics = re.sub("\[(.*?)\]", "", lyrics)
    lyrics = re.sub("\((.*?)\)", "", lyrics)
    lyrics = re.sub("\*(.*?)\*", "", lyrics)
    lyrics = lyrics.strip().replace("  ", " ")

    s_lyrics = re.split("(?<=[a-z])(?=[A-Z])", lyrics)
    long = {"in'": "ing", "'em": "them", "'til": "until", "'cause": "because", "i'ma": 'i am going to'}

    for num in range(len(s_lyrics)):
        s_lyrics[num] = s_lyrics[num].lower().replace("in' ", "ing ").replace("'em", "them").replace("'til", "until").replace("'cause", "because").replace("i'ma", "i am going to").replace("\u200b", "")

    stop_words = ["a", "are", "i", "is", "must", "be", "an", "the", "and", "but", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just"]

    for num in range(len(s_lyrics)):
        words = s_lyrics[num].split(" ")
        filtered_words = [word for word in words if word not in stop_words]

        s_lyrics[num] = " ".join(filtered_words)

    # print(s_lyrics)
    # print("------------------")

    return s_lyrics

def sentiment_analysis(lyrics):
    print(lyrics)
    str_tweet = ' '.join(lyrics)
    text_object = NRCLex(str_tweet)

    data = text_object.raw_emotion_scores
    print(data)

    # add all info to a dataframe
    # after all songs are done - sum up each column and see which category is most popular in playlist
    # another dataframe to store information of each song - add artist and song name to beginning of dict

def client_auth():
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    song = input("Enter a song name: ")
    result = sp.search(song)
    song_id = result["tracks"]["items"][0]["id"]
    
    track = sp.track(song_id)
    print(track["album"]["artists"][0]["name"])

# look at google doc to start with actual model to predict emotion

# model_train()


app = Flask(__name__)

@app.route("/")
def main():
    return render_template("home.html")
 
if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port = 3000)