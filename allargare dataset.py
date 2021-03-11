from typing import List
import lyricsgenius
import pandas as pd
from config import client_access_token

def get_songs(artist_name="", max_songs=2):
    for attempt in range(3):
        try:
            genius = lyricsgenius.Genius(client_access_token)
            artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
            return artist.songs
        except:
            return get_songs(artist_name)
        else:
            break
    else:
        return ["error"]


def add_new_songs(new_songs=[], songs=[]):
    songs_to_add = []
    for ns in new_songs:
        if ns not in songs:
            songs_to_add.append(ns)
    return songs_to_add


if __name__ == '__main__':

    df = pd.read_csv("datasets/df_prova.csv")

    mini = df[0:10]

    new_songs= []

    for index, row in mini.iterrows():
        artista = str(row["Artista"]).replace("Featuring", "").split()[0:2]
        artisti_check = []
        if artista not in "".join(artisti_check):
            songs_genius = get_songs(artista)
            provisoria = add_new_songs(new_songs=songs_genius, songs= mini["Titolo"].to_list())
            for p in provisoria:
                new_songs.append(p)

