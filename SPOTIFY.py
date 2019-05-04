import os
import pandas as pd
import numpy as np
import pickle
import random
from collections import Counter

class SPOTIFY:
    def __init__(self, max_words=1, threshold=0.001):
        self.THRESHOLD = threshold

        self.ALBUMS_PATH = './Data/Spotify/albums/'
        self.LYRICS_PATH = './Data/Spotify/lyrics/'
        self.AUDIO_PATH = './Data/Spotify/audio/'

        with open('./Data/Spotify/SPOTIFY.pickle', 'rb') as f:
            save = pickle.load(f)
            self.DATA = save['SPOTIFY']
            self.ARTIST_RELATED = save['artist_related']
            self.VOCAB_TO_INT = save['vocab_to_int']
            self.INT_TO_VOCAB = save['int_to_vocab']
            self.TSNE_RESULT_COS = save['tsne_result_cos']
            self.TSNE_RESULT_EUC = save['tsne_result_euc']
            del save

        album_files = []
        for _, track in self.DATA.iterrows():
            album_files.append(track['album_id'] + '.jpg')

        self.DATA['album_art_file'] = album_files

        self.MAX_WORDS = int(len(self.VOCAB_TO_INT) * max_words)

        with open('./Data/Spotify/data_checkpoint.pickle', 'rb') as f:
            save = pickle.load(f)
            self.ARTISTS = save['artists']
            self.FUTURE_ARTISTS = save['future_artists']
            self.ALBUMS = save['albums']
            self.TRACKS = save['tracks']
            self.YEARS_SORTED = save['years_sorted']
            self.ARTIST_YEAR = save['artist_year']
            self.COUNTRIES_SORTED = save['countries_sorted']
            self.ARTIST_COUNTRY = save['artist_country']
            self.GENRES_SORTED = save['genres_sorted']
            self.ARTIST_GENRE = save['artist_genre']   
            del save

        with open('./Data/Spotify/embedding_model.emb.pickle', 'rb') as f:
            save = pickle.load(f)
            self.EMB_TABLE = save['embedding_lookup']
            del save

        with open('./Data/Spotify/artist_related.pickle', 'rb') as f:
            save = pickle.load(f)
            self.ID_TO_INT = save['id_to_int']
            self.INT_TO_ID = save['int_to_id']
            del save

        self.P_DROP = self.__get_subsampling_drop_prob()

        self.EMB_DIM = len(self.DATA.iloc[0].embedding_vector)

    def __get_subsampling_drop_prob(self):
        int_words = []
        for _, track in self.DATA.iterrows():
            for word in track['lyrics'].split():
                int_words.append(min(self.MAX_WORDS + 1, self.VOCAB_TO_INT[word]))
                
        word_counts = Counter(int_words)
        
        total_count = len(int_words)
        freqs = {word: count/total_count for word, count in word_counts.items()}
        p_drop = {word: 1 - np.sqrt(self.THRESHOLD/freqs[word]) for word in word_counts}
        return p_drop

    def lyric_words_to_int(self, lyric):
        int_words = []
        for word in lyric.split():
            int_words.append(min(self.MAX_WORDS + 1, self.VOCAB_TO_INT[word]))
        int_words_dropped = [word for word in int_words if random.random() < (1 - self.P_DROP[word])]
        return int_words_dropped

    