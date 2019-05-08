import os
import pandas as pd
import numpy as np
import pickle
import random
from collections import Counter
from sklearn.preprocessing import StandardScaler

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

        scaler_lat = StandardScaler()
        scaler_lat.fit(self.DATA.lat.values.reshape(-1, 1))
        lat_min, lat_max = min(self.DATA.lat.values), max(self.DATA.lat.values)

        scaler_lng = StandardScaler()
        scaler_lng.fit(self.DATA.lng.values.reshape(-1, 1))
        lng_min, lng_max = min(self.DATA.lng.values), max(self.DATA.lng.values)

        scaler_year = StandardScaler()
        scaler_year.fit(np.array(self.DATA.year.values.reshape(-1, 1), dtype=np.float32))
        year_min, year_max = min(self.DATA.year.values), max(self.DATA.year.values)

        embedding_vector_unit = []
        lat_scaled = []
        lat_norm = []
        lng_scaled = []
        lng_norm = []
        year_scaled = []
        year_norm = []

        for _, track in self.DATA.iterrows():
            embedding_vector_unit.append(track.embedding_vector / np.linalg.norm(track.embedding_vector))

            lat_scaled.append(scaler_lat.transform([[track.lat]])[0][0])
            lng_scaled.append(scaler_lng.transform([[track.lng]])[0][0])
            year_scaled.append(scaler_year.transform([[float(track.year)]])[0][0])
            
            lat_norm.append((track.lat - lat_min)/(lat_max - lat_min))
            lng_norm.append((track.lng - lng_min)/(lng_max - lng_min))
            year_norm.append((track.year - year_min)/(year_max - year_min))

        self.DATA['embedding_vector_unit'] = embedding_vector_unit
        self.DATA['lat_scaled'] = lat_scaled
        self.DATA['lat_norm'] = lat_norm
        self.DATA['lng_scaled'] = lng_scaled
        self.DATA['lng_norm'] = lng_norm
        self.DATA['year_scaled'] = year_scaled
        self.DATA['year_norm'] = year_norm

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

    