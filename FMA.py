import os
import pandas as pd
import ast
import numpy as np

class FreeMusicArchive:
    def __init__(self, fma_set, sr=44100):
        self.NB_AUDIO_SAMPLES = 1321967
        self.SAMPLING_RATE = sr
        self.META_FEATURES_PATH = './Data/fma_metadata/features.csv'
        self.META_GENRES_PATH = './Data/fma_metadata/genres.csv'
        self.META_TRACKS_PATH = './Data/fma_metadata/tracks.csv'
        self.META_ECHO_PATH = './Data/fma_metadata/echonest.csv'
        self.DATA_PATH = './Data/fma_large'

        # MP3 file IDs with 0 second of audio.
        self.FILES_NO_AUDIO = [1486, 5574, 65753, 80391, 98558, 98559, 98560, 98571, 99134, 105247, 108925, 126981, 127336, 133297, 143992]
        # MP3 train file IDs with less than 30 seconds of audio.
        self.FILES_SHORT = [98565, 98566, 98567, 98568, 98569, 108924]
        # MP3 files that are wierd or distorted
        self.FILES_DISTORTED = [107535, 48949, 44374]
        self.FILES_FAULTY = self.FILES_NO_AUDIO + self.FILES_SHORT + self.FILES_DISTORTED

        self.TRACKS, self.PARTITION = self.get_metadata(fma_set)

        self.TOP_GENRES, self.CLASS_MAP, self.NUM_CLASSES = self.get_top_genres(fma_set)

        self.FEATURES = self.load(self.META_FEATURES_PATH)

    def load(self, filepath):
        filename = os.path.basename(filepath)
        if 'features' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'echonest' in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

        if 'genres' in filename:
            return pd.read_csv(filepath, index_col=0)

        if 'tracks' in filename:
            tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

            COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                       ('track', 'genres'), ('track', 'genres_all')]
            for column in COLUMNS:
                tracks[column] = tracks[column].map(ast.literal_eval)

            COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                       ('album', 'date_created'), ('album', 'date_released'),
                       ('artist', 'date_created'), ('artist', 'active_year_begin'),
                       ('artist', 'active_year_end')]
            for column in COLUMNS:
                tracks[column] = pd.to_datetime(tracks[column])

            SUBSETS = ('small', 'medium', 'large')
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.api.types.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

            COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                       ('album', 'type'), ('album', 'information'),
                       ('artist', 'bio')]
            for column in COLUMNS:
                tracks[column] = tracks[column].astype('category')

            return tracks

    def get_audio_path(self, track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(self.DATA_PATH, tid_str[:3], tid_str + '.mp3')

    def get_metadata(self, fma_set):
        tracks = self.load(self.META_TRACKS_PATH)
        tracks = tracks.drop(self.FILES_FAULTY)

        if fma_set == 'full':
            ss = tracks.loc[tracks.set.subset <= 'large']
            partition = {}
            partition['training'] = ss.loc[ss.set.split == 'training'].index.values
            partition['validation'] = ss.loc[ss.set.split == 'validation'].index.values
            partition['test'] = ss.loc[ss.set.split == 'test'].index.values
            return ss, partition
        else:
            ss = tracks.loc[tracks.set.subset <= fma_set]
            partition = {}
            partition['training'] = ss.loc[ss.set.split == 'training'].index.values
            partition['validation'] = ss.loc[ss.set.split == 'validation'].index.values
            partition['test'] = ss.loc[ss.set.split == 'test'].index.values
            return ss, partition

    def get_top_genres(self, fma_set):
        if fma_set == 'small':
            class_map = {'Electronic':0, 'Experimental':1, 'Folk':2, 'Hip-Hop':3, 'Instrumental':4, 'International':5, 'Pop':6, 'Rock':7}
            labels = {}
            for index, row in self.TRACKS.track.iterrows():
                labels[index] = class_map[row['genre_top']]

            return labels, class_map, len(class_map)
        else:
            labels = {}
            class_map = {}
            cnt = 0
            for cl in self.TRACKS.track.genre_top.cat.categories:
                class_map[cl] = cnt
                cnt += 1

            for index, row in self.TRACKS.track.iterrows():
                labels[index] = class_map[row['genre_top']]

            return labels, class_map, len(class_map)

    def get_filenames(self, filepath):
        names = os.listdir(filepath)
        names.remove('README.txt')
        names.remove('checksums')

        files = []
        for name in names:
            i_names = os.listdir(filepath + f'/{name}/')
            for n in i_names:
                if int(n[:6]) in self.FILES_FAULTY:
                    continue
                files.append(filepath + f'/{name}/{n}')

        return np.asarray(files)