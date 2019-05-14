import time as tm
import numpy as np
import pandas as pd

import keras
from keras.models import Model, load_model
from keras import layers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

from keras.datasets import cifar100

import h5py
import pickle
from PIL import Image

import SPOTIFY
SPOTIFY = SPOTIFY.SPOTIFY()

class DataGeneratorRNN(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, emb_dim, batch_size, dim, features, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.dim = dim
        self.features = features
        self.data_path = './Data/features/spotify_' + features + '.hdf5'
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, self.dim[1], self.dim[0])) 
        Y = np.empty((self.batch_size, self.emb_dim))

        with h5py.File(self.data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = f['data'][str(ID)]
                Y[i,:] = self.labels[ID]
        
        return X.reshape(X.shape[0], *self.dim), Y

class DataGeneratorCNN(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, emb_dim, batch_size, dim, features, n_channels=1, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.dim = dim
        self.features = features
        self.data_path = './Data/features/spotify_' + features + '.hdf5'
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, *self.dim))       
        Y = np.empty((self.batch_size, self.emb_dim))

        with h5py.File(self.data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = f['data'][str(ID)]
                Y[i,:] = self.labels[ID]
        
        return X.reshape(X.shape[0], *self.dim, 1), Y

class DataGeneratorALBUMS(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, track_ids, SPOTIFY, batch_size, shuffle=True):
        'Initialization'
        self.track_ids = track_ids
        self.SPOTIFY = SPOTIFY
        self.album_art_path = SPOTIFY.ALBUMS_PATH
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.track_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        track_ids_temp = [self.track_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(track_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.track_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def __data_generation(self, track_ids_temp):
        'Generates data containing batch_size samples'
        
        Y = np.empty((self.batch_size, self.SPOTIFY.EMB_DIM))
        X = np.empty((self.batch_size, 640, 640, 3))
        
        for batch_index, track_id in enumerate(track_ids_temp): 
            img = Image.open(self.album_art_path + SPOTIFY.DATA.loc[track_id].album_art_file)
            img.load()
            data = np.asarray(img)
            X[batch_index,:] = data / 255.0
            Y[batch_index,:] = self.SPOTIFY.DATA.loc[track_id].embedding_vector

        return X, Y

class DataGeneratorLYRICS(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, track_ids, SPOTIFY, batch_size, shuffle=True):
        'Initialization'
        self.track_ids = track_ids
        self.SPOTIFY = SPOTIFY
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.track_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        track_ids_temp = [self.track_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(track_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.track_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def __data_generation(self, track_ids_temp):
        'Generates data containing batch_size samples'
        
        lyric_lengths = []
        transformed_lyrics = []
        
        Y = np.empty((self.batch_size, self.SPOTIFY.EMB_DIM))
        
        for batch_index, track_id in enumerate(track_ids_temp):
            lyrics = self.SPOTIFY.lyric_words_to_int(self.SPOTIFY.DATA.loc[track_id].lyrics)
            lyric_lengths.append(len(lyrics))
            transformed_lyrics.append(lyrics)
            
            Y[batch_index,:] = self.SPOTIFY.DATA.loc[track_id].embedding_vector

        max_length = max(lyric_lengths)

        X = np.zeros((self.batch_size, max_length), dtype=np.int)
        
        for batch_index, lyric in enumerate(transformed_lyrics):
            X[batch_index,(max_length-len(lyric)):] = lyric

        return X, Y

def get_predictions(model_path, features, desc):
    K.clear_session()
    
    batch_size = 21

    model = load_model(model_path)

    track_ids = list(SPOTIFY.DATA.index.values)

    if features == 'mel_scaled_stft':
        gen = DataGeneratorCNN(
             list_IDs=track_ids, 
             labels=SPOTIFY.DATA[['embedding_vector']].to_dict()['embedding_vector'], 
             emb_dim=800, 
             batch_size=batch_size, 
             dim=(256, 643), 
             features='mel_scaled_stft',
             shuffle=False
        )

    if features == 'cqt':
        gen = DataGeneratorCNN(
             list_IDs=track_ids, 
             labels=SPOTIFY.DATA[['embedding_vector']].to_dict()['embedding_vector'], 
             emb_dim=800, 
             batch_size=batch_size, 
             dim=(168, 643), 
             features='cqt',
             shuffle=False
        )

    if features == 'chroma':
        gen = DataGeneratorRNN(
             list_IDs=track_ids, 
             labels=SPOTIFY.DATA[['embedding_vector']].to_dict()['embedding_vector'], 
             emb_dim=800, 
             batch_size=batch_size, 
             dim=(643, 12), 
             features='chroma',
             shuffle=False
        )

    if features == 'mfcc':
        gen = DataGeneratorRNN(
             list_IDs=track_ids, 
             labels=SPOTIFY.DATA[['embedding_vector']].to_dict()['embedding_vector'], 
             emb_dim=800, 
             batch_size=batch_size, 
             dim=(643, 12), 
             features='mfcc',
             shuffle=False
        )

    if features == 'albums':
        gen = DataGeneratorALBUMS(
            track_ids=track_ids,
            SPOTIFY=SPOTIFY,
            batch_size=batch_size,
            shuffle=False
        )

    if features == 'lyrics':
        gen = DataGeneratorLYRICS(
            track_ids=track_ids,
            SPOTIFY=SPOTIFY,
            batch_size=61,
            shuffle=False
        )

    vector_out = model.predict_generator(
        generator=gen,
        verbose=1
    )

    data = FILE.create_group(desc)

    for track_id, pred in zip(track_ids, vector_out):
        data[str(track_id)] = pred
        
if __name__ == '__main__':
    FILE = h5py.File('./Results/spotify/predictions.hdf5', 'a')
    
    get_predictions("./Models/rnn/cos/spotify.chroma.RNN_LARGE.hdf5", 'chroma', 'chroma_rnn')
    get_predictions("./Models/rnn/cos/spotify.mfcc.RNN_LARGE.hdf5", 'mfcc', 'mfcc_rnn')
    get_predictions("./Models/cnn/cos/spotify.cqt.Freq.hdf5", 'cqt', 'cqt_time_cnn')
    get_predictions("./Models/cnn/cos/spotify.cqt.Simple.hdf5", 'cqt', 'cqt_simple_cnn')
    get_predictions("./Models/cnn/cos/spotify.cqt.Time.hdf5", 'cqt', 'cqt_freq_cnn')
    get_predictions("./Models/cnn/cos/spotify.mel_scaled_stft.Freq.hdf5", 'mel_scaled_stft', 'mel_scaled_stft_freq_cnn')
    get_predictions("./Models/cnn/cos/spotify.mel_scaled_stft.Simple.hdf5", 'mel_scaled_stft', 'mel_scaled_stft_simple_cnn')
    get_predictions("./Models/cnn/cos/spotify.mel_scaled_stft.Time.hdf5", 'mel_scaled_stft', 'mel_scaled_stft_time_cnn')
    get_predictions("./Models/spotify/1.0.001.Lyrics.hdf5", 'lyrics', 'lyrics_rnn') # SPOTIFY(max_words=1, threshold=0.001)
    get_predictions("./Models/spotify/Albums.hdf5", 'albums', 'albums_cnn')

    FILE.close()