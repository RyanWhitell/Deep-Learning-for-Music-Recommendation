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

def get_predictions(model_path, features):
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

    return vector_out

if __name__ == '__main__':
    chroma_rnn = get_predictions("./Models/rnn/cos/spotify.chroma.RNN_LARGE.hdf5", 'chroma')
    mfcc_rnn = get_predictions("./Models/rnn/cos/spotify.mfcc.RNN_LARGE.hdf5", 'mfcc')

    cqt_freq = get_predictions("./Models/cnn/cos/spotify.cqt.Freq.hdf5", 'cqt')
    cqt_time = get_predictions("./Models/cnn/cos/spotify.cqt.Time.hdf5", 'cqt')

    mel_scaled_stft_freq = get_predictions("./Models/cnn/cos/spotify.mel_scaled_stft.Freq.hdf5", 'mel_scaled_stft')
    mel_scaled_stft_time = get_predictions("./Models/cnn/cos/spotify.mel_scaled_stft.Time.hdf5", 'mel_scaled_stft')

    lyrics = get_predictions("./Models/spotify/1.0.001.Lyrics.hdf5", 'lyrics') # SPOTIFY(max_words=1, threshold=0.001)
    albums = get_predictions("./Models/spotify/Albums.hdf5", 'albums')

    chroma_rnn_list = []
    for pred in chroma_rnn:
        chroma_rnn_list.append(list(pred))
    del chroma_rnn

    mfcc_rnn_list = []
    for pred in mfcc_rnn:
        mfcc_rnn_list.append(list(pred))
    del mfcc_rnn

    cqt_freq_list = []
    for pred in cqt_freq:
        cqt_freq_list.append(list(pred))
    del cqt_freq

    cqt_time_list = []
    for pred in cqt_time:
        cqt_time_list.append(list(pred))
    del cqt_time

    mel_scaled_stft_freq_list = []
    for pred in mel_scaled_stft_freq:
        mel_scaled_stft_freq_list.append(list(pred))
    del mel_scaled_stft_freq

    mel_scaled_stft_time_list = []
    for pred in mel_scaled_stft_time:
        mel_scaled_stft_time_list.append(list(pred))
    del mel_scaled_stft_time

    lyrics_list = []
    for pred in lyrics:
        lyrics_list.append(list(pred))
    del lyrics

    albums_list = []
    for pred in albums:
        albums_list.append(list(pred))
    del albums

    predictions = pd.DataFrame(data={}, index=list(SPOTIFY.DATA.index.values))

    predictions['chroma_rnn'] = chroma_rnn_list
    predictions['mfcc_rnn'] = mfcc_rnn_list

    predictions['cqt_freq'] = cqt_freq_list
    predictions['cqt_time'] = cqt_time_list
    
    predictions['mel_scaled_stft_freq'] = mel_scaled_stft_freq_list
    predictions['mel_scaled_stft_time'] = mel_scaled_stft_time_list

    predictions['lyrics'] = lyrics_list
    predictions['albums'] = albums_list

    with open('./Results/spotify/predictions.pickle', 'wb') as f:
        save = {
            'predictions': predictions
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        del save