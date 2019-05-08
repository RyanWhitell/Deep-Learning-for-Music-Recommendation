import time
import numpy as np
import random
import pandas as pd

import keras
from keras.models import Model, load_model
from keras import layers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

import h5py
import pickle
import argparse

import SPOTIFY

parser = argparse.ArgumentParser(description="trains a stacked ensemble model")
parser.add_argument('--lyrics', action='store_true')
parser.add_argument('--albums', action='store_true')
parser.add_argument('--location', action='store_true')
parser.add_argument('--year', action='store_true')
parser.add_argument('-q', '--quick', default=False, help='runs each test quickly to ensure they will run')
args = parser.parse_args()

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, track_ids, SPOTIFY, add_lyrics, add_albums, add_location, add_year, batch_size, shuffle=True):
        'Initialization'
        self.track_ids = track_ids
        self.SPOTIFY = SPOTIFY
        self.add_lyrics = add_lyrics 
        self.add_albums = add_albums
        self.add_location = add_location 
        self.add_year = add_year
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
        chroma_rnn = np.empty((self.batch_size, 800))
        mfcc_rnn = np.empty((self.batch_size, 800))
        cqt_freq = np.empty((self.batch_size, 800))
        cqt_time = np.empty((self.batch_size, 800))
        mel_scaled_stft_freq = np.empty((self.batch_size, 800))
        mel_scaled_stft_time = np.empty((self.batch_size, 800))
        lyrics = np.empty((self.batch_size, 800))
        albums = np.empty((self.batch_size, 800))
        lat = np.zeros((self.batch_size, 1))
        lng = np.zeros((self.batch_size, 1))
        year = np.empty((self.batch_size, 1))
        target = np.empty((self.batch_size, 800))
        
        with h5py.File('./Results/spotify/predictions.hdf5','r') as f:
            for batch_index, track_id in enumerate(track_ids_temp): 
                chroma_rnn[batch_index,:] = f['chroma_rnn'][str(track_id)] / np.linalg.norm(f['chroma_rnn'][str(track_id)])
                mfcc_rnn[batch_index,:] = f['mfcc_rnn'][str(track_id)] / np.linalg.norm(f['mfcc_rnn'][str(track_id)])
                cqt_freq[batch_index,:] = f['cqt_freq_cnn'][str(track_id)] / np.linalg.norm(f['cqt_freq_cnn'][str(track_id)])
                cqt_time[batch_index,:] = f['cqt_time_cnn'][str(track_id)] / np.linalg.norm(f['cqt_time_cnn'][str(track_id)])
                mel_scaled_stft_freq[batch_index,:] = f['mel_scaled_stft_freq_cnn'][str(track_id)] / np.linalg.norm(f['mel_scaled_stft_freq_cnn'][str(track_id)])
                mel_scaled_stft_time[batch_index,:] = f['mel_scaled_stft_time_cnn'][str(track_id)] / np.linalg.norm(f['mel_scaled_stft_time_cnn'][str(track_id)])
                lyrics[batch_index,:] = f['lyrics_rnn'][str(track_id)] / np.linalg.norm(f['lyrics_rnn'][str(track_id)])
                albums[batch_index,:] = f['albums_cnn'][str(track_id)] / np.linalg.norm(f['albums_cnn'][str(track_id)])
                lat[batch_index,:] = self.SPOTIFY.DATA.loc[track_id].lat_norm
                lng[batch_index,:] = self.SPOTIFY.DATA.loc[track_id].lng_norm
                year[batch_index,:] = self.SPOTIFY.DATA.loc[track_id].year_norm
                target[batch_index,:] = self.SPOTIFY.DATA.loc[track_id].embedding_vector

        return_list = [chroma_rnn, mfcc_rnn, cqt_freq, cqt_time, mel_scaled_stft_freq, mel_scaled_stft_time]
        
        if self.add_lyrics:
            return_list.append(lyrics)
        if self.add_albums:
            return_list.append(albums)
        if self.add_location:
            return_list.append(lat)
            return_list.append(lng)
        if self.add_year:
            return_list.append(year)

        return return_list, target

def train_model(model, model_name, add_lyrics, add_albums, add_location, add_year, quick):
    print('*' * 117)
    print(f'Training model: {model_name}')
    print('*' * 117)

    time_start = time.perf_counter()
    
    if quick:
        #results_model_name = './Results/spotify/DELETE.' + model_name
        model_name = './Models/spotify/DELETE.' + model_name
    else:
        #results_model_name = './Results/spotify/' + model_name
        model_name = './Models/spotify/' + model_name

    train_list = list(SPOTIFY.DATA.loc[SPOTIFY.DATA.split == 'train'].index.values)
    val_list = list(SPOTIFY.DATA.loc[SPOTIFY.DATA.split == 'val'].index.values)
    test_list = list(SPOTIFY.DATA.loc[SPOTIFY.DATA.split == 'test'].index.values)

    if quick:
        epochs = 1
        train_list = train_list[:64]
        val_list = train_list[:64]
        test_list = test_list[:64]
    else:
        epochs = 500

    training_generator = DataGenerator(
        track_ids=train_list,
        SPOTIFY=SPOTIFY,
        add_lyrics=add_lyrics, 
        add_albums=add_albums, 
        add_location=add_location, 
        add_year=add_year,
        batch_size=8
    )

    val_generator = DataGenerator(
        track_ids=val_list,
        SPOTIFY=SPOTIFY,
        add_lyrics=add_lyrics, 
        add_albums=add_albums, 
        add_location=add_location, 
        add_year=add_year,
        batch_size=1,
        shuffle=False
    )

    test_generator = DataGenerator(
        track_ids=test_list,
        SPOTIFY=SPOTIFY,
        add_lyrics=add_lyrics, 
        add_albums=add_albums, 
        add_location=add_location, 
        add_year=add_year,
        batch_size=1,
        shuffle=False
    )

    model.compile(
        loss=keras.losses.cosine_proximity,
        optimizer=keras.optimizers.Adam(lr=0.00001),
        metrics=['mean_squared_error']
    )

    checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True) 
    callbacks_list = [checkpoint, early_stop]

    history = model.fit_generator(
        generator=training_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list
    )

    with open(model_name + '.history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    predictions = model.predict_generator(
        generator=test_generator,
        verbose=1
    )

    predictions_df = pd.DataFrame(data={}, index=test_list)

    predictions_list = []
    for pred in predictions:
        predictions_list.append(list(pred))

    predictions_df['prediction'] = predictions_list

    with open(model_name + '.test_predictions.pkl', 'wb') as file_pi:
        save = {
            'test_predictions':predictions_df
        }
        pickle.dump(save, file_pi)
    
    print('*' * 117)
    print(f'Total execution time:{(time.perf_counter()-time_start)/60:.2f} min')
    print('*' * 117)   
    
def STACKED_ENS(add_lyrics, add_albums, add_location, add_year):
    in_chroma_rnn = layers.Input(shape=(800,), name='in_chroma_rnn')
    in_mfcc_rnn = layers.Input(shape=(800,), name='in_mfcc_rnn')
    in_cqt_freq = layers.Input(shape=(800,), name='in_cqt_freq')
    in_cqt_time = layers.Input(shape=(800,), name='in_cqt_time')
    in_mel_scaled_stft_freq = layers.Input(shape=(800,), name='in_mel_scaled_stft_freq')
    in_mel_scaled_stft_time = layers.Input(shape=(800,), name='in_mel_scaled_stft_time')
    in_lyrics = layers.Input(shape=(800,), name='in_lyrics')
    in_albums = layers.Input(shape=(800,), name='in_albums')
    in_lat = layers.Input(shape=(1,), name='in_lat')
    in_lng = layers.Input(shape=(1,), name='in_lng')
    in_year = layers.Input(shape=(1,), name='in_year')

    chroma_rnn = layers.Dense(1024, activation='relu', name='chroma_rnn_input')(in_chroma_rnn)
    chroma_rnn = layers.Dense(512, activation='relu', name='chroma_rnn_dim_red')(chroma_rnn)

    mfcc_rnn = layers.Dense(1024, activation='relu', name='mfcc_rnn_input')(in_mfcc_rnn)
    mfcc_rnn = layers.Dense(512, activation='relu', name='mfcc_rnn_dim_red')(mfcc_rnn)

    cqt_freq = layers.Dense(1024, activation='relu', name='cqt_freq_input')(in_cqt_freq)
    cqt_time = layers.Dense(1024, activation='relu', name='cqt_time_input')(in_cqt_time)
    cqt_time_freq_cnn = layers.concatenate([cqt_time,cqt_freq], name='cqt_time_freq')
    cqt_time_freq_cnn = layers.Dense(2048, activation='relu', name='cqt_time_freq_fc1')(cqt_time_freq_cnn)
    cqt_time_freq_cnn = layers.Dense(1024, activation='relu', name='cqt_time_freq_fc2')(cqt_time_freq_cnn)
    cqt_time_freq_cnn = layers.Dense(512, activation='relu', name='cqt_time_freq__dim_red')(cqt_time_freq_cnn)

    mel_scaled_stft_freq = layers.Dense(1024, activation='relu', name='mel_scaled_stft_freq_input')(in_mel_scaled_stft_freq)
    mel_scaled_stft_time = layers.Dense(1024, activation='relu', name='mel_scaled_stft_time_input')(in_mel_scaled_stft_time)
    mel_scaled_stft_time_freq_cnn = layers.concatenate([mel_scaled_stft_time,mel_scaled_stft_freq], name='mel_scaled_stft_time_freq')
    mel_scaled_stft_time_freq_cnn = layers.Dense(2048, activation='relu', name='mel_scaled_stft_time_freq_fc1')(mel_scaled_stft_time_freq_cnn)
    mel_scaled_stft_time_freq_cnn = layers.Dense(1024, activation='relu', name='mel_scaled_stft_time_freq_fc2')(mel_scaled_stft_time_freq_cnn)
    mel_scaled_stft_time_freq_cnn = layers.Dense(512, activation='relu', name='mel_scaled_stft_time_freq__dim_red')(mel_scaled_stft_time_freq_cnn)

    lyrics = layers.Dense(1024, activation='relu', name='lyrics_input')(in_lyrics)
    lyrics = layers.Dense(512, activation='relu', name='lyrics_dim_red')(lyrics)

    albums = layers.Dense(1024, activation='relu', name='albums_input')(in_albums)
    albums = layers.Dense(512, activation='relu', name='albums_dim_red')(albums)

    lat = layers.Dense(128, activation='relu', name='lat_input')(in_lat)
    lng = layers.Dense(128, activation='relu', name='lng_input')(in_lng)
    location = layers.concatenate([lat, lng], name='lat_lng_concat')
    location = layers.Dense(512, activation='relu', name='location_input')(location)
    
    year = layers.Dense(512, activation='relu', name='year_input')(in_year)
    
    x = layers.concatenate([chroma_rnn, mfcc_rnn, cqt_time_freq_cnn, mel_scaled_stft_time_freq_cnn], name='concat_audio_base')

    input_list = [in_chroma_rnn, in_mfcc_rnn, in_cqt_freq, in_cqt_time, in_mel_scaled_stft_freq, in_mel_scaled_stft_time]

    if add_lyrics:
        x = layers.concatenate([x, lyrics], name='concat_lyrics')
        input_list.append(in_lyrics)
    if add_albums:
        x = layers.concatenate([x, albums], name='concat_albums')
        input_list.append(in_albums)
    if add_location:
        x = layers.concatenate([x, location], name='concat_location')
        input_list.append(in_lat)
        input_list.append(in_lng)
    if add_year:
        x = layers.concatenate([x, year], name='concat_year')
        input_list.append(in_year)

    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    
    x = layers.Dense(2048, activation='relu', name='fc2')(x)

    x = layers.Dense(1024, activation='relu', name='fc3')(x)

    pred = layers.Dense(800, activation=None, name='output')(x)
    
    return Model(inputs=input_list, outputs=pred)

if __name__ == '__main__':
    SPOTIFY = SPOTIFY.SPOTIFY()

    model_name = "STACKED_ENS"
    if args.lyrics:
        model_name = 'LYRICS_' + model_name
    if args.albums:
        model_name = 'ALBUMS_' + model_name
    if args.location:
        model_name = 'LOCATION_' + model_name
    if args.year:
        model_name = 'YEAR_' + model_name

    ################# STACKED_ENS ################
    K.clear_session()
    model = STACKED_ENS(add_lyrics=args.lyrics, add_albums=args.albums, add_location=args.location, add_year=args.year)
    model.summary()
    train_model(model=model, model_name=model_name, add_lyrics=args.lyrics, add_albums=args.albums, add_location=args.location, add_year=args.year, quick=args.quick)