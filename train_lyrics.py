import pickle
import pandas as pd
import numpy as np
import random
import time

import keras
from keras.models import Model, load_model, Sequential
from keras import layers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

import SPOTIFY

import argparse

parser = argparse.ArgumentParser(description="trains a model")
parser.add_argument('-m', '--max-words', default=1, help='Percentage of the vocabulary to keep')
parser.add_argument('-t', '--threshold', default=0.001, help='Subsampling drop threshold. 0.001 drops around 30%. Higher values drop less.')
parser.add_argument('-q', '--quick', default=False, help='runs each test quickly to ensure they will run')
args = parser.parse_args()

class DataGenerator(keras.utils.Sequence):
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
    
def train_model(model, model_name, max_words, threshold, quick):
    print('*' * 117)
    print(f'Training model: {model_name}')
    print('*' * 117)

    time_start = time.perf_counter()
    
    if quick:
        model_name = './Models/spotify/DELETE.' + str(max_words) + '.' + str(threshold) + '.' + model_name
    else:
        model_name = './Models/spotify/' + str(max_words) + '.' + str(threshold) + '.' + model_name

    train_list = list(SPOTIFY.DATA.loc[SPOTIFY.DATA.split == 'train'].index.values)
    val_list = list(SPOTIFY.DATA.loc[SPOTIFY.DATA.split == 'val'].index.values)

    if quick:
        epochs = 1
        train_list = train_list[:64]
        val_list = train_list[:64]
    else:
        epochs = 500

    training_generator = DataGenerator(
        track_ids=train_list,
        SPOTIFY=SPOTIFY,
        batch_size=16
    )

    val_generator = DataGenerator(
        track_ids=val_list,
        SPOTIFY=SPOTIFY,
        batch_size=16
    )

    ################################################# lr=0.00005
    model.compile(
        loss=keras.losses.cosine_proximity,
        optimizer=keras.optimizers.Adam(lr = 0.00005, clipnorm=1.),
        metrics=['mean_squared_error']
    )

    checkpoint = ModelCheckpoint(model_name + '_lr_00005.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True) 
    callbacks_list = [checkpoint, early_stop]

    history = model.fit_generator(
        generator=training_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list
    )
    
    with open(model_name + '_lr_00005.history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    ################################################# lr=0.00001
    model.compile(
        loss=keras.losses.cosine_proximity,
        optimizer=keras.optimizers.Adam(lr=0.00001, clipnorm=1.),
        metrics=['mean_squared_error']
    )

    checkpoint = ModelCheckpoint(model_name + '_lr_00001.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True) 
    callbacks_list = [checkpoint, early_stop]

    history = model.fit_generator(
        generator=training_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list
    )

    with open(model_name + '_lr_00001.history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    ################################################# lr=0.000001
    model.compile(
        loss=keras.losses.cosine_proximity,
        optimizer=keras.optimizers.Adam(lr=0.000001, clipnorm=1.),
        metrics=['mean_squared_error']
    )

    checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True) 
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

    with open(model_name + '.emb.pickle', 'wb') as f:
        save = {
            'embedding_lookup': np.asarray(model.get_layer('embedding').get_weights()[0])
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        del save

    print('*' * 117)
    print(f'Total execution time:{(time.perf_counter()-time_start)/60:.2f} min')
    print('*' * 117)   
    
def LYRICS_RNN(num_classes, emb_size):
    model = Sequential() 
    model.add(layers.Embedding(emb_size, output_dim=200, input_length=None, name='embedding')) 
    model.add(layers.CuDNNLSTM(units=256, return_sequences=True, name='rnn_layer_1')) 
    model.add(layers.CuDNNLSTM(units=512, return_sequences=True, name='rnn_layer_2')) 
    model.add(layers.CuDNNLSTM(units=1024, return_sequences=False, name='rnn_out')) 
    model.add(layers.Dense(num_classes, name='logits')) 
    return model

if __name__ == '__main__':
    K.clear_session()
    SPOTIFY = SPOTIFY.SPOTIFY(max_words=float(args.max_words), threshold=float(args.threshold))
    model = LYRICS_RNN(num_classes=SPOTIFY.EMB_DIM, emb_size=int(SPOTIFY.MAX_WORDS + 2)) # + 1 for empy words (0) and infr words (MAX_WORDS + 1)
    model.summary() 
    train_model(model=model, model_name='Lyrics', max_words=args.max_words, threshold=args.threshold, quick=args.quick)