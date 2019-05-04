import pickle
import pandas as pd
import numpy as np
import random
import time

import keras
from keras.models import Model, load_model
from keras import layers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from keras.applications.mobilenet_v2 import MobileNetV2
from PIL import Image

import SPOTIFY

import argparse

parser = argparse.ArgumentParser(description="trains a model")
parser.add_argument('-q', '--quick', default=False, help='runs each test quickly to ensure they will run')
args = parser.parse_args()

class DataGenerator(keras.utils.Sequence):
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

def train_model(model_name, quick):
    print('*' * 117)
    print(f'Training model: {model_name}')
    print('*' * 117)

    base_model = MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(640, 640, 3),
        pooling='avg'
    )

    output = layers.Dense(800, activation='linear')(base_model.output)
    
    model = Model(base_model.input, output)

    model.summary() 

    time_start = time.perf_counter()
    
    if quick:
        model_name = './Models/spotify/DELETE.' + model_name
    else:
        model_name = './Models/spotify/' + model_name

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
        batch_size=6
    )

    val_generator = DataGenerator(
        track_ids=val_list,
        SPOTIFY=SPOTIFY,
        batch_size=6
    )

    ################################################# lr=0.00001
    model.compile(
        loss=keras.losses.cosine_proximity,
        optimizer=keras.optimizers.Adam(lr=0.00001),
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
        optimizer=keras.optimizers.Adam(lr=0.000001),
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

    print('*' * 117)
    print(f'Total execution time:{(time.perf_counter()-time_start)/60:.2f} min')
    print('*' * 117)   
   
if __name__ == '__main__':
    K.clear_session()
    SPOTIFY = SPOTIFY.SPOTIFY()
    train_model(model_name='Albums', quick=args.quick)