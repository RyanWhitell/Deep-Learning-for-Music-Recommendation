import time as tm
import numpy as np

import keras
from keras.models import Model, load_model
from keras import layers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

from keras.datasets import cifar100

import h5py
import pickle
import argparse

import FMA
import SPOTIFY

parser = argparse.ArgumentParser(description="trains a model")
parser.add_argument('-d', '--dataset', required=True, help='dataset to use: fma_med, cifar100')
parser.add_argument('-t', '--test', default='', help='test to carry out: single genre classification (sgc). multi genre classification (mgc)')
parser.add_argument('-f', '--features', default='', help='which features to use: stft, stft_halved, mel_scaled_stft, cqt, chroma, mfcc')
parser.add_argument('-q', '--quick', default=False, help='runs each test quickly to ensure they will run')
args = parser.parse_args()

"""
Valid test combinations:
fma_med -> sgc -> stft, stft_halved, mel_scaled_stft, cqt, chroma, mfcc
fma_large -> mgc -> stft, stft_halved, mel_scaled_stft, cqt, chroma, mfcc
"""

class DataGeneratorSPOTIFY(keras.utils.Sequence):
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

class DataGeneratorFMA(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_classes, features, dataset, test_type, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.features = features
        self.dataset = dataset
        self.data_path = './Data/features/' + dataset + '_' + features + '.hdf5'
        self.test_type = test_type
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

        if self.test_type == 'sgc':
            y = np.empty((self.batch_size), dtype=int)
        if self.test_type == 'mgc':
            y = np.empty((self.batch_size, self.n_classes), dtype=int)

        with h5py.File(self.data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = f['data'][str(ID)]
                if self.test_type == 'sgc':
                    y[i] = self.labels[ID]
                if self.test_type == 'mgc':
                    y[i,:] = self.labels[ID]
            
        if self.test_type == 'sgc':
            return X.reshape(X.shape[0], *self.dim), keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.test_type == 'mgc':
            return X.reshape(X.shape[0], *self.dim), y

def train_model(model, model_name, dim, features, dataset, test_type, quick):
    try:
        if quick:
            model_name = './Models/rnn/' + test_type + '/' + 'DELETE.' + dataset + '.' + features + '.' + model_name
        else:
            model_name = './Models/rnn/' + test_type + '/' + dataset + '.' + features + '.' + model_name

        print('*' * 117)
        print(f'Training model: {model_name}')
        print('*' * 117)

        time_start = tm.perf_counter()

        if dataset in ['fma_med', 'fma_large']:
            train_list = FMA.PARTITION['training']
            val_list = FMA.PARTITION['validation']
            
            if dataset == 'fma_med':
                labels = FMA.TOP_GENRES
            else:
                labels = FMA.ALL_GENRES

            num_classes = FMA.NUM_CLASSES

            if test_type == 'sgc':
                model.compile(
                    loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['categorical_accuracy']
                )
                checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
                early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=20, mode='max') 
                callbacks_list = [checkpoint, early_stop]
            elif test_type == 'mgc':
                model.compile(
                    loss=keras.losses.binary_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['categorical_accuracy', 'binary_accuracy', 'mean_squared_error']
                )
                checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min') 
                callbacks_list = [checkpoint, early_stop]
            else:
                raise Exception('Unknown test type!')

            if quick:
                epochs = 1
                train_list = train_list[:64]
                val_list = train_list[:64]
            else:
                epochs = 500

            training_generator = DataGeneratorFMA(
                list_IDs=train_list,
                labels=labels,
                batch_size=8,
                dim=dim,
                n_classes=num_classes,
                features=features,
                dataset=dataset,
                test_type=test_type
            )

            vbs = 1
            for i in range(1,17):
                if len(val_list) % i == 0:
                    vbs = i

            val_generator = DataGeneratorFMA(
                list_IDs=val_list,
                labels=labels,
                batch_size=vbs,
                dim=dim,
                n_classes=num_classes,
                features=features,
                dataset=dataset,
                test_type=test_type
            )

            history = model.fit_generator(
                generator=training_generator,
                validation_data=val_generator,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks_list
            )

        elif dataset == 'spotify':
            train_list = list(SPOTIFY.DATA.loc[SPOTIFY.DATA.split == 'train'].index.values)
            val_list = list(SPOTIFY.DATA.loc[SPOTIFY.DATA.split == 'val'].index.values)
            labels = labels = SPOTIFY.DATA[['embedding_vector']].to_dict()['embedding_vector']
            emb_dim = SPOTIFY.EMB_DIM

            if quick:
                epochs = 1
                train_list = train_list[:64]
                val_list = train_list[:64]
            else:
                epochs = 500
    
            training_generator = DataGeneratorSPOTIFY(
                list_IDs=train_list,
                labels=labels,
                emb_dim=emb_dim,
                batch_size=8,
                dim=dim,
                features=features,
            )

            vbs = 1
            for i in range(1,17):
                if len(val_list) % i == 0:
                    vbs = i

            val_generator = DataGeneratorSPOTIFY(
                list_IDs=val_list,
                labels=labels,
                emb_dim=emb_dim,
                batch_size=vbs,
                dim=dim,
                features=features
            )

            ################################################# lr = 0.00005
            if test_type == 'cos':
                model.compile(
                    loss=keras.losses.cosine_proximity,
                    optimizer=keras.optimizers.Adam(lr = 0.00005, clipnorm=1.),
                    metrics=['mean_squared_error']
                )
            elif test_type == 'mse':
                model.compile(
                    loss=keras.losses.mean_squared_error,
                    optimizer=keras.optimizers.Adam(lr = 0.00005, clipnorm=1.),
                    metrics=['cosine_proximity']
                )
            else:
                raise Exception('Unknown test type!')

            checkpoint = ModelCheckpoint(model_name + '_pre_train_lr_00005.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True) 
            callbacks_list = [checkpoint, early_stop]

            history = model.fit_generator(
                generator=training_generator,
                validation_data=val_generator,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks_list
            )

            with open(model_name + '_pre_train_lr_00005.history.pkl', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

            ################################################# lr = 0.00001
            if test_type == 'cos':
                model.compile(
                    loss=keras.losses.cosine_proximity,
                    optimizer=keras.optimizers.Adam(lr = 0.00001, clipnorm=1.),
                    metrics=['mean_squared_error']
                )
            elif test_type == 'mse':
                model.compile(
                    loss=keras.losses.mean_squared_error,
                    optimizer=keras.optimizers.Adam(lr = 0.00001, clipnorm=1.),
                    metrics=['cosine_proximity']
                )
            else:
                raise Exception('Unknown test type!')

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
        print(f'Total execution time:{(tm.perf_counter()-time_start)/60:.2f} min')
        print('*' * 117)   

    except Exception as e:
        print('!' * 117)
        print(f'{model_name} EXCEPTION')
        print(e)
        print('!' * 117)

def RNN(input_shape, test_type, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.CuDNNGRU(units=256, return_sequences=True, name='rnn_layer_1')(inputs)

    x = layers.CuDNNGRU(units=256, return_sequences=True, name='rnn_layer_2')(x)

    x = layers.CuDNNGRU(units=256, return_sequences=False, name='rnn_out')(x)

    x = layers.Dense(num_classes, name='logits')(x)
    
    if test_type == 'sgc':
        output_activation = 'softmax'
    elif test_type == 'mgc':
        output_activation = 'sigmoid'
    elif test_type in ['cos', 'mse']:
        output_activation = 'linear'
    
    pred = layers.Activation(output_activation, name=output_activation)(x)

    return  Model(inputs=inputs, outputs=pred)

def RNN_LARGE(input_shape, test_type, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.CuDNNLSTM(units=256, return_sequences=True, name='rnn_layer_1')(inputs)

    x = layers.CuDNNLSTM(units=512, return_sequences=True, name='rnn_layer_2')(x)

    x = layers.CuDNNLSTM(units=1024, return_sequences=False, name='rnn_out')(x)

    x = layers.Dense(num_classes, name='logits')(x)
    
    if test_type == 'sgc':
        output_activation = 'softmax'
    elif test_type == 'mgc':
        output_activation = 'sigmoid'
    elif test_type in ['cos', 'mse']:
        output_activation = 'linear'
    
    pred = layers.Activation(output_activation, name=output_activation)(x)

    return  Model(inputs=inputs, outputs=pred)


if __name__ == '__main__':

    if args.dataset in ['fma_med', 'fma_large', 'spotify']:
        if args.dataset == 'fma_med':
            FMA = FMA.FreeMusicArchive('medium', 22050)
            num_classes = FMA.NUM_CLASSES

        elif args.dataset == 'fma_large':
            FMA = FMA.FreeMusicArchive('large', 22050)
            num_classes = FMA.NUM_CLASSES

        elif args.dataset == 'spotify':
            SPOTIFY = SPOTIFY.SPOTIFY()
            num_classes = SPOTIFY.EMB_DIM

        if args.features == 'stft':
            freq, time = 2049, 643  
            dim = (time, freq)

        elif args.features == 'stft_halved':
            freq, time = 2049//2, 643
            dim = (time, freq)

        elif args.features == 'mel_scaled_stft':
            freq, time = 256, 643
            dim = (time, freq)

        elif args.features == 'cqt':
            freq, time = 168, 643
            dim = (time, freq)

        elif args.features in ['chroma', 'mfcc']:
            freq, time = 12, 643
            dim = (time, freq)

    else:
        raise Exception('Wrong dataset/feature combination!')

    if args.dataset in ['fma_med', 'fma_large']:
        ################## RNN ################
        K.clear_session()
        model = RNN(input_shape=dim, test_type=args.test, num_classes=num_classes)
        model.summary()
        train_model(model=model, model_name='RNN', dim=dim, features=args.features, dataset=args.dataset, test_type=args.test, quick=args.quick) 
    
    if args.dataset == 'spotify':
        ################# RNN_LARGE ################
        K.clear_session()
        model = RNN_LARGE(input_shape=dim, test_type=args.test, num_classes=num_classes)
        model.summary()
        train_model(model=model, model_name='RNN_LARGE', dim=dim, features=args.features, dataset=args.dataset, test_type=args.test, quick=args.quick) 