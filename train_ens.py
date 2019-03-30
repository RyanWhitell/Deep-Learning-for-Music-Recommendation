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

parser = argparse.ArgumentParser(description="trains a model")
parser.add_argument('-d', '--dataset', required=True, help='dataset to use: fma_med, cifar100')
parser.add_argument('-t', '--test', default='', help='test to carry out: single genre classification (sgc). multi genre classification (mgc)')
parser.add_argument('-q', '--quick', default=False, help='runs each test quickly to ensure they will run')
args = parser.parse_args()

"""
Valid test combinations:
fma_med -> sgc
fma_large -> mgc
"""

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, n_classes, dataset, test_type, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dataset = dataset
        self.stft_data_path = './Data/features/' + dataset + '_stft_halved.hdf5'
        self.mel_data_path = './Data/features/' + dataset + '_mel_scaled_stft.hdf5'
        self.cqt_data_path = './Data/features/' + dataset + '_cqt.hdf5'
        self.chroma_data_path = './Data/features/' + dataset + '_chroma.hdf5'
        self.mfcc_data_path = './Data/features/' + dataset + '_mfcc.hdf5'
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
        X_stft   = np.empty((self.batch_size, 1024, 643))
        X_mel    = np.empty((self.batch_size, 256,  643))
        X_cqt    = np.empty((self.batch_size, 168,  643))
        X_chroma = np.empty((self.batch_size, 12,   643))
        X_mfcc   = np.empty((self.batch_size, 12,   643))
        
        if self.test_type == 'sgc':
            y = np.empty((self.batch_size), dtype=int)
        if self.test_type == 'mgc':
            y = np.empty((self.batch_size, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_temp):
            if self.test_type == 'sgc':
                y[i] = self.labels[ID]
            if self.test_type == 'mgc':
                y[i,:] = self.labels[ID]
            
        with h5py.File(self.stft_data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X_stft[i,] = f['data'][str(ID)]
        with h5py.File(self.mel_data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X_mel[i,] = f['data'][str(ID)]
        with h5py.File(self.cqt_data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X_cqt[i,] = f['data'][str(ID)]
        with h5py.File(self.chroma_data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X_chroma[i,] = f['data'][str(ID)]
        with h5py.File(self.mfcc_data_path,'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                X_mfcc[i,] = f['data'][str(ID)]
        
        X_stft   = X_stft.reshape(X_stft.shape[0],     1024, 643, 1)
        X_mel    = X_mel.reshape(X_mel.shape[0],       256,  643, 1)
        X_cqt    = X_cqt.reshape(X_cqt.shape[0],       168,  643, 1)
        X_chroma = X_chroma.reshape(X_chroma.shape[0], 643,  12)
        X_mfcc   = X_mfcc.reshape(X_mfcc.shape[0],     643,  12)

        if self.test_type == 'sgc':
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return list([X_stft, X_mel, X_cqt, X_chroma, X_mfcc]), y

def train_model(model, model_name, dataset, test_type, quick):
    try:
        if quick:
            model_name = './Models/ens/' + test_type + '/' + 'DELETE.' + dataset + '.' + model_name
        else:
            model_name = './Models/ens/' + test_type + '/' + dataset + '.' + model_name

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
                    optimizer=keras.optimizers.Adam(lr=0.0001),
                    metrics=['categorical_accuracy']
                )
            elif test_type == 'mgc':
                model.compile(
                    loss=keras.losses.binary_crossentropy,
                    optimizer=keras.optimizers.Adam(lr=0.0001),
                    metrics=['categorical_accuracy']
                )
            else:
                raise Exception('Unknown test type!')

            checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
            early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=10, mode='max') 
            callbacks_list = [checkpoint, early_stop]

            if quick:
                epochs = 1
                train_list = train_list[:64]
                val_list = train_list[:64]
            else:
                epochs = 500

            training_generator = DataGenerator(
                list_IDs=train_list,
                labels=labels,
                batch_size=8,
                n_classes=num_classes,
                dataset=dataset,
                test_type=test_type
            )

            vbs = 1
            for i in range(1,17):
                if len(val_list) % i == 0:
                    vbs = i

            val_generator = DataGenerator(
                list_IDs=val_list,
                labels=labels,
                batch_size=vbs,
                n_classes=num_classes,
                dataset=dataset,
                test_type=test_type,
                shuffle=False
            )

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

def ENS(dataset, test_type, num_classes, quick):
    if quick:
        stft_time_path  = './Models/cnn/' + test_type + '/' + 'DELETE.' + dataset + '.stft_halved.Time.hdf5'
        mel_time_path   = './Models/cnn/' + test_type + '/' + 'DELETE.' + dataset + '.mel_scaled_stft.Time.hdf5'
        cqt_time_path   = './Models/cnn/' + test_type + '/' + 'DELETE.' + dataset + '.cqt.Time.hdf5'
        
        stft_freq_path  = './Models/cnn/' + test_type + '/' + 'DELETE.' + dataset + '.stft_halved.Freq.hdf5'
        mel_freq_path   = './Models/cnn/' + test_type + '/' + 'DELETE.' + dataset + '.mel_scaled_stft.Freq.hdf5'
        cqt_freq_path   = './Models/cnn/' + test_type + '/' + 'DELETE.' + dataset + '.cqt.Freq.hdf5'
        
        chroma_rnn_path = './Models/rnn/' + test_type + '/' + 'DELETE.' + dataset + '.chroma.RNN.hdf5'
        mfcc_rnn_path   = './Models/rnn/' + test_type + '/' + 'DELETE.' + dataset + '.mfcc.RNN.hdf5'
    else:
        stft_time_path  = './Models/cnn/' + test_type + '/' + dataset + '.stft_halved.Time.hdf5'
        mel_time_path   = './Models/cnn/' + test_type + '/' + dataset + '.mel_scaled_stft.Time.hdf5'
        cqt_time_path   = './Models/cnn/' + test_type + '/' + dataset + '.cqt.Time.hdf5'
        
        stft_freq_path  = './Models/cnn/' + test_type + '/' + dataset + '.stft_halved.Freq.hdf5'
        mel_freq_path   = './Models/cnn/' + test_type + '/' + dataset + '.mel_scaled_stft.Freq.hdf5'
        cqt_freq_path   = './Models/cnn/' + test_type + '/' + dataset + '.cqt.Freq.hdf5'
        
        chroma_rnn_path = './Models/rnn/' + test_type + '/' + dataset + '.chroma.RNN.hdf5'
        mfcc_rnn_path   = './Models/rnn/' + test_type + '/' + dataset + '.mfcc.RNN.hdf5'

    stft_time_model  = load_model(stft_time_path)
    mel_time_model   = load_model(mel_time_path)
    cqt_time_model   = load_model(cqt_time_path)
    stft_freq_model  = load_model(stft_freq_path)
    mel_freq_model   = load_model(mel_freq_path)
    cqt_freq_model   = load_model(cqt_freq_path)
    chroma_rnn_model = load_model(chroma_rnn_path)
    mfcc_rnn_model   = load_model(mfcc_rnn_path)
    
    stft_time_model  = Model(inputs=stft_time_model.input, outputs=stft_time_model.get_layer('logits').output)
    mel_time_model   = Model(inputs=mel_time_model.input, outputs=mel_time_model.get_layer('logits').output)
    cqt_time_model   = Model(inputs=cqt_time_model.input, outputs=cqt_time_model.get_layer('logits').output)
    stft_freq_model  = Model(inputs=stft_freq_model.input, outputs=stft_freq_model.get_layer('logits').output)
    mel_freq_model   = Model(inputs=mel_freq_model.input, outputs=mel_freq_model.get_layer('logits').output)
    cqt_freq_model   = Model(inputs=cqt_freq_model.input, outputs=cqt_freq_model.get_layer('logits').output)
    chroma_rnn_model = Model(inputs=chroma_rnn_model.input, outputs=chroma_rnn_model.get_layer('logits').output)
    mfcc_rnn_model   = Model(inputs=mfcc_rnn_model.input, outputs=mfcc_rnn_model.get_layer('logits').output)
    
    for layer in stft_time_model.layers:
        layer.trainable = False
    for layer in mel_time_model.layers:
        layer.trainable = False
    for layer in cqt_time_model.layers:
        layer.trainable = False
    for layer in stft_freq_model.layers:
        layer.trainable = False
    for layer in mel_freq_model.layers:
        layer.trainable = False
    for layer in cqt_freq_model.layers:
        layer.trainable = False
    for layer in chroma_rnn_model.layers:
        layer.trainable = False
    for layer in mfcc_rnn_model.layers:
        layer.trainable = False

    X_stft   = layers.Input(shape=(1024, 643, 1))
    X_mel    = layers.Input(shape=(256,  643, 1))
    X_cqt    = layers.Input(shape=(168,  643, 1))
    X_chroma = layers.Input(shape=(643,  12))
    X_mfcc   = layers.Input(shape=(643,  12))
    
    stft_t = stft_time_model(X_stft)
    mel_t = mel_time_model(X_mel)
    cqt_t = cqt_time_model(X_cqt)
    
    stft_f = stft_freq_model(X_stft)
    mel_f = mel_freq_model(X_mel)
    cqt_f = cqt_freq_model(X_cqt)
    
    chroma = chroma_rnn_model(X_chroma)
    mfcc = mfcc_rnn_model(X_mfcc)
    
    x = layers.concatenate([stft_t, mel_t, cqt_t, stft_f, mel_f, cqt_f, chroma, mfcc], name='All_Model_Logits')
    
    x = layers.Dense(1024, kernel_regularizer=layers.regularizers.l2(0.002), name='fc_1')(x)
    x = layers.Activation('relu', name='fc_1_relu')(x)
    x = layers.Dropout(0.5, name='fc_1_dropout')(x)
    
    x = layers.Dense(512, kernel_regularizer=layers.regularizers.l2(0.002), name='fc_2')(x)
    x = layers.Activation('relu', name='fc_2_relu')(x)
    x = layers.Dropout(0.5, name='fc_2_dropout')(x)
    
    x = layers.Dense(256, kernel_regularizer=layers.regularizers.l2(0.002), name='fc_3')(x)
    x = layers.Activation('relu', name='fc_3_relu')(x)
    x = layers.Dropout(0.5, name='fc_3_dropout')(x)
    
    x = layers.Dense(num_classes, name='logits')(x)
    
    if test_type == 'sgc':
        output_activation = 'softmax'
    elif test_type == 'mgc':
        output_activation = 'sigmoid'

    pred = layers.Activation(output_activation, name=output_activation)(x)
    
    return Model(inputs=[X_stft, X_mel, X_cqt, X_chroma, X_mfcc], outputs=pred)

if __name__ == '__main__':
    if args.dataset in ['fma_med', 'fma_large']:
        if args.dataset == 'fma_med':
            FMA = FMA.FreeMusicArchive('medium', 22050)
        elif args.dataset == 'fma_large':
            FMA = FMA.FreeMusicArchive('large', 22050)
        num_classes = FMA.NUM_CLASSES

    ################# ENS ################
    K.clear_session()
    model = ENS(dataset=args.dataset, test_type=args.test, num_classes=num_classes, quick=args.quick)
    model.summary()
    train_model(model=model, model_name='ENS', dataset=args.dataset, test_type=args.test, quick=args.quick) 