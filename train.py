import time as tm
import numpy as np

import keras
from keras.models import Model, load_model
from keras import layers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import to_categorical

import h5py
import pickle
import argparse

import FMA

parser = argparse.ArgumentParser(description="trains a model")
parser.add_argument('-d', '--dataset', required=True, help='dataset to use: fma_med')
parser.add_argument('-t', '--test', required=True, help='test to carry out: sgc')
parser.add_argument('-f', '--features', required=True, help='which features to use: stft, stft_halved, mel_scaled_stft')
parser.add_argument('-q', '--quick', default=False, help='runs each test quickly to ensure they will run')
args = parser.parse_args()

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_classes, features, dataset, n_channels=1, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes

        self.features = features
        self.dataset = dataset

        if self.features == 'stft_halved':
            self.data_path = './Data/features/' + dataset + '_stft.hdf5'
            self.halved = True
        else: 
            self.data_path = './Data/features/' + dataset + '_' + features + '.hdf5'
            self.halved = False

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
        y = np.empty((self.batch_size), dtype=int)

        with h5py.File(self.data_path) as f:
            for i, ID in enumerate(list_IDs_temp):
                if self.halved:
                    X[i,] = f['data'][str(ID)][0:1024,:]
                else:
                    X[i,] = f['data'][str(ID)]
                y[i] = self.labels[ID]
            
        return X.reshape(X.shape[0], *self.dim, 1), keras.utils.to_categorical(y, num_classes=self.n_classes)

def train_model(model, model_name, dim, features, dataset, quick):
    try:
        if quick:
            model_name = './Models/' + 'DELETE.' + dataset + '.' + features + '.' + model_name
        else:
            model_name = './Models/' + dataset + '.' + features + '.' + model_name

        print('*' * 117)
        print(f'Training model: {model_name}')
        print('*' * 117)

        time_start = tm.perf_counter()

        if dataset == 'fma_med':
            train_list = FMA.PARTITION['training']
            val_list = FMA.PARTITION['validation']
            labels = FMA.TOP_GENRES
            num_classes = FMA.NUM_CLASSES

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['acc']
        )

        checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_acc', patience=20, mode='max') 
        callbacks_list = [checkpoint, early_stop]

        if quick:
            epochs = 1
            train_list = train_list[:64]
            val_list = train_list[:64]
        else:
            epochs = 100

        training_generator = DataGenerator(
            list_IDs=train_list,
            labels=labels,
            batch_size=8,
            dim=dim,
            n_classes=num_classes,
            features=features,
            dataset=dataset
        )

        vbs = 1
        for i in range(1,17):
            if len(val_list) % i == 0:
                vbs = i

        val_generator = DataGenerator(
            list_IDs=val_list,
            labels=labels,
            batch_size=vbs,
            dim=dim,
            n_classes=num_classes,
            features=features,
            dataset=dataset,
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

def Simple(features, input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    ############## INPUT LAYER ##############
    x = layers.Conv2D(8, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='input1_1')(inputs)
    x = layers.BatchNormalization(name='input1_1_bn')(x)
    x = layers.Activation('relu', name='input1_1_relu')(x)

    if features == 'stft' or features == 'stft_halved':
        x = layers.Conv2D(16, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='input1_2')(x)
        x = layers.BatchNormalization(name='input1_2_bn')(x)
        x = layers.Activation('relu', name='input1_2_relu')(x)
    
    if features == 'mel_scaled_stft':
        x = layers.Conv2D(16, (3, 3), strides=(1, 1), use_bias=False, name='input1_2')(x)
        x = layers.BatchNormalization(name='input1_2_bn')(x)
        x = layers.Activation('relu', name='input1_2_relu')(x)

    ############## HIDDEN LAYER 1  ##############
    x = layers.SeparableConv2D(32, (3, 3), use_bias=False, name='block1_1')(x)
    x = layers.BatchNormalization(name='block1_1_bn')(x)
    x = layers.Activation('relu', name='block1_1_relu')(x)

    x = layers.SeparableConv2D(64, (3, 3), use_bias=False, name='block1_2')(x)
    x = layers.BatchNormalization(name='block1_2_bn')(x)
 
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block1_mp')(x)
 
    x = layers.Activation('relu', name='block1_2_relu')(x)
    
    ############## HIDDEN LAYER 2  ##############
    x = layers.SeparableConv2D(128, (3, 3), use_bias=False, name='block2')(x)
    x = layers.BatchNormalization(name='block2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_mp')(x)

    x = layers.Activation('relu', name='block2_relu')(x)
    
    ############## HIDDEN LAYER 3  ##############
    x = layers.SeparableConv2D(256, (3, 3), use_bias=False, name='block3')(x)
    x = layers.BatchNormalization(name='block3_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_mp')(x)

    x = layers.Activation('relu', name='block3_relu')(x)
    
    ############## HIDDEN LAYER 4  ##############
    x = layers.SeparableConv2D(512, (3, 3), use_bias=False, name='block4')(x)
    x = layers.BatchNormalization(name='block4_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_mp')(x)

    x = layers.Activation('relu', name='block4_relu')(x)
    
    ############## OUTPUT LAYER ##############
    x = layers.SeparableConv2D(1024, (3, 3), use_bias=False, name='out')(x)
    x = layers.BatchNormalization(name='out_bn')(x)
    x = layers.Activation('relu', name='out_relu')(x)
        
    x = layers.GlobalAveragePooling2D(name='GAP')(x)

    x = layers.Dense(num_classes, name='logits')(x)
    
    pred = layers.Activation('softmax', name='softmax')(x)

    return Model(inputs=inputs, outputs=pred)

def Time(iks, input_shape, num_classes):
    time = input_shape[1]

    inputs = layers.Input(shape=input_shape)

    ############## INPUT LAYER ##############
    w1 = layers.Conv2D(2, (1, time//iks[0]), strides=(1, 2), use_bias=False, padding='same', name='input_over_' + str(iks[0]))(inputs)
    w1 = layers.BatchNormalization(name='input_over_' + str(iks[0]) + '_bn')(w1)
    w1 = layers.Activation('relu', name='input_over_' + str(iks[0]) + '_relu')(w1)

    w2 = layers.Conv2D(2, (1, time//iks[1]), strides=(1, 2), use_bias=False, padding='same', name='input_over_' + str(iks[1]))(inputs)
    w2 = layers.BatchNormalization(name='input_over_' + str(iks[1]) + '_bn')(w2)
    w2 = layers.Activation('relu', name='input_over_' + str(iks[1]) + '_relu')(w2)
    
    w3 = layers.Conv2D(2, (1, time//iks[2]), strides=(1, 2), use_bias=False, padding='same', name='input_over_' + str(iks[2]))(inputs)
    w3 = layers.BatchNormalization(name='input_over_' + str(iks[2]) + '_bn')(w3)
    w3 = layers.Activation('relu', name='input_over_' + str(iks[2]) + '_relu')(w3)
    
    w4 = layers.Conv2D(2, (1, time//iks[3]), strides=(1, 2), use_bias=False, padding='same', name='input_over_' + str(iks[3]))(inputs)
    w4 = layers.BatchNormalization(name='input_over_' + str(iks[3]) + '_bn')(w4)
    w4 = layers.Activation('relu', name='input_over_' + str(iks[3]) + '_relu')(w4)

    w5 = layers.Conv2D(2, (1, time//iks[4]), strides=(1, 2), use_bias=False, padding='same', name='input_over_' + str(iks[4]))(inputs)
    w5 = layers.BatchNormalization(name='input_over_' + str(iks[4]) + '_bn')(w5)
    w5 = layers.Activation('relu', name='input_over_' + str(iks[4]) + '_relu')(w5)

    w6 = layers.Conv2D(2, (1, time//iks[5]), strides=(1, 2), use_bias=False, padding='same', name='input_over_' + str(iks[5]))(inputs)
    w6 = layers.BatchNormalization(name='input_over_' + str(iks[5]) + '_bn')(w6)
    w6 = layers.Activation('relu', name='input_over_' + str(iks[5]) + '_relu')(w6)
    
    x = layers.concatenate([w1, w2, w3, w4, w5, w6], axis=3, name='inputs')

    ############## HIDDEN LAYER 1  ##############
    x = layers.SeparableConv2D(16, (1, 9), use_bias=False, name='block1')(x)
    x = layers.BatchNormalization(name='block1_bn')(x)

    x = layers.MaxPooling2D((2, 1), strides=(2, 1), padding='same', name='block1_mp_freq')(x)

    x = layers.Activation('relu', name='block1_relu')(x)

    ############## HIDDEN LAYER 2  ##############
    x = layers.SeparableConv2D(32, (1, 9), use_bias=False, name='block2')(x)
    x = layers.BatchNormalization(name='block2_bn')(x)

    x = layers.MaxPooling2D((2, 1), strides=(2, 1), padding='same', name='block2_mp_freq')(x)

    x = layers.Activation('relu', name='block2_relu')(x)

    ############## HIDDEN LAYER 3  ##############
    x = layers.SeparableConv2D(64, (1, 9), use_bias=False, name='block3')(x)
    x = layers.BatchNormalization(name='block3_bn')(x)

    x = layers.MaxPooling2D((2, 1), strides=(2, 1), padding='same', name='block3_mp_freq')(x)

    x = layers.Activation('relu', name='block3_relu')(x)
    
    ############## OUTPUT LAYER ##############
    x = layers.SeparableConv2D(128, (1, 9), use_bias=False, name='preflat')(x)
    x = layers.BatchNormalization(name='preflat_bn')(x)
    x = layers.Activation('relu', name='preflat_relu')(x)

    x = layers.SeparableConv2D(256, (int(x.shape[1]), 1), use_bias=False, name='freq_flat')(x)
    x = layers.BatchNormalization(name='freq_flat_bn')(x)
    x = layers.Activation('relu', name='freq_flat_relu')(x)   
        
    x = layers.GlobalAveragePooling2D(name='GAP')(x)

    x = layers.Dense(num_classes, name='logits')(x)
    
    pred = layers.Activation('softmax', name='softmax')(x)

    return Model(inputs=inputs, outputs=pred)

def Freq(iks, input_shape, num_classes):
    freq = input_shape[0]
    
    inputs = layers.Input(shape=input_shape)

    ############## INPUT LAYER ##############
    h1 = layers.Conv2D(2, (freq//iks[0], 1), strides=(2, 1), use_bias=False, padding='same', name='input_over_' + str(iks[0]))(inputs)
    h1 = layers.BatchNormalization(name='input_over_' + str(iks[0]) + '_bn')(h1)
    h1 = layers.Activation('relu', name='input_over_' + str(iks[0]) + '_relu')(h1)
    
    h2 = layers.Conv2D(2, (freq//iks[1], 1), strides=(2, 1), use_bias=False, padding='same', name='input_over_' + str(iks[1]))(inputs)
    h2 = layers.BatchNormalization(name='input_over_' + str(iks[1]) + '_bn')(h2)
    h2 = layers.Activation('relu', name='input_over_' + str(iks[1]) + '_relu')(h2)
    
    h3 = layers.Conv2D(2, (freq//iks[2], 1), strides=(2, 1), use_bias=False, padding='same', name='input_over_' + str(iks[2]))(inputs)
    h3 = layers.BatchNormalization(name='input_over_' + str(iks[2]) + '_bn')(h3)
    h3 = layers.Activation('relu', name='input_over_' + str(iks[2]) + '_relu')(h3)
    
    h4 = layers.Conv2D(2, (freq//iks[3], 1), strides=(2, 1), use_bias=False, padding='same', name='input_over_' + str(iks[3]))(inputs)
    h4 = layers.BatchNormalization(name='input_over_' + str(iks[3]) + '_bn')(h4)
    h4 = layers.Activation('relu', name='input_over_' + str(iks[3]) + '_relu')(h4)

    h5 = layers.Conv2D(2, (freq//iks[4], 1), strides=(2, 1), use_bias=False, padding='same', name='input_over_' + str(iks[4]))(inputs)
    h5 = layers.BatchNormalization(name='input_over_' + str(iks[4]) + '_bn')(h5)
    h5 = layers.Activation('relu', name='input_over_' + str(iks[4]) + '_relu')(h5)

    h6 = layers.Conv2D(2, (freq//iks[5], 1), strides=(2, 1), use_bias=False, padding='same', name='input_over_' + str(iks[5]))(inputs)
    h6 = layers.BatchNormalization(name='input_over_' + str(iks[5]) + '_bn')(h6)
    h6 = layers.Activation('relu', name='input_over_' + str(iks[5]) + '_relu')(h6)
    
    x = layers.concatenate([h1, h2, h3, h4, h5, h6], axis=3, name='inputs')

    ############## HIDDEN LAYER 1  ##############
    x = layers.SeparableConv2D(16, (9, 1), use_bias=False, name='block1')(x)
    x = layers.BatchNormalization(name='block1_bn')(x)

    x = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same', name='block1_mp_time')(x)

    x = layers.Activation('relu', name='block1_relu')(x)

    ############## HIDDEN LAYER 2  ##############
    x = layers.SeparableConv2D(32, (9, 1), use_bias=False, name='block2')(x)
    x = layers.BatchNormalization(name='block2_bn')(x)

    x = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same', name='block2_mp_time')(x)

    x = layers.Activation('relu', name='block2_relu')(x)

    ############## HIDDEN LAYER 3  ##############
    x = layers.SeparableConv2D(64, (9, 1), use_bias=False, name='block3')(x)
    x = layers.BatchNormalization(name='block3_bn')(x)

    x = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='same', name='block3_mp_time')(x)

    x = layers.Activation('relu', name='block3_relu')(x)

    ############## OUTPUT LAYER ##############
    x = layers.SeparableConv2D(128, (9, 1), use_bias=False, name='preflat')(x)
    x = layers.BatchNormalization(name='preflat_bn')(x)
    x = layers.Activation('relu', name='preflat_relu')(x)

    x = layers.SeparableConv2D(256, (1, int(x.shape[2])), use_bias=False, name='time_flat')(x)
    x = layers.BatchNormalization(name='time_flat_bn')(x)
    x = layers.Activation('relu', name='time_flat_relu')(x)

    x = layers.GlobalAveragePooling2D(name='GAP')(x)

    x = layers.Dense(num_classes, name='logits')(x)
    
    pred = layers.Activation('softmax', name='softmax')(x)

    return Model(inputs=inputs, outputs=pred)
  
def TimeFreq(features, dataset, input_shape,  num_classes, quick):
    if quick:
        time_path = './Models/' + 'DELETE.' + dataset + '.' + features + '.' + 'Time.hdf5'
        freq_path = './Models/' + 'DELETE.' + dataset + '.' + features + '.' + 'Freq.hdf5'
    else:
        time_path = './Models/' + dataset + '.' + features + '.' + 'Time.hdf5'
        freq_path = './Models/' + dataset + '.' + features + '.' + 'Freq.hdf5'

    time_model = load_model(time_path)
    freq_model = load_model(freq_path)

    time_model = Model(inputs=time_model.input, outputs=time_model.get_layer('logits').output)
    freq_model = Model(inputs=freq_model.input, outputs=freq_model.get_layer('logits').output)
    
    for layer in time_model.layers:
        layer.trainable = False
    for layer in freq_model.layers:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    
    t = time_model(inputs)
    f = freq_model(inputs)
    
    x = layers.concatenate([t,f], name='Time_Freq')
    
    x = layers.Dense(256, kernel_regularizer=layers.regularizers.l2(0.002), name='fc_1')(x)
    x = layers.Activation('relu', name='fc_1_relu')(x)
    x = layers.Dropout(0.5, name='fc_1_dropout')(x)
    
    x = layers.Dense(128, kernel_regularizer=layers.regularizers.l2(0.002), name='fc_2')(x)
    x = layers.Activation('relu', name='fc_2_relu')(x)
    x = layers.Dropout(0.5, name='fc_2_dropout')(x)
    
    x = layers.Dense(num_classes, name='logits')(x)
    
    pred = layers.Activation('softmax', name='softmax')(x)
    
    return Model(inputs=inputs, outputs=pred)


if __name__ == '__main__':
    if args.dataset == 'fma_med':
        FMA = FMA.FreeMusicArchive('medium', 22050)
        num_classes = FMA.NUM_CLASSES

    if args.features == 'stft':
        freq, time = 2049, 643  
        dim = (freq, time)
        fiks = [6, 12, 32, 64, 128, 256] # 341, 170, 64, 32, 16, 8
        tiks = [4, 8, 16, 32, 64, 96]    # 160, 80, 40, 20, 10, 6

    if args.features == 'stft_halved':
        freq, time = 2049//2, 643
        dim = (freq, time)
        fiks = [6, 12, 32, 64, 128, 256] # 170, 85, 32, 16, 8, 4
        tiks = [4, 8, 16, 32, 64, 96]    # 160, 80, 40, 20, 10, 6

    if args.features == 'mel_scaled_stft':
        freq, time = 256, 643
        dim = (freq, time)
        fiks = [6, 8, 12, 24, 32, 64] # 42, 32, 21, 10, 8, 4
        tiks = [4, 8, 16, 32, 64, 96] # 160, 80, 40, 20, 10, 6

    ################ Freq ################
    K.clear_session()
    model = Freq(iks=fiks, input_shape=(*dim, 1), num_classes=num_classes)
    model.summary()
    train_model(model=model, model_name='Freq', dim=dim, features=args.features, dataset=args.dataset, quick=args.quick) 

    ################ Time ################
    K.clear_session()
    model = Time(iks=tiks, input_shape=(*dim, 1), num_classes=num_classes)
    model.summary()
    train_model(model=model, model_name='Time', dim=dim, features=args.features, dataset=args.dataset, quick=args.quick)  

    ################ Simple ################
    K.clear_session()
    model = Simple(features=args.features, input_shape=(*dim, 1), num_classes=num_classes)
    model.summary()
    train_model(model=model, model_name='Simple', dim=dim, features=args.features, dataset=args.dataset, quick=args.quick)  

    ################ TimeFreq ################
    K.clear_session()
    model = TimeFreq(features=args.features, dataset=args.dataset, input_shape=(*dim, 1), num_classes=num_classes, quick=args.quick)
    model.summary()
    train_model(model=model, model_name='TimeFreq', dim=dim, features=args.features, dataset=args.dataset, quick=args.quick)       