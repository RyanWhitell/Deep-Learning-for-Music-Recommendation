import os

import numpy as np
import sklearn
from sklearn import metrics
import pandas as pd

import keras
from keras.models import load_model
from keras import backend as K
from keras.utils import to_categorical

import h5py
import pickle

import argparse

import warnings
warnings.filterwarnings("ignore")

import FMA

parser = argparse.ArgumentParser(description="gets testing results from a model")
parser.add_argument('-d', '--dataset', required=True, help='dataset to use: fma_med')
parser.add_argument('-t', '--test', default='', help='test to carry out: single genre classification (sgc)')
args = parser.parse_args()

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
        
        y = np.empty((self.batch_size), dtype=int)
        
        for i, ID in enumerate(list_IDs_temp):
            y[i] = self.labels[ID]
            
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

def test_model_sgc(model_name, dataset):
    K.clear_session()
    
    model_path = './Models/ens/sgc/' + dataset + '.' + model_name + '.hdf5'

    model = load_model(model_path)

    if dataset == 'fma_med':
        test_list = FMA.PARTITION['test']
        labels = FMA.TOP_GENRES
        num_classes = FMA.NUM_CLASSES

        tbs = 1
        for i in range(1,17):
            if len(test_list) % i == 0:
                tbs = i
        
        test_generator = DataGenerator(
            list_IDs=test_list,
            labels=labels,
            batch_size=tbs,
            n_classes=num_classes,
            dataset=dataset,
            test_type='sqc',
            shuffle=False
        )

        softmax_out = model.predict_generator(
            generator=test_generator,
            verbose=1
        )


    predicted = []
    for pred in softmax_out:
        predicted.append(pred.argmax())
    
    predicted = np.array(predicted)

    acc = sum(predicted == y_true) / len(predicted)
    f1_macro = metrics.f1_score(y_true, predicted, average='macro')
    f1_micro = metrics.f1_score(y_true, predicted, average='micro')
    f1_weighted = metrics.f1_score(y_true, predicted, average='weighted')
    
    print(f'{model_path}\nacc {acc:.5f} f1_macro {f1_macro:.5f} f1_micro {f1_micro:.5f} f1_weighted {f1_weighted:.5f}\n')
    
    return softmax_out, acc, f1_macro, f1_micro, f1_weighted

if __name__ == '__main__':
    if args.dataset == 'fma_med':
        FMA = FMA.FreeMusicArchive('medium', 22050)

        y_true = []
        for i in FMA.PARTITION['test']:
            y_true.append(FMA.TOP_GENRES[i])

        y_true = np.array(y_true)    
        y_true_oh = keras.utils.to_categorical(y_true, num_classes=FMA.NUM_CLASSES)

    else:
        raise Exception('Wrong dataset!')


    results = pd.DataFrame({
        'name':['ENS']
    })

    hist_path = './Models/ens/' + args.test + '/' + args.dataset + '.' + 'ENS' + '.history.pkl'
    print(hist_path)
    hist = pickle.load(open(hist_path, "rb" ))

    results['hist_acc'] = [hist['categorical_accuracy']]
    results['hist_val_acc'] = [hist['val_categorical_accuracy']]
    results['epochs'] = len(hist['categorical_accuracy'])
    
    function = {'sgc':test_model_sgc}

    sm, acc, macro, micro, weighted = function[args.test](model_name='ENS', dataset=args.dataset)

    results['accuracy'] = acc 
    results['f1_macro'] = macro
    results['f1_micro'] = micro
    results['f1_weighted'] = weighted
    results['softmax_out'] = [sm]

    save_path = './Results/ens/' + args.test + '/' + args.dataset + '.pkl'

    results.to_pickle(save_path)