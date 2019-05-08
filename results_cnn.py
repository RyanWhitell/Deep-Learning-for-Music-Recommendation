import os

import numpy as np
import sklearn
from sklearn import metrics
import pandas as pd

import keras
from keras.models import load_model
from keras import backend as K
from keras.utils import to_categorical

from keras.datasets import cifar100

import h5py
import pickle

import argparse

import warnings
warnings.filterwarnings("ignore")

import FMA

parser = argparse.ArgumentParser(description="gets testing results from a model")
parser.add_argument('-d', '--dataset', required=True, help='dataset to use: fma_med, fma_large, cifar100')
parser.add_argument('-t', '--test', default='', help='test to carry out: single genre classification (sgc). multi genre classification (mgc)')
parser.add_argument('-f', '--features', default='', help='features to use: stft, stft_halved, mel_scaled_stft, cqt, chroma, mfcc')
args = parser.parse_args()

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_classes, features, dataset, test_type, n_channels=1, shuffle=True):
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
            return X.reshape(X.shape[0], *self.dim, 1), keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.test_type == 'mgc':
            return X.reshape(X.shape[0], *self.dim, 1), y

def test_model_sgc(model_name, dim, features, dataset):
    K.clear_session()
    
    model_path = './Models/cnn/sgc/' + dataset + '.' + features + '.' + model_name + '.hdf5'

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
            dim=dim,
            n_classes=num_classes,
            features=features,
            dataset=dataset,
            test_type='sgc',
            shuffle=False
        )

        softmax_out = model.predict_generator(
            generator=test_generator,
            verbose=1
        )

    elif dataset == 'cifar100':
        softmax_out = model.predict(
            x=CIFAR_X,
            batch_size=10,
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

def test_model_mgc(model_name, dim, features, dataset):
    K.clear_session()
    
    model_path = './Models/cnn/mgc/' + dataset + '.' + features + '.' + model_name + '.hdf5'

    model = load_model(model_path)

    if dataset == 'fma_large':
        test_list = FMA.PARTITION['test']
        labels = FMA.ALL_GENRES
        num_classes = FMA.NUM_CLASSES

        tbs = 1
        for i in range(1,17):
            if len(test_list) % i == 0:
                tbs = i
        
        test_generator = DataGenerator(
            list_IDs=test_list,
            labels=labels,
            batch_size=tbs,
            dim=dim,
            n_classes=num_classes,
            features=features,
            dataset=dataset,
            test_type='mgc',
            shuffle=False
        )

        sigmoid_out = model.predict_generator(
            generator=test_generator,
            verbose=1
        )

    predicted = np.asarray(np.around(sigmoid_out), dtype=np.int)

    hamm = metrics.hamming_loss(y_true, predicted)
    mse = np.mean((y_true - sigmoid_out)**2)

    print(f'{model_path}\nhamm {hamm:.6f} mse {mse:.6f}')
    
    return sigmoid_out, hamm, mse

if __name__ == '__main__':
    if args.dataset in ['fma_med', 'fma_large']:
        if args.features == 'stft':
            freq, time = 2049, 643  
            dim = (freq, time)

        if args.features == 'stft_halved':
            freq, time = 2049//2, 643
            dim = (freq, time)

        if args.features == 'mel_scaled_stft':
            freq, time = 256, 643
            dim = (freq, time)

        if args.features == 'cqt':
            freq, time = 168, 643
            dim = (freq, time)

        if args.features in ['chroma', 'mfcc']:
            freq, time = 12, 643
            dim = (freq, time)
        
        if args.dataset == 'fma_med':
            FMA = FMA.FreeMusicArchive('medium', 22050)

            y_true = []
            for i in FMA.PARTITION['test']:
                y_true.append(FMA.TOP_GENRES[i])

            y_true = np.array(y_true)    
            y_true_oh = keras.utils.to_categorical(y_true, num_classes=FMA.NUM_CLASSES)
        
        if args.dataset == 'fma_large':
            FMA = FMA.FreeMusicArchive('large', 22050)

            y_true = []
            for i in FMA.PARTITION['test']:
                y_true.append(FMA.ALL_GENRES[i])

            y_true = np.array(y_true)
            y_true_oh = y_true

    elif args.dataset == 'cifar100':
        (_, _), (CIFAR_X, y_true) = cifar100.load_data(label_mode='fine')
        y_true_oh =  keras.utils.to_categorical(y_true, num_classes=100)

        y_true = y_true.reshape(-1)

        args.features = 'cifar100'
        num_classes = 100
        freq, time = 32, 32
        dim = (freq, time, 3)
    
    else:
        raise Exception('Wrong dataset!')

    if args.test == 'sgc':

        if args.dataset == 'fma_med':
            results = pd.DataFrame({
                'name':['Time','Freq','Simple','TimeFreq']
            })
        else: 
            results = pd.DataFrame({
                'name':['Time','Freq','Simple']
            })

        hist_acc = []
        hist_val_acc = []

        accuracy = []
        f1_macro = []
        f1_micro = []
        f1_weighted = []
        softmax_out = []

        epochs = []

        for index, row in results.iterrows():
            hist_path = './Models/cnn/' + args.test + '/' + args.dataset + '.' + args.features + '.' + row['name'] + '.history.pkl'
            
            hist=pickle.load(open(hist_path, "rb" ))
            hist_acc.append(hist['acc'])
            hist_val_acc.append(hist['val_acc'])
            epochs.append(len(hist['acc']))

            sm, acc, macro, micro, weighted = test_model_sgc(model_name=row['name'], dim=dim, features=args.features, dataset=args.dataset)
            accuracy.append(acc) 
            f1_macro.append(macro) 
            f1_micro.append(micro) 
            f1_weighted.append(weighted)
            softmax_out.append(sm) 

        results['hist_acc'] = hist_acc 
        results['hist_val_acc'] = hist_val_acc 

        results['accuracy'] = accuracy 
        results['f1_macro'] = f1_macro
        results['f1_micro'] = f1_micro
        results['f1_weighted'] = f1_weighted
        results['softmax_out'] = softmax_out 

        results['epochs'] = epochs

        save_path = './Results/cnn/' + args.test + '/' + args.dataset + '.' + args.features  + '.pkl'

        results.to_pickle(save_path)

    if args.test == 'mgc':
        results = pd.DataFrame({
            'name':['Time','Freq','Simple']
        })

        hist_cat_acc = []
        hist_bin_acc = []
        hist_mse = []
        hist_val_loss = []

        hamm = []
        mse = []
        sigmoid_out = []

        epochs = []

        for index, row in results.iterrows():
            hist_path = './Models/cnn/' + args.test + '/' + args.dataset + '.' + args.features + '.' + row['name'] + '.history.pkl'
            
            hist=pickle.load(open(hist_path, "rb" ))
            hist_cat_acc.append(hist['categorical_accuracy'])
            hist_bin_acc.append(hist['binary_accuracy'])
            hist_mse.append(hist['mean_squared_error'])
            hist_val_loss.append(hist['val_loss'])
            epochs.append(len(hist['categorical_accuracy']))

            sig_out, h, m = test_model_mgc(model_name=row['name'], dim=dim, features=args.features, dataset=args.dataset)
            sigmoid_out.append(sig_out) 
            hamm.append(h) 
            mse.append(m) 

        results['hist_cat_acc'] = hist_cat_acc 
        results['hist_bin_acc'] = hist_bin_acc 
        results['hist_mse'] = hist_mse 
        results['hist_val_loss'] = hist_val_loss

        results['hamm'] = hamm
        results['mse'] = mse
        results['sigmoid_out'] = sigmoid_out 

        results['epochs'] = epochs

        save_path = './Results/cnn/' + args.test + '/' + args.dataset + '.' + args.features  + '.pkl'

        results.to_pickle(save_path)