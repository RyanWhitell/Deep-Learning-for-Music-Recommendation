import os

import h5py
import time
import multiprocessing
from tqdm import tqdm
import sklearn

import pandas as pd
import ast

import librosa as lr
import numpy as np

import argparse

import FMA

parser = argparse.ArgumentParser(description="extracts features")
parser.add_argument('-d', '--dataset', required=True, help='dataset to use: fma_med')
parser.add_argument('-f', '--features', required=True, help='which features to extract: stft, mel_scaled_stft')
parser.add_argument('-q', '--quick', default=False, help='runs each extraction quickly to ensure they will extract')
args = parser.parse_args()

def get_fma_stft(track_id):
    scaler = sklearn.preprocessing.StandardScaler()

    sr=22050
    n_fft=4096
    hop_length=1024
    win_length=4096
    window='hann'

    tid_str = '{:06d}'.format(track_id)
    file_path = os.path.join('Data/fma_large', tid_str[:3], tid_str + '.mp3')

    y, _ = lr.load(path=file_path, sr=sr)
    
    spectrum = lr.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    
    return track_id, scaler.fit_transform(lr.amplitude_to_db(np.abs(spectrum[:,:643])))

def extract_fma_stft(ids, fma_set, quick):
    if quick:
        ids = ids[:100]
        f = h5py.File('./Data/features/DELETE.fma_' + fma_set + '_stft.hdf5', 'a')
    else:
        f = h5py.File('./Data/features/fma_' + fma_set + '_stft.hdf5', 'a')

    data = f.create_group('data')
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for i, spec in tqdm(pool.imap_unordered(get_fma_stft, ids), total=len(ids)):
        data[str(i)] = spec

    f.close()


def get_fma_mel_scaled_stft(track_id):
    scaler = sklearn.preprocessing.StandardScaler()

    sr=22050
    n_fft=4096
    hop_length=1024
    win_length=4096
    window='hann'
    
    tid_str = '{:06d}'.format(track_id)
    file_path = os.path.join('Data/fma_large', tid_str[:3], tid_str + '.mp3')

    y, _ = lr.load(path=file_path, sr=sr)
    
    mel_spec = lr.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=256)
    
    return track_id, scaler.fit_transform(mel_spec[:,:643])

def extract_fma_mel_scaled_stft(ids, fma_set, quick):
    if quick:
        ids = ids[:100]
        f = h5py.File('./Data/features/DELETE.fma_' + fma_set + '_mel_scaled_stft.hdf5', 'a')
    else:
        f = h5py.File('./Data/features/fma_' + fma_set + '_mel_scaled_stft.hdf5', 'a')

    data = f.create_group('data')
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for i, spec in tqdm(pool.imap_unordered(get_fma_mel_scaled_stft, ids), total=len(ids)):
        data[str(i)] = spec

    f.close()


def get_fma_cqt(track_id):
    scaler = sklearn.preprocessing.StandardScaler()

    sr=22050
    #?n_fft=4096
    hop_length=1024
    #?win_length=4096
    window='hann'
    
    tid_str = '{:06d}'.format(track_id)
    file_path = os.path.join('Data/fma_large', tid_str[:3], tid_str + '.mp3')

    y, _ = lr.load(path=file_path, sr=sr)
    
    cqt = np.abs(lr.core.cqt(y=y, sr=sr, hop_length=hop_length, window=window, n_bins=84*2, bins_per_octave=12*2))
    
    return track_id, scaler.fit_transform(lr.amplitude_to_db(cqt[:,:643]))

def extract_fma_cqt(ids, fma_set, quick):
    if quick:
        ids = ids[:100]
        f = h5py.File('./Data/features/DELETE.fma_' + fma_set + '_cqt.hdf5', 'a')
    else:
        f = h5py.File('./Data/features/fma_' + fma_set + '_cqt.hdf5', 'a')

    data = f.create_group('data')
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for i, spec in tqdm(pool.imap_unordered(get_fma_cqt, ids), total=len(ids)):
        data[str(i)] = spec

    f.close()




if __name__=='__main__':
    print('File Start...')
    file_start = time.perf_counter()

    if args.dataset == 'fma_med':
        FMA = FMA.FreeMusicArchive('medium', 22050)
        ids = FMA.TRACKS.index.values

        if args.features == 'stft':
            extract_fma_stft(ids, 'med', args.quick)
        if args.features == 'mel_scaled_stft':
            extract_fma_mel_scaled_stft(ids, 'med', args.quick)
        if args.features == 'cqt':
            extract_fma_cqt(ids, 'med', args.quick)

    print(f'Total file execution time: {time.perf_counter()-file_start:.2f}s')