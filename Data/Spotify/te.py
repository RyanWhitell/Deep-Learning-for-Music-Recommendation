import pickle
import h5py
import numpy as np
import random
import time

from sklearn.neighbors import KNeighborsClassifier

import keras
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers
from keras import backend as K

with open('artist_related.pickle', 'rb') as f:
    save = pickle.load(f)
    artist_related = save['artist_related']
    del save

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, artist_related, batch_size, shuffle=True):
        'Initialization'
        self.artist_ids = list(artist_related.keys())
        self.artist_related = artist_related
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.artist_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        artist_ids_temp = [self.artist_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(artist_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.artist_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, artist_ids_temp):
        'Generates data containing batch_size samples'
    
        artist = []
        context = []
        related = []
        
        for artist_id in artist_ids_temp:
            for related_artist_id in self.artist_related[artist_id]:
                artist.append(artist_id)
                context.append(related_artist_id)
                related.append(1)
            for _ in range(len(self.artist_related[artist_id])):
                artist.append(artist_id)
                context.append(random.choice(self.artist_ids))
                related.append(0)
        
        artist = np.asarray(artist, dtype=np.int).reshape(-1,1)
        context = np.asarray(context, dtype=np.int).reshape(-1,1)
        related = np.asarray(related, dtype=np.int).reshape(-1,1)
        
        return [artist, context], related

def artist_emb_model(emb_dim):
    input_target = layers.Input((1,), name='input_target')
    input_context = layers.Input((1,), name='input_context')

    embedding = layers.Embedding(len(artist_related), emb_dim, input_length=1, name='embedding')

    target = embedding(input_target)
    target = layers.Reshape((emb_dim, 1), name='reshape_target')(target)

    context = embedding(input_context)
    context = layers.Reshape((emb_dim, 1), name='reshape_context')(context)

    similarity_measure = layers.dot([target, context], axes=1, normalize=True, name='cos_similarity')
    similarity_measure = layers.Reshape((1,), name='reshape_cos_similarity')(similarity_measure)
    output = layers.Dense(1, activation='sigmoid', name='sigmoid')(similarity_measure)
    
    return Model(inputs=[input_target, input_context], outputs=output)

def print_acc(emb_table):
    classifier = KNeighborsClassifier(n_neighbors=20)  
    classifier.fit(emb_table, range(len(emb_table))) 
    
    classifier_cos = KNeighborsClassifier(n_neighbors=20, metric='cosine')  
    classifier_cos.fit(emb_table, range(len(emb_table))) 

    artist_range = range(len(artist_related))
    
    id_sim = 0    
    id_sim_cos = 0
    for _ in range(1000):      
        
        a = random.choice(artist_range)
        
        emb_related = classifier.kneighbors(emb_table[a].reshape(1, -1), len(artist_related[a]))
        emb_related_cos = classifier_cos.kneighbors(emb_table[a].reshape(1, -1), len(artist_related[a]))

        spot_id = set(artist_related[a])
        emb_id = set(emb_related[1][0]) 
        emb_id_cos = set(emb_related_cos[1][0])
        
        id_sim += float(len(emb_id & spot_id)) / len(emb_id | spot_id) * 100
        id_sim_cos += float(len(emb_id_cos & spot_id)) / len(emb_id_cos | spot_id) * 100

    mse_loss = 0
    mse_mean = 0
    rand_mse_loss = 0
    rand_mse_mean = 0
    for a, r in artist_related.items():
        related_indices = list(r)
        random_indices = random.sample(artist_range, len(related_indices))
        repeated_artist_indices = [a] * len(related_indices)

        mse_vals = (np.square(emb_table[repeated_artist_indices] - emb_table[related_indices])).mean(axis=1)
        mse_loss += mse_vals.sum(axis=0)
        mse_mean += mse_vals.mean(axis=0)
        
        rand_mse_vals = (np.square(emb_table[repeated_artist_indices] - emb_table[random_indices])).mean(axis=1)
        rand_mse_loss += rand_mse_vals.sum(axis=0)
        rand_mse_mean += rand_mse_vals.mean(axis=0)
        
    print(f'ID Similarity MSE    : {id_sim / 1000:.20f}')
    print(f'ID Similarity COS    : {id_sim_cos / 1000:.20f}')
    print(f'Related Total MSE    : {mse_loss:.20f}')
    print(f'Random  Total MSE    : {rand_mse_loss:.20f}')
    print(f'Total MSE Ratio      : {rand_mse_loss / mse_loss:.20f}')
    print(f'Related Average MSE  : {mse_mean / len(artist_related):.20f}')
    print(f'Random  Average MSE  : {rand_mse_mean / len(artist_related):.20f}')
    print(f'Average MSE Ratio    : {(rand_mse_mean / len(artist_related)) / (mse_mean / len(artist_related)):.20f}')

    with open(model_name + '_temp.emb.pickle', 'wb') as f:
        save = {
            'embedding_lookup': emb_table
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        del save

class PrintAccCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print_acc(self.model.get_layer('embedding').get_weights()[0])

model_name ='embedding_model'
emb_dim  = 800
epochs = 1000

model = artist_emb_model(emb_dim)
model.summary()

############################### patience = 1 : batch_size = 32 : lr = 0.001
checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='loss', patience=1, mode='min', restore_best_weights=True) 
print_acc_callback = PrintAccCallback()

callbacks_list = [checkpoint, early_stop, print_acc_callback]

gen = DataGenerator(
    artist_related=artist_related,
    batch_size=32,
    shuffle=True
)

model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam()
)

model.fit_generator(
    generator=gen,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list
)

############################### patience = 1 : batch_size = 16 : lr = 0.0005
checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='loss', patience=1, mode='min', restore_best_weights=True) 
print_acc_callback = PrintAccCallback()

callbacks_list = [checkpoint, early_stop, print_acc_callback]
gen = DataGenerator(
    artist_related=artist_related,
    batch_size=16,
    shuffle=True
)

model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(lr=0.0005)
)

model.fit_generator(
    generator=gen,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list
)

############################### patience = 2 : batch_size = 1 : lr = 0.00001
checkpoint = ModelCheckpoint(model_name + '.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='loss', patience=2, mode='min', restore_best_weights=True) 
print_acc_callback = PrintAccCallback()

callbacks_list = [checkpoint, early_stop, print_acc_callback]

gen = DataGenerator(
    artist_related=artist_related,
    batch_size=1,
    shuffle=True
)

model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(lr=0.00001)
)

model.fit_generator(
    generator=gen,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list
)

with open(model_name + '.emb.pickle', 'wb') as f:
    save = {
        'embedding_lookup': np.asarray(model.get_layer('embedding').get_weights()[0])
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    del save