import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math

def convert_labels(y, class_map):
    inv_map = {v: k for k, v in class_map.items()}
    labels = []
    for x in y:
        labels.append(inv_map[x])
    
    return np.array(labels)

def get_metrics(logits, predicted, y_true_oh, y_true, class_map, plot_cm=True, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    genres_never = set(np.unique(y_true)) - set(np.unique(predicted))
    print('  {} out of {} genres were never predicted: {}'.format(len(genres_never), len(np.unique(y_true)), genres_never))

    loss = metrics.log_loss(y_true_oh, logits)
    print('  Mean log loss: {:.7f}'.format(loss))

    acc = sum(predicted == y_true) / len(predicted)
    print('  Accuracy: {:.2%}'.format(acc))

    f1 = metrics.f1_score(y_true, predicted, average='micro')
    print('  Micro F1 score: {:.2%}'.format(f1))

    f1 = metrics.f1_score(y_true, predicted, average='weighted')
    print('  Weighted F1 score: {:.2%}'.format(f1))
    
    if plot_cm:
        plot_cm(predicted, y_true, class_map, figsize=(15,10), normalize=normalize, title='Confusion matrix', cmap=plt.cm.Blues)

def plot_cm(predicted, y_true, class_map, figsize=(15,10), normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    y_true = convert_labels(y_true, class_map)
    predicted = convert_labels(predicted, class_map)
    
    cm = metrics.confusion_matrix(y_true, predicted, labels=list(class_map.keys()))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.rcParams['figure.figsize'] = figsize
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_map.keys()))
    plt.xticks(tick_marks, class_map.keys(), rotation=45)
    plt.yticks(tick_marks, class_map.keys())

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def get_conv_output(out_channels, kh, kw, sh, sw, ih, iw, ic, padding='VALID', framework='KERAS'):
    if (framework == 'KERAS'):
        model = Sequential()
        model.add(Conv2D(out_channels, kernel_size=(kh,kw), strides=(sh,sw), input_shape=(ih, iw, ic), padding=padding, name='conv'))
        
        out_h = model.get_layer('conv').output_shape[1]
        out_w = model.get_layer('conv').output_shape[2]
        out_c = model.get_layer('conv').output_shape[3]
        print(out_h, out_w, out_c)
    if (framework == 'TORCH'):
        if (padding == 'VALID'):
            ph, pw = 0, 0
        if (padding == 'SAME'):
            if (kh % 2 == 0):
                ph = kh//2
            else:
                ph = math.ceil(kh//2)
            if (kw % 2 == 0):
                pw = kw//2
            else:
                pw = math.ceil(kw//2)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(ic, out_channels, kernel_size=(kh,kw), stride=(sh,sw), padding=(ph,pw), bias=False)

            def forward(self, x):
                return self.conv1(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
        model = Net().to(device)

        summary(model, (ic, ih, iw))

        names = os.listdir(filepath)
        names.remove('README.txt')
        names.remove('checksums')

        files = []
        for name in names:
            i_names = os.listdir(filepath + f'/{name}/')
            for n in i_names:
                if int(n[:6]) in self.FILES_FAULTY:
                    continue
                files.append(filepath + f'/{name}/{n}')

        return np.asarray(files)