::python train_cnn.py -d=cifar100 -q=True

::python train_cnn.py -d=fma_med -t=sgc -f=stft -q=True
::python train_cnn.py -d=fma_med -t=sgc -f=stft_halved -q=True
::python train_cnn.py -d=fma_med -t=sgc -f=mel_scaled_stft -q=True
::python train_cnn.py -d=fma_med -t=sgc -f=cqt -q=True
::python train_cnn.py -d=fma_med -t=sgc -f=chroma -q=True
::python train_cnn.py -d=fma_med -t=sgc -f=mfcc -q=True

