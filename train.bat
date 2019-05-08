::::::::::::::::::::::::
::::::: CIFAR100 :::::::
::::::::::::::::::::::::
::python train_cnn.py -d=cifar100 


:::::::::::::::::::::::
::::::: FMA_MED :::::::
:::::::::::::::::::::::
::python train_cnn.py -d=fma_med -t=sgc -f=stft
::python train_cnn.py -d=fma_med -t=sgc -f=stft_halved
::python train_cnn.py -d=fma_med -t=sgc -f=mel_scaled_stft
::python train_cnn.py -d=fma_med -t=sgc -f=cqt
::python train_cnn.py -d=fma_med -t=sgc -f=chroma
::python train_cnn.py -d=fma_med -t=sgc -f=mfcc

::python train_rnn.py -d=fma_med -t=sgc -f=mfcc
::python train_rnn.py -d=fma_med -t=sgc -f=chroma

::python train_ens.py -d=fma_med -t=sgc


:::::::::::::::::::::::::
::::::: FMA_LARGE :::::::
:::::::::::::::::::::::::
::python train_cnn.py -d=fma_large -t=mgc -f=cqt
::python train_cnn.py -d=fma_large -t=mgc -f=mel_scaled_stft
::python train_cnn.py -d=fma_large -t=mgc -f=stft_halved

::python train_rnn.py -d=fma_large -t=mgc -f=mfcc 
::python train_rnn.py -d=fma_large -t=mgc -f=chroma 


:::::::::::::::::::::::
::::::: SPOTIFY :::::::
:::::::::::::::::::::::
::python train_cnn.py -d=spotify -t=cos -f=cqt
::python train_cnn.py -d=spotify -t=cos -f=mel_scaled_stft

::python train_rnn.py -d=spotify -t=cos -f=mfcc
::python train_rnn.py -d=spotify -t=cos -f=chroma

::python train_lyrics.py

::python train_albums.py

::python get_pred_spotify.py

::python train_spot_ens.py 
::python train_spot_ens.py --lyrics 
::python train_spot_ens.py --albums 
::python train_spot_ens.py --location 
::python train_spot_ens.py --year 
::python train_spot_ens.py --lyrics --albums --location --year