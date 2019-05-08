::::::::::::::::::::::::
::::::: CIFAR100 :::::::
::::::::::::::::::::::::
::python results_cnn.py -d=cifar100 -f=cifar100 -t=sgc


:::::::::::::::::::::::
::::::: FMA_MED :::::::
:::::::::::::::::::::::
::python results_cnn.py -d=fma_med -t=sgc -f=stft
::python results_cnn.py -d=fma_med -t=sgc -f=stft_halved
::python results_cnn.py -d=fma_med -t=sgc -f=mel_scaled_stft
::python results_cnn.py -d=fma_med -t=sgc -f=cqt
::python results_cnn.py -d=fma_med -t=sgc -f=chroma
::python results_cnn.py -d=fma_med -t=sgc -f=mfcc

::python results_rnn.py -d=fma_med -t=sgc -f=mfcc
::python results_rnn.py -d=fma_med -t=sgc -f=chroma

::python results_ens.py -d=fma_med -t=sgc


:::::::::::::::::::::::::
::::::: FMA_LARGE :::::::
:::::::::::::::::::::::::
::python results_cnn.py -d=fma_large -t=mgc -f=stft_halved
::python results_cnn.py -d=fma_large -t=mgc -f=mel_scaled_stft
::python results_cnn.py -d=fma_large -t=mgc -f=cqt

::python results_rnn.py -d=fma_large -t=mgc -f=mfcc
::python results_rnn.py -d=fma_large -t=mgc -f=chroma


:::::::::::::::::::::::
::::::: SPOTIFY :::::::
:::::::::::::::::::::::
::python get_pred_spotify.py