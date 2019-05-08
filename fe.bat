:::::::::::::::::::::::
::::::: FMA_MED :::::::
:::::::::::::::::::::::
::python fe.py -d=fma_med -f=stft 
::python fe.py -d=fma_med -f=stft_halved 
::python fe.py -d=fma_med -f=mel_scaled_stft 
::python fe.py -d=fma_med -f=cqt 
::python fe.py -d=fma_med -f=chroma 
::python fe.py -d=fma_med -f=mfcc 


:::::::::::::::::::::::::
::::::: FMA_LARGE :::::::
:::::::::::::::::::::::::
::python fe.py -d=fma_large -f=stft_halved
::python fe.py -d=fma_large -f=mel_scaled_stft 
::python fe.py -d=fma_large -f=cqt 
::python fe.py -d=fma_large -f=chroma 
::python fe.py -d=fma_large -f=mfcc


:::::::::::::::::::::::
::::::: SPOTIFY :::::::
:::::::::::::::::::::::
::python fe.py -d=spotify -f=mel_scaled_stft 
::python fe.py -d=spotify -f=cqt
::python fe.py -d=spotify -f=chroma 
::python fe.py -d=spotify -f=mfcc 