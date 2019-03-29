::python fe.py -d=fma_med -f=stft -q=True 
::python fe.py -d=fma_med -f=stft_halved -q=True
::python fe.py -d=fma_med -f=mel_scaled_stft -q=True
::python fe.py -d=fma_med -f=cqt -q=True
::python fe.py -d=fma_med -f=chroma -q=True
::python fe.py -d=fma_med -f=mfcc -q=True

::python fe.py -d=fma_large -f=stft_halved -q=True
::python fe.py -d=fma_large -f=mel_scaled_stft -q=True
::python fe.py -d=fma_large -f=cqt -q=True
::python fe.py -d=fma_large -f=chroma -q=True
::python fe.py -d=fma_large -f=mfcc -q=True