::python train.py -d=fma_med -t=sgc -f=stft -q=True
::python train.py -d=fma_med -t=sgc -f=stft_halved -q=True
::python train.py -d=fma_med -t=sgc -f=mel_scaled_stft -q=True
python train.py -d=fma_med -t=sgc -f=cqt
python results.py -d=fma_med -f=cqt