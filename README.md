# NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling

For Official repo visit [here](https://github.com/mindslab-ai/nuwave).

## Train :
```
python3 train.py  chkpt_dir --max_steps 1000
```

## Inference :
```
python3 inference.py .\CHKPT\weights-107184.pt .\sample\tation_RetailSample_1.wav -o "output.wav"
```
