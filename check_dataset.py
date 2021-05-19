import os
import glob
import random
import numpy as np
import torchaudio as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from params import params
from tqdm import tqdm
import torch

class MelFromDisk():
    def __init__(self, params):
        self.params = params
        self.path = params.path
        self.wav_list = glob.glob(
            os.path.join(self.path, "**", "*.wav"), recursive=True
        )

        self.mapping = [i for i in range(len(self.wav_list))]
        self.downsample = T.transforms.Resample(
            params.new_sample_rate,
            params.sample_rate,
            resampling_method="sinc_interpolation",
        )

    def check_dataset(self, idx):
        wavpath = self.wav_list[idx]
        id = os.path.basename(wavpath).split(".")[0]
        audio, sr = T.load_wav(wavpath)
        if self.params.new_sample_rate != sr:
            print(id)
            #raise ValueError(f"Invalid sample rate {sr}.")
        #audio = torch.clamp(audio[0].squeeze(0) / 32767.5, -1.0, 1.0)

        start = np.random.randint(0, audio.shape[1] - self.params.n_segment - 1)
        #print(audio.shape)
        if audio.shape[0] == 2:
            audio = audio[0, :]
        audio = audio.squeeze(0)[start : start + self.params.n_segment]
        audio = audio / 32767.5
        #print(id, audio.shape)
        lr_audio = self.downsample(audio)
        lr_audio = lr_audio / 32767.5


        return {"audio": audio, "lr_audio": lr_audio, "id": id}

if __name__ == "__main__":
    M = MelFromDisk(params)
    for i in tqdm(range(0, len(M.wav_list))):
        try:
            out = M.check_dataset(i)
        except Exception as e:
            print(e)
            continue
