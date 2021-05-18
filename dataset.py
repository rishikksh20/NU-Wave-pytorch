import os
import glob
import random
import numpy as np
import torchaudio as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def create_dataloader(params, train, is_distributed=False):
    dataset = MelFromDisk(params, train)

    return DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        shuffle=not is_distributed,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )


class MelFromDisk(Dataset):
    def __init__(self, params, train):
        self.params = params
        self.train = train
        self.path = params.path
        self.wav_list = glob.glob(
            os.path.join(self.path, "**", "*.wav"), recursive=True
        )

        self.mapping = [i for i in range(len(self.wav_list))]
        self.downsample = T.transforms.Resample(
            params.new_sample_rate, params.sample_rate,
            resampling_method='sinc_interpolation'
        )

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wavpath = self.wav_list[idx]
        id = os.path.basename(wavpath).split(".")[0]

        audio, sr = T.load_wav(wavpath)
        if self.params.new_sample_rate != sr:
            raise ValueError(f'Invalid sample rate {sr}.')
        # audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)


        start = np.random.randint(
            0, audio.shape[1] - self.params.n_segment - 1
        )
        audio = audio.squeeze(0)[start: start + self.params.n_segment]
        lr_audio = self.downsample(audio)

        audio = audio / 32767.5
        lr_audio = lr_audio / 32767.5

        return {
            'audio': audio,
            'lr_audio': lr_audio,
            'id': id
        }