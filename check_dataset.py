import os
import glob
import numpy as np
import torchaudio as T
from params import params
from tqdm import tqdm


class AudioDataset:
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
            raise ValueError(f"Invalid sample rate {sr}.")

        start = np.random.randint(0, audio.shape[1] - self.params.n_segment - 1)

        if audio.shape[0] == 2:
            audio = audio[0, :]
        audio = audio.squeeze(0)[start : start + self.params.n_segment]
        audio = audio / 32767.5

        lr_audio = self.downsample(audio)
        lr_audio = lr_audio / 32767.5

        return {"audio": audio, "lr_audio": lr_audio, "id": id}


if __name__ == "__main__":
    M = AudioDataset(params)
    for i in tqdm(range(0, len(M.wav_list))):
        try:
            out = M.check_dataset(i)
        except Exception as e:
            print(e)
            continue
