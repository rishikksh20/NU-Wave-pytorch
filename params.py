import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


params = AttrDict(
    # Training params
    path="/mnt/dev/jean/wavs_44k_jean_nugan/",
    data_dir="./preprocessed/",
    batch_size=4,
    learning_rate=2e-4,
    max_grad_norm=None,
    # Data params
    sample_rate=22050,
    n_mels=256,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Probably an error in paper.
    n_segment=32768,  # For 44.1KHz -> 22.05 kHz n_segment mod 2
    new_sample_rate=44100,
    # Model params
    input_channels=1,
    output_channels=1,
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    noise_schedule=np.linspace(1e-6, 0.006, 1000).tolist(),
    inference_noise_schedule=[
        0.000001,
        0.000002,
        0.00001,
        0.0001,
        0.001,
        0.01,
        0.1,
        0.9,
    ],
)
