import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser

from params import AttrDict, params as base_params
from model import NUWave


models = {}


def predict(lr_audio, model_dir=None, params=None, device=torch.device("cuda")):
    # Lazy load model.
    if not model_dir in models:
        if os.path.exists(f"{model_dir}/weights.pt"):
            checkpoint = torch.load(f"{model_dir}/weights.pt")
        else:
            checkpoint = torch.load(model_dir, map_location=device)
        model = NUWave(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[model_dir] = model

    model = models[model_dir]
    model.params.override(params)
    with torch.no_grad():
        beta = np.array(model.params.inference_noise_schedule)
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        # Expand rank 2 tensors by adding a batch dimension.
        if len(lr_audio.shape) == 1:
            lr_audio = lr_audio.unsqueeze(0)
        lr_audio = lr_audio.to(device)

        audio = torch.randn(lr_audio.shape[0], 2 * lr_audio.shape[-1], device=device)
        noise_scale = torch.from_numpy(alpha_cum ** 0.5).float().unsqueeze(1).to(device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = (1 - alpha[n]) / (1 - alpha_cum[n]) ** 0.5
            audio = c1 * (
                audio - c2 * model(audio, lr_audio, noise_scale[n]).squeeze(1)
            )
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = (
                    (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                ) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
    return audio, model.params.new_sample_rate


def main(args):
    lr_audio, sr = torchaudio.load(args.audio_path)
    if 22050 != sr:
        raise ValueError(f"Invalid sample rate {sr}.")
    params = {}
    # if args.noise_schedule:
    #     params["noise_schedule"] = torch.from_numpy(np.load(args.noise_schedule))
    
    audio, sr = predict(lr_audio, model_dir=args.model_dir, params=params)
    torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="runs inference on a spectrogram file generated by wavegrad.preprocess"
    )
    parser.add_argument(
        "model_dir",
        help="directory containing a trained model (or full path to weights.pt file)",
    )
    parser.add_argument(
        "audio_path",
        help="path to a low resolution file generated",
    )

    parser.add_argument("--output", "-o", default="output.wav", help="output file name")
    main(parser.parse_args())
