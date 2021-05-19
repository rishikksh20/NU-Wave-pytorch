import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


LINEAR_SCALE = 50000
GAMMA = 1 // 16


class PositionalEncoding(nn.Module):
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(
            half_dim
        )
        exponents = 10 ** -(exponents * GAMMA)
        exponents = LINEAR_SCALE * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.embedding = PositionalEncoding(n_channels)
        self.projection1 = Linear(n_channels, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, noise_level):
        x = self.embedding(noise_level)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, noise_scale):

        noise_scale = self.diffusion_projection(noise_scale).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + noise_scale
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class NUWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.factor = 2
        self.input_projection = Conv1d(
            params.input_channels, params.residual_channels, 1
        )
        self.conditioner_projection = Conv1d(
            params.input_channels, params.residual_channels, 1
        )
        self.diffusion_embedding = DiffusionEmbedding(params.residual_channels)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    params.n_mels,
                    params.residual_channels,
                    2 ** (i % params.dilation_cycle_length),
                )
                for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        self.output_projection = Conv1d(
            params.residual_channels, params.output_channels, 1
        )
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, lr_audio, noise_scale):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = silu(x)

        noise_scale = self.diffusion_embedding(noise_scale)
        lr_audio = lr_audio.unsqueeze(1)
        cond = F.interpolate(lr_audio, size=lr_audio.shape[-1] * self.factor)
        cond = self.conditioner_projection(cond)
        cond = silu(cond)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, noise_scale)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = silu(x)
        x = self.output_projection(x)
        return x
