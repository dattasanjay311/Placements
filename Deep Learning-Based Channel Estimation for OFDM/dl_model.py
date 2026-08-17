"""
dl_model.py

A compact 1D CNN that refines the noisy, interpolated LS channel estimate
into a more accurate frequency-domain channel estimate.

Input : (batch, 2, N_FFT)  -- channel 0 = Re{H_LS}, channel 1 = Im{H_LS}
Output: (batch, 2, N_FFT)  -- refined Re/Im estimate of H

Design choices (for interview defense):
  - The subcarrier axis is treated as a 1D "spatial" axis: nearby subcarriers
    are correlated (coherence bandwidth), so local convolutions can learn to
    denoise/interpolate the initial LS estimate the same way an image
    denoising CNN exploits local pixel correlation. This mirrors the
    image-based channel-estimation approach in Soltani et al., "Deep Learning-
    Based Channel Estimation," IEEE Comm. Letters, 2019 (ChannelNet).
  - Residual connection: the network learns a *correction* to the LS estimate
    rather than the full mapping from scratch. LS is already unbiased (just
    noisy), so learning "how to denoise LS" is an easier, better-conditioned
    task than learning "channel estimation" from zero, and it guarantees the
    network can always fall back to something close to LS if the correction
    isn't confident.
"""

import torch
import torch.nn as nn


class ChannelEstimatorCNN(nn.Module):
    def __init__(self, n_fft=64, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden, 2, kernel_size=9, padding=4),
        )

    def forward(self, x):
        return x + self.net(x)   # residual: learn the correction to LS


if __name__ == "__main__":
    model = ChannelEstimatorCNN()
    dummy = torch.randn(8, 2, 64)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}, trainable params: {n_params:,}")
