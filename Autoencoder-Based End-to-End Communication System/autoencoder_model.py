"""
autoencoder_model.py

The autoencoder communication system, following O'Shea & Hoydis (2017) and
the standard reference architecture (matching MathWorks' Communications
Toolbox implementation of the same idea): two dense layers for the encoder,
two for the decoder, with a differentiable AWGN channel layer sandwiched
directly between them.

The crucial architectural difference from a normal classifier: the
"channel" is a non-trainable layer INSIDE the forward pass, not something
applied afterward to pre-generated data. Gradients flow from the
cross-entropy loss, through the channel (trivial: d(x+n)/dx = 1), all the
way back into the encoder's weights. This is what lets the transmitter and
receiver be optimized JOINTLY for a specific channel and objective, rather
than being designed separately (modulation, then coding) as in classical
systems.

  Encoder (Tx):  one-hot message (M) -> Dense(M, ReLU) -> Dense(n) -> energy normalization -> x
  Channel:       y = x + n,  n ~ N(0, sigma^2 I)     (AWGN, non-trainable)
  Decoder (Rx):  y (n) -> Dense(M, ReLU) -> Dense(M) -> logits (softmax applied by the loss)
"""

import math
import torch
import torch.nn as nn


class EnergyNormalize(nn.Module):
    """Rescales each example so ||x||^2 = n exactly (the 'Energy constraint' in
    the literature: every transmitted block uses the same total energy)."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * math.sqrt(self.n) / (norm + 1e-12)


class Encoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M, M),
            nn.ReLU(inplace=True),
            nn.Linear(M, n),
        )
        self.normalize = EnergyNormalize(n)

    def forward(self, one_hot):
        x = self.net(one_hot)
        return self.normalize(x)


class Decoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, M),
            nn.ReLU(inplace=True),
            nn.Linear(M, M),
        )

    def forward(self, y):
        return self.net(y)   # raw logits; nn.CrossEntropyLoss applies softmax internally


class Autoencoder(nn.Module):
    def __init__(self, M, n):
        super().__init__()
        self.M, self.n = M, n
        self.encoder = Encoder(M, n)
        self.decoder = Decoder(M, n)

    def forward(self, one_hot, noise_std):
        x = self.encoder(one_hot)
        noise = torch.randn_like(x) * noise_std
        y = x + noise
        logits = self.decoder(y)
        return logits, x


if __name__ == "__main__":
    M, n = 16, 7
    model = Autoencoder(M, n)
    batch = 8
    msgs = torch.randint(0, M, (batch,))
    one_hot = torch.nn.functional.one_hot(msgs, M).float()

    logits, x = model(one_hot, noise_std=0.5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"(n={n}, k={int(math.log2(M))}, M={M}) autoencoder: {n_params} trainable params")
    print(f"Encoder output shape: {tuple(x.shape)}  |  Decoder logits shape: {tuple(logits.shape)}")
    energy = torch.sum(x ** 2, dim=-1)
    print(f"Per-example transmitted energy ||x||^2 (should all equal n={n}): "
          f"{energy.detach().numpy().round(4)}")
