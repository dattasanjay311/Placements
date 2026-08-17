"""
train.py

Trains the autoencoder for two configurations:
  - (n=2, k=2, M=4)  : small enough to directly visualize the learned
                        constellation in 2D.
  - (n=7, k=4, M=16) : the classical (7,4) block-code comparison point,
                        benchmarked against Hamming(7,4) in evaluate_bler.py.

Both are trained at a fixed Eb/No (matching the standard reference setup:
low enough to produce a learnable, nonzero error rate, high enough that
training isn't dominated by pure noise). Since the "dataset" here is
generated on the fly (random messages + fresh channel noise every batch),
there is no train/val split in the usual sense -- every batch is data the
model has never seen before, so overfitting in the classical sense isn't
possible; what's tracked instead is whether block error rate on freshly
sampled batches keeps improving.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoencoder_model import Autoencoder
from channel import ebno_db_to_noise_std

TRAIN_EBNO_DB = 3.0
BATCH_SIZE = 512
BATCHES_PER_EPOCH = 200
EPOCHS = 60
LR = 1e-3


def block_error_rate(logits, msgs):
    preds = torch.argmax(logits, dim=-1)
    return (preds != msgs).float().mean().item()


def train_one_config(M, n, tag, seed=0):
    torch.manual_seed(seed)
    k = int(math.log2(M))
    rate = k / n
    noise_std = ebno_db_to_noise_std(TRAIN_EBNO_DB, rate)
    print(f"\n=== Training ({n},{k}) autoencoder, M={M}, rate={rate:.3f}, "
          f"training Eb/No={TRAIN_EBNO_DB}dB -> noise_std={noise_std:.4f} ===")

    model = Autoencoder(M, n)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    loss_hist, bler_hist = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_bler = 0.0, 0.0
        for _ in range(BATCHES_PER_EPOCH):
            msgs = torch.randint(0, M, (BATCH_SIZE,))
            one_hot = torch.nn.functional.one_hot(msgs, M).float()

            opt.zero_grad()
            logits, _ = model(one_hot, noise_std)
            loss = loss_fn(logits, msgs)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_bler += block_error_rate(logits, msgs)
        sched.step()

        loss_hist.append(running_loss / BATCHES_PER_EPOCH)
        bler_hist.append(running_bler / BATCHES_PER_EPOCH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss {loss_hist[-1]:.4f} | "
                  f"train BLER {bler_hist[-1]:.4f}")

    torch.save(model.state_dict(), f"model_{tag}.pt")
    print(f"Final train BLER: {bler_hist[-1]:.4f}  (saved model_{tag}.pt)")
    return loss_hist, bler_hist


def main():
    os.makedirs("results", exist_ok=True)

    loss_22, bler_22 = train_one_config(M=4, n=2, tag="n2k2")
    loss_74, bler_74 = train_one_config(M=16, n=7, tag="n7k4")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (loss_hist, bler_hist, title) in zip(
        axes, [(loss_22, bler_22, "(2,2) autoencoder, M=4"),
               (loss_74, bler_74, "(7,4) autoencoder, M=16")]
    ):
        ax2 = ax.twinx()
        l1, = ax.plot(loss_hist, color="tab:blue", label="Cross-entropy loss")
        l2, = ax2.plot(bler_hist, color="tab:red", label="Block error rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss", color="tab:blue")
        ax2.set_ylabel("Block error rate", color="tab:red")
        ax2.set_yscale("log")
        ax.set_title(f"{title}\n(training Eb/No = {TRAIN_EBNO_DB} dB)")
        ax.legend(handles=[l1, l2], loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    print("\nSaved results/training_curves.png")


if __name__ == "__main__":
    main()
