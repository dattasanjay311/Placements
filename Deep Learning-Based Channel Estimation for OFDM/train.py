"""
train.py

Trains the CNN channel estimator on the generated dataset and saves:
  - trained weights (model.pt)
  - training/validation loss curve (results/training_loss.png)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dl_model import ChannelEstimatorCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3


def main():
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(0)

    d = np.load("data/dataset.npz")
    X_train = torch.from_numpy(d["X_train"])
    Y_train = torch.from_numpy(d["Y_train"])
    X_val = torch.from_numpy(d["X_val"]).to(DEVICE)
    Y_val = torch.from_numpy(d["Y_val"]).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = ChannelEstimatorCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    # Baseline: LS-only MSE on the validation set (i.e. "do nothing" correction)
    with torch.no_grad():
        baseline_val_mse = loss_fn(X_val, Y_val).item()

    train_losses, val_losses = [], []
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), Y_val).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "model.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f}")

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"LS-only (no correction) val MSE : {baseline_val_mse:.5f}")
    print(f"Best CNN val MSE                : {best_val:.5f}")
    print(f"Relative improvement over LS    : {100*(1 - best_val/baseline_val_mse):.1f}%")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.axhline(baseline_val_mse, color="gray", linestyle="--", label="LS-only (no NN) val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.title("Training curve: CNN channel estimator")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_loss.png", dpi=150)
    print("Saved results/training_loss.png")


if __name__ == "__main__":
    main()
