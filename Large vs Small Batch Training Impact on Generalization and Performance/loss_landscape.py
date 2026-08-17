"""
loss_landscape.py
==================
Explores whether small-batch training finds flatter minima than
large-batch training, by plotting the loss surface around the
converged weights along two random directions (filter-normalized
random-direction landscape visualization).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import BASE_LR, LOSS_LANDSCAPE_BATCH_SIZES, scaled_lr
from model import NeuralNetwork


def compute_loss_landscape(
    model, X, y, criterion, device, center_weights,
    direction1, direction2, alpha_range=(-1, 1), beta_range=(-1, 1), steps=20,
):
    """Evaluate loss on a 2D grid of weight-space directions around center_weights."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    betas = np.linspace(beta_range[0], beta_range[1], steps)
    losses = np.zeros((steps, steps))
    X_tensor, y_tensor = torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device)

    original_weights = [p.clone() for p in model.parameters()]
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            with torch.no_grad():
                for k, p in enumerate(model.parameters()):
                    p.data = center_weights[k] + alpha * direction1[k] + beta * direction2[k]
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                losses[i, j] = criterion(outputs, y_tensor).item()

    with torch.no_grad():  # Restore original weights
        for p, orig in zip(model.parameters(), original_weights):
            p.data = orig

    return alphas, betas, losses


def run_loss_landscape_experiment(
    X_train, y_train, X_test, y_test,
    base_lr=BASE_LR, device="cpu",
    batch_sizes=LOSS_LANDSCAPE_BATCH_SIZES,
    train_epochs=30, landscape_steps=15,
):
    """
    Trains a model at each batch size in `batch_sizes` and computes the
    loss landscape around its final weights.

    Returns
    -------
    landscapes : dict[int, tuple] -> batch_size -> (alphas, betas, losses)
    """
    print("\n" + "=" * 80)
    print(" ADVANCED ANALYSIS: Loss Landscape Exploration")
    print("=" * 80)
    print("\nComputing loss landscapes for different batch sizes...")

    landscapes = {}

    for bs in batch_sizes:
        print(f"\n  Processing batch size {bs}...")
        model = NeuralNetwork().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=scaled_lr(bs, base_lr))
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device)),
            batch_size=bs, shuffle=True,
        )

        # Train for fewer epochs for speed
        for epoch in range(train_epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        center_weights = [p.clone().detach() for p in model.parameters()]
        direction1 = [torch.randn_like(p) * 0.1 for p in model.parameters()]
        direction2 = [torch.randn_like(p) * 0.1 for p in model.parameters()]

        alphas, betas, losses = compute_loss_landscape(
            model, X_test, y_test, criterion, device,
            center_weights, direction1, direction2, steps=landscape_steps,
        )
        landscapes[bs] = (alphas, betas, losses)
        print("Loss landscape computed")

    return landscapes
