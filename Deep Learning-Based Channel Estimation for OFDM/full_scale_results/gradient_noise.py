"""
gradient_noise.py
==================
Measures how much a mini-batch gradient estimate varies from batch to
batch, at different batch sizes. Small batches -> noisier gradients.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import GRADIENT_NOISE_BATCH_SIZES
from model import NeuralNetwork


def compute_gradient_variance(X_train, y_train, batch_sizes_to_test, num_batches=20, device="cpu"):
    """
    For each batch size, sample `num_batches` mini-batch gradients of the
    first layer's weights and return the mean per-parameter variance.
    """
    X_tensor, y_tensor = torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device)
    results = {}

    for bs in batch_sizes_to_test:
        model = NeuralNetwork().to(device)
        criterion = nn.CrossEntropyLoss()
        gradients = []

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=bs, shuffle=True)
        loader_iter = iter(loader)
        for _ in range(min(num_batches, len(loader))):
            batch_X, batch_y = next(loader_iter)
            model.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            grad = model.fc1.weight.grad.clone().detach().cpu().numpy().flatten()
            gradients.append(grad)

        gradients = np.array(gradients)
        results[bs] = np.var(gradients, axis=0).mean()

    return results


def run_gradient_noise_experiment(
    X_train, y_train, batch_sizes=GRADIENT_NOISE_BATCH_SIZES, num_batches=20, device="cpu"
):
    print("\n" + "=" * 80)
    print(" GRADIENT NOISE ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing gradient variance across different batch sizes...")

    grad_variance = compute_gradient_variance(X_train, y_train, batch_sizes, num_batches, device)

    print("\nGradient Variance Results:")
    for bs, var in grad_variance.items():
        print(f"  Batch size {bs:3d}: Variance = {var:.6f}")

    return grad_variance
