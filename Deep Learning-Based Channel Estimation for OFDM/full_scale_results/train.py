"""
train.py
========
Core training/evaluation loop used by every experiment in this project.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import NeuralNetwork


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size,
    learning_rate=0.001,
    epochs=50,
    device="cpu",
    verbose=False,
):
    """
    Train a NeuralNetwork with the given batch size and track metrics.

    Returns
    -------
    dict with keys:
        history             - per-epoch train/test loss & accuracy, epoch time
        final_train_acc
        final_test_acc
        generalization_gap  - final_train_acc - final_test_acc
        avg_epoch_time
        convergence_epoch   - first epoch where test accuracy exceeds 80%
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model / loss / optimizer
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_time": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_time = time.time() - start_time

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = 100 * (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(100 * correct / total)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Acc: {history['train_acc'][-1]:.2f}%, Test Acc: {test_acc:.2f}%"
            )

    generalization_gap = history["train_acc"][-1] - history["test_acc"][-1]
    convergence_epoch = (
        np.argmax(np.array(history["test_acc"]) > 80) if max(history["test_acc"]) > 80 else epochs
    )

    return {
        "history": history,
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "generalization_gap": generalization_gap,
        "avg_epoch_time": np.mean(history["epoch_time"]),
        "convergence_epoch": convergence_epoch,
    }
