"""
model.py
========
The feedforward neural network architecture used across all experiments.
Kept constant everywhere so that batch size is the only variable being studied.

Architecture:
    Input(20) -> Dense(64) -> ReLU -> Dropout(0.2)
              -> Dense(32) -> ReLU -> Dropout(0.2)
              -> Output(2)
"""

import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Simple feedforward neural network for binary classification."""

    def __init__(self, input_size=20, hidden_sizes=(64, 32), num_classes=2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
