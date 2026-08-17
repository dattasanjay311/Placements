"""
config.py
=========
Global configuration: random seeds, device selection, and experiment
hyperparameters shared across the whole project.
"""

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# Experiment hyperparameters
# ----------------------------------------------------------------------------
BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]
BASE_LR = 0.001          # Learning rate at the reference batch size (32)
REFERENCE_BATCH_SIZE = 32
EPOCHS = 50

# Batch sizes used for the smaller, focused sub-experiments
LR_SCALING_TEST_BATCH_SIZES = [32, 128, 512]
LOSS_LANDSCAPE_BATCH_SIZES = [32, 512]
GRADIENT_NOISE_BATCH_SIZES = [8, 32, 128, 512]


def scaled_lr(batch_size, base_lr=BASE_LR, reference=REFERENCE_BATCH_SIZE):
    """Linear Scaling Rule: lr = base_lr * (batch_size / reference)."""
    return base_lr * (batch_size / reference)


def print_banner():
    print("=" * 80)
    print("BATCH SIZE ANALYSIS PROJECT")
    print("=" * 80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
