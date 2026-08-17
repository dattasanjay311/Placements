"""
lr_scaling_analysis.py
=======================
Focused experiment comparing a FIXED learning rate against a LINEARLY
SCALED learning rate, on the Large dataset, across a few batch sizes.
Demonstrates why the Linear Scaling Rule matters for large-batch training.
"""

from config import BASE_LR, EPOCHS, LR_SCALING_TEST_BATCH_SIZES, scaled_lr
from train import train_model


def run_lr_scaling_experiment(
    X_large_train, y_large_train, X_large_test, y_large_test,
    base_lr=BASE_LR, epochs=EPOCHS, device="cpu",
    test_batch_sizes=LR_SCALING_TEST_BATCH_SIZES,
):
    """
    Returns
    -------
    fixed_lr_results, scaled_lr_results : dict[int, dict]
        batch_size -> output of train_model(), for the fixed-LR and
        scaled-LR runs respectively.
    """
    print("\n DETAILED ANALYSIS: Learning Rate Scaling Impact")
    print("=" * 80)
    print("\nComparing Fixed LR vs Scaled LR for Large Dataset...")

    fixed_lr_results = {}
    scaled_lr_results = {}

    for bs in test_batch_sizes:
        print(f"\n  Batch size {bs}:")

        print(f" Training with fixed LR ({base_lr})...", end=" ")
        result_fixed = train_model(
            X_large_train, y_large_train, X_large_test, y_large_test,
            batch_size=bs, learning_rate=base_lr, epochs=epochs, device=device,
        )
        fixed_lr_results[bs] = result_fixed
        print(f" Test Acc: {result_fixed['final_test_acc']:.2f}%")

        lr_scaled = scaled_lr(bs, base_lr)
        print(f" Training with scaled LR ({lr_scaled:.4f})...", end=" ")
        result_scaled = train_model(
            X_large_train, y_large_train, X_large_test, y_large_test,
            batch_size=bs, learning_rate=lr_scaled, epochs=epochs, device=device,
        )
        scaled_lr_results[bs] = result_scaled
        print(f" Test Acc: {result_scaled['final_test_acc']:.2f}%")

    return fixed_lr_results, scaled_lr_results
