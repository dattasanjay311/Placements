"""
experiments.py
==============
Runs the main grid of experiments: every dataset x every batch size,
with the learning rate linearly scaled to the batch size.
"""

import pandas as pd

from config import BASE_LR, EPOCHS, scaled_lr
from train import train_model


def run_experiments(datasets, batch_sizes, base_lr=BASE_LR, epochs=EPOCHS, device="cpu"):
    """
    Train a model for every (dataset, batch_size) combination.

    Returns
    -------
    results : dict[str, dict[int, dict]]
        results[dataset_name][batch_size] -> output of train_model()
    """
    print("\n RUNNING EXPERIMENTS...")
    print("=" * 80)

    results = {dataset_name: {} for dataset_name in datasets.keys()}

    for dataset_name, (X_tr, y_tr, X_te, y_te) in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 80}")

        for batch_size in batch_sizes:
            print(f"\n  Training with batch size: {batch_size}...", end=" ")

            lr = scaled_lr(batch_size, base_lr)
            result = train_model(
                X_tr, y_tr, X_te, y_te,
                batch_size=batch_size,
                learning_rate=lr,
                epochs=epochs,
                device=device,
                verbose=False,
            )

            results[dataset_name][batch_size] = result
            print(f" Test Acc: {result['final_test_acc']:.2f}%, Gen Gap: {result['generalization_gap']:.2f}%")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)

    return results


def build_results_dataframe(results, datasets, batch_sizes):
    """Flatten the nested `results` dict into a tidy pandas DataFrame."""
    print("\n ANALYZING RESULTS...")

    rows = []
    for dataset_name in datasets.keys():
        for batch_size in batch_sizes:
            r = results[dataset_name][batch_size]
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Batch Size": batch_size,
                    "Train Accuracy": r["final_train_acc"],
                    "Test Accuracy": r["final_test_acc"],
                    "Generalization Gap": r["generalization_gap"],
                    "Avg Epoch Time": r["avg_epoch_time"],
                    "Convergence Epoch": r["convergence_epoch"],
                }
            )

    results_df = pd.DataFrame(rows)
    print("\n Results Summary:")
    print(results_df.to_string(index=False))
    return results_df
