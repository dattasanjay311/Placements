"""
main.py
=======
Entry point. Reproduces the full notebook end-to-end:

    1. Prepare Small / Medium / Large datasets
    2. Run the main batch-size x dataset experiment grid
    3. Analyze + visualize results
    4. Run the learning-rate scaling sub-experiment
    5. Print key findings & recommendations
    6. Run the loss-landscape sub-experiment
    7. Run the gradient-noise sub-experiment
    8. Export results to CSV

Run with:  python main.py
"""

import os

from config import BASE_LR, BATCH_SIZES, DEVICE, EPOCHS, print_banner
from data import prepare_datasets
from experiments import build_results_dataframe, run_experiments
from gradient_noise import run_gradient_noise_experiment
from loss_landscape import run_loss_landscape_experiment
from lr_scaling_analysis import run_lr_scaling_experiment
from report import print_key_findings, print_recommendations
from visualize import (
    plot_gradient_noise,
    plot_loss_landscape,
    plot_lr_scaling,
    plot_main_dashboard,
)

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_banner()

    # 1. Data
    datasets, datasets_info = prepare_datasets()

    # 2. Main experiment grid
    results = run_experiments(datasets, BATCH_SIZES, BASE_LR, EPOCHS, DEVICE)
    results_df = build_results_dataframe(results, datasets, BATCH_SIZES)

    # 3. Visualize main results
    plot_main_dashboard(results_df, results, datasets, output_dir=OUTPUT_DIR)

    # 4. Learning rate scaling sub-experiment (on the Large dataset)
    X_large_train, y_large_train, X_large_test, y_large_test = datasets["Large (50K)"]
    fixed_lr_results, scaled_lr_results = run_lr_scaling_experiment(
        X_large_train, y_large_train, X_large_test, y_large_test,
        base_lr=BASE_LR, epochs=EPOCHS, device=DEVICE,
    )
    plot_lr_scaling(
        fixed_lr_results, scaled_lr_results,
        test_batch_sizes=list(fixed_lr_results.keys()), output_dir=OUTPUT_DIR,
    )

    # 5. Key findings & recommendations
    print_key_findings(results_df, datasets)
    print_recommendations()

    # 6. Loss landscape sub-experiment (on the Medium dataset)
    X_med_train, y_med_train, X_med_test, y_med_test = datasets["Medium (10K)"]
    landscapes = run_loss_landscape_experiment(
        X_med_train, y_med_train, X_med_test, y_med_test, base_lr=BASE_LR, device=DEVICE,
    )
    plot_loss_landscape(landscapes, output_dir=OUTPUT_DIR)

    # 7. Gradient noise sub-experiment (on the Medium dataset)
    grad_variance = run_gradient_noise_experiment(X_med_train, y_med_train, device=DEVICE)
    plot_gradient_noise(grad_variance, output_dir=OUTPUT_DIR)

    # 8. Export results
    print("\n" + "=" * 80)
    print(" EXPORTING RESULTS")
    print("=" * 80)
    csv_path = os.path.join(OUTPUT_DIR, "batch_size_experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f" Saved: {csv_path}")


if __name__ == "__main__":
    main()
