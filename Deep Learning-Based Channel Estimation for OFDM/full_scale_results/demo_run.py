"""
demo_run.py
===========
A reduced-scale run of the full pipeline (smaller datasets, fewer epochs)
so it completes in a few minutes and produces real, reproducible output.
Uses the exact same modules/logic as main.py -- only the scale is reduced.
"""
import os
import time

import data as data_mod
from config import BASE_LR, DEVICE, print_banner
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

# ---- reduced scale for a fast demo run ----
data_mod.DATASET_SIZES = {
    "Small (1K)": 600,
    "Medium (10K)": 3000,
    "Large (50K)": 8000,
}
BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]
EPOCHS = 10

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

t0 = time.time()
print_banner()

datasets, datasets_info = data_mod.prepare_datasets()

results = run_experiments(datasets, BATCH_SIZES, BASE_LR, EPOCHS, DEVICE)
results_df = build_results_dataframe(results, datasets, BATCH_SIZES)

plot_main_dashboard(results_df, results, datasets, output_dir=OUTPUT_DIR)

X_large_train, y_large_train, X_large_test, y_large_test = datasets["Large (50K)"]
fixed_lr_results, scaled_lr_results = run_lr_scaling_experiment(
    X_large_train, y_large_train, X_large_test, y_large_test,
    base_lr=BASE_LR, epochs=EPOCHS, device=DEVICE,
)
plot_lr_scaling(fixed_lr_results, scaled_lr_results,
                 test_batch_sizes=list(fixed_lr_results.keys()), output_dir=OUTPUT_DIR)

print_key_findings(results_df, datasets)
print_recommendations()

X_med_train, y_med_train, X_med_test, y_med_test = datasets["Medium (10K)"]
landscapes = run_loss_landscape_experiment(
    X_med_train, y_med_train, X_med_test, y_med_test,
    base_lr=BASE_LR, device=DEVICE, train_epochs=15, landscape_steps=12,
)
plot_loss_landscape(landscapes, output_dir=OUTPUT_DIR)

grad_variance = run_gradient_noise_experiment(X_med_train, y_med_train, device=DEVICE)
plot_gradient_noise(grad_variance, output_dir=OUTPUT_DIR)

csv_path = os.path.join(OUTPUT_DIR, "batch_size_experiment_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n Saved: {csv_path}")

print(f"\nTOTAL DEMO RUN TIME: {time.time()-t0:.1f}s")
