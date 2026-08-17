"""
visualize.py
============
All plotting functions. Each `plot_*` function saves a PNG to `output_dir`
and calls plt.show(). Kept separate from the analysis/experiment logic so
figures can be regenerated without re-running training.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

DATASET_COLORS = {
    "Small (1K)": "#FF6B6B",
    "Medium (10K)": "#4ECDC4",
    "Large (50K)": "#45B7D1",
}


def plot_main_dashboard(results_df, results, datasets, output_dir="."):
    """The 7-panel dashboard: accuracy, gen. gap, speed, convergence, heatmaps, curves."""
    print("\n CREATING VISUALIZATIONS...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # PLOT 1: Test Accuracy vs Batch Size (Main Finding)
    ax1 = fig.add_subplot(gs[0, :])
    for name in datasets.keys():
        data = results_df[results_df["Dataset"] == name]
        ax1.plot(data["Batch Size"], data["Test Accuracy"], marker="o", lw=2.5, ms=8,
                  label=name, color=DATASET_COLORS[name])
    ax1.set_xscale("log")
    ax1.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Impact of Batch Size on Test Accuracy Across Dataset Sizes",
                   fontsize=14, fontweight="bold", pad=20)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10, frameon=True, shadow=True)

    # PLOT 2: Generalization Gap vs Batch Size
    ax2 = fig.add_subplot(gs[1, 0])
    for name in datasets.keys():
        data = results_df[results_df["Dataset"] == name]
        ax2.plot(data["Batch Size"], data["Generalization Gap"], marker="s",
                  label=name, color=DATASET_COLORS[name])
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Generalization Gap (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Generalization Gap vs Batch Size", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=9)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # PLOT 3: Training Time vs Batch Size
    ax3 = fig.add_subplot(gs[1, 1])
    for name in datasets.keys():
        data = results_df[results_df["Dataset"] == name]
        ax3.plot(data["Batch Size"], data["Avg Epoch Time"], marker="^",
                  label=name, color=DATASET_COLORS[name])
    ax3.set_xscale("log")
    ax3.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Avg Epoch Time (seconds)", fontsize=11, fontweight="bold")
    ax3.set_title("Training Efficiency vs Batch Size", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", ls="--", alpha=0.5)
    ax3.legend(fontsize=9)

    # PLOT 4: Convergence Speed
    ax4 = fig.add_subplot(gs[1, 2])
    for name in datasets.keys():
        data = results_df[results_df["Dataset"] == name]
        ax4.plot(data["Batch Size"], data["Convergence Epoch"], marker="d",
                  label=name, color=DATASET_COLORS[name])
    ax4.set_xscale("log")
    ax4.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Epochs to Converge", fontsize=11, fontweight="bold")
    ax4.set_title("Convergence Speed vs Batch Size", fontsize=12, fontweight="bold")
    ax4.grid(True, which="both", ls="--", alpha=0.5)
    ax4.legend(fontsize=9)

    # PLOT 5: Heatmap - Test Accuracy
    ax5 = fig.add_subplot(gs[2, 0])
    pivot_test = results_df.pivot(index="Dataset", columns="Batch Size", values="Test Accuracy")
    sns.heatmap(pivot_test, annot=True, fmt=".1f", cmap="RdYlGn",
                cbar_kws={"label": "Test Accuracy (%)"}, ax=ax5, vmin=70, vmax=95)
    ax5.set_title("Test Accuracy Heatmap", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax5.set_ylabel("")

    # PLOT 6: Heatmap - Generalization Gap
    ax6 = fig.add_subplot(gs[2, 1])
    pivot_gap = results_df.pivot(index="Dataset", columns="Batch Size", values="Generalization Gap")
    sns.heatmap(pivot_gap, annot=True, fmt=".1f", cmap="RdYlGn_r",
                cbar_kws={"label": "Gen Gap (%)"}, ax=ax6)
    ax6.set_title("Generalization Gap Heatmap", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax6.set_ylabel("")

    # PLOT 7: Training Curves for Medium Dataset
    ax7 = fig.add_subplot(gs[2, 2])
    medium_batch_sizes_to_plot = [8, 64, 512]
    for bs in medium_batch_sizes_to_plot:
        history = results["Medium (10K)"][bs]["history"]
        ax7.plot(history["test_acc"], linewidth=2, label=f"Batch {bs}")
    ax7.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax7.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax7.set_title("Learning Curves (Medium Dataset)", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.5)
    ax7.legend(fontsize=9)

    plt.suptitle("Comprehensive Batch Size Analysis: Impact on Model Performance",
                 fontsize=16, fontweight="bold", y=0.995)

    path = os.path.join(output_dir, "batch_size_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f" Saved: {path}")
    plt.show()


def plot_lr_scaling(fixed_lr_results, scaled_lr_results, test_batch_sizes, output_dir="."):
    """Bar charts comparing fixed vs. linearly-scaled learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(test_batch_sizes))
    width = 0.35

    # Plot 1: Test Accuracy Comparison
    ax = axes[0]
    fixed_accs = [fixed_lr_results[bs]["final_test_acc"] for bs in test_batch_sizes]
    scaled_accs = [scaled_lr_results[bs]["final_test_acc"] for bs in test_batch_sizes]
    bars1 = ax.bar(x - width / 2, fixed_accs, width, label="Fixed LR", color="#FF6B6B", alpha=0.8)
    bars2 = ax.bar(x + width / 2, scaled_accs, width, label="Scaled LR", color="#4ECDC4", alpha=0.8)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Impact of Learning Rate Scaling")
    ax.set_xticks(x, test_batch_sizes)
    ax.legend()
    ax.bar_label(bars1, padding=3, fmt="%.1f%%")
    ax.bar_label(bars2, padding=3, fmt="%.1f%%")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    # Plot 2: Generalization Gap Comparison
    ax = axes[1]
    fixed_gaps = [fixed_lr_results[bs]["generalization_gap"] for bs in test_batch_sizes]
    scaled_gaps = [scaled_lr_results[bs]["generalization_gap"] for bs in test_batch_sizes]
    ax.bar(x - width / 2, fixed_gaps, width, label="Fixed LR", color="#FF6B6B", alpha=0.8)
    ax.bar(x + width / 2, scaled_gaps, width, label="Scaled LR", color="#4ECDC4", alpha=0.8)
    ax.set_ylabel("Generalization Gap (%)")
    ax.set_title("Generalization Gap: Fixed vs Scaled LR")
    ax.set_xticks(x, test_batch_sizes)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "lr_scaling_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\n Saved: {path}")
    plt.show()


def plot_loss_landscape(landscapes, output_dir="."):
    """3D surface plots of the loss landscape for each batch size."""
    batch_sizes = list(landscapes.keys())
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(16, 6), subplot_kw={"projection": "3d"})
    if len(batch_sizes) == 1:
        axes = [axes]

    for idx, bs in enumerate(batch_sizes):
        ax = axes[idx]
        alphas, betas, losses = landscapes[bs]
        X_mesh, Y_mesh = np.meshgrid(alphas, betas)
        surf = ax.plot_surface(X_mesh, Y_mesh, np.log(losses), cmap="viridis", alpha=0.9, edgecolor="none")
        ax.set_title(
            f"Loss Landscape - Batch Size {bs}\n{'(Sharp Minima)' if bs == max(batch_sizes) else '(Flat Minima)'}",
            fontsize=12, fontweight="bold", pad=20,
        )
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_zlabel("Log Loss")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.view_init(elev=25, azim=45)

    plt.suptitle("Loss Landscape Comparison: Small vs Large Batch Training", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_landscape_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\n Saved: {path}")
    plt.show()


def plot_gradient_noise(grad_variance, output_dir="."):
    """Bar chart of gradient variance per batch size."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    batch_sizes_list = list(grad_variance.keys())
    variances = list(grad_variance.values())
    bars = ax.bar(range(len(batch_sizes_list)), variances,
                   color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"][: len(batch_sizes_list)], alpha=0.8)
    ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gradient Variance", fontsize=12, fontweight="bold")
    ax.set_title("Gradient Noise vs Batch Size\n(Higher variance = More noise)",
                  fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(range(len(batch_sizes_list)), batch_sizes_list)
    ax.grid(True, alpha=0.3, axis="y")
    ax.bar_label(bars, fmt="%.4f", padding=3)

    plt.tight_layout()
    path = os.path.join(output_dir, "gradient_noise_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\n Saved: {path}")
    plt.show()
