# Large Batch vs Small Batch Training: A Comprehensive Analysis

A deep dive into how batch size affects neural network generalization,
convergence speed, and training efficiency — originally a single Colab
notebook, split here into a reusable Python project.

## Project Overview

- Analyze how batch size affects model generalization, convergence, and performance.
- Compare training dynamics across three synthetic datasets of different sizes (1K / 10K / 50K samples).
- Visualize the relationship between batch size, accuracy, and training speed.
- Investigate the impact of the Linear Scaling Rule (batch size ↔ learning rate).
- Explore *why* small batches generalize better via loss-landscape and gradient-noise analysis.

## Research Questions

1. How does batch size affect model generalization?
2. Do smaller batches consistently generalize better?
3. How does batch size impact convergence speed?
4. What is the relationship between batch size and learning rate?
5. Is there a critical batch size beyond which accuracy saturates or degrades?
6. How does dataset size influence the optimal batch size selection?

## Project Structure

```
batch_size_analysis/
├── main.py                  # Entry point — runs the full pipeline end-to-end
├── config.py                 # Seeds, device, batch sizes, learning rate, epochs
├── data.py                   # Builds the Small (1K) / Medium (10K) / Large (50K) datasets
├── model.py                  # NeuralNetwork architecture (Dense 64 → 32 → 2)
├── train.py                  # train_model(): training/eval loop with metric tracking
├── experiments.py             # Runs the dataset x batch-size experiment grid
├── lr_scaling_analysis.py    # Fixed-LR vs Linearly-Scaled-LR comparison
├── loss_landscape.py         # 2D loss-surface visualization around converged weights
├── gradient_noise.py         # Gradient variance vs batch size
├── visualize.py              # All plotting functions (dashboard, heatmaps, 3D surfaces)
├── report.py                 # Prints key findings & recommendations
├── requirements.txt
└── outputs/                  # PNG charts + results CSV are written here
```

Each module does one job and can be imported independently — e.g. you can
`import model` and `import train` in your own script without pulling in the
plotting or experiment-orchestration code.

## Neural Network Architecture

Kept constant across every experiment so batch size is the only variable
under study:

```
Input (20 features)
  → Dense(64) → ReLU → Dropout(0.2)
  → Dense(32) → ReLU → Dropout(0.2)
  → Output(2)
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

This reproduces the full notebook flow:

1. **Data prep** — builds the Small/Medium/Large datasets (`data.py`)
2. **Main experiment grid** — trains a model for every dataset × batch size
   `[1, 8, 32, 64, 128, 256, 512]` combination, scaling the learning rate
   linearly with batch size (`experiments.py`)
3. **Dashboard** — a 7-panel figure: accuracy vs batch size, generalization
   gap, training speed, convergence speed, two heatmaps, and learning curves
   (`visualize.py` → `outputs/batch_size_analysis.png`)
4. **LR scaling experiment** — fixed vs. scaled learning rate on the Large
   dataset (`lr_scaling_analysis.py` → `outputs/lr_scaling_analysis.png`)
5. **Key findings & recommendations** printed to the console (`report.py`)
6. **Loss landscape** — 3D loss surfaces for batch sizes 32 vs 512, showing
   flat vs. sharp minima (`loss_landscape.py` → `outputs/loss_landscape_analysis.png`)
7. **Gradient noise analysis** — gradient variance across batch sizes
   `[8, 32, 128, 512]` (`gradient_noise.py` → `outputs/gradient_noise_analysis.png`)
8. **Export** — final metrics table saved to `outputs/batch_size_experiment_results.csv`

To run a single stage, import just what you need, e.g.:

```python
from config import DEVICE, BASE_LR, EPOCHS
from data import prepare_datasets
from experiments import run_experiments, build_results_dataframe

datasets, info = prepare_datasets()
results = run_experiments(datasets, [8, 32, 128], BASE_LR, EPOCHS, DEVICE)
df = build_results_dataframe(results, datasets, [8, 32, 128])
```

## Key Findings

- **No single best batch size** — the optimal choice depends on dataset
  size, compute budget, and whether you're optimizing for accuracy or speed.
- **Generalization favors smaller batches** — smaller batches consistently
  show a smaller generalization gap, likely by settling into flatter minima.
- **Learning rate scaling is non-negotiable** — large-batch training only
  matches small-batch accuracy when the learning rate is scaled up
  proportionally (Linear Scaling Rule).
- **Larger batches are hardware-efficient** — faster epochs, but sometimes
  need more epochs total to converge, eating into that speed advantage.
- **Practical starting point** — a batch size of 32 or 64 is a robust
  default; tune from there based on your accuracy/time trade-off.

## Requirements

See `requirements.txt`. Core dependencies: `torch`, `torchvision`, `numpy`,
`pandas`, `scikit-learn`, `matplotlib`, `seaborn`.
