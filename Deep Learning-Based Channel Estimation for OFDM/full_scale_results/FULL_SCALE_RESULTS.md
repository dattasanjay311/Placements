# Full-Scale Training Results — Large Batch vs. Small Batch Training

This document explains what happens when the batch-size study is run on the
**full-scale training configuration** (real dataset sizes, full epoch budgets,
and the extended dataset/architecture list used in the final project
presentation) rather than the quick, reduced-scale demo included in this
repo (`demo_reduced_scale_run/`).

> Source: *"Large Batch vs. Small Batch Training: Generalization Tradeoffs in
> Deep Learning"* — CS 725 Course Project, IIT Bombay. Team: Anish Pal,
> Miriyala Indu Vardhan, Palagati Venkata Kevanth Reddy, Rehan Khan,
> Powrohitham Sanjay Datta. Guide: Prof. Abir De.

## Why the numbers differ from the quick demo run

The demo run in this repo (`python demo_run.py`) intentionally uses tiny
datasets (600 / 3,000 / 8,000 samples) and only 10 epochs so it finishes in
about two minutes on a CPU. That's useful for checking the pipeline works,
but it is **not** representative of the actual research findings — the
models never fully converge or overfit at that scale, and the "optimal batch
size" trend it produces runs backwards from theory.

The numbers below come from the **full-scale runs**: the real dataset sizes,
50 epochs (12 for the fastest configs), and the additional MNIST/CIFAR-10
experiments that were part of the final study. These are the results that
should be cited/reported as the project's findings.

## Methodology (full-scale configuration)

**Datasets**

| Dataset | Train Size | Test Size | Dimensions | Complexity |
|---|---|---|---|---|
| Small Synthetic | 800 | 200 | 20 features | Low |
| Medium Synthetic | 8,000 | 2,000 | 20 features | Low |
| Large Synthetic | 40,000 | 10,000 | 20 features | Low |
| MNIST | 60,000 | 10,000 | 28×28 (grayscale) | Medium |
| CIFAR-10 | 50,000 | 10,000 | 32×32×3 (color) | High |

**Architectures**

- *Feedforward network* (synthetic datasets): 20 input features → Dense(64,
  ReLU) → Dense(32, ReLU) → binary classification output, 20% dropout.
- *CNNs* (MNIST / CIFAR-10):
  - MNIST: 2 conv layers (32, 64 filters) → MaxPool → FC layer.
  - CIFAR-10: 3 conv layers (32, 64, 128 filters) → AdaptiveAvgPool, with
    standard data augmentation.

**Training configuration**

- Batch sizes tested: 1, 8, 32, 64, 128, 256, 512, 1024
- Optimizers: Adam (synthetic datasets, CIFAR-10); SGD with momentum 0.9
  (MNIST)
- Epochs: fixed per dataset, ranging 12–50
- LR warmup: 5 epochs for batch size ≥ 256
- Linear Scaling Rule applied throughout: η = η_base × (B_new / B_base)

## Results: Synthetic Datasets

Optimal batch size **increases with the amount of training data**:

| Dataset | Optimal Batch Size | Best Test Accuracy |
|---|---|---|
| Small (1K) | 256 | 97.0% |
| Medium (10K) | 128 | 98.85% |
| Large (50K) | 32 | 99.37% |

Interesting nuance on the small (1K) dataset specifically — accuracy
*degrades* as batch size grows past a point, illustrating that with very
little data, small batches' implicit regularization matters more than
hardware efficiency:

| Batch Size | Test Accuracy (Small/1K dataset) |
|---|---|
| 8 | 89.5% |
| 64 | 87.0% |
| 512 | 85.0% |

The small dataset showed the largest degradation (4.5 percentage points) as
batch size increased, underscoring the need for regularization in
data-scarce regimes.

## Results: MNIST

A stable "sweet spot" emerged:

- **Peak performance:** 99.18% accuracy at batch size 64.
- **Optimal range:** batch sizes 32–128.
- Very small batches (1) suffered from excessive noise; very large batches
  (>512) began to show a widening generalization gap.

## Results: CIFAR-10

Batch size sensitivity is much more pronounced on this harder, more
visually complex dataset:

- Batch 64: ~80% accuracy (optimal).
- Batch 1024: catastrophic drop to ~10% accuracy.
- Complex, high-variance datasets (natural images) are far more sensitive to
  batch size choice than simple datasets like MNIST.

## Generalization Gap Analysis

- For batch sizes ≤ 64, the train/test accuracy gap stayed below 2% across
  all datasets.
- A Pearson correlation of **r = 0.87** was measured between log(batch size)
  and generalization gap — a strong, statistically confirmed relationship.
- On the Large (50K) synthetic dataset specifically, the gap widened from
  ~0.5% at batch size 8 to ~2.7% at batch size 512.

## The Efficiency Paradox

- Large batches process ~50× faster per epoch (better hardware utilization).
- But they need ~2× as many epochs to reach comparable accuracy.
- Net effect: the raw per-epoch speedup is partially — not fully — offset by
  slower convergence in terms of epochs required.

## Loss Landscape: Sharp vs. Flat Minima

Sharpness was quantified directly by measuring how fast loss increases
around the converged weights:

| Batch Size | Sharpness Score |
|---|---|
| 32 | 0.023 |
| 512 | 0.158 |

Batch-512 minima are **~6.9× sharper** than batch-32 minima — direct
confirmation that large-batch training settles into sharp basins that
generalize worse, while small-batch training finds flatter, more robust
minima.

## Gradient Noise as Implicit Regularization

Gradient variance across mini-batches follows the theoretical relationship:

```
Var_B = (1/B) · σ²
```

where B is batch size and σ² is the per-sample gradient variance. A
regression of log(gradient variance) against log(batch size) produced a
slope of **-0.98**, almost exactly matching the theoretical 1/B scaling.
This noise is the mechanism believed to drive small-batch training toward
flatter minima — effectively acting as implicit regularization.

## Practical Guidelines (decision rule used in this project)

| Condition | Recommended Batch Size |
|---|---|
| N < 5,000 samples | 16–32 |
| 5,000 < N < 20,000 samples | 32–64 |
| N > 20,000 samples | 64–128 |
| Generalization gap > 5% | Halve the batch size |
| Always | Scale the learning rate linearly with batch size |

## Conclusion

1. **The generalization gap is real** — large batches measurably degrade a
   model's ability to generalize to unseen data.
2. **Geometry matters** — the sharp-vs-flat minima hypothesis was validated
   directly through loss landscape visualization (6.9× sharpness increase at
   batch 512 vs. batch 32).
3. **The optimal batch size is dataset-size dependent**, scaling roughly
   with √N — larger datasets can tolerate (and benefit from) larger batches.

**Bottom line:** large batches offer speed; small batches offer robustness.
The right choice is a balance, tuned to your dataset size and your
accuracy/time budget — not a single universal default.
