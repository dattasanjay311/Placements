# Autoencoder-Based End-to-End Communication System

A from-scratch implementation of the O'Shea & Hoydis "learn the whole
physical layer" idea: instead of designing modulation and channel coding
separately, a transmitter and receiver are represented as a single neural
network with a real, non-trainable AWGN channel embedded *inside* the
forward pass, and trained jointly end-to-end. No modulation theory or
coding theory is given to the network — it discovers both from gradient
descent alone.

Pure Python (NumPy + PyTorch). No hardware, no toolboxes.

## Why this project

Classical wireless systems design channel coding and modulation as two
separate, independently-optimized blocks, each with a closed-form
mathematical model. The autoencoder framing (O'Shea & Hoydis, 2017) treats
the *entire chain* — transmitter, channel, receiver — as one differentiable
system and optimizes it jointly for a single objective: minimize the
probability that the receiver recovers the wrong message. This is the
foundational result behind the "AI-native air interface" research direction
increasingly discussed for 6G, and it's structurally very different from
classical channel estimation (project 1): this is a *classification*
problem (cross-entropy loss, softmax output), and the channel is a live,
differentiable layer inside the network rather than something used only to
generate training data beforehand.

## System model

| Concept | This project's implementation |
|---|---|
| Message | One of $M = 2^k$ possible messages, one-hot encoded |
| Encoder (Tx) | Dense($M\to M$, ReLU) → Dense($M \to n$) → energy normalization |
| Channel | AWGN, $y = x + n$, $n\sim\mathcal{N}(0,\sigma^2 I)$ — **live inside the forward pass** |
| Decoder (Rx) | Dense($n\to M$, ReLU) → Dense($M\to M$) → (softmax via the loss) |
| Constraint | $\|x\|^2 = n$ exactly for every transmitted block (energy normalization) |
| Rate | $R = k/n$ bits/channel use |
| Loss | Cross-entropy between the true message and the decoder's output |

**The key architectural difference from project 1**: the channel here is a
non-trainable *layer*, not a data-generation step. Gradients flow from the
loss, through the channel ($\partial(x+n)/\partial x = 1$, trivial but
essential), all the way back into the encoder's weights — this is what lets
transmitter and receiver be optimized *jointly*, discovering a
constellation/code matched to the exact channel and objective, rather than
combining a hand-designed modulation scheme with a separately-designed code.

### Getting Eb/N0 right (and proving it)

Comparing schemes with different rates (e.g. a rate-1 uncoded baseline
against a rate-4/7 coded scheme) requires normalizing by *information bit*
energy, not raw signal power — otherwise a scheme that "spends" more
channel uses per message looks unfairly penalized or rewarded. Given the
energy constraint $\|x\|^2=n$, energy per information bit is $E_b = n/k =
1/R$, and combined with the standard real-AWGN convention $\sigma^2=N_0/2$:

$$\sigma^2 = \frac{1}{2 R \,(E_b/N_0)_{\text{linear}}}$$

This formula is not taken on faith — `channel.py` simulates uncoded BPSK
with it and checks the result against the textbook closed-form
$\text{BER}=Q(\sqrt{2E_b/N_0})$, matching to within 3% (Monte Carlo noise)
across a 0–8 dB sweep before the formula is trusted for anything else.

## Two experiments

**(n=2, k=2, M=4)** — small enough that the encoder's output *is* a 2D
point, so the learned "constellation" can be plotted directly.

**(n=7, k=4, M=16)** — the classical (7,4) block-code comparison point,
benchmarked against a real Hamming(7,4) implementation (not a textbook
formula copied in — built programmatically from its parity-check matrix and
exhaustively verified: minimum distance 3, and **all 112** single-bit-error
cases across all 16 codewords decode correctly before it's trusted as a
baseline).

## Results

### The learned constellation

With zero knowledge of modulation theory, gradient descent alone
rediscovers something extremely close to a phase-rotated QPSK constellation
— the same conclusion classical theory reaches analytically for "4 equal
energy points under AWGN, minimize error" (maximize minimum pairwise
distance ⇒ corners of a square):

| Message | Learned point | Energy |
|---|---|---|
| 00 | (−1.307, +0.539) | 2.000 |
| 01 | (+0.534, +1.310) | 2.000 |
| 10 | (−0.550, −1.303) | 2.000 |
| 11 | (+1.300, −0.557) | 2.000 |

All 4 edges measure 1.99–2.02 (theoretical perfect square: 2.000); both
diagonals measure exactly 2.828 (theoretical: 2.828). See
`results/constellation.png` for the constellation plotted together with the
receiver's learned decision regions — each point sits at the center of its
own region, maximizing noise margin, exactly as theory predicts it should.

### BLER vs. Eb/N0 — learned vs. classical coding

*(Adaptive Monte Carlo, minimum 200 block errors or 4,000,000 samples per
point; full numbers and per-point sample counts in `results/summary.txt`.)*

| Eb/N0 (dB) | Uncoded | Hamming HDD | Autoencoder | Hamming MLD |
|---:|---:|---:|---:|---:|
| 0 | 0.2795 | 0.2649 | 0.1942 | 0.1770 |
| 3 (training pt.) | 0.0879 | 0.0721 | 0.0377 | 0.0297 |
| 5 | 0.0245 | 0.0144 | 0.0046 | 0.0033 |
| 8 | 0.00072 | 0.00028 | 0.00002 | 0.00001 |

At Eb/N0 needed to reach BLER = $10^{-2}$: uncoded BPSK needs 6.01 dB,
Hamming hard-decision needs 5.38 dB (a **0.63 dB** coding gain — matching
the ~0.6 dB documented in MathWorks' independent reference implementation
of this same experiment), the **autoencoder needs 4.28 dB**, and the
Hamming ML-decoding upper bound needs 4.18 dB. In other words, **the
autoencoder closes 91% of the gap between hard-decision and
maximum-likelihood Hamming decoding** — approaching the best a *fixed*
(7,4) code could possibly achieve, despite starting with no coding-theory
knowledge at all. See `results/bler_comparison.png`.

The Hamming hard-decision Monte Carlo curve is cross-checked against a
closed-form formula (exact for this code, since Hamming(7,4) is a *perfect*
single-error-correcting code: it always correctly decodes ≤1 bit error and
always mis-corrects ≥2 bit errors) — the two match to within 8.35% max
relative error, consistent with expected statistical noise at a ~200-error
sample size, confirming the simulation pipeline itself is correct
independent of the learned model.

**A methodological note for full transparency**: the lowest-BLER points
(around 8 dB, for Hamming-MLD and the autoencoder) hit the 4,000,000-sample
cap before reaching the 200-error target (only ~40–80 errors observed
there), so those specific points carry more Monte Carlo variance than the
rest of the curve — worth knowing before quoting them as precise.

## Project structure

```
autoencoder_comms/
├── channel.py                  # Eb/No <-> noise variance, verified against textbook BPSK BER
├── hamming_baseline.py          # (7,4) Hamming code, built + exhaustively verified programmatically
├── autoencoder_model.py          # Encoder / AWGN channel layer / Decoder
├── train.py                       # Trains both (2,2) and (7,4) configs
├── evaluate_bler.py                 # Monte Carlo BLER comparison, all 4 schemes
├── visualize_constellation.py        # Constellation + decision-region plot
├── model_n2k2.pt / model_n7k4.pt      # Trained weights
├── requirements.txt
└── results/
    ├── training_curves.png
    ├── constellation.png
    ├── bler_comparison.png
    └── summary.txt
```

## How to run

```bash
pip install -r requirements.txt

python3 channel.py                  # verify Eb/No formula against BPSK theory (~instant)
python3 hamming_baseline.py         # exhaustively verify the Hamming(7,4) code (~instant)
python3 autoencoder_model.py        # sanity-check model shapes + energy constraint (~instant)
python3 train.py                    # trains both configs -> model_*.pt (~1 min on CPU)
python3 visualize_constellation.py  # -> results/constellation.png (~instant)
python3 evaluate_bler.py            # -> results/bler_comparison.png, summary.txt (~1-3 min)
```

## Talking points for interviews

- **Why the channel has to be *inside* the network, not applied to
  pre-generated data**: joint optimization of Tx and Rx is only possible if
  gradients can flow through the channel back to the encoder. If the channel
  were applied outside the computation graph (as in project 1's data
  generation), the encoder could never learn — there'd be no path for the
  loss to influence its weights.
- **Why Eb/N0, not raw SNR**: schemes here use different numbers of channel
  uses (4, 7) for the same 4 information bits. Comparing them on Eb/N0
  rather than per-symbol SNR is what makes the comparison fair — and this
  project doesn't just assert the conversion formula, it derives it and
  checks it against the one unambiguous ground truth (theoretical BPSK BER)
  before trusting it anywhere else.
- **Why build Hamming(7,4) from its parity-check matrix instead of using a
  library or a memorized table**: correctness of the *baseline* matters as
  much as correctness of the learned model — a bad baseline would make the
  comparison meaningless either way. Exhaustively checking all 112
  single-error cases (not sampling a few) is proof, not just plausibility.
- **Why the autoencoder lands between Hamming-HDD and Hamming-MLD, not
  beyond MLD**: ML decoding is the *provably optimal* decoder for a fixed
  (7,4) code — nothing can beat it while stuck with that exact code. The
  autoencoder's achievement isn't beating that bound, it's getting to 91%
  of the way there while also designing its own code and modulation jointly,
  with no coding theory given to it at all.
- **Why train at a fixed Eb/No (3 dB) rather than a range**, unlike project
  1's channel estimator: this follows the original paper's methodology
  directly, and creates a natural comparison point with project 1's opposite
  choice — worth being able to discuss both trade-offs (a single well-chosen
  operating point vs. robustness across a range) if asked.

## Possible extensions

- Train (7,4) at several different Eb/No values and compare BLER curves
  (the reference implementation shows performance shifts closer to MLD as
  training Eb/No decreases toward 1–2 dB) — an easy, direct ablation on top
  of what's already built.
- Extend to a Rayleigh fading channel instead of AWGN, following O'Shea &
  Hoydis's own fading-channel experiments.
- Try (2,4) with average-power (rather than energy) normalization, which
  the literature shows converges to an amplitude-and-phase-shift-keyed
  constellation instead of a pure PSK ring — a nice illustration of how the
  power constraint shapes what the network discovers.
- Validate over real hardware with an SDR, following the natural next paper
  in this line of work (Dörner et al., "Deep Learning Based Communication
  Over the Air," 2018) — sending the *actual* learned waveform through a
  real RF channel instead of a simulated one.
