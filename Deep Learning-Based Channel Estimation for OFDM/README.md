# Deep Learning-Based Channel Estimation for OFDM Systems

A from-scratch OFDM simulator with three channel estimators — classical **LS**,
classical **MMSE**, and a **CNN** trained to denoise the LS estimate — compared
on estimation accuracy (NMSE) and end-to-end bit error rate (BER).

Everything is implemented in pure Python (NumPy for the OFDM physical layer,
PyTorch for the CNN). No MATLAB, no toolboxes, no hardware required.

## Why this project

Channel estimation is the step in every OFDM-based standard (LTE, 5G NR,
Wi-Fi) where the receiver figures out how the wireless channel distorted the
transmitted signal, using a handful of known pilot symbols scattered across
the resource grid. Classical estimators (LS, MMSE) are the industry baseline;
learning-based estimation is an active research direction (e.g. Ye et al.,
*"Power of Deep Learning for Channel Estimation and Signal Detection in OFDM
Systems,"* IEEE Wireless Comm. Letters, 2017; Soltani et al., *"Deep
Learning-Based Channel Estimation,"* IEEE Comm. Letters, 2019) because a
network can implicitly learn channel statistics and denoising behavior that
would otherwise require hand-derived, model-specific formulas. This project
reproduces that comparison end-to-end, including the physical layer that
generates the training data, not just a pre-packaged dataset.

## System model

| Parameter | Value |
|---|---|
| Subcarriers (N_FFT) | 64 |
| Cyclic prefix (N_CP) | 16 |
| Modulation | QPSK, Gray-coded, unit energy |
| Pilot pattern | Comb-type, every 4th subcarrier (16 pilots / 48 data) |
| Channel | Rayleigh multipath, 8 taps, exponential power-delay profile |
| Fading | Block fading (constant per OFDM symbol, independent across symbols) |
| Training SNR range | 0–25 dB |

The simulator implements the **actual time-domain chain** (IFFT → cyclic
prefix → linear convolution with the channel → AWGN → CP removal → FFT) in
`ofdm_system.py`, and a **vectorized frequency-domain equivalent**
(`Y[k] = H[k]X[k] + N[k]`) used for fast bulk dataset generation. These two
are not independent implementations — the frequency-domain form is only exact
because `N_CP ≥ L-1` (standard OFDM design rule that eliminates ISI/ICI); a
`sanity_check()` in `ofdm_system.py` confirms the two paths agree numerically
to within `1e-19` relative error before either is trusted for the rest of the
project.

## Estimators compared

1. **LS (Least Squares)** — `Ĥ = Y_pilot / X_pilot` at pilot subcarriers,
   linearly interpolated across the band. No knowledge of channel statistics
   required; noisy, and has an interpolation-error floor from the comb
   spacing that doesn't vanish even at high SNR.
2. **MMSE (genie-aided)** — combines the LS pilot estimate with the *true*
   channel frequency-correlation matrix (derived analytically from the known
   power-delay profile) and noise variance. This is the standard idealized
   upper bound used in the literature: it assumes side information a real
   receiver would need to separately estimate.
3. **CNN (proposed)** — a compact 1D convolutional network
   (`dl_model.py`, ~11.7K parameters) that takes the interpolated LS estimate
   as input and learns a **residual correction** toward the true channel:
   `H_pred = H_LS + CNN(H_LS)`. Trained once across the full 0–25 dB SNR
   range so a single model generalizes across channel conditions, rather than
   overfitting one noise level. Treats the 64 subcarriers as a 1D "spatial"
   axis, exploiting the same local-correlation idea (coherence bandwidth) that
   motivates image-denoising CNNs — nearby subcarriers fade similarly, so
   local convolutions can learn to smooth out LS noise.

## Results

*(Full numbers in `results/summary.txt`; test set = 4,000 fresh OFDM symbols
per SNR point, independent of training data.)*

**Channel estimation accuracy (Normalized MSE):**

| SNR (dB) | LS | MMSE (genie) | CNN | CNN vs LS |
|---:|---:|---:|---:|---:|
| 0  | 0.713 | 0.282 | 0.325 | −54.5% |
| 5  | 0.238 | 0.122 | 0.146 | −38.5% |
| 10 | 0.093 | 0.046 | 0.059 | −36.4% |
| 15 | 0.045 | 0.015 | 0.022 | −51.2% |
| 20 | 0.030 | 0.005 | 0.010 | −68.7% |
| 25 | 0.026 | 0.002 | 0.005 | −79.0% |

Averaged over the 0–25 dB training range, the **CNN cuts NMSE by ~55%
relative to LS**, while requiring no explicit knowledge of channel statistics
(unlike MMSE). The gap to genie-aided MMSE is expected and is itself a useful
result: MMSE is an idealized bound given information a real system doesn't
have for free.

**End-to-end BER (QPSK, zero-forcing equalization):** the CNN tracks MMSE
closely from 15 dB upward and beats LS by 32–88% in that range; see
`results/ber_vs_snr.png`. At very low SNR (0–5 dB) the estimator matters less
because noise dominates the error floor regardless of channel-estimate
quality — all three curves converge near the "Perfect CSI" bound's shape.

**Training:** converges in 40 epochs (~5.5 min on a single CPU core, no GPU
needed); validation MSE tracks training MSE closely throughout (see
`results/training_loss.png`) — no overfitting, despite the small model.

## Project structure

```
ofdm_channel_estimation/
├── ofdm_system.py            # QPSK mod/demod, channel model, OFDM tx/rx (time + freq domain)
├── classical_estimators.py   # LS and genie-aided MMSE estimators
├── dl_model.py                # CNN architecture (residual channel-estimate refiner)
├── generate_dataset.py        # Builds train/val/test sets across SNR range -> data/dataset.npz
├── train.py                   # Trains the CNN, saves model.pt + results/training_loss.png
├── evaluate.py                # Compares LS / MMSE / CNN -> NMSE & BER plots + summary.txt
├── model.pt                    # Trained weights (reproducible via train.py)
├── results/
│   ├── training_loss.png
│   ├── mse_vs_snr.png
│   ├── ber_vs_snr.png
│   └── summary.txt
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

python3 ofdm_system.py         # sanity-check the physical layer (~instant)
python3 generate_dataset.py    # builds data/dataset.npz (~10s)
python3 train.py               # trains the CNN -> model.pt (~5 min on CPU)
python3 evaluate.py            # produces all plots + results/summary.txt (~30s)
```

## Talking points for interviews

- **Why the CP condition matters:** `N_CP ≥ L-1` is what turns a linear
  convolution into a circular one after CP removal, which is the entire
  reason OFDM can equalize a multipath channel with a single per-subcarrier
  division instead of time-domain deconvolution. This project verifies that
  condition numerically rather than assuming it.
- **Why MMSE beats LS, and why the gap grows with SNR:** LS has a systematic
  interpolation-error floor from comb pilot spacing that doesn't shrink with
  SNR; MMSE's use of the true correlation structure lets it keep improving.
  This is visible directly in the NMSE plot (LS plateaus past ~15 dB, MMSE
  keeps falling).
- **Why residual learning for the CNN:** LS is already unbiased, just noisy —
  learning a correction on top of it is a better-conditioned problem than
  learning channel estimation from raw received samples, and it guarantees
  the network never does worse than "no correction" as a fallback.
- **Why the CNN doesn't match genie-aided MMSE:** MMSE assumes perfect
  knowledge of channel statistics and noise variance; the CNN only ever sees
  noisy pilot observations and must infer that structure implicitly from
  data. Closing that gap without side information is exactly the open
  research question this class of methods targets.

## Possible extensions

- Compare against an LSTM/Transformer estimator that exploits *time*
  correlation across consecutive OFDM symbols (this project uses block
  fading — one channel realization per symbol — so there's no time
  correlation to exploit yet).
- Validate on a 3GPP-standard channel model (TDL/CDL from the 5G NR spec)
  instead of a generic exponential PDP.
- Feed real captured samples from an RTL-SDR into the trained model instead
  of simulated data.
- Extend to MIMO-OFDM (per-antenna-pair channel matrices instead of a single
  SISO response).
