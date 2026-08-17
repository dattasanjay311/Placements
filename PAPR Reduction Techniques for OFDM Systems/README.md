# PAPR Reduction Techniques for OFDM Systems

## Overview
OFDM (used in LTE, 5G NR, and Wi-Fi) suffers from high **Peak-to-Average
Power Ratio (PAPR)** because the transmitted signal is a sum of many
independently modulated subcarriers, which can constructively add up to
large instantaneous peaks. High PAPR forces the power amplifier (PA) to
operate with a large back-off, reducing power efficiency — a critical
concern in modem/RF design.

This project simulates a 64-subcarrier QPSK OFDM system in Python and
compares three standard PAPR reduction techniques against an uncoded
baseline, using Monte Carlo simulation over 20,000 OFDM symbols.

## Techniques implemented
1. **Clipping** — hard-limits the time-domain signal amplitude at a
   threshold relative to the RMS power. Simple and effective but
   introduces nonlinear distortion (in-band + out-of-band), which
   degrades BER and spectral mask compliance.
2. **Selective Mapping (SLM)** — generates U independent phase-rotated
   candidate versions of the OFDM symbol and transmits the one with the
   lowest PAPR (with the chosen phase sequence sent as side information).
   Distortion-free.
3. **Partial Transmit Sequence (PTS)** — partitions subcarriers into V
   sub-blocks, applies a phase rotation to each sub-block's IFFT output,
   and searches phase combinations to minimize PAPR. Distortion-free.

## Results (PAPR at CCDF = 1e-2, i.e. exceeded by only 1% of symbols)

| Method    | PAPR (dB) | Reduction vs baseline |
|-----------|-----------|------------------------|
| Baseline  | 9.72      | —                       |
| Clipping  | 4.72      | 5.01 dB                 |
| SLM (U=8) | 7.18      | 2.55 dB                 |
| PTS (V=4) | 7.36      | 2.37 dB                 |

See `papr_ccdf.png` for the full CCDF curves.

## Key takeaway
Clipping gives the largest raw PAPR reduction but at the cost of signal
distortion (increased BER, spectral regrowth) — a trade-off real modem
designs must manage carefully. SLM and PTS achieve meaningful,
distortion-free reduction at the cost of extra computation and (for SLM)
transmitting side information.

## How to run
```bash
python3 papr_reduction.py
```
Requires `numpy` and `matplotlib`.

## Suggested resume bullet
> Simulated OFDM PAPR reduction techniques (Clipping, SLM, PTS) in Python
> via Monte Carlo analysis (20K symbols), achieving up to 5 dB PAPR
> reduction; characterized distortion-free (SLM/PTS) vs. distortion-based
> (clipping) trade-offs relevant to PA back-off and modem power efficiency.
