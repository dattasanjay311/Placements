"""
PAPR Reduction Techniques for OFDM Systems
--------------------------------------------
Simulates an OFDM transmitter and compares three PAPR (Peak-to-Average
Power Ratio) reduction techniques against an uncoded baseline:

    1. Clipping
    2. Selective Mapping (SLM)
    3. Partial Transmit Sequence (PTS)

Output:
    - CCDF plot (Pr[PAPR > PAPR0]) comparing all methods
    - Printed summary table of PAPR@1e-2 (dB) for each method

Author: <you>
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Parameters -----------------------------
N_SUBCARRIERS   = 64          # number of OFDM subcarriers
OVERSAMPLE      = 4           # oversampling factor (for accurate PAPR via IFFT)
N_SYMBOLS       = 20000       # number of OFDM symbols to simulate (Monte Carlo)
MOD_ORDER       = 4           # QPSK

# SLM parameters
SLM_CANDIDATES  = 8           # number of phase-rotated candidate sequences (U)

# PTS parameters
PTS_SUBBLOCKS   = 4           # number of partitions (V)
PTS_PHASES      = [1, -1, 1j, -1j]  # allowed rotation factors per sub-block

# Clipping parameter
CLIP_RATIO_DB   = 4.0         # clipping threshold relative to RMS power (dB)

rng = np.random.default_rng(42)


# --------------------------- Helper functions ---------------------------
def qpsk_symbols(n):
    """Generate n random QPSK symbols with unit average energy."""
    bits = rng.integers(0, 2, size=(n, 2))
    i = 1 - 2 * bits[:, 0]
    q = 1 - 2 * bits[:, 1]
    return (i + 1j * q) / np.sqrt(2)


def ofdm_modulate(freq_symbols, oversample=OVERSAMPLE):
    """IFFT with zero-padding (oversampling) to get the time-domain OFDM signal."""
    n = len(freq_symbols)
    padded = np.zeros(n * oversample, dtype=complex)
    padded[: n // 2] = freq_symbols[: n // 2]
    padded[-(n // 2):] = freq_symbols[n // 2:]
    return np.fft.ifft(padded) * oversample


def papr_db(time_signal):
    """PAPR in dB for a single OFDM time-domain symbol."""
    power = np.abs(time_signal) ** 2
    peak = np.max(power)
    avg = np.mean(power)
    return 10 * np.log10(peak / avg)


def ccdf(papr_values, grid):
    """Complementary CDF: Pr[PAPR > x] for each x in grid."""
    papr_values = np.asarray(papr_values)
    return np.array([(papr_values > x).mean() for x in grid])


# --------------------------- PAPR reduction methods ---------------------------
def baseline_papr():
    """Standard OFDM, no PAPR reduction."""
    vals = np.zeros(N_SYMBOLS)
    for k in range(N_SYMBOLS):
        freq = qpsk_symbols(N_SUBCARRIERS)
        time = ofdm_modulate(freq)
        vals[k] = papr_db(time)
    return vals


def clipping_papr(clip_ratio_db=CLIP_RATIO_DB):
    """Hard clipping of the time-domain signal amplitude."""
    vals = np.zeros(N_SYMBOLS)
    for k in range(N_SYMBOLS):
        freq = qpsk_symbols(N_SUBCARRIERS)
        time = ofdm_modulate(freq)
        rms = np.sqrt(np.mean(np.abs(time) ** 2))
        clip_level = rms * 10 ** (clip_ratio_db / 20)
        mag = np.abs(time)
        scale = np.minimum(1, clip_level / (mag + 1e-12))
        clipped = time * scale
        vals[k] = papr_db(clipped)
    return vals


def slm_papr(u=SLM_CANDIDATES):
    """Selective Mapping: generate U phase-rotated candidates, keep the
    lowest-PAPR one (as the transmitter would)."""
    vals = np.zeros(N_SYMBOLS)
    for k in range(N_SYMBOLS):
        freq = qpsk_symbols(N_SUBCARRIERS)
        best = np.inf
        for _ in range(u):
            phases = np.exp(1j * rng.uniform(0, 2 * np.pi, N_SUBCARRIERS))
            candidate_freq = freq * phases
            time = ofdm_modulate(candidate_freq)
            p = papr_db(time)
            if p < best:
                best = p
        vals[k] = best
    return vals


def pts_papr(v=PTS_SUBBLOCKS, phases=PTS_PHASES):
    """Partial Transmit Sequence: split subcarriers into V disjoint
    sub-blocks, rotate each sub-block's IFFT output by a phase factor,
    and pick the phase combination that minimizes PAPR."""
    vals = np.zeros(N_SYMBOLS)
    subblock_size = N_SUBCARRIERS // v

    # Precompute all phase combinations for the sub-blocks (first block fixed
    # to phase 1 to avoid redundant search)
    from itertools import product
    combos = list(product(phases, repeat=v - 1))

    for k in range(N_SYMBOLS):
        freq = qpsk_symbols(N_SUBCARRIERS)

        # Partition into V sub-blocks (interleaved partitioning for better performance)
        sub_freqs = []
        for b in range(v):
            block = np.zeros(N_SUBCARRIERS, dtype=complex)
            block[b::v] = freq[b::v]
            sub_freqs.append(block)

        sub_times = [ofdm_modulate(sf) for sf in sub_freqs]

        best = np.inf
        for combo in combos:
            full_combo = (1,) + combo
            combined = sum(c * t for c, t in zip(full_combo, sub_times))
            p = papr_db(combined)
            if p < best:
                best = p
        vals[k] = best
    return vals


# --------------------------------- Main ---------------------------------
def main():
    print("Simulating baseline OFDM ...")
    baseline = baseline_papr()

    print("Simulating Clipping ...")
    clipped = clipping_papr()

    print("Simulating SLM ...")
    slm = slm_papr()

    print("Simulating PTS (this is the slowest) ...")
    # PTS is combinatorially expensive, use fewer symbols for speed
    global N_SYMBOLS
    n_symbols_backup = N_SYMBOLS
    N_SYMBOLS = 2000
    pts = pts_papr()
    N_SYMBOLS = n_symbols_backup

    # ---- CCDF plot ----
    grid = np.linspace(2, 13, 200)
    ccdf_baseline = ccdf(baseline, grid)
    ccdf_clipped = ccdf(clipped, grid)
    ccdf_slm = ccdf(slm, grid)
    ccdf_pts = ccdf(pts, grid)

    plt.figure(figsize=(7, 5))
    plt.semilogy(grid, ccdf_baseline, label="No reduction (baseline)", linewidth=2)
    plt.semilogy(grid, ccdf_clipped, label=f"Clipping ({CLIP_RATIO_DB} dB)", linewidth=2)
    plt.semilogy(grid, ccdf_slm, label=f"SLM (U={SLM_CANDIDATES})", linewidth=2)
    plt.semilogy(grid, ccdf_pts, label=f"PTS (V={PTS_SUBBLOCKS})", linewidth=2)
    plt.xlabel("PAPR$_0$ (dB)")
    plt.ylabel(r"CCDF: Pr[PAPR > PAPR$_0$]")
    plt.title(f"PAPR CCDF Comparison (N={N_SUBCARRIERS} subcarriers, QPSK)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("papr_ccdf.png", dpi=150)
    print("Saved plot to papr_ccdf.png")

    # ---- Summary table: PAPR at CCDF = 1e-2 ----
    def papr_at_prob(vals, prob=1e-2):
        sorted_vals = np.sort(vals)
        idx = int((1 - prob) * len(sorted_vals))
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    print("\n--- Summary: PAPR0 (dB) at CCDF = 1e-2 ---")
    results = {
        "Baseline": papr_at_prob(baseline),
        "Clipping": papr_at_prob(clipped),
        "SLM": papr_at_prob(slm),
        "PTS": papr_at_prob(pts),
    }
    for name, val in results.items():
        reduction = results["Baseline"] - val
        print(f"{name:10s}: {val:5.2f} dB   (reduction vs baseline: {reduction:5.2f} dB)")

    return results


if __name__ == "__main__":
    main()
