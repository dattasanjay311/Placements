"""
channel.py

AWGN channel utilities for the autoencoder communication system, and the
Eb/N0 <-> noise-variance conversion used throughout the project.

--- Deriving the Eb/N0 -> noise variance relationship ---

The autoencoder sends k information bits using n real-valued channel uses
per block, at code rate R = k/n [bits/channel use]. The transmitter's
output is normalized to an ENERGY constraint: ||x||^2 = n exactly, for
every transmitted block (i.e. average unit power per real dimension).

Given that constraint:
    Eb (energy per information bit) = (total block energy) / k = n/k = 1/R

The standard real-baseband AWGN convention defines noise variance per real
dimension as sigma^2 = N0/2 (N0 = one-sided noise power spectral density).
Combining:
    Eb/N0 = Eb / (2 * sigma^2)
    =>  sigma^2 = Eb / (2 * (Eb/N0)) = 1 / (2 * R * (Eb/N0)_linear)

This is the formula implemented below. It is NOT taken on faith: `verify()`
at the bottom simulates uncoded BPSK (R=1) with this exact noise model and
confirms the Monte Carlo BER matches the textbook closed-form
BER = Q(sqrt(2*Eb/N0)) -- the one unambiguous, universally-agreed reference
point for what "Eb/N0" means. Only after that check passes is the formula
trusted for the coded/learned schemes, where no such simple closed form
exists to check against directly.
"""

import numpy as np
from scipy.special import erfc


def ebno_db_to_noise_std(ebno_db, rate):
    """rate = k/n. Returns the per-real-dimension AWGN standard deviation."""
    ebno_linear = 10 ** (np.asarray(ebno_db) / 10)
    noise_var = 1.0 / (2 * rate * ebno_linear)
    return np.sqrt(noise_var)


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


def bpsk_theoretical_ber(ebno_db):
    """Textbook closed-form BER for uncoded BPSK over real AWGN."""
    ebno_linear = 10 ** (np.asarray(ebno_db) / 10)
    return qfunc(np.sqrt(2 * ebno_linear))


def awgn(x, noise_std, rng=None):
    rng = rng or np.random.default_rng()
    return x + rng.normal(0, noise_std, size=x.shape)


def verify(n_bits=2_000_000, seed=0):
    """
    Simulates uncoded BPSK (rate R=1, unit energy per bit) using this
    module's exact noise model, and compares against the closed-form BER.
    """
    rng = np.random.default_rng(seed)
    ebno_range_db = [0, 2, 4, 6, 8]
    print(f"{'Eb/N0 (dB)':>10} | {'Simulated BER':>14} | {'Theoretical BER':>16} | {'Rel. error':>10}")
    max_rel_err = 0.0
    for ebno_db in ebno_range_db:
        bits = rng.integers(0, 2, size=n_bits)
        x = 1 - 2 * bits.astype(float)          # BPSK: 0->+1, 1->-1, unit energy (rate R=1 => Eb=1/R=1)
        noise_std = ebno_db_to_noise_std(ebno_db, rate=1.0)
        y = awgn(x, noise_std, rng=rng)
        bits_hat = (y < 0).astype(int)
        sim_ber = np.mean(bits_hat != bits)
        theo_ber = bpsk_theoretical_ber(ebno_db)
        rel_err = abs(sim_ber - theo_ber) / theo_ber
        max_rel_err = max(max_rel_err, rel_err)
        print(f"{ebno_db:10d} | {sim_ber:14.6f} | {theo_ber:16.6f} | {rel_err:9.2%}")
    return max_rel_err


if __name__ == "__main__":
    max_err = verify()
    print(f"\nMax relative error across all points: {max_err:.2%}")
    assert max_err < 0.05, "Noise model does not match theoretical BPSK BER -- formula is wrong!"
    print("PASSED: Eb/N0 -> noise variance formula matches theoretical BPSK BER.")
