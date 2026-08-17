"""
ofdm_system.py

Core OFDM transceiver building blocks:
  - QPSK modulation / demodulation
  - Multipath Rayleigh fading channel generation (exponential power-delay profile)
  - A time-domain OFDM tx/rx chain (explicit cyclic prefix insertion/removal and
    linear convolution with the channel) used for verification / demonstration.
  - A fast, vectorized frequency-domain channel model used for bulk dataset
    generation and Monte Carlo BER simulation.

Why two channel-application paths exist:
    Once the cyclic prefix (CP) is at least as long as the channel (N_CP >= L-1),
    the standard OFDM CP derivation shows that removing the CP at the receiver
    turns the linear convolution into an N_FFT-point CIRCULAR convolution of the
    transmit samples with the channel. By the circular convolution theorem this
    means, in the frequency domain:

        Y[k] = H[k] * X[k] + N[k]        for every subcarrier k

    This is exact (not an approximation) given the CP condition, so it is used
    directly for fast, vectorized batch generation instead of re-simulating
    every sample in the time domain. `sanity_check()` at the bottom verifies the
    two paths agree numerically.
"""

import numpy as np

# ---------------------------------------------------------------------------
# System parameters (LTE/Wi-Fi-like OFDM numerology, scaled down for a
# tractable student/portfolio project)
# ---------------------------------------------------------------------------
N_FFT = 64                              # number of OFDM subcarriers
N_CP = 16                               # cyclic prefix length (samples)
N_TAPS = 8                              # number of multipath channel taps (L)
PILOT_STEP = 4                          # comb-type pilot spacing
PILOT_IDX = np.arange(0, N_FFT, PILOT_STEP)
DATA_IDX = np.array([k for k in range(N_FFT) if k not in PILOT_IDX])
N_PILOT = len(PILOT_IDX)
N_DATA = len(DATA_IDX)
PILOT_SYMBOL = (1 + 1j) / np.sqrt(2)    # fixed, known unit-energy QPSK pilot

assert N_CP >= N_TAPS - 1, "CP must be >= L-1 to avoid ISI/ICI"


# ---------------------------------------------------------------------------
# QPSK modulation
# ---------------------------------------------------------------------------
def qpsk_modulate(bits):
    """Gray-coded, unit-energy QPSK. bits: (..., 2) of {0,1} -> complex symbols (...,)"""
    bits = np.asarray(bits)
    i = 1 - 2 * bits[..., 0]
    q = 1 - 2 * bits[..., 1]
    return (i + 1j * q) / np.sqrt(2)


def qpsk_demodulate(symbols):
    """Complex symbols (...,) -> bits (..., 2) via sign (nearest constellation point) detection."""
    b0 = (np.real(symbols) < 0).astype(int)
    b1 = (np.imag(symbols) < 0).astype(int)
    return np.stack([b0, b1], axis=-1)


# ---------------------------------------------------------------------------
# Channel generation
# ---------------------------------------------------------------------------
def generate_channel(batch_size, num_taps=N_TAPS, decay=2.5, rng=None):
    """
    Rayleigh multipath channel, exponential power-delay profile (PDP).
    Returns:
      h      : (batch, num_taps) complex time-domain taps, E[sum_l |h_l|^2] = 1
      H_freq : (batch, N_FFT) complex frequency response = FFT(zero-padded h),
               E[|H[k]|^2] = 1 for every subcarrier k.
    """
    rng = rng or np.random.default_rng()
    tap_idx = np.arange(num_taps)
    pdp = np.exp(-tap_idx / decay)
    pdp = pdp / pdp.sum()                       # normalize total tap energy to 1

    std = np.sqrt(pdp / 2)                      # per-tap std for real & imag parts
    h_real = rng.standard_normal((batch_size, num_taps)) * std
    h_imag = rng.standard_normal((batch_size, num_taps)) * std
    h = h_real + 1j * h_imag

    h_padded = np.zeros((batch_size, N_FFT), dtype=complex)
    h_padded[:, :num_taps] = h
    H_freq = np.fft.fft(h_padded, axis=-1)
    return h, H_freq


# ---------------------------------------------------------------------------
# Resource-grid assembly
# ---------------------------------------------------------------------------
def build_freq_grid(data_symbols, pilot_symbols=None):
    """
    data_symbols : (batch, N_DATA) complex
    pilot_symbols: (batch, N_PILOT) complex, defaults to the fixed known pilot
    returns X    : (batch, N_FFT) complex resource grid
    """
    batch = data_symbols.shape[0]
    if pilot_symbols is None:
        pilot_symbols = np.full((batch, N_PILOT), PILOT_SYMBOL)
    X = np.zeros((batch, N_FFT), dtype=complex)
    X[:, DATA_IDX] = data_symbols
    X[:, PILOT_IDX] = pilot_symbols
    return X


# ---------------------------------------------------------------------------
# Time-domain OFDM tx/rx (verification / demonstration path)
# ---------------------------------------------------------------------------
def ofdm_tx_time_domain(X_freq):
    """X_freq: (batch, N_FFT) -> time-domain waveform with CP: (batch, N_FFT+N_CP)"""
    x = np.fft.ifft(X_freq, axis=-1)
    x_cp = np.concatenate([x[:, -N_CP:], x], axis=-1)
    return x_cp


def apply_channel_time_domain(x_cp, h, snr_db):
    """
    Linear convolution with a per-example multipath channel + AWGN.
    x_cp: (batch, N_FFT+N_CP), h: (batch, N_TAPS)
    Loops over the batch (used only for small demo batches / the sanity check,
    not for bulk dataset generation).
    """
    batch = h.shape[0]
    out_len = x_cp.shape[1]
    y = np.zeros((batch, out_len), dtype=complex)
    for i in range(batch):
        conv = np.convolve(x_cp[i], h[i])
        y[i] = conv[:out_len]

    snr_lin = 10 ** (snr_db / 10)
    noise_power = 1.0 / snr_lin          # Es = 1 (E[|H|^2]=1, E[|X|^2]=1)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape)
    )
    return y + noise


def ofdm_rx_time_domain(y_cp):
    """Remove CP + FFT. y_cp: (batch, N_FFT+N_CP) -> Y_freq: (batch, N_FFT)"""
    y = y_cp[:, N_CP:]
    return np.fft.fft(y, axis=-1)


# ---------------------------------------------------------------------------
# Fast vectorized frequency-domain equivalent (bulk dataset / Monte Carlo)
# ---------------------------------------------------------------------------
def apply_channel_freq_domain(X_freq, H_freq, snr_db, rng=None):
    """
    Y[k] = H[k] X[k] + N[k]  (exact given N_CP >= N_TAPS - 1, see module docstring).
    X_freq, H_freq: (batch, N_FFT) complex
    """
    rng = rng or np.random.default_rng()
    snr_lin = 10 ** (snr_db / 10)
    noise_power = 1.0 / snr_lin
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(X_freq.shape) + 1j * rng.standard_normal(X_freq.shape)
    )
    return H_freq * X_freq + noise


def sanity_check(n_trials=200, seed=0):
    """Confirms the time-domain and frequency-domain channel paths agree (noiseless)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_trials, N_DATA, 2))
    data_syms = qpsk_modulate(bits)
    X = build_freq_grid(data_syms)
    h, H = generate_channel(n_trials, rng=rng)

    x_cp = ofdm_tx_time_domain(X)
    y_cp = apply_channel_time_domain(x_cp, h, snr_db=200)  # ~noiseless
    Y_time = ofdm_rx_time_domain(y_cp)

    Y_freq = H * X  # noiseless frequency-domain equivalent

    rel_err = np.mean(np.abs(Y_time - Y_freq) ** 2) / np.mean(np.abs(Y_freq) ** 2)
    return rel_err


if __name__ == "__main__":
    rel_err = sanity_check()
    print(f"N_FFT={N_FFT}, N_CP={N_CP}, N_TAPS={N_TAPS}, N_PILOT={N_PILOT}, N_DATA={N_DATA}")
    print(f"Time-domain vs frequency-domain relative MSE (should be ~1e-20 or smaller): {rel_err:.3e}")
    assert rel_err < 1e-10, "Time-domain and frequency-domain paths disagree!"
    print("PASSED: CP-based time-domain simulation matches the frequency-domain model exactly.")
