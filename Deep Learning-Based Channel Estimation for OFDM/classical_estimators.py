"""
classical_estimators.py

Classical pilot-based channel estimators for comb-type OFDM:
  - Least Squares (LS) at pilot subcarriers, linearly interpolated across the band.
  - Linear MMSE (LMMSE), using the *true* second-order channel statistics
    (frequency correlation matrix derived analytically from the assumed
    power-delay profile). This is the standard "genie-aided" MMSE benchmark
    used in the channel-estimation literature: it assumes perfect knowledge
    of channel statistics + noise variance, so it upper-bounds what a
    linear, statistics-aware estimator can achieve. LS needs no such
    knowledge; MMSE trades that requirement for better accuracy.
"""

import numpy as np
from ofdm_system import N_FFT, N_TAPS, PILOT_IDX


def ls_estimate(Y_freq, X_freq, pilot_idx=PILOT_IDX):
    """
    LS estimate at pilot positions, linearly interpolated (real & imaginary
    parts separately, with circular wraparound since pilots are periodic)
    across all N_FFT subcarriers.

    Y_freq, X_freq: (batch, N_FFT) complex
    returns:
      H_ls_full  : (batch, N_FFT) complex   -- interpolated estimate, all subcarriers
      H_ls_pilot : (batch, N_PILOT) complex -- raw LS estimate at pilot positions only
    """
    H_ls_pilot = Y_freq[:, pilot_idx] / X_freq[:, pilot_idx]
    batch = Y_freq.shape[0]
    H_ls_full = np.zeros((batch, N_FFT), dtype=complex)
    all_idx = np.arange(N_FFT)
    for i in range(batch):
        re = np.interp(all_idx, pilot_idx, H_ls_pilot[i].real, period=N_FFT)
        im = np.interp(all_idx, pilot_idx, H_ls_pilot[i].imag, period=N_FFT)
        H_ls_full[i] = re + 1j * im
    return H_ls_full, H_ls_pilot


def theoretical_freq_correlation(num_taps=N_TAPS, decay=2.5, n_fft=N_FFT):
    """
    R_HH[k1,k2] = E[H[k1] H[k2]^*], derived analytically from the exponential
    power-delay profile via R_HH = F * diag(pdp) * F^H, where F is the
    n_fft x num_taps DFT sub-matrix (F[k,l] = exp(-j*2*pi*k*l/n_fft)).
    """
    tap_idx = np.arange(num_taps)
    pdp = np.exp(-tap_idx / decay)
    pdp = pdp / pdp.sum()

    k = np.arange(n_fft).reshape(-1, 1)
    l = np.arange(num_taps).reshape(1, -1)
    F = np.exp(-1j * 2 * np.pi * k * l / n_fft)      # (n_fft, num_taps)
    R_HH = F @ np.diag(pdp) @ F.conj().T              # (n_fft, n_fft)
    return R_HH


def mmse_estimate(H_ls_pilot, snr_db, pilot_idx=PILOT_IDX, num_taps=N_TAPS, decay=2.5):
    """
    Linear MMSE estimate combining the LS pilot estimate with known channel
    statistics:
        H_mmse = R_HH[:,pilot] @ (R_HH[pilot,pilot] + noise_power * I)^-1 @ H_ls_pilot
    (the noise_power*I term uses |X_pilot|=1 for unit-energy QPSK pilots, so
    the usual (X_p X_p^H)^-1 scaling in the textbook formula is just I).

    H_ls_pilot: (batch, N_PILOT) complex
    returns H_mmse_full: (batch, N_FFT) complex
    """
    R_HH = theoretical_freq_correlation(num_taps, decay, N_FFT)
    R_pp = R_HH[np.ix_(pilot_idx, pilot_idx)]
    R_fp = R_HH[:, pilot_idx]

    snr_lin = 10 ** (snr_db / 10)
    noise_power = 1.0 / snr_lin
    W = R_fp @ np.linalg.inv(R_pp + noise_power * np.eye(len(pilot_idx)))  # (N_FFT, N_PILOT)

    H_mmse_full = (W @ H_ls_pilot.T).T   # (batch, N_FFT)
    return H_mmse_full


if __name__ == "__main__":
    from ofdm_system import (
        N_DATA, qpsk_modulate, generate_channel, build_freq_grid, apply_channel_freq_domain,
    )

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2000, N_DATA, 2))
    X = build_freq_grid(qpsk_modulate(bits))
    h, H_true = generate_channel(2000, rng=rng)

    for snr_db in [0, 10, 20]:
        Y = apply_channel_freq_domain(X, H_true, snr_db, rng=rng)
        H_ls, H_ls_pilot = ls_estimate(Y, X)
        H_mmse = mmse_estimate(H_ls_pilot, snr_db)
        nmse_ls = np.mean(np.abs(H_ls - H_true) ** 2) / np.mean(np.abs(H_true) ** 2)
        nmse_mmse = np.mean(np.abs(H_mmse - H_true) ** 2) / np.mean(np.abs(H_true) ** 2)
        print(f"SNR {snr_db:3d} dB | LS NMSE = {nmse_ls:.4f} | MMSE NMSE = {nmse_mmse:.4f}")
