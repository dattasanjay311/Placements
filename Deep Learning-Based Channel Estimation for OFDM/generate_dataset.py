"""
generate_dataset.py

Builds train/val/test datasets for the DL channel estimator.

For each SNR level in SNR_RANGE_DB, generates N_PER_SNR OFDM symbols with
independent random multipath channels and random QPSK data, computes the LS
estimate (network input) and the true channel (label), and stores them.
Training across a range of SNRs (rather than one fixed SNR) lets the CNN
learn a single model that generalizes across channel conditions, rather than
overfitting to one noise level.

Saves everything to data/dataset.npz.
"""

import os
import numpy as np
from ofdm_system import (
    N_DATA, qpsk_modulate, generate_channel, build_freq_grid, apply_channel_freq_domain,
)
from classical_estimators import ls_estimate

SNR_RANGE_DB = [0, 5, 10, 15, 20, 25]
N_PER_SNR_TRAIN = 8000
N_PER_SNR_VAL = 1000
N_PER_SNR_TEST = 2000


def make_split(n_per_snr, seed):
    rng = np.random.default_rng(seed)
    H_ls_list, H_true_list, snr_list = [], [], []

    for snr_db in SNR_RANGE_DB:
        bits = rng.integers(0, 2, size=(n_per_snr, N_DATA, 2))
        data_syms = qpsk_modulate(bits)
        X_grid = build_freq_grid(data_syms)
        h, H_true = generate_channel(n_per_snr, rng=rng)

        Y = apply_channel_freq_domain(X_grid, H_true, snr_db, rng=rng)
        H_ls_full, _ = ls_estimate(Y, X_grid)

        H_ls_list.append(H_ls_full)
        H_true_list.append(H_true)
        snr_list.append(np.full(n_per_snr, snr_db))

    H_ls_all = np.concatenate(H_ls_list, axis=0)
    H_true_all = np.concatenate(H_true_list, axis=0)
    snr_all = np.concatenate(snr_list, axis=0)

    perm = rng.permutation(len(snr_all))
    return H_ls_all[perm], H_true_all[perm], snr_all[perm]


def complex_to_2ch(H):
    """(N, N_FFT) complex -> (N, 2, N_FFT) float32; channel 0 = real, channel 1 = imag."""
    return np.stack([H.real, H.imag], axis=1).astype(np.float32)


def main():
    os.makedirs("data", exist_ok=True)

    print("Generating training set...")
    H_ls_tr, H_true_tr, snr_tr = make_split(N_PER_SNR_TRAIN, seed=1)
    print("Generating validation set...")
    H_ls_va, H_true_va, snr_va = make_split(N_PER_SNR_VAL, seed=2)
    print("Generating test set...")
    H_ls_te, H_true_te, snr_te = make_split(N_PER_SNR_TEST, seed=3)

    np.savez_compressed(
        "data/dataset.npz",
        X_train=complex_to_2ch(H_ls_tr), Y_train=complex_to_2ch(H_true_tr), snr_train=snr_tr,
        X_val=complex_to_2ch(H_ls_va), Y_val=complex_to_2ch(H_true_va), snr_val=snr_va,
        X_test=complex_to_2ch(H_ls_te), Y_test=complex_to_2ch(H_true_te), snr_test=snr_te,
    )
    print(f"Train: {H_ls_tr.shape[0]}  Val: {H_ls_va.shape[0]}  Test: {H_ls_te.shape[0]}")
    print(f"SNR levels covered: {SNR_RANGE_DB} dB")
    print("Saved to data/dataset.npz")


if __name__ == "__main__":
    main()
