"""
evaluate.py

Compares LS, MMSE, and the trained CNN estimator across SNR levels using two
metrics:
  1. Normalized channel-estimation MSE (NMSE) vs SNR
  2. End-to-end BER (zero-forcing equalization + QPSK detection) vs SNR,
     including a "perfect CSI" bound for reference.

Saves: results/mse_vs_snr.png, results/ber_vs_snr.png, results/summary.txt
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ofdm_system import (
    N_DATA, DATA_IDX, qpsk_modulate, qpsk_demodulate,
    generate_channel, build_freq_grid, apply_channel_freq_domain,
)
from classical_estimators import ls_estimate, mmse_estimate
from dl_model import ChannelEstimatorCNN

SNR_RANGE_DB = [0, 5, 10, 15, 20, 25, 30]
N_TEST = 4000
SEED = 123


def complex_to_2ch(H):
    return np.stack([H.real, H.imag], axis=1).astype(np.float32)


def two_ch_to_complex(H2):
    return H2[:, 0, :] + 1j * H2[:, 1, :]


def nmse(H_hat, H_true):
    return np.mean(np.abs(H_hat - H_true) ** 2) / np.mean(np.abs(H_true) ** 2)


def run_ber(H_hat, Y, bits_true):
    """Zero-forcing equalization on data subcarriers + QPSK detection -> BER."""
    X_hat_data = Y[:, DATA_IDX] / H_hat[:, DATA_IDX]
    bits_hat = qpsk_demodulate(X_hat_data)
    return np.mean(bits_hat != bits_true)


def main():
    model = ChannelEstimatorCNN()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(SEED)

    R = {"snr": SNR_RANGE_DB, "nmse_ls": [], "nmse_mmse": [], "nmse_dl": [],
         "ber_ls": [], "ber_mmse": [], "ber_dl": [], "ber_perfect": []}

    for snr_db in SNR_RANGE_DB:
        bits = rng.integers(0, 2, size=(N_TEST, N_DATA, 2))
        data_syms = qpsk_modulate(bits)
        X_grid = build_freq_grid(data_syms)
        h, H_true = generate_channel(N_TEST, rng=rng)
        Y = apply_channel_freq_domain(X_grid, H_true, snr_db, rng=rng)

        H_ls, H_ls_pilot = ls_estimate(Y, X_grid)
        H_mmse = mmse_estimate(H_ls_pilot, snr_db)
        with torch.no_grad():
            H_dl = two_ch_to_complex(model(torch.from_numpy(complex_to_2ch(H_ls))).numpy())

        R["nmse_ls"].append(nmse(H_ls, H_true))
        R["nmse_mmse"].append(nmse(H_mmse, H_true))
        R["nmse_dl"].append(nmse(H_dl, H_true))

        R["ber_ls"].append(run_ber(H_ls, Y, bits))
        R["ber_mmse"].append(run_ber(H_mmse, Y, bits))
        R["ber_dl"].append(run_ber(H_dl, Y, bits))
        R["ber_perfect"].append(run_ber(H_true, Y, bits))

        print(f"SNR {snr_db:2d} dB | NMSE  LS={R['nmse_ls'][-1]:.4f}  MMSE={R['nmse_mmse'][-1]:.4f}  "
              f"CNN={R['nmse_dl'][-1]:.4f} | BER  LS={R['ber_ls'][-1]:.4f}  MMSE={R['ber_mmse'][-1]:.4f}  "
              f"CNN={R['ber_dl'][-1]:.4f}  Perfect={R['ber_perfect'][-1]:.4f}")

    # --- NMSE plot ---
    plt.figure(figsize=(6.5, 4.5))
    plt.semilogy(R["snr"], R["nmse_ls"], "o-", label="LS")
    plt.semilogy(R["snr"], R["nmse_mmse"], "s-", label="MMSE (genie-aided)")
    plt.semilogy(R["snr"], R["nmse_dl"], "^-", label="CNN (proposed)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Normalized MSE")
    plt.title("Channel Estimation Accuracy vs SNR")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("results/mse_vs_snr.png", dpi=150)

    # --- BER plot ---
    plt.figure(figsize=(6.5, 4.5))
    plt.semilogy(R["snr"], R["ber_ls"], "o-", label="LS")
    plt.semilogy(R["snr"], R["ber_mmse"], "s-", label="MMSE (genie-aided)")
    plt.semilogy(R["snr"], R["ber_dl"], "^-", label="CNN (proposed)")
    plt.semilogy(R["snr"], R["ber_perfect"], "k--", label="Perfect CSI (bound)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("End-to-End BER vs SNR (QPSK, ZF equalization)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("results/ber_vs_snr.png", dpi=150)

    with open("results/summary.txt", "w") as f:
        f.write("SNR(dB) | NMSE_LS | NMSE_MMSE | NMSE_CNN | BER_LS  | BER_MMSE| BER_CNN | BER_PerfectCSI\n")
        for i, snr in enumerate(R["snr"]):
            f.write(f"{snr:7d} | {R['nmse_ls'][i]:.5f} | {R['nmse_mmse'][i]:.5f} | "
                    f"{R['nmse_dl'][i]:.5f} | {R['ber_ls'][i]:.5f} | {R['ber_mmse'][i]:.5f} | "
                    f"{R['ber_dl'][i]:.5f} | {R['ber_perfect'][i]:.5f}\n")

    print("\nSaved results/mse_vs_snr.png, results/ber_vs_snr.png, results/summary.txt")


if __name__ == "__main__":
    main()
