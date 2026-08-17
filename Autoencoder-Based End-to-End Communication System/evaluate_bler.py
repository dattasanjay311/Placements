"""
evaluate_bler.py

Monte Carlo Block Error Rate (BLER) vs Eb/No comparison across four schemes,
all sending k=4 information bits per block, evaluated at the SAME Eb/No
(hence the careful rate-aware noise calibration in channel.py -- this is
what makes the comparison fair despite the schemes using different numbers
of channel uses):

  1. Uncoded BPSK      (n=4, rate=1,   no redundancy)
  2. Hamming(7,4) HDD   (n=7, rate=4/7, hard-decision syndrome decoding)
  3. Hamming(7,4) MLD   (n=7, rate=4/7, maximum-likelihood decoding -- the
                         best any receiver could do with this fixed code)
  4. Autoencoder (7,4)  (n=7, rate=4/7, jointly learned modulation+coding)

Sampling is adaptive per Eb/No point: keep drawing batches until either a
minimum error count is reached (for a statistically reliable estimate even
at low BLER) or a sample cap is hit (to bound runtime at high Eb/No, where
BLER can be extremely small).

A closed-form theoretical curve for Hamming(7,4) hard-decision decoding is
also computed and overlaid as a cross-check against the Monte Carlo result
for that same scheme -- if they don't match, something in the simulation
is wrong.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import ebno_db_to_noise_std, qfunc, bpsk_theoretical_ber
from hamming_baseline import encode, hard_decision_decode, ml_decode
from autoencoder_model import Autoencoder

K = 4
EBNO_RANGE_DB = list(range(-2, 9))     # -2 .. 8 dB
MIN_ERRORS = 200
MAX_SAMPLES = 4_000_000
BATCH = 20_000
SEED = 2024


def adaptive_bler(sim_batch_fn, min_errors=MIN_ERRORS, max_samples=MAX_SAMPLES, batch=BATCH):
    total_errors, total_blocks = 0, 0
    while total_errors < min_errors and total_blocks < max_samples:
        n_err = sim_batch_fn(batch)
        total_errors += n_err
        total_blocks += batch
    return total_errors / total_blocks, total_blocks


def sim_uncoded_bpsk(batch_size, ebno_db, rng):
    noise_std = ebno_db_to_noise_std(ebno_db, rate=1.0)
    bits = rng.integers(0, 2, size=(batch_size, K))
    x = 1 - 2 * bits.astype(float)
    y = x + rng.normal(0, noise_std, size=x.shape)
    bits_hat = (y < 0).astype(int)
    return int(np.sum(np.any(bits_hat != bits, axis=1)))


def sim_hamming(decoder, batch_size, ebno_db, rng):
    rate = K / 7
    noise_std = ebno_db_to_noise_std(ebno_db, rate)
    msgs = rng.integers(0, 2, size=(batch_size, K))
    codewords = encode(msgs)
    x = 1 - 2 * codewords.astype(float)
    y = x + rng.normal(0, noise_std, size=x.shape)
    if decoder == "hard":
        msgs_hat = hard_decision_decode((y < 0).astype(int))
    else:
        msgs_hat = ml_decode(y)
    return int(np.sum(np.any(msgs_hat != msgs, axis=1)))


def sim_autoencoder(model, batch_size, ebno_db, rng, M=16, n=7):
    rate = K / n
    noise_std = ebno_db_to_noise_std(ebno_db, rate)
    msgs = rng.integers(0, M, size=batch_size)
    with torch.no_grad():
        one_hot = torch.nn.functional.one_hot(torch.from_numpy(msgs), M).float()
        x = model.encoder(one_hot)
        y = x + torch.randn_like(x) * noise_std
        preds = torch.argmax(model.decoder(y), dim=-1).numpy()
    return int(np.sum(preds != msgs))


def hamming_hdd_theoretical(ebno_db):
    """Closed-form BLER for Hamming(7,4) hard-decision decoding: exact given
    the BSC-after-hard-decision assumption, since this code corrects all
    single-bit errors and mis-corrects all 2+-bit errors (a perfect code)."""
    rate = K / 7
    ebno_linear = 10 ** (np.asarray(ebno_db) / 10)
    p = qfunc(np.sqrt(2 * rate * ebno_linear))       # per-coded-bit error prob
    p0 = (1 - p) ** 7
    p1 = 7 * p * (1 - p) ** 6
    return 1 - p0 - p1


def main():
    rng = np.random.default_rng(SEED)

    model = Autoencoder(M=16, n=7)
    model.load_state_dict(torch.load("model_n7k4.pt", map_location="cpu"))
    model.eval()

    results = {"ebno": EBNO_RANGE_DB, "uncoded": [], "hamming_hdd": [],
               "hamming_mld": [], "autoencoder": []}
    sample_counts = {"uncoded": [], "hamming_hdd": [], "hamming_mld": [], "autoencoder": []}

    for ebno_db in EBNO_RANGE_DB:
        bler_u, n_u = adaptive_bler(lambda b: sim_uncoded_bpsk(b, ebno_db, rng))
        bler_hh, n_hh = adaptive_bler(lambda b: sim_hamming("hard", b, ebno_db, rng))
        bler_hm, n_hm = adaptive_bler(lambda b: sim_hamming("ml", b, ebno_db, rng))
        bler_ae, n_ae = adaptive_bler(lambda b: sim_autoencoder(model, b, ebno_db, rng))

        results["uncoded"].append(bler_u)
        results["hamming_hdd"].append(bler_hh)
        results["hamming_mld"].append(bler_hm)
        results["autoencoder"].append(bler_ae)
        for key, cnt in zip(["uncoded", "hamming_hdd", "hamming_mld", "autoencoder"],
                             [n_u, n_hh, n_hm, n_ae]):
            sample_counts[key].append(cnt)

        print(f"Eb/No {ebno_db:3d} dB | Uncoded={bler_u:.5f} | Hamming-HDD={bler_hh:.5f} | "
              f"Hamming-MLD={bler_hm:.5f} | Autoencoder={bler_ae:.5f}")

    # Closed-form cross-check for Hamming HDD
    theo_hdd = hamming_hdd_theoretical(EBNO_RANGE_DB)
    max_rel_err = np.max(np.abs(np.array(results["hamming_hdd"]) - theo_hdd) /
                          np.maximum(theo_hdd, 1e-6))
    print(f"\nHamming-HDD Monte Carlo vs closed-form: max relative error = {max_rel_err:.2%}")

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.semilogy(results["ebno"], results["uncoded"], "o-", label="Uncoded BPSK (rate 1)")
    plt.semilogy(results["ebno"], results["hamming_hdd"], "s-", label="Hamming(7,4), hard decision")
    plt.semilogy(EBNO_RANGE_DB, theo_hdd, "k:", linewidth=1, label="Hamming(7,4) HDD, closed-form")
    plt.semilogy(results["ebno"], results["hamming_mld"], "d-", label="Hamming(7,4), ML decoding")
    plt.semilogy(results["ebno"], results["autoencoder"], "^-", linewidth=2.2,
                 label="Autoencoder (7,4) -- learned", color="tab:red")
    plt.xlabel("$E_b/N_0$ (dB)")
    plt.ylabel("Block Error Rate")
    plt.title("BLER Comparison: Learned vs Classical (7,4) Schemes, k=4 info bits/block")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, which="both")
    plt.ylim(bottom=1e-5)
    plt.tight_layout()
    plt.savefig("results/bler_comparison.png", dpi=150)
    print("Saved results/bler_comparison.png")

    with open("results/summary.txt", "w") as f:
        f.write("EbNo(dB) | Uncoded | HammingHDD | HammingMLD | Autoencoder | HammingHDD_closedform\n")
        for i, ebno in enumerate(EBNO_RANGE_DB):
            f.write(f"{ebno:8d} | {results['uncoded'][i]:.5f} | {results['hamming_hdd'][i]:.5f} | "
                    f"{results['hamming_mld'][i]:.5f} | {results['autoencoder'][i]:.5f} | "
                    f"{theo_hdd[i]:.5f}\n")
        f.write(f"\nHamming-HDD Monte Carlo vs closed-form max relative error: {max_rel_err:.2%}\n")
        f.write(f"\nSample counts per Eb/No point (adaptive, min {MIN_ERRORS} errors or "
                f"{MAX_SAMPLES} samples cap):\n")
        for key in sample_counts:
            f.write(f"  {key}: {sample_counts[key]}\n")
    print("Saved results/summary.txt")


if __name__ == "__main__":
    main()
