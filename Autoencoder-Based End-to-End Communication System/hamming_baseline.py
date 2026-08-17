"""
hamming_baseline.py

A (7,4) Hamming code -- the classical baseline the autoencoder is compared
against -- constructed programmatically rather than copied from a textbook
table, and exhaustively verified against the two defining properties of the
code before it's trusted for anything else:
  1. Minimum distance between any two distinct codewords is exactly 3
     (this is *why* the code can correct any single-bit error).
  2. Syndrome decoding correctly recovers the original message after
     ANY single-bit flip, for EVERY one of the 16 codewords (all 7*16
     single-error cases checked, not sampled).

Construction: the parity-check matrix H is built so its 7 columns are the
binary representations of 1..7. Columns that are single-bit (powers of 2:
1, 2, 4) become the parity positions; the rest become data positions. This
is the standard Hamming(7,4) construction, but built and checked here
rather than assumed correct.

Two decoders are provided:
  - hard_decision_decode: BPSK hard-decision + syndrome lookup (what a
    real low-complexity receiver does)
  - ml_decode: maximum-likelihood / minimum-Euclidean-distance decoding
    against all 16 valid modulated codewords (the best any receiver could
    do with this fixed code -- an upper bound on Hamming(7,4)'s own
    performance, used as a second baseline)
"""

import numpy as np
import itertools

N, K = 7, 4


def _build_code():
    # Columns of H = binary representations of 1..7 (3-bit, MSB first)
    ints = np.arange(1, 8)
    H = np.array([[(i >> (2 - b)) & 1 for i in ints] for b in range(3)])  # (3,7)

    single_bit = [i for i in range(7) if bin(ints[i]).count("1") == 1]      # parity cols
    multi_bit = [i for i in range(7) if i not in single_bit]                # data cols
    assert len(single_bit) == 3 and len(multi_bit) == 4

    # Build G (4x7): data bits pass straight through at `multi_bit` columns;
    # parity bits at `single_bit` columns are solved so that H @ codeword = 0 (mod 2).
    G = np.zeros((K, N), dtype=int)
    for row, col in enumerate(multi_bit):
        G[row, col] = 1
    # For each data bit (one at a time), find what parity pattern it forces, via H.
    for row, dcol in enumerate(multi_bit):
        col_syndrome = H[:, dcol]                      # H's column at this data position
        for pcol_idx, pcol in enumerate(single_bit):
            # H[:, pcol] is a standard basis vector (single 1) by construction.
            bit_idx = np.argmax(H[:, pcol])
            G[row, pcol] = col_syndrome[bit_idx]
    return H, G, single_bit, multi_bit


H, G, PARITY_COLS, DATA_COLS = _build_code()


def encode(data_bits):
    """data_bits: (..., 4) -> codeword: (..., 7), all mod 2."""
    return (data_bits @ G) % 2


def _syndrome_table():
    """Maps each of the 7 nonzero syndromes -> which bit position it flags."""
    table = {}
    for pos in range(N):
        e = np.zeros(N, dtype=int)
        e[pos] = 1
        syn = tuple((H @ e) % 2)
        table[syn] = pos
    return table


SYNDROME_TABLE = _syndrome_table()


def hard_decision_decode(received_bits):
    """received_bits: (..., 7) hard bits -> recovered data bits: (..., 4)."""
    shape = received_bits.shape[:-1]
    flat = received_bits.reshape(-1, N)
    corrected = flat.copy()
    for i in range(flat.shape[0]):
        syn = tuple((H @ flat[i]) % 2)
        if syn != (0, 0, 0):
            pos = SYNDROME_TABLE.get(syn)
            if pos is not None:
                corrected[i, pos] ^= 1
    return corrected[:, DATA_COLS].reshape(*shape, K)


def all_codewords_bpsk():
    """All 16 (7,4) codewords, BPSK modulated (0->+1, 1->-1). Returns (16,7) and (16,4) messages."""
    messages = np.array(list(itertools.product([0, 1], repeat=K)))
    codewords = encode(messages)
    bpsk = 1 - 2 * codewords.astype(float)
    return bpsk, messages, codewords


def ml_decode(received_soft):
    """
    Maximum-likelihood decoding: nearest (min Euclidean distance) of the 16
    valid BPSK-modulated codewords. received_soft: (..., 7) real values.
    """
    bpsk_book, messages, _ = all_codewords_bpsk()
    shape = received_soft.shape[:-1]
    flat = received_soft.reshape(-1, N)
    dists = np.linalg.norm(flat[:, None, :] - bpsk_book[None, :, :], axis=-1)  # (batch, 16)
    best = np.argmin(dists, axis=-1)
    return messages[best].reshape(*shape, K)


def _verify():
    print(f"Parity positions (0-indexed): {PARITY_COLS}")
    print(f"Data positions (0-indexed):   {DATA_COLS}\n")

    messages = np.array(list(itertools.product([0, 1], repeat=K)))
    codewords = encode(messages)

    # Property 1: all 16 codewords distinct, and minimum pairwise Hamming distance is 3.
    assert len(set(map(tuple, codewords))) == 16, "Codewords are not all distinct!"
    min_dist = N
    for i in range(16):
        for j in range(i + 1, 16):
            d = np.sum(codewords[i] != codewords[j])
            min_dist = min(min_dist, d)
    print(f"Minimum distance between any two codewords: {min_dist} (must be 3)")
    assert min_dist == 3, "Minimum distance is not 3 -- this is not a valid Hamming(7,4) code!"

    # Property 2: exhaustively test every single-bit flip on every codeword.
    total_checks, failures = 0, 0
    for msg, cw in zip(messages, codewords):
        for flip_pos in range(N):
            corrupted = cw.copy()
            corrupted[flip_pos] ^= 1
            recovered = hard_decision_decode(corrupted.reshape(1, N))[0]
            total_checks += 1
            if not np.array_equal(recovered, msg):
                failures += 1
    print(f"Single-bit-error correction: {total_checks - failures}/{total_checks} correct "
          f"(every codeword x every single-bit-flip position)")
    assert failures == 0, f"{failures} single-bit-error cases failed to decode correctly!"

    print("\nPASSED: valid (7,4) Hamming code -- min distance 3, corrects all single-bit errors.")


if __name__ == "__main__":
    _verify()
