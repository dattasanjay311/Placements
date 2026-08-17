"""
visualize_constellation.py

For the (n=2, k=2, M=4) autoencoder, the encoder's output is literally a
2D point -- exactly like a constellation diagram. This script:
  1. Feeds each of the 4 possible messages through the trained encoder and
     plots where they land.
  2. Feeds a fine grid over the 2D plane through the trained DECODER to
     visualize the receiver's learned decision regions -- i.e. what the
     network would decide for any possible received point, not just the
     noiseless transmitted ones.

This is the single most famous result from this line of research: with no
concept of "modulation" ever given to it, gradient descent alone discovers
a symmetric, maximally-separated constellation, because that's what
minimizes decoding error under AWGN with an energy constraint -- the same
conclusion classical communication theory reaches analytically for this
exact setup (4 equal-energy points, AWGN, minimize error <=> maximize
minimum pairwise distance <=> put them at the corners of a square).
"""

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoencoder_model import Autoencoder

M, n = 4, 2


def main():
    model = Autoencoder(M, n)
    model.load_state_dict(torch.load("model_n2k2.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        one_hot = torch.eye(M)
        tx_points = model.encoder(one_hot).numpy()   # (4, 2)

        grid_range = np.linspace(-3, 3, 400)
        gx, gy = np.meshgrid(grid_range, grid_range)
        grid_points = torch.from_numpy(
            np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
        )
        logits = model.decoder(grid_points)
        decisions = torch.argmax(logits, dim=-1).numpy().reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(gx, gy, decisions, levels=np.arange(-0.5, M, 1), cmap="Pastel1", alpha=0.8)
    colors = plt.cm.Set1(np.linspace(0, 1, M))
    for i in range(M):
        ax.scatter(*tx_points[i], s=220, color=colors[i], edgecolor="black",
                   linewidth=1.2, zorder=5, label=f"message {i} ({format(i, '02b')})")
    circle = plt.Circle((0, 0), math.sqrt(n), fill=False, linestyle="--",
                         color="gray", linewidth=1, label=f"energy circle ($\\|x\\|^2$={n})")
    ax.add_artist(circle)
    ax.set_xlabel("Channel-use dimension 1")
    ax.set_ylabel("Channel-use dimension 2")
    ax.set_title("Learned (2,2) Autoencoder Constellation + Decision Regions\n"
                 "(no modulation theory given to the network -- purely learned)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("results/constellation.png", dpi=150)
    print("Saved results/constellation.png")

    # Quantify how close to a "perfect" symmetric constellation it actually is
    print("\nLearned constellation points (message: (x, y), energy):")
    for i in range(M):
        e = np.sum(tx_points[i] ** 2)
        print(f"  {i} ({format(i, '02b')}): ({tx_points[i,0]:+.3f}, {tx_points[i,1]:+.3f})  "
              f"energy={e:.3f}")

    pairwise = []
    for i in range(M):
        for j in range(i + 1, M):
            pairwise.append(np.linalg.norm(tx_points[i] - tx_points[j]))
    edge_dist, diag_dist = math.sqrt(2 * n), 2 * math.sqrt(n)
    print(f"\nPairwise distances between the 4 points: "
          f"{[f'{d:.3f}' for d in sorted(pairwise)]}")
    print(f"(A perfect square constellation on this energy circle has 4 edges of "
          f"length {edge_dist:.3f} and 2 diagonals of length {diag_dist:.3f})")


if __name__ == "__main__":
    main()
