#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "/_spirale/outputs"
N = 10_000  # liczby 1..N

def sieve_is_prime(n: int) -> np.ndarray:
    """Zwraca maskę bool o długości n+1: True dla liczb pierwszych."""
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    limit = int(np.sqrt(n))
    for p in range(2, limit + 1):
        if is_prime[p]:
            is_prime[p*p:n+1:p] = False
    return is_prime

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n = np.arange(1, N + 1, dtype=np.float64)            # 1..N
    r = np.sqrt(n)
    theta = n                                            # radiany
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # maska liczb pierwszych
    is_prime = sieve_is_prime(N)
    prime_mask = is_prime[1:]                            # bo n zaczyna się od 1

    # rysunek
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    # tło: wszystkie liczby (delikatne)
    ax.scatter(x, y, s=0.5, alpha=0.25, linewidths=0, marker='.', zorder=1)
    # wyróżnienie: liczby pierwsze (ciemne)
    ax.scatter(x[prime_mask], y[prime_mask],
               s=4, alpha=0.9, linewidths=0, marker='.', zorder=2)

    ax.set_aspect("equal", adjustable="box")
    lim = np.sqrt(N)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Spirala Sacksa (N={N})")

    out_path = os.path.join(OUT_DIR, f"sacks_spiral_N{N}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Zapisano: {out_path}")

if __name__ == "__main__":
    main()
