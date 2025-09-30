#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2a_make_canonical.py
Tworzy kanoniczne współrzędne Spirali Sacksa dla liczb 1..N oraz maskę liczb pierwszych.
Zapisuje plik .npz (dane) i .png (podgląd).

Wyjścia domyślne:
  /home/admin2/Documents/repos/pwpw/_spirale/outputs/canonical/canonical_sacks_N{N}.npz
  /home/admin2/Documents/repos/pwpw/_spirale/outputs/canonical/canonical_sacks_N{N}.png
"""

import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- podstawy ----------

def sieve_bool(n: int) -> np.ndarray:
    """Zwraca maskę bool długości n+1: True dla liczb pierwszych (0..n)."""
    if n < 2:
        m = np.zeros(n + 1, dtype=bool)
        return m
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    limit = int(n ** 0.5) + 1
    for p in range(2, limit):
        if is_prime[p]:
            is_prime[p * p : n + 1 : p] = False
    return is_prime

def sacks_coords(N: int, theta_mode: str = "theta=n") -> tuple[np.ndarray, np.ndarray]:
    """
    Zwraca (x,y) dla n=1..N w układzie Sacksa:
      r = sqrt(n), theta = n (radiany)
    Dodatkowo opcja 'theta=golden*n' (nie używana później, ale zostawiamy na wszelki).
    """
    GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5.0))
    ns = np.arange(1, N + 1, dtype=np.float64)
    r = np.sqrt(ns)
    theta = ns * (GOLDEN_ANGLE if theta_mode == "theta=golden*n" else 1.0)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


# ---------- zapis ----------

def save_npz(out_npz: Path, N: int, ns: np.ndarray,
             x: np.ndarray, y: np.ndarray, prime_mask_1N: np.ndarray) -> None:
    """Zapisuje skonstruowane tablice do pliku .npz."""
    np.savez(
        out_npz,
        N=np.int32(N),
        n_all=ns.astype(np.int32),
        x_all=x.astype(np.float64),
        y_all=y.astype(np.float64),
        is_prime_1N=prime_mask_1N.astype(np.bool_),
        n_prime=ns[prime_mask_1N].astype(np.int32),
        x_prime=x[prime_mask_1N].astype(np.float64),
        y_prime=y[prime_mask_1N].astype(np.float64),
    )

def save_preview_png(out_png: Path, N: int, x: np.ndarray, y: np.ndarray, prime_mask_1N: np.ndarray) -> None:
    """Szybki podgląd: rysuje tylko liczby pierwsze."""
    xs = x[prime_mask_1N]
    ys = y[prime_mask_1N]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(xs, ys, s=2, alpha=0.9, marker='.', linewidths=0)
    ax.set_aspect('equal', 'box')
    lim = math.sqrt(N) + 2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Canonical Sacks spiral (N={N})")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10_000, help="Zakres liczb 1..N")
    ap.add_argument("--theta", type=str, default="theta=n",
                    choices=["theta=n", "theta=golden*n"], help="Model kąta (domyślnie klasyczny)")
    ap.add_argument("--outdir", type=str,
                    default="/home/admin2/Documents/repos/pwpw/_spirale/outputs/canonical",
                    help="Folder wyjściowy")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N = int(args.N)
    print(f"[step2a] Buduję kanon Sacksa dla N={N}, {args.theta}")

    # 1) liczby pierwsze
    is_prime_0N = sieve_bool(N)
    prime_mask_1N = is_prime_0N[1:]         # maska dla 1..N
    piN = int(prime_mask_1N.sum())
    print(f"[step2a] π({N}) = {piN} primes")

    # 2) współrzędne Sacksa
    ns = np.arange(1, N + 1, dtype=np.int32)
    x, y = sacks_coords(N, args.theta)

    # sanity: zasięg promienia
    r_min = float(np.sqrt(1))
    r_max = float(np.sqrt(N))
    print(f"[step2a] r in [{r_min:.3f}, {r_max:.3f}]  |  xy range ~ [-{r_max:.3f}, {r_max:.3f}]")

    # 3) zapis danych
    out_npz = outdir / f"canonical_sacks_N{N}.npz"
    save_npz(out_npz, N, ns, x, y, prime_mask_1N)
    print(f"[step2a] Zapisano dane: {out_npz}")

    # 4) podgląd
    out_png = outdir / f"canonical_sacks_N{N}.png"
    save_preview_png(out_png, N, x, y, prime_mask_1N)
    print(f"[step2a] Zapisano podgląd: {out_png}")

    # 5) metadane (opcjonalne)
    meta = {
        "N": N,
        "theta": args.theta,
        "num_primes": piN,
        "r_min": r_min,
        "r_max": r_max,
        "files": {
            "npz": str(out_npz),
            "preview_png": str(out_png),
        },
    }
    meta_path = outdir / f"canonical_sacks_N{N}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[step2a] Zapisano metadane: {meta_path}")

    print("[step2a] DONE.")

if __name__ == "__main__":
    main()
