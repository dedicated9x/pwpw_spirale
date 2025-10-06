#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2a_ml_make_dataset.py
Syntetyczny dataset dla ETAPU 2 (ML regressja transformacji).
Tworzy chmury punktów w pikselach z losową geometrią, brakami, fałszywkami i jitterem.

Wyjścia:
  <outdir>/ds/{split}/sample_000000.npz   (points, true_points, false_points, H, model, present_mask, image_size, meta)
  <outdir>/vis/{split}/sample_000000_model-HOM_K1160_fp12.png   (wizualizacja chmury)

Domyślnie korzysta z kanonu z step2a:
  /home/admin2/Documents/repos/pwpw/_spirale/outputs/canonical/canonical_sacks_N10000.npz
"""

import os
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
from tqdm import tqdm





# ---------- narzędzia bazowe ----------

def load_canonical(npz_path: str, flip_y: bool = True) -> Dict[str, np.ndarray]:
    """Wczytaj kanon Sacksa (tylko prime’y). Opcjonalnie odbij y->-y, by dopasować układ obrazu (y w dół)."""
    d = np.load(npz_path)
    n_prime = d["n_prime"].astype(np.int32)      # indeksy 1..N liczb pierwszych
    x = d["x_prime"].astype(np.float64)
    y = d["y_prime"].astype(np.float64)
    if flip_y:
        y = -y
    S = np.stack([x, y], axis=1)                # (Np,2) kanon w jednostkach Sacksa
    # r i kąt kanoniczny (θ≈n) – przydatne do dropoutów radialnych/klinowych
    r = np.sqrt((S[:, 0]**2 + S[:, 1]**2))
    theta = n_prime.astype(np.float64)          # dla Sacksa: θ=n (radiany)
    return {
        "S": S,                  # (Np,2) kanoniczne XY
        "n_prime": n_prime,      # (Np,)
        "r": r,                  # (Np,)
        "theta": theta,          # (Np,)
        "r_max": float(r.max()),
        "Np": int(S.shape[0]),
    }

def H_similarity(scale: float, theta: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    H = np.array([
        [scale*c, -scale*s, tx],
        [scale*s,  scale*c, ty],
        [0.0,      0.0,     1.0]
    ], dtype=np.float64)
    return H

def H_affine(a11: float, a12: float, a21: float, a22: float, tx: float, ty: float) -> np.ndarray:
    return np.array([
        [a11, a12, tx],
        [a21, a22, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

def H_perspective(h31: float, h32: float) -> np.ndarray:
    """Minimalna perspektywa: modyfikuje wiersz [h31,h32,1]."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [h31, h32, 1.0]
    ], dtype=np.float64)

def compose(*Hs: np.ndarray) -> np.ndarray:
    """Mnożenie macierzy homografii (prawa do lewej): compose(A, B) = A @ B."""
    M = np.eye(3, dtype=np.float64)
    for H in Hs:
        M = H @ M
    return M

def apply_H(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Zastosuj homografię do punktów (N,2)."""
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xh = np.concatenate([X, ones], axis=1)      # (N,3)
    Yh = (H @ Xh.T).T                            # (N,3)
    w = Yh[:, 2:3]
    Y = Yh[:, :2] / np.maximum(1e-12, w)
    return Y

def sample_image_size(rng: np.random.Generator,
                      short_min: int = 1200,
                      short_max: int = 2000,
                      aspect_min: float = 1.2,
                      aspect_max: float = 1.8) -> Tuple[int, int]:
    """Losuj rozmiar obrazu (W,H)."""
    short = int(rng.integers(short_min, short_max + 1))
    aspect = float(rng.uniform(aspect_min, aspect_max))
    W = int(max(short, round(short * aspect)))
    H = int(short)
    # wylosuj też możliwość odwrócenia orientacji (portret/landscape)
    if rng.random() < 0.3:
        W, H = H, W
    return W, H

def sample_transform(rng: np.random.Generator,
                     model: str,
                     r_max: float,
                     W: int, H: int) -> np.ndarray:
    """
    Losuje transformację, która sensownie wypełnia kadr.
    Zwraca homografię 3x3 mapującą kanon -> piksele.
    """
    cx, cy = W * 0.5, H * 0.5

    # bazowa skala tak, by promień ~ r_max mapował się na ~0.45*min(W,H)
    target = 0.45 * min(W, H)
    s0 = float(target / max(1e-6, r_max))

    # lekka modyfikacja skali i losowa rotacja
    s = s0 * float(rng.uniform(0.85, 1.15))
    theta = float(rng.uniform(-math.pi, math.pi))
    tx = float(cx + rng.uniform(-0.08, 0.08) * W)
    ty = float(cy + rng.uniform(-0.08, 0.08) * H)

    H_sim = H_similarity(s, theta, tx, ty)

    if model == "SIM":
        return H_sim

    if model == "AFF":
        # dodaj lekki shear/anizotropię
        shear_x = float(rng.uniform(-0.08, 0.08))
        shear_y = float(rng.uniform(-0.08, 0.08))
        A = np.array([[1.0, shear_x],
                      [shear_y, 1.0]], dtype=np.float64)
        # skalowanie anizotropowe
        sx = float(rng.uniform(0.95, 1.05))
        sy = float(rng.uniform(0.95, 1.05))
        A = A @ np.diag([sx, sy])
        # rotacja w affinie zostaje już w H_sim; tu A stosujemy w układzie kanonu
        H_aff_local = H_affine(A[0,0], A[0,1], A[1,0], A[1,1], 0.0, 0.0)
        return compose(H_sim, H_aff_local)

    # HOM: affina + niewielka perspektywa
    h31 = float(rng.uniform(-0.002, 0.002))
    h32 = float(rng.uniform(-0.002, 0.002))
    H_p = H_perspective(h31, h32)

    # doraźna lekka affina jak wyżej
    shear_x = float(rng.uniform(-0.06, 0.06))
    shear_y = float(rng.uniform(-0.06, 0.06))
    A = np.array([[1.0, shear_x],
                  [shear_y, 1.0]], dtype=np.float64)
    sx = float(rng.uniform(0.95, 1.06))
    sy = float(rng.uniform(0.95, 1.06))
    A = A @ np.diag([sx, sy])
    H_aff_local = H_affine(A[0,0], A[0,1], A[1,0], A[1,1], 0.0, 0.0)

    return compose(H_sim, H_aff_local, H_p)

def dropout_mask(rng: np.random.Generator,
                 r: np.ndarray,
                 theta: np.ndarray,
                 base_p: float,
                 radial_extra_max: float,
                 wedge_p: float) -> np.ndarray:
    """
    Buduje maskę (True=ZACHOWAJ) z globalnym dropoutem, radialnym wzrostem i klinem kątowym.
    """
    N = r.shape[0]
    keep = np.ones(N, dtype=bool)

    # global
    if base_p > 1e-9:
        keep &= rng.random(N) > base_p

    # radialny: liniowy wzrost prawdopodobieństwa do 'radial_extra_max' przy max(r)
    if radial_extra_max > 1e-6:
        r01 = (r - r.min()) / max(1e-9, (r.max() - r.min()))
        p_rad = r01 * radial_extra_max
        keep &= rng.random(N) > p_rad

    # klin kątowy
    if rng.random() < wedge_p:
        width = rng.uniform(math.radians(60), math.radians(120))
        phi0 = rng.uniform(-math.pi, math.pi)
        # oddal punkty wewnątrz klina o dodatkowe prawdopodobieństwo 0.05..0.15
        p_add = rng.uniform(0.05, 0.15)
        ang = (theta - phi0 + math.pi) % (2*math.pi) - math.pi
        in_wedge = np.abs(ang) < (width * 0.5)
        kill = rng.random(N) < (p_add * in_wedge.astype(float))
        keep &= ~kill

    return keep

def add_jitter(rng: np.random.Generator, P: np.ndarray) -> np.ndarray:
    """Dodaj losowy (czasem anizotropowy) jitter subpikselowy."""
    N = P.shape[0]
    sigma = rng.uniform(0.2, 1.2)
    if rng.random() < 0.5:
        # anizotropia w losowym kierunku
        f = rng.uniform(1.3, 1.8)
        ang = rng.uniform(-math.pi, math.pi)
        R = np.array([[math.cos(ang), -math.sin(ang)],
                      [math.sin(ang),  math.cos(ang)]])
        S = np.diag([sigma * f, sigma])
        cov = R @ S @ S @ R.T
        noise = rng.multivariate_normal(mean=[0,0], cov=cov, size=N)
    else:
        noise = rng.normal(0.0, sigma, size=(N, 2))
    return P + noise

def add_false_positives(rng: np.random.Generator, W: int, H: int,
                        avoid: np.ndarray, min_dist: float = 3.0) -> np.ndarray:
    """Dodaj FP w losowych pozycjach, z ominięciem zbyt bliskich prawdziwych punktów."""
    lam = rng.uniform(5, 40)   # średnia Poissona
    K = int(rng.poisson(lam))
    if K == 0:
        return np.zeros((0, 2), dtype=np.float64)
    FP = rng.uniform([0, 0], [W, H], size=(K, 2))
    if avoid.size == 0:
        return FP
    # odfiltruj zbyt bliskie
    from scipy.spatial import cKDTree  # jeśli nie chcesz SciPy, zamień na proste KD/siatkę
    tree = cKDTree(avoid)
    d, _ = tree.query(FP, k=1)
    return FP[d >= min_dist]

def quantize_maybe(rng: np.random.Generator, P: np.ndarray) -> np.ndarray:
    """Z prawd. 0.5 dodaj efekt kwantyzacji (0.1 px lub do intów)."""
    if rng.random() < 0.5:
        step = 1.0 if rng.random() < 0.5 else 0.1
        return (np.round(P / step) * step)
    return P


# ---------- generator próbek ----------

def make_sample(rng: np.random.Generator,
                canon: Dict[str, np.ndarray],
                model_probs=(0.2, 0.3, 0.5),
                short_min=1200, short_max=2000,
                aspect_min=1.2, aspect_max=1.8,
                base_dropout_max=0.05,
                radial_extra_max=0.05,
                wedge_prob=0.3,
                min_points_keep=800) -> Dict:
    """
    Tworzy jedną próbkę: punkty obserwowane + metadane i ground-truth transformację.
    Gwarantuje min_points_keep prawdziwych punktów po wszystkich filtrach (resampluje).
    """
    S = canon["S"]
    r = canon["r"]
    theta = canon["theta"]
    r_max = canon["r_max"]
    Np = canon["Np"]

    # los modelu
    model = np.random.choice(["SIM", "AFF", "HOM"], p=np.array(model_probs)/np.sum(model_probs))

    # wylosuj obraz
    W, H = sample_image_size(rng, short_min=short_min, short_max=short_max,
                             aspect_min=aspect_min, aspect_max=aspect_max)

    # homografia
    Hgt = sample_transform(rng, model, r_max, W, H)

    # transformuj wszystkie prime’y
    S_pix = apply_H(S, Hgt)  # (Np,2)

    # zbuduj maskę keep (dropout)
    base_p = rng.uniform(0.0, base_dropout_max)
    keep = dropout_mask(rng, r, theta, base_p=base_p,
                        radial_extra_max=rng.uniform(0.0, radial_extra_max),
                        wedge_p=wedge_prob)

    # crop do kadru
    in_img = ((S_pix[:, 0] >= 0) & (S_pix[:, 0] < W) &
              (S_pix[:, 1] >= 0) & (S_pix[:, 1] < H))
    keep &= in_img

    # jeśli za mało – spróbuj jeszcze raz (losujemy parametry od nowa)
    if keep.sum() < min_points_keep:
        return make_sample(rng, canon, model_probs, short_min, short_max,
                           aspect_min, aspect_max, base_dropout_max,
                           radial_extra_max, wedge_prob, min_points_keep)

    # jitter + kwantyzacja
    P_true = add_jitter(rng, S_pix[keep])
    P_true = quantize_maybe(rng, P_true)

    # FP
    P_false = add_false_positives(rng, W, H, avoid=P_true, min_dist=3.0)

    # final „obserwacje” = true ∪ false (bez etykiet)
    P_obs = P_true
    if P_false.size > 0:
        P_obs = np.concatenate([P_obs, P_false], axis=0)

    # present mask względem Np (kolejność jak w kanonie)
    present_mask = keep.astype(np.bool_)

    meta = {
        "model": model,
        "W": int(W),
        "H": int(H),
        "rms_sigma": "subpx",
        "base_dropout": float(base_p),
        "radial_extra_max": float(radial_extra_max),
        "wedge_prob": float(wedge_prob),
        "n_true": int(P_true.shape[0]),
        "n_false": int(P_false.shape[0]),
        "n_total": int(P_obs.shape[0]),
    }

    return {
        "points": P_obs.astype(np.float32),               # (K,2) mieszanka true+false
        "true_points": P_true.astype(np.float32),         # (M,2) tylko prawdziwe
        "false_points": P_false.astype(np.float32),       # (F,2)
        "present_mask": present_mask,                     # (Np,) bool
        "H": Hgt.astype(np.float64),                      # (3,3) homografia GT
        "image_size": np.array([W, H], dtype=np.int32),   # (2,)
        "model": model,                                   # 'SIM'/'AFF'/'HOM'
        "meta": meta,
    }


# ---------- wizualizacja ----------

def draw_points(W: int, H: int, true_pts: np.ndarray, false_pts: np.ndarray) -> np.ndarray:
    """Rysuje punkty na białym tle: true=zielone, false=żółte."""
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    # true
    for x, y in true_pts:
        cv2.circle(img, (int(round(x)), int(round(y))), 2, (80, 200, 80), -1, lineType=cv2.LINE_AA)
    # false
    for x, y in false_pts:
        cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 215, 255), -1, lineType=cv2.LINE_AA)
    return img

# ---------- domyślne ścieżki ----------
DEFAULT_CANON = Path(__file__).parents[0] / "_outputs/_spirale/step2a_create_canonical/canonical_sacks_N10000.npz"
DEFAULT_OUT   = Path(__file__).parents[0] / "_outputs/_spirale/step3a_make_dataset"


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=str, default=DEFAULT_CANON,
                    help="Plik z kanonem (z step2a)")
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUT,
                    help="Folder bazowy wyjść (ds/ i vis/ polecą tu)")
    ap.add_argument("--num", type=int, default=2000, help="Liczba próbek do wygenerowania")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                    help="Nazwa splitu")
    ap.add_argument("--seed", type=int, default=123, help="Seed PRNG")
    # rozkłady geometrii
    ap.add_argument("--p_sim_aff_hom", type=str, default="0.2,0.3,0.5",
                    help="Prawdopodobieństwa modelu geometrii (SIM,AFF,HOM)")
    # rozmiar obrazu
    ap.add_argument("--short_min", type=int, default=1200)
    ap.add_argument("--short_max", type=int, default=2000)
    ap.add_argument("--aspect_min", type=float, default=1.2)
    ap.add_argument("--aspect_max", type=float, default=1.8)
    # widoczność i szum
    ap.add_argument("--base_dropout_max", type=float, default=0.05)
    ap.add_argument("--radial_extra_max", type=float, default=0.05)
    ap.add_argument("--wedge_prob", type=float, default=0.3)
    ap.add_argument("--min_points_keep", type=int, default=800,
                    help="Minimalna liczba zachowanych prawdziwych punktów; inaczej resample")
    # wizualizacje
    ap.add_argument("--vis_every", type=int, default=1,
                    help="Co ile próbek zapisywać wizualizację (1=każda)")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.outdir)
    ds_dir = out_base / "ds" / args.split
    vis_dir = out_base / "vis" / args.split
    ds_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    pvals = [float(x) for x in args.p_sim_aff_hom.split(",")]
    if len(pvals) != 3:
        raise ValueError("--p_sim_aff_hom: podaj 3 liczby (SIM,AFF,HOM)")
    model_probs = tuple(pvals)

    rng = np.random.default_rng(args.seed)
    canon = load_canonical(args.canonical, flip_y=True)

    # manifest
    manifest = []

    for i in tqdm(range(args.num), desc=f"gen {args.split}", ncols=100):
        sample = make_sample(
            rng, canon,
            model_probs=model_probs,
            short_min=args.short_min, short_max=args.short_max,
            aspect_min=args.aspect_min, aspect_max=args.aspect_max,
            base_dropout_max=args.base_dropout_max,
            radial_extra_max=args.radial_extra_max,
            wedge_prob=args.wedge_prob,
            min_points_keep=args.min_points_keep
        )

        stem = f"sample_{i:06d}"
        npz_path = ds_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            points=sample["points"],
            true_points=sample["true_points"],
            false_points=sample["false_points"],
            present_mask=sample["present_mask"],
            H=sample["H"],
            image_size=sample["image_size"],
            model=np.array(sample["model"]),
            meta=np.array(str(sample["meta"]))
        )

        if (i % args.vis_every) == 0:
            W, H = map(int, sample["image_size"])
            vis = draw_points(W, H, sample["true_points"], sample["false_points"])
            tag = f"{stem}_model-{sample['model']}_K{sample['meta']['n_total']}_fp{sample['meta']['n_false']}.png"
            cv2.imwrite(str(vis_dir / tag), vis)

        manifest.append({
            "i": i,
            "npz": str(npz_path),
            "model": sample["model"],
            "n_true": int(sample["meta"]["n_true"]),
            "n_false": int(sample["meta"]["n_false"]),
            "n_total": int(sample["meta"]["n_total"]),
            "W": int(sample["meta"]["W"]),
            "H": int(sample["meta"]["H"]),
        })

    # zapis manifestu
    import json
    man_path = out_base / f"manifest_{args.split}.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DONE] Zapisano: {len(manifest)} próbek, manifest: {man_path}")
    print(f"       DS: {ds_dir}")
    print(f"       VIS: {vis_dir}")


if __name__ == "__main__":
    main()
