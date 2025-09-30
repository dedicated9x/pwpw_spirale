#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2b_fit_transform.py  (v4 – fixed Procrustes + canonical y-flip)
ETAP 2: Dopasowanie transformacji (similarity) między detekcjami z heury a kanonem Sacksa.

Zmiany vs v3:
- Procrustes (Umeyama) naprawiony: C = Xc.T @ Yc / n, poprawny wybór R i skali.
- Domyślnie flipujemy kanon po osi Y (y -> -y), żeby pracować w układzie obrazu (y w dół).
- reflection-aware: --reflection {auto,allow,forbid}, poprawnie działa.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import cv2

from _spirale_src.step1_rs.random_search_sacks_points_v3 import get_points

# --- hardcoded params (Twoje) ---
HARD_PARAMS = {
    'clahe_clip': 3.71,
    'clahe_grid': 12,
    'gauss_blur': 3,
    'method': 'GAUSS',
    'thresh_type': 'BIN_INV',
    'adapt_block': 20,
    'adapt_C': 10,
    'morph_open': 3,
    'morph_close': 1,
    'min_area': 25,
    'max_area': 1750,
    'min_circularity': 0.67,
    'max_elongation': 1.83,
    'dot_radius': 6,
}

# --- domyślne ścieżki ---
DEFAULT_IMG = "/home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317_cut_shifted.jpg"
DEFAULT_CAN = "/home/admin2/Documents/repos/pwpw/_spirale/outputs/canonical/canonical_sacks_N10000.npz"
OUT_BASE    = Path("/home/admin2/Documents/repos/pwpw/_spirale/outputs/fit")

# ----------------------------- utils -----------------------------

def load_canonical(npz_path: str, flip_y: bool) -> Dict[str, np.ndarray]:
    d = np.load(npz_path)
    x = d["x_prime"].astype(np.float64)
    y = d["y_prime"].astype(np.float64)
    if flip_y:
        y = -y  # dopasowujemy do układu obrazu (oś y w dół)
    return {
        "N": int(d["N"]),
        "x_prime": x,
        "y_prime": y,
    }

def pack_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([x, y], axis=1)

def procrustes_similarity_umeyama(X: np.ndarray, Y: np.ndarray, allow_reflection: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Znajdź (s, R, t), które minimalizuje || Y - (s R X + t) ||_F.
    Implementacja wg Umeyama (1991).
    - X, Y: (N,2)
    - allow_reflection: jeśli False, wymuszamy det(R)=+1 (czysta rotacja).
    """
    assert X.shape == Y.shape and X.shape[1] == 2
    n = X.shape[0]
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    # kowariancja X->Y
    C = (Xc.T @ Yc) / n  # 2x2

    U, S, Vt = np.linalg.svd(C)
    # D = diag(1, sign) zapewnia odpowiedni det(R)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1.0
    if not allow_reflection and D[1, 1] < 0:
        # wymuś det=+1
        D[1, 1] = 1.0

    R = U @ D @ Vt
    varX = (Xc**2).sum() / n
    s = float(np.trace(np.diag(S) @ D) / varX)
    t = muY - s * (R @ muX)

    return s, R, t

def apply_similarity(P: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # wiersze jako wektory: (P @ R^T)
    return (s * (P @ R.T)) + t

def nn_unique_pairs(S_t: np.ndarray, P: np.ndarray):
    if S_t.size == 0 or P.size == 0:
        return (np.array([], int), np.array([], int), np.array([], float))
    d2 = (S_t[:, None, 0] - P[None, :, 0])**2 + (S_t[:, None, 1] - P[None, :, 1])**2
    idxP = np.argmin(d2, axis=1)
    dist = np.sqrt(np.min(d2, axis=1))
    best = {}
    for iS, iP, d in zip(range(S_t.shape[0]), idxP.tolist(), dist.tolist()):
        if (iP not in best) or (d < best[iP][1]):
            best[iP] = (iS, d)
    if not best:
        return (np.array([], int), np.array([], int), np.array([], float))
    pairs_S, pairs_P, pairs_D = zip(*[(iS, iP, d) for iP, (iS, d) in best.items()])
    return (np.array(pairs_S, int), np.array(pairs_P, int), np.array(pairs_D, float))

def trimmed_icp_similarity(S: np.ndarray, P: np.ndarray,
                           keep_ratio: float,
                           max_iters: int,
                           tol_rms: float,
                           allow_reflection: bool):
    # Inicjacja: skala po RMS + centrowanie, bez rotacji
    muS, muP = S.mean(axis=0), P.mean(axis=0)
    s = float(np.sqrt(((P - muP)**2).sum() / ((S - muS)**2).sum()))
    R = np.eye(2)
    t = muP - s * (R @ muS)

    history = []
    last_rms = None

    for it in range(1, max_iters + 1):
        S_t = apply_similarity(S, s, R, t)
        idxS, idxP, d = nn_unique_pairs(S_t, P)
        if idxS.size == 0:
            return {"ok": False, "reason": "no_pairs"}

        keep = int(max(4, math.floor(keep_ratio * idxS.size)))
        kidx = np.argsort(d)[:keep]
        idxS_k, idxP_k, d_k = idxS[kidx], idxP[kidx], d[kidx]

        s_new, R_new, t_new = procrustes_similarity_umeyama(S[idxS_k], P[idxP_k], allow_reflection=allow_reflection)

        rms = float(np.sqrt(np.mean(d_k**2)))
        p95 = float(np.quantile(d_k, 0.95))
        inl_ratio = keep / max(1, idxS.size)
        detR = float(np.linalg.det(R_new))
        history.append({"iter": it, "rms": rms, "p95": p95, "pairs": int(idxS.size),
                        "kept": int(keep), "inliers_ratio": inl_ratio, "detR": detR})
        print(f"[ICP {'REFL' if allow_reflection else 'ROT '}] it={it:02d} | pairs={idxS.size} kept={keep} "
              f"| rms={rms:.3f} p95={p95:.3f} inliers={inl_ratio:.3f} det(R)={detR:+.3f}")

        if last_rms is not None and abs(last_rms - rms) < tol_rms:
            s, R, t = s_new, R_new, t_new
            print("[ICP] Δrms < tol → stop.")
            break
        s, R, t = s_new, R_new, t_new
        last_rms = rms

    # final stats
    S_t = apply_similarity(S, s, R, t)
    idxS, idxP, d = nn_unique_pairs(S_t, P)
    keep = int(max(4, math.floor(keep_ratio * idxS.size)))
    kidx = np.argsort(d)[:keep]
    d_k = d[kidx]
    stats = {
        "rms_px": float(np.sqrt(np.mean(d_k**2))),
        "p95_px": float(np.quantile(d_k, 0.95)),
        "inliers_ratio": keep / max(1, idxS.size),
        "iters": len(history),
        "detR": float(np.linalg.det(R)),
    }
    return {
        "ok": True, "s": float(s), "R": R.astype(float), "t": t.astype(float),
        "inliers_S": idxS[kidx], "inliers_P": idxP[kidx], "inliers_dist": d_k,
        "history": history, "stats": stats
    }

def save_detections_csv(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("x_px,y_px\n")
        for x, y in pts:
            f.write(f"{int(x)},{int(y)}\n")

def save_transform_json(path: Path, res: Dict[str, Any], chosen_mode: str, flip_y: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    R = res["R"]
    angle = math.atan2(R[1,0], R[0,0])
    M = np.zeros((2,3), dtype=float)
    M[:,:2] = res["s"] * R
    M[:,2]  = res["t"]
    out = {
        "model": "similarity",
        "chosen_mode": chosen_mode,   # 'forbid' (rot) lub 'allow' (odbicie)
        "canonical_flipped_y": flip_y,
        "scale": res["s"],
        "rotation_rad": angle,
        "rotation_deg": angle * 180.0 / math.pi,
        "tx": float(res["t"][0]),
        "ty": float(res["t"][1]),
        "matrix_2x3": M.tolist(),
        "stats": res["stats"],
        "history": res["history"],
        "hardcoded_params": HARD_PARAMS,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def draw_overlay_fit(img_path: str, P: np.ndarray, S: np.ndarray, trans: Dict[str, Any],
                     inliers_S: np.ndarray, inliers_P: np.ndarray, out_path: Path) -> None:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    s, R, t = trans["s"], trans["R"], trans["t"]
    S_t = apply_similarity(S, s, R, t)

    vis = cv2.addWeighted(bgr, 0.80, np.zeros_like(bgr), 0.20, 0)
    for x, y in P:
        cv2.circle(vis, (int(x), int(y)), 3, (80, 220, 80), -1, lineType=cv2.LINE_AA)           # detekcje
    for x, y in S_t:
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (255, 160, 0), 1, lineType=cv2.LINE_AA) # kanon po T
    for iS, iP in zip(inliers_S.tolist(), inliers_P.tolist()):
        xs, ys = S_t[iS]; xp, yp = P[iP]
        cv2.line(vis, (int(round(xs)), int(round(ys))), (int(round(xp)), int(round(yp))), (180,180,180), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

# ----------------------------- CLI -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, default=DEFAULT_IMG)
    ap.add_argument("--canonical", type=str, default=DEFAULT_CAN)
    ap.add_argument("--flip_canonical_y", type=str, default="true", choices=["true","false"],
                    help="Jeśli true (domyślnie), zamienia y->-y w kanonie, by dopasować układ obrazu (y w dół).")
    ap.add_argument("--keep_ratio", type=float, default=0.85)
    ap.add_argument("--max_iters", type=int, default=30)
    ap.add_argument("--tol_rms", type=float, default=1e-4)
    ap.add_argument("--reflection", type=str, default="auto", choices=["auto","allow","forbid"],
                    help="auto testuje oba warianty i wybiera lepszy")
    return ap.parse_args()

def main():
    args = parse_args()
    flip_y = (args.flip_canonical_y.lower() == "true")

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    can = load_canonical(args.canonical, flip_y=flip_y)
    S = pack_xy(can["x_prime"], can["y_prime"])

    print(f"[fit] Obraz: {img_path.name}")
    print(f"[fit] Kanon: N={10000}, primes={len(S)}  (flip_y={flip_y})")
    print(f"[fit] Heura (hardcoded): {HARD_PARAMS}")

    # ETAP 1: detekcje
    pts = get_points(str(img_path), HARD_PARAMS)
    P = np.array(pts, dtype=np.float64)
    print(f"[fit] Detekcje (heura): {len(P)}")

    out_dir = OUT_BASE / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    det_csv = out_dir / "detections.csv"
    save_detections_csv(det_csv, P)

    # ETAP 2: dopasowanie – tryby reflection
    modes = []
    if args.reflection == "auto":
        modes = [False, True]
    elif args.reflection == "allow":
        modes = [True]
    else:
        modes = [False]

    best = None
    best_mode = None
    for allow_reflection in modes:
        res = trimmed_icp_similarity(
            S=S, P=P,
            keep_ratio=float(args.keep_ratio),
            max_iters=int(args.max_iters),
            tol_rms=float(args.tol_rms),
            allow_reflection=allow_reflection
        )
        if not res.get("ok", False):
            print(f"[fit] Try ({'allow' if allow_reflection else 'forbid'}) failed: {res.get('reason')}")
            continue
        detR = res["stats"]["detR"]
        print(f"[fit] FINAL ({'allow' if allow_reflection else 'forbid'}) | rms={res['stats']['rms_px']:.3f}px "
              f"p95={res['stats']['p95_px']:.3f}px inliers={res['stats']['inliers_ratio']:.3f} det(R)={detR:+.3f}")
        if (best is None) or (res['stats']['rms_px'] < best['stats']['rms_px']):
            best = res
            best_mode = "allow" if allow_reflection else "forbid"

    if best is None:
        print("[fit][ERROR] Żadne dopasowanie nie wyszło.")
        return

    trn_json = out_dir / "transform.json"
    overlay  = out_dir / "overlay_fit.png"
    save_transform_json(trn_json, best, chosen_mode=best_mode, flip_y=flip_y)
    draw_overlay_fit(str(img_path), P, S, best, best["inliers_S"], best["inliers_P"], overlay)

    if best_mode == "allow":
        print("[fit] Wybrano wariant z ODBICIEM (det(R)<0).")
    print(f"[fit] Zapisano do: {out_dir}")

if __name__ == "__main__":
    main()
