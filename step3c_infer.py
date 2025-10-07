#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2b_ml_infer_batch.py
Batch inferencja PointNet-regresora homografii dla WSZYSTKICH plików bezpośrednio w _inputs (bez rekurencji).
Wizka: zielone = detekcje z heury, pomarańczowe = kanon po transformacji ML.
"""

import json
import math
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- heurystyczna detekcja punktów ---
from _library.random_search_sacks_points_v3 import get_points  # zwraca floaty

# --- domyślne ścieżki ---
BASE_DIR        = Path(__file__).parents[0]
DEFAULT_INPUTS  = BASE_DIR / "_inputs"
DEFAULT_CANON   = BASE_DIR / "_outputs/_spirale/step2a_create_canonical/canonical_sacks_N10000.npz"
DEFAULT_MODEL   = BASE_DIR / "_outputs/_spirale/step3b_train/last.pt"
DEFAULT_OUTDIR  = BASE_DIR / "_outputs/_spirale/step3c_infer"

# --- heura (domyślne parametry) ---
HEUR_PARAMS = {
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

# ---------------------- narzędzia: homografia/normalizacja ----------------------

def load_canonical(npz_path: str, flip_y: bool = True) -> np.ndarray:
    """Zwraca (Np,2) kanon XY (Sacks primes). Flip Y (y->-y) jak w treningu."""
    d = np.load(npz_path)
    x = d["x_prime"].astype(np.float32)
    y = d["y_prime"].astype(np.float32)
    if flip_y:
        y = -y
    return np.stack([x, y], axis=1)

def norm_mat_torch(W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Macierz 3x3: piksele -> [-1,1] (ten sam scale dla x,y = 0.5*max(W,H))."""
    B = W.shape[0]
    cx = W.float() * 0.5
    cy = H.float() * 0.5
    alpha = torch.maximum(W.float(), H.float()) * 0.5
    M = torch.zeros(B, 3, 3, dtype=torch.float32, device=W.device)
    M[:, 0, 0] = 1.0 / alpha
    M[:, 1, 1] = 1.0 / alpha
    M[:, 0, 2] = -cx / alpha
    M[:, 1, 2] = -cy / alpha
    M[:, 2, 2] = 1.0
    return M

def to_norm_coords_torch(xy_px: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    cx = W.float() * 0.5
    cy = H.float() * 0.5
    alpha = torch.maximum(W.float(), H.float()) * 0.5
    x = (xy_px[..., 0] - cx[:, None]) / alpha[:, None]
    y = (xy_px[..., 1] - cy[:, None]) / alpha[:, None]
    return torch.stack([x, y], dim=-1)

def apply_H_torch(pts: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """pts: (B,N,2), H: (B,3,3) -> (B,N,2)"""
    B, N, _ = pts.shape
    ones = torch.ones(B, N, 1, dtype=pts.dtype, device=pts.device)
    P = torch.cat([pts, ones], dim=-1)
    Y = torch.bmm(P, H.transpose(1, 2))
    w = torch.clamp(Y[..., 2:3], min=1e-8)
    return Y[..., :2] / w

def H_from_params(params: torch.Tensor) -> torch.Tensor:
    """params (B,8) → H (B,3,3) z h33=1"""
    B = params.shape[0]
    a,b,c,d,e,f,g,h = torch.chunk(params, 8, dim=1)
    H = torch.zeros(B,3,3, dtype=params.dtype, device=params.device)
    H[:,0,0]=a[:,0]; H[:,0,1]=b[:,0]; H[:,0,2]=c[:,0]
    H[:,1,0]=d[:,0]; H[:,1,1]=e[:,0]; H[:,1,2]=f[:,0]
    H[:,2,0]=g[:,0]; H[:,2,1]=h[:,0]; H[:,2,2]=1.0
    return H

# ---------------------- PointNet (musi pasować do treningu) ----------------------

class PointNetRegressor(nn.Module):
    def __init__(self, in_dim=2, feat_dim=256, out_dim=8):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256, feat_dim), nn.ReLU(True),
            nn.Linear(feat_dim, 256), nn.ReLU(True),
            nn.Linear(256, 128), nn.ReLU(True),
            nn.Linear(128, out_dim),
        )
        nn.init.zeros_(self.mlp2[-1].weight)
        nn.init.zeros_(self.mlp2[-1].bias)

    def forward(self, pts_norm):  # (B,N,2)
        f = self.mlp1(pts_norm)           # (B,N,256)
        g = torch.max(f, dim=1).values    # (B,256)
        out = self.mlp2(g)                # (B,8)
        return out

# ---------------------- dopasowanie: metryki i overlay ----------------------

def mutual_nn_pairs(A: np.ndarray, B: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual-NN z bramkowaniem tau [px]. Zwraca (idxA, idxB, dists)."""
    if A.size == 0 or B.size == 0:
        return np.array([], int), np.array([], int), np.array([], float)
    DA = np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(axis=2))  # (Na,Nb)
    iB = DA.argmin(axis=1)
    dAB = DA[np.arange(A.shape[0]), iB]
    iA = DA.argmin(axis=0)
    mask = (np.arange(A.shape[0]) == iA[iB]) & (dAB <= tau)
    idxA = np.where(mask)[0]
    idxB = iB[mask]
    d = dAB[mask]
    return idxA.astype(int), idxB.astype(int), d.astype(float)

def draw_overlay(img_path: str, P_px: np.ndarray, S_px: np.ndarray,
                 pairs: Tuple[np.ndarray, np.ndarray], out_path: Path,
                 draw_lines: bool = True):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    vis = cv2.addWeighted(bgr, 0.82, np.zeros_like(bgr), 0.18, 0)
    for x, y in P_px:  # detekcje (zielone)
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (80,220,80), -1, lineType=cv2.LINE_AA)
    for x, y in S_px:  # kanon po ML (pomarańcz)
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (255,160,0), -1, lineType=cv2.LINE_AA)
    if draw_lines and pairs[0].size > 0:
        ia, ib = pairs
        for a, b in zip(ia.tolist(), ib.tolist()):
            xa, ya = S_px[a]; xb, yb = P_px[b]
            cv2.line(vis, (int(round(xa)), int(round(ya))),
                          (int(round(xb)), int(round(yb))), (180,180,180), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

# ---------------------- inferencja dla jednego obrazu ----------------------

def infer_one(img_path: str,
              model_ckpt: str,
              canonical_npz: str,
              out_dir: Path,
              heur_params: Dict,
              max_pts: int = 1024,
              tau_px: float = 6.0,
              device: str = "cuda"):
    device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 1) heura -> punkty
    pts = get_points(img_path, heur_params)   # List[(x,y)] float
    P_px = np.array(pts, dtype=np.float32)
    if P_px.size == 0:
        raise RuntimeError(f"Brak punktów z heury dla {img_path}")

    # 2) kanon (flip_y=True jak w treningu)
    S = load_canonical(canonical_npz, flip_y=True).astype(np.float32)

    # 3) przygotuj batch (1,N,2) i normalizację do [-1,1]
    H_img, W_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape
    W = torch.tensor([W_img], dtype=torch.float32, device=device)
    H = torch.tensor([H_img], dtype=torch.float32, device=device)

    # subsample/pad P do max_pts
    K = P_px.shape[0]
    if K >= max_pts:
        sel = np.random.choice(K, max_pts, replace=False)
        P_px_used = P_px[sel]
    else:
        add = np.random.choice(K, max_pts - K, replace=True)
        P_px_used = np.concatenate([P_px, P_px[add]], axis=0)
    P_px_t = torch.from_numpy(P_px_used[None, ...]).to(device)  # (1,N,2)

    N_pix = norm_mat_torch(W, H)                                 # (1,3,3)
    P_norm = to_norm_coords_torch(P_px_t, W, H)                  # (1,N,2)

    # 4) model
    model = PointNetRegressor(in_dim=2, feat_dim=256, out_dim=8).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        params = model(P_norm)           # (1,8)
        H_norm = H_from_params(params)   # (1,3,3)

    # 5) do pikseli
    N_inv = torch.linalg.inv(N_pix)
    H_px = torch.bmm(N_inv, H_norm)      # (1,3,3)
    H_px_np = H_px.squeeze(0).cpu().numpy()

    # 6) przewidź pozycje kanonu w pikselach
    S_t = apply_H_torch(torch.from_numpy(S[None, ...]).to(device), H_px).squeeze(0).cpu().numpy()

    # 7) metryki (mutual-NN, bramka tau_px)
    ia, ib, d = mutual_nn_pairs(S_t, P_px, tau=tau_px)
    keep = max(4, int(0.85 * ia.size)) if ia.size > 0 else 0
    if ia.size > 0:
        kidx = np.argsort(d)[:keep]
        rms = float(np.sqrt(np.mean(d[kidx]**2)))
        p95 = float(np.quantile(d[kidx], 0.95))
        inliers_ratio = float(keep / max(1, ia.size))
    else:
        rms = p95 = float("inf")
        inliers_ratio = 0.0

    # 8) zapisy
    img_stem = Path(img_path).stem
    out_img  = out_dir / img_stem / "overlay_pointnet.png"
    out_json = out_dir / img_stem / "transform_pointnet.json"
    draw_overlay(img_path, P_px, S_t, (ia, ib), out_img, draw_lines=True)

    meta = {
        "image": str(img_path),
        "model_ckpt": str(model_ckpt),
        "tau_px": tau_px,
        "stats": {"rms_px": rms, "p95_px": p95, "inliers_ratio": inliers_ratio,
                  "matched_pairs": int(ia.size)},
        "H_norm": H_norm.squeeze(0).cpu().numpy().tolist(),
        "H_px": H_px_np.tolist(),
        "W": int(W_img), "H": int(H_img),
        "heur_params": HEUR_PARAMS,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[infer] {img_stem}: RMS={rms:.3f}px  p95={p95:.3f}px  inliers={inliers_ratio:.3f}  pairs={ia.size}")
    return rms, p95, inliers_ratio

# ---------------------- CLI ----------------------

def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_dir", type=Path, default=DEFAULT_INPUTS,
                    help="Folder obrazów – tylko pliki bezpośrednio w katalogu (bez rekurencji).")
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Ścieżka do .pt (best/last).")
    ap.add_argument("--canonical", type=str, default=str(DEFAULT_CANON), help="Kanon z step2a (flip_y=True jak w treningu).")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Folder wyników.")
    ap.add_argument("--params_json", type=str, default="", help="(Opcjonalnie) JSON z parametrami heury.")
    ap.add_argument("--max_pts", type=int, default=1024, help="Ile punktów z chmury użyć (subsample/pad).")
    ap.add_argument("--tau_px", type=float, default=6.0, help="Próg (px) dla par Mutual-NN.")
    ap.add_argument("--cpu", action="store_true", help="Wymuś inferencję na CPU.")
    return ap.parse_args()

# ---------------------- main (batch) ----------------------

def list_input_files(inputs_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in inputs_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

def main():
    args = parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    # params heury
    heur_params = HEUR_PARAMS
    if args.params_json:
        with open(args.params_json, "r", encoding="utf-8") as f:
            heur_params = json.load(f)

    device = "cpu" if args.cpu else "cuda"

    inputs_dir: Path = args.inputs_dir
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Brak katalogu wejściowego: {inputs_dir}")

    paths = list_input_files(inputs_dir)
    if not paths:
        raise FileNotFoundError(f"Brak obrazów w {inputs_dir}")

    print(f"[infer] Plików do przetworzenia: {len(paths)}  |  device={device}")

    ok = 0
    for pth in tqdm(paths, desc="infer batch", unit="img"):
        try:
            infer_one(str(pth), args.model, args.canonical, out_dir, heur_params,
                      max_pts=args.max_pts, tau_px=args.tau_px, device=device)
            ok += 1
        except Exception as e:
            print(f"[infer][ERR] {pth.name}: {e}")

    print(f"\n[SUMMARY] OK: {ok}/{len(paths)} | OUT: {out_dir}")

if __name__ == "__main__":
    main()
