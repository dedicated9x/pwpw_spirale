#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2b_ml_train.py
Trenuje PointNet-regresor homografii H (3x3, h33=1) na datasetcie z step2a_ml_make_dataset.py.

Wejście:  .npz z polami: points [K,2], H [3,3], image_size [W,H], model, ...
Kanon:    wczytywany z canonical_sacks_N10000.npz (flip_y=True) – jak w generatorze.

Strata:   L = w_param * L1(Ĥ, H*) + w_cfd * Chamfer( Ĥ(S), P_norm ) + w_cbd * Chamfer( P_norm, Ĥ(S) )
           gdzie P_norm = znormalizowane punkty obserwacji do [-1,1] (centrowanie w (W/2,H/2), skala=0.5*max(W,H))

Uwaga: batchowanie z losowym podpróbkowaniem do max_pts (domyślnie 1024) dla stabilnej pamięci.
"""

import os
import json
import math
import glob
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# --------------------------- utils: homografia i normalizacja ---------------------------

def to_norm_coords(xy: torch.Tensor, W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Mapuje piksele -> [-1,1] w obu osiach (ten sam scale dla x,y).
    xy: (B,N,2), W/H: (B,)
    """
    cx = W.float() * 0.5
    cy = H.float() * 0.5
    alpha = torch.maximum(W.float(), H.float()) * 0.5  # 0.5 * max(W,H)
    x = (xy[..., 0] - cx[:, None]) / alpha[:, None]
    y = (xy[..., 1] - cy[:, None]) / alpha[:, None]
    return torch.stack([x, y], dim=-1)

def norm_mat(W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Zwraca macierz 3x3, która mapuje piksele -> [-1,1] (B,3,3)
    """
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

def apply_H_torch(pts: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Zastosuj H do punktów.
    pts: (B,N,2), H: (B,3,3). Zwraca (B,N,2).
    """
    B, N, _ = pts.shape
    ones = torch.ones(B, N, 1, dtype=pts.dtype, device=pts.device)
    P = torch.cat([pts, ones], dim=-1)                   # (B,N,3)
    Y = torch.bmm(P, H.transpose(1, 2))                  # (B,N,3)
    w = torch.clamp(Y[..., 2:3], min=1e-8)
    return Y[..., :2] / w

def H_from_params(params: torch.Tensor) -> torch.Tensor:
    """
    params: (B,8) -> H: [[a,b,c],[d,e,f],[g,h,1]]
    Renormalizuje tak, by h33=1.
    """
    B = params.shape[0]
    a,b,c,d,e,f,g,h = torch.chunk(params, 8, dim=1)
    H = torch.zeros(B,3,3, dtype=params.dtype, device=params.device)
    H[:,0,0]=a[:,0]; H[:,0,1]=b[:,0]; H[:,0,2]=c[:,0]
    H[:,1,0]=d[:,0]; H[:,1,1]=e[:,0]; H[:,1,2]=f[:,0]
    H[:,2,0]=g[:,0]; H[:,2,1]=h[:,0]; H[:,2,2]=1.0
    # renorm:
    scale = H[:,2,2:3]  # ==1, ale zostawiamy dla ogólności
    H = H / scale.unsqueeze(-1)
    return H

def H_to_params(H: torch.Tensor) -> torch.Tensor:
    """
    H: (B,3,3) -> (B,8) z h33=1
    """
    H = H / H[:,2,2:3].unsqueeze(-1)
    return torch.stack([H[:,0,0],H[:,0,1],H[:,0,2],H[:,1,0],H[:,1,1],H[:,1,2],H[:,2,0],H[:,2,1]], dim=1)


# --------------------------- dataset ---------------------------

class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir: str, canonical_npz: str, max_pts: int = 1024):
        self.files = sorted(glob.glob(str(Path(ds_dir) / "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"Brak plików .npz w {ds_dir}")
        can = np.load(canonical_npz)
        # kanon (flip_y=True jak w generatorze)
        self.S = np.stack([can["x_prime"], -can["y_prime"]], axis=1).astype(np.float32)  # (1229,2)
        self.max_pts = max_pts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        P = d["points"].astype(np.float32)          # (K,2) w pikselach
        H_gt = d["H"].astype(np.float32)            # (3,3) kanon->pix
        W, H = d["image_size"].astype(np.float32)   # (2,)

        # subsample/pad do max_pts
        K = P.shape[0]
        if K >= self.max_pts:
            sel = np.random.choice(K, self.max_pts, replace=False)
            P = P[sel]
        else:
            # dosampeluj z powtórzeniami
            add = np.random.choice(K, self.max_pts - K, replace=True)
            P = np.concatenate([P, P[add]], axis=0)

        sample = {
            "points_px": P,               # (max_pts,2)
            "H_gt_px": H_gt,              # (3,3)
            "W": W, "H": H,               # scalary
            "S": self.S,                  # (Np,2) stały kanon
        }
        return sample


def collate(batch):
    # wszystko stałej wielkości dzięki max_pts; tylko zamiana na tensory
    P = torch.from_numpy(np.stack([b["points_px"] for b in batch], axis=0))     # (B,N,2)
    W = torch.from_numpy(np.stack([b["W"] for b in batch], axis=0))             # (B,)
    H = torch.from_numpy(np.stack([b["H"] for b in batch], axis=0))             # (B,)
    Hgt_px = torch.from_numpy(np.stack([b["H_gt_px"] for b in batch], axis=0))  # (B,3,3)
    S = torch.from_numpy(batch[0]["S"]).unsqueeze(0).repeat(P.shape[0],1,1)     # (B,Ns,2)
    return P, W, H, Hgt_px, S


# --------------------------- PointNet regresor ---------------------------

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
        # inicjalizacja blisko homografii identyczności (H ~ I)
        nn.init.zeros_(self.mlp2[-1].weight)
        nn.init.zeros_(self.mlp2[-1].bias)

    def forward(self, pts_norm):  # (B,N,2)
        B, N, _ = pts_norm.shape
        f = self.mlp1(pts_norm)           # (B,N,256)
        g = torch.max(f, dim=1).values    # (B,256)
        out = self.mlp2(g)                # (B,8)
        return out


# --------------------------- Chamfer ---------------------------

def chamfer_bidirectional(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A: (B,Na,2), B: (B,Nb,2). Zwraca (mean min_A->B, mean min_B->A).
    """
    # dystanse kwadratowe przez broadcast
    # D[b,i,j] = ||A[b,i]-B[b,j]||^2
    D = torch.cdist(A, B, p=2)  # (B,Na,Nb)
    d_ab = torch.min(D, dim=2).values  # (B,Na)
    d_ba = torch.min(D, dim=1).values  # (B,Nb)
    return d_ab.mean(), d_ba.mean()


# --------------------------- trening ---------------------------

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # dane
    train_ds = CloudDataset(str(Path(args.dataset) / "ds" / "train"), args.canonical, args.max_pts)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_loader = None
    val_dir = Path(args.dataset) / "ds" / "val"
    if val_dir.exists():
        val_ds = CloudDataset(str(val_dir), args.canonical, args.max_pts)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, collate_fn=collate, drop_last=False)

    # model
    model = PointNetRegressor(in_dim=2, feat_dim=256, out_dim=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train e{epoch}/{args.epochs}", ncols=100)
        loss_avg = 0.0

        for P_px, W, H, Hgt_px, S in pbar:
            P_px = P_px.to(device); W=W.to(device); H=H.to(device); Hgt_px=Hgt_px.to(device); S=S.to(device)

            # normalizacja
            N_pix = norm_mat(W, H)                           # (B,3,3)
            P = to_norm_coords(P_px, W, H)                   # (B,N,2)
            Hgt = torch.bmm(N_pix, Hgt_px)                   # GT w przestrzeni normowanej

            # forward
            params = model(P)                                 # (B,8)
            H_pred = H_from_params(params)                    # (B,3,3)

            # Chamfer: H_pred(S) vs P
            S_pred = apply_H_torch(S, H_pred)                 # (B,Ns,2)
            cf_fwd, cf_bwd = chamfer_bidirectional(S_pred, P)

            # Param loss na 8 elementach
            tgt_params = H_to_params(Hgt).detach()
            l_param = F.l1_loss(params, tgt_params)

            loss = args.w_param * l_param + args.w_cf * cf_fwd + args.w_cb * cf_bwd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_avg = 0.9*loss_avg + 0.1*loss.item() if loss_avg>0 else loss.item()
            pbar.set_postfix(loss=f"{loss_avg:.4f}", lp=f"{l_param.item():.3f}",
                             cf=f"{cf_fwd.item():.3f}", cb=f"{cf_bwd.item():.3f}")

        # walidacja (opcjonalna)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0; n_batches = 0
                for P_px, W, H, Hgt_px, S in val_loader:
                    P_px = P_px.to(device); W=W.to(device); H=H.to(device); Hgt_px=Hgt_px.to(device); S=S.to(device)
                    N_pix = norm_mat(W, H)
                    P = to_norm_coords(P_px, W, H)
                    Hgt = torch.bmm(N_pix, Hgt_px)
                    params = model(P)
                    H_pred = H_from_params(params)
                    S_pred = apply_H_torch(S, H_pred)
                    cf_fwd, cf_bwd = chamfer_bidirectional(S_pred, P)
                    tgt_params = H_to_params(Hgt).detach()
                    l_param = F.l1_loss(params, tgt_params)
                    loss = args.w_param * l_param + args.w_cf * cf_fwd + args.w_cb * cf_bwd
                    val_loss += loss.item(); n_batches += 1
                val_loss /= max(1, n_batches)
            # save best
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(),
                            "args": vars(args)}, outdir / "best.pt")
        # save last
        torch.save({"model": model.state_dict(),
                    "args": vars(args)}, outdir / "last.pt")

    print(f"[DONE] modelem zapisane do: {outdir}/best.pt i last.pt")

PATH_DATASET = Path(__file__).parents[1] / "_outputs/_spirale/step3a_make_dataset"
PATH_CANON = Path(__file__).parents[1] / "_outputs/_spirale/step2a_create_canonical/canonical_sacks_N10000.npz"
PATH_OUTPUT_MODEL = Path(__file__).parents[1] / "_outputs/_spirale/step3b_train"

# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default=str(PATH_DATASET),
                    help="Folder bazowy datasetu (z podfolderami ds/train, ds/val, ...)")
    ap.add_argument("--canonical", type=str,
                    default=str(PATH_CANON),
                    help="Kanon z step2a (używamy do S)")
    ap.add_argument("--outdir", type=str,
                    default=str(PATH_OUTPUT_MODEL),
                    help="Gdzie zapisać model i config")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_pts", type=int, default=1024, help="Ile punktów z obserwacji używać na próbkę")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    # wagi strat
    ap.add_argument("--w_param", type=float, default=1.0)
    ap.add_argument("--w_cf", type=float, default=1.0)
    ap.add_argument("--w_cb", type=float, default=0.2)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(1337)
    np.random.seed(1337)
    train_loop(args)
