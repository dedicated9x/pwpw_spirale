#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_search_sacks_points.py
Random search parametrów detekcji kropek na zdjęciu spirali Sacksa.

Wejście:
  --in /home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317.jpg

Wyjścia (tylko overlay PNG):
  /home/admin2/Documents/repos/pwpw/_spirale/outputs/random_search/overlay_<paramy>.png
"""

import os
import cv2
import math
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

# ---- domyślne ścieżki ----
DEFAULT_IN  = "/home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317.jpg"
# DEFAULT_OUT = "/home/admin2/Documents/repos/pwpw/_spirale/outputs/random_search"
DEFAULT_OUT = "/home/admin2/Documents/repos/pwpw/_spirale/outputs/random_search_narrow"

# ---- detekcja (parametry będą podawane z random searcha) ----
def preprocess_gray(bgr, clahe_clip, clahe_grid, gauss_blur):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    gray = clahe.apply(gray)
    if gauss_blur > 0:
        k = gauss_blur | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray

def make_mask(gray, method, thresh_type, adapt_block, adapt_C, morph_open, morph_close):
    # method: 'GAUSS' | 'MEAN'
    # thresh_type: 'BIN' | 'BIN_INV'
    meth = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 'GAUSS' else cv2.ADAPTIVE_THRESH_MEAN_C
    ttype = cv2.THRESH_BINARY_INV if thresh_type == 'BIN_INV' else cv2.THRESH_BINARY
    blk = max(3, adapt_block | 1)
    th = cv2.adaptiveThreshold(gray, 255, meth, ttype, blk, adapt_C)

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_open+1, 2*morph_open+1))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_close+1, 2*morph_close+1))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    return th

def shape_features(cnt):
    area = cv2.contourArea(cnt)
    per  = cv2.arcLength(cnt, True)
    circ = (4*math.pi*area/(per*per)) if per > 1e-6 else 0.0
    elong = 1.0
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        a = max(MA, ma) / 2.0
        b = max(1e-6, min(MA, ma) / 2.0)
        elong = a / b
    return area, circ, elong

def detect_points(bgr, params):
    gray = preprocess_gray(bgr, params['clahe_clip'], params['clahe_grid'], params['gauss_blur'])
    mask = make_mask(gray, params['method'], params['thresh_type'],
                     params['adapt_block'], params['adapt_C'],
                     params['morph_open'], params['morph_close'])

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        area, circ, elong = shape_features(c)
        if area < params['min_area'] or area > params['max_area']:
            continue
        if circ < params['min_circularity'] or elong > params['max_elongation']:
            continue
        M = cv2.moments(c)
        if M['m00'] <= 0:
            continue
        cx = float(M['m10']/M['m00'])
        cy = float(M['m01']/M['m00'])
        pts.append((cx, cy))
    return pts

def draw_overlay(bgr, pts, dot_radius=4):
    vis = bgr.copy()
    # lekko przyciemnij tło
    vis = cv2.addWeighted(vis, 0.85, np.zeros_like(vis), 0.15, 0)
    # same kropki (bez labeli)
    r = max(1, int(dot_radius))
    for (x, y) in pts:
        cv2.circle(vis, (int(round(x)), int(round(y))), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return vis

# # ---- random search przestrzeń ----
# def sample_params(rng):
#     return {
#         # preprocess
#         'clahe_clip'      : rng.uniform(1.5, 4.0),
#         'clahe_grid'      : rng.integers(4, 12),        # tile grid
#         'gauss_blur'      : int(rng.choice([0, 3, 5, 7])),
#         # adaptive threshold
#         'method'          : rng.choice(['GAUSS', 'MEAN']),
#         'thresh_type'     : rng.choice(['BIN', 'BIN_INV']),
#         'adapt_block'     : int(rng.integers(15, 61)),  # musi być nieparzyste (wymusimy w kodzie)
#         'adapt_C'         : float(rng.integers(2, 15)), # dodatnie = surowsze; można też dać ujemne, ale startujmy od +2..14
#         # morfologia
#         'morph_open'      : int(rng.integers(0, 4)),    # 0..3
#         'morph_close'     : int(rng.integers(0, 5)),    # 0..4
#         # filtr komponentów
#         'min_area'        : int(rng.integers(8, 60)),
#         'max_area'        : int(rng.integers(300, 3500)),
#         'min_circularity' : float(rng.uniform(0.40, 0.85)),
#         'max_elongation'  : float(rng.uniform(1.6, 3.5)),
#         # rysunek
#         'dot_radius'      : int(rng.integers(2, 7)),
#     }

# ---- random search przestrzeń (zawężona pod Twoje wyniki) ----
def sample_params(rng):
    return {
        # preprocess
        'clahe_clip'      : rng.uniform(2.6, 3.9),
        'clahe_grid'      : int(rng.integers(9, 13)),           # 9–12
        'gauss_blur'      : int(rng.choice([3, 5])),            # lekki blur

        # adaptive threshold
        'method'          : ('GAUSS' if rng.random() < 0.75 else 'MEAN'),
        'thresh_type'     : 'BIN_INV',                          # kropki ciemniejsze od tła
        'adapt_block'     : int(rng.integers(17, 24)),          # małe, lokalne okno (odd wymusi kod)
        'adapt_C'         : float(rng.integers(8, 14)),         # 8–13

        # morfologia
        'morph_open'      : 3,                                  # sprawdzony filtr drobnicy
        'morph_close'     : int(rng.integers(1, 3)),            # 1–2

        # filtr komponentów
        'min_area'        : int(rng.integers(20, 41)),          # 20–40 px²
        'max_area'        : int(rng.integers(1200, 2201)),      # 1200–2200 px²
        'min_circularity' : float(rng.uniform(0.65, 0.78)),     # ostrzej odcina śmieci
        'max_elongation'  : float(rng.uniform(1.8, 2.2)),       # nie dopuszczamy “robaków”

        # rysunek
        'dot_radius'      : int(rng.integers(4, 7)),            # 4–6
    }


def params_to_tag(p):
    # zwięzły, ale pełny tag w nazwie pliku
    tag = (
        f"cl{p['clahe_clip']:.2f}"
        f"_cg{p['clahe_grid']}"
        f"_gb{p['gauss_blur']}"
        f"_{p['method']}_{p['thresh_type']}"
        f"_ab{p['adapt_block']}"
        f"_aC{int(p['adapt_C'])}"
        f"_mo{p['morph_open']}"
        f"_mc{p['morph_close']}"
        f"_minA{p['min_area']}"
        f"_maxA{p['max_area']}"
        f"_circ{p['min_circularity']:.2f}"
        f"_elong{p['max_elongation']:.2f}"
        f"_dr{p['dot_radius']}"
    )
    # plikowa higiena: kropki → 'p'
    return tag.replace('.', 'p')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="Ścieżka do obrazu wejściowego")
    ap.add_argument("--outdir", dest="out_dir", default=DEFAULT_OUT, help="Folder wynikowy (tylko overlay)")
    ap.add_argument("--trials", type=int, default=150, help="Liczba prób losowych")
    ap.add_argument("--seed", type=int, default=42, help="Seed PRNG")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bgr = cv2.imread(args.in_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Nie mogę wczytać: {args.in_path}")

    rng = np.random.default_rng(args.seed)

    best = {"count": -1, "tag": None, "path": None}

    for _ in tqdm(range(args.trials), desc="Random search", ncols=100):
        p = sample_params(rng)

        pts = detect_points(bgr, p)
        vis = draw_overlay(bgr, pts, dot_radius=p['dot_radius'])

        tag = params_to_tag(p)
        out_path = os.path.join(args.out_dir, f"overlay_{tag}_n{len(pts)}.png")
        cv2.imwrite(out_path, vis)

        # opcjonalnie: pamiętaj najlepszy (po liczbie punktów)
        if len(pts) > best["count"]:
            best.update({"count": len(pts), "tag": tag, "path": out_path})

    # zapisz metadane best
    if best["path"]:
        meta = {
            "best_count": best["count"],
            "best_tag": best["tag"],
            "best_overlay_path": best["path"]
        }
        with open(os.path.join(args.out_dir, "best.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[BEST] n={best['count']} → {best['path']}")

if __name__ == "__main__":
    main()
