#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_search_sacks_points_v3.py
Random search parametrów detekcji kropek na zdjęciu spirali Sacksa
z funkcją detect_points(input_img, output_img, params_dict)
oraz get_points(img_path, params_dict) -> List[Tuple[int,int]].

Wejście:
  --in /home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317.jpg

Wyjścia (tylko overlay PNG):
  /home/admin2/Documents/repos/pwpw/_spirale/outputs/random_search_narrow_v2/overlay_<paramy>_n{count}.png
"""

import os
import cv2
import math
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

# ---- domyślne ścieżki ----
DEFAULT_IN  = "/home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317.jpg"
DEFAULT_OUT = "/home/admin2/Documents/repos/pwpw/_spirale/outputs/random_search_narrow_v3"

# ---------------------- PIPELINE ----------------------

def preprocess_gray(bgr, clahe_clip, clahe_grid, gauss_blur):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    gray = clahe.apply(gray)
    if gauss_blur > 0:
        k = gauss_blur | 1  # nieparzysty
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray

def make_mask(gray, method, thresh_type, adapt_block, adapt_C, morph_open, morph_close):
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

def _shape_features(cnt):
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

def _detect_points_from_bgr(bgr, params):
    gray = preprocess_gray(bgr, params['clahe_clip'], params['clahe_grid'], params['gauss_blur'])
    mask = make_mask(gray, params['method'], params['thresh_type'],
                     params['adapt_block'], params['adapt_C'],
                     params['morph_open'], params['morph_close'])

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        area, circ, elong = _shape_features(c)
        if area < params['min_area'] or area > params['max_area']:
            continue
        if circ < params['min_circularity'] or elong > params['max_elongation']:
            continue
        M = cv2.moments(c)
        if M['m00'] <= 0:
            continue
        cx = float(M['m10']/M['m00'])
        cy = float(M['m01']/M['m00'])
        pts.append((cx, cy))  # floaty (subpiksel)
    return pts

def _draw_overlay(bgr, pts, dot_radius=4):
    vis = bgr.copy()
    vis = cv2.addWeighted(vis, 0.85, np.zeros_like(vis), 0.15, 0)  # lekkie przyciemnienie tła
    r = max(1, int(dot_radius))
    for (x, y) in pts:
        cv2.circle(vis, (int(round(x)), int(round(y))), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return vis

# ---------------------- PUBLIC API ----------------------

def detect_points(input_img: str, output_img: str, params_dict: dict):
    """
    Czyta obraz z input_img, wykrywa kropki wg params_dict,
    renderuje overlay (same kropki) i zapisuje do output_img.
    Zwraca dict z liczbą detekcji i listą punktów (floaty).
    """
    bgr = cv2.imread(input_img, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Nie mogę wczytać: {input_img}")

    pts = _detect_points_from_bgr(bgr, params_dict)
    vis = _draw_overlay(bgr, pts, dot_radius=params_dict.get('dot_radius', 4))
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    cv2.imwrite(output_img, vis)
    return {"count": len(pts), "points": pts}

def get_points(img_path: str, params_dict: dict) -> List[Tuple[int, int]]:
    """
    Zwraca listę centroidów jako inty (x, y), zaokrąglając subpikselowe wyniki.
    Nie zapisuje żadnego obrazu. Używaj w ETAPIE 2 (dopasowanie do wzorca).
    """
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Nie mogę wczytać: {img_path}")

    pts_float = _detect_points_from_bgr(bgr, params_dict)
    # Zaokrąglij do intów (zgodnie z wymaganym typem List[Tuple[int,int]])
    pts_int: List[Tuple[int, int]] = [(int(round(x)), int(round(y))) for (x, y) in pts_float]
    return pts_int

# ---------------------- RANDOM SEARCH ----------------------

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
    return tag.replace('.', 'p')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN, help="Ścieżka do obrazu wejściowego")
    ap.add_argument("--outdir", dest="out_dir", default=DEFAULT_OUT, help="Folder wynikowy (tylko overlay)")
    ap.add_argument("--trials", type=int, default=150, help="Liczba prób losowych")
    ap.add_argument("--seed", type=int, default=42, help="Seed PRNG")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    best = {"count": -1, "tag": None, "path": None}

    for _ in tqdm(range(args.trials), desc="Random search", ncols=100):
        p = sample_params(rng)
        tag = params_to_tag(p)

        tmp_path = os.path.join(args.out_dir, f"overlay_{tag}_tmp.png")
        res = detect_points(args.in_path, tmp_path, p)
        count = res["count"]

        final_path = os.path.join(args.out_dir, f"overlay_{tag}_n{count}.png")
        try:
            os.replace(tmp_path, final_path)
        except FileNotFoundError:
            final_path = None

        if final_path and count > best["count"]:
            best.update({"count": count, "tag": tag, "path": final_path})

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
