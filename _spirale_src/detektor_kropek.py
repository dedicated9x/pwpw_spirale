#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_sacks_points.py
Detekcja kropek (wydruk Spirali Sacksa) z fotki i eksport chmury punktów.

Wejście:
  /home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317.jpg

Wyjścia (do OUT_DIR):
  - points.csv           (x_px,y_px,r_eff_px,area,circularity)
  - overlay.png          (oryginał + markery detekcji)
  - debug_gray.png       (po CLAHE)
  - debug_thresh.png     (maska progowania po morfologii)

Uruchom:
  python3 extract_sacks_points.py
"""

import os
import csv
import math
import argparse
from typing import Tuple, List

import cv2
import numpy as np

# --- ŚCIEŻKI ---
IN_PATH  = "/_spirale/inputs/PXL_20250925_061456317.jpg"
OUT_DIR  = "/_spirale/outputs"

# --- PARAMETRY DETEKCJI (dostosujesz pod swój druk/telefon) ---
CLAHE_CLIP         = 2.0
CLAHE_GRID         = 8
GAUSS_BLUR         = 3        # kernel size (nieparzysty); 0 = wyłącz
ADAPT_BLOCK        = 35       # rozmiar okna dla adaptiveThreshold (nieparzysty)
ADAPT_C            = 7        # stała odejmowana; większa → mniej czułe
MORPH_OPEN         = 1        # promień otwarcia (0=wyłącz)
MORPH_CLOSE        = 2        # promień domknięcia (0=wyłącz)

MIN_AREA_PX        = 10       # minimalna liczba pikseli komponentu
MAX_AREA_PX        = 2_500    # maksymalna (zabezp. przed brudem/napisem)
MIN_CIRCULARITY    = 0.55     # 4πA/P^2
MAX_ELONGATION     = 3.0      # max. stosunek półosi elipsy (a/b)

DRAW_IDX_EVERY     = 1        # podpisywać każdy punkt

# --- NARZĘDZIA ---

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def to_gray_norm(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    gray = clahe.apply(gray)
    if GAUSS_BLUR and GAUSS_BLUR > 0:
        k = max(1, GAUSS_BLUR | 1)  # wymuś nieparzysty
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray

def adaptive_mask(gray: np.ndarray) -> np.ndarray:
    # Próba obu biegunów: kropki ciemne na jasnym tle lub odwrotnie
    th1 = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,
                                max(3, ADAPT_BLOCK | 1),
                                ADAPT_C)
    th2 = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                max(3, ADAPT_BLOCK | 1),
                                ADAPT_C)
    # Wybierz gęstszą, ale nie zalaną maskę
    ratio1 = th1.mean() / 255.0
    ratio2 = th2.mean() / 255.0
    th = th1 if 0.001 < ratio1 < 0.5 else th2

    # Morfologia
    if MORPH_OPEN > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MORPH_OPEN+1, 2*MORPH_OPEN+1))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    if MORPH_CLOSE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MORPH_CLOSE+1, 2*MORPH_CLOSE+1))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    return th

def components(mask: np.ndarray):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return num, labels, stats, centroids  # stats: [x,y,w,h,area]

def shape_features(contour: np.ndarray) -> Tuple[float, float, Tuple[float,float,float]]:
    # Circularity
    area = cv2.contourArea(contour)
    per  = cv2.arcLength(contour, True)
    circ = (4 * math.pi * area / (per * per)) if per > 1e-6 else 0.0

    # Elipsa
    if len(contour) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)  # MA: major, ma: minor
        a = max(MA, ma) / 2.0
        b = max(1e-6, min(MA, ma) / 2.0)
        elong = a / b
        ellipse = (cx, cy, a, b, angle)
    else:
        elong = 1.0
        M = cv2.moments(contour)
        cx = (M["m10"] / (M["m00"] + 1e-9))
        cy = (M["m01"] / (M["m00"] + 1e-9))
        ellipse = (cx, cy, 0.0, 0.0, 0.0)

    return circ, elong, ellipse

def detect_points(bgr: np.ndarray):
    gray = to_gray_norm(bgr)
    mask = adaptive_mask(gray)

    # kontury po masce
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue
        circ, elong, ellipse = shape_features(cnt)
        if circ < MIN_CIRCULARITY or elong > MAX_ELONGATION:
            continue

        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # promień efektywny ~ z obszaru
        r_eff = math.sqrt(area / math.pi)
        kept.append({
            "cx": cx, "cy": cy, "r_eff": r_eff,
            "area": float(area), "circularity": float(circ),
            "ellipse": ellipse, "contour": cnt
        })

    return gray, mask, kept

def draw_overlay(bgr: np.ndarray, points: List[dict]) -> np.ndarray:
    vis = bgr.copy()

    # Półprzezroczyste przygaszenie tła — lepszy kontrast markerów
    overlay = vis.copy()
    overlay[:] = (0, 0, 0)
    vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)

    for i, p in enumerate(sorted(points, key=lambda x: x["r_eff"], reverse=True)):
        cx, cy = int(round(p["cx"])), int(round(p["cy"]))
        r  = max(2, int(round(p["r_eff"])))

        # 1) elipsa konturu (cyjan)
        (ex, ey, a, b, ang) = p["ellipse"]
        if a > 0 and b > 0:
            cv2.ellipse(vis, (int(ex), int(ey)), (int(a), int(b)), ang, 0, 360, (255, 255, 0), 1, cv2.LINE_AA)

        # 2) centroid jako ZIELONY KWADRAT z czarną obwódką
        s = max(3, min(9, r))  # rozmiar kwadratu zależny od r
        pt1 = (cx - s, cy - s)
        pt2 = (cx + s, cy + s)
        cv2.rectangle(vis, pt1, pt2, (0, 0, 0), 3, cv2.LINE_AA)     # obrys
        cv2.rectangle(vis, pt1, pt2, (80, 220, 80), 2, cv2.LINE_AA) # wypełnienie/linia

        # 3) ID (biały z cieniem)
        if DRAW_IDX_EVERY and (i % DRAW_IDX_EVERY == 0):
            label = str(i)
            cv2.putText(vis, label, (cx + s + 3, cy - s - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (cx + s + 3, cy - s - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis

def save_csv(path: str, pts: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_px", "y_px", "r_eff_px", "area_px2", "circularity"])
        for p in pts:
            w.writerow([f"{p['cx']:.3f}", f"{p['cy']:.3f}", f"{p['r_eff']:.3f}", f"{p['area']:.3f}", f"{p['circularity']:.4f}"])

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=IN_PATH, help="Ścieżka do obrazu wejściowego")
    parser.add_argument("--outdir", dest="out_dir", default=OUT_DIR, help="Folder wyjściowy")
    args = parser.parse_args()

    in_path = args.in_path
    out_dir = args.out_dir
    ensure_dir(out_dir)

    bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Nie mogę wczytać: {in_path}")

    gray, mask, pts = detect_points(bgr)

    # sort stabilny: po x potem y dla powtarzalności (nie wymagane)
    pts_sorted = sorted(pts, key=lambda p: (p["cy"], p["cx"]))

    # zapisy
    csv_path   = os.path.join(out_dir, "points.csv")
    overlay    = draw_overlay(bgr, pts_sorted)
    overlay_p  = os.path.join(out_dir, "overlay.png")
    gray_p     = os.path.join(out_dir, "debug_gray.png")
    mask_p     = os.path.join(out_dir, "debug_thresh.png")

    save_csv(csv_path, pts_sorted)
    cv2.imwrite(overlay_p, overlay)
    cv2.imwrite(gray_p, gray)
    cv2.imwrite(mask_p, mask)

    print(f"[OK] Zapisano:")
    print(f"  - {csv_path}")
    print(f"  - {overlay_p}")
    print(f"  - {gray_p}")
    print(f"  - {mask_p}")
    print(f"Detekcji: {len(pts_sorted)}")

if __name__ == "__main__":
    main()
