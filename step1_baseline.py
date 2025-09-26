# path_img = "/home/admin2/Downloads/Shape detection/ShapeDetector raw/PXL_20250925_050204575.jpg"
#
# width = 3072
# height = 4080

import os
from pathlib import Path
import cv2 as cv
import numpy as np
from math import pi

# --- I/O ---
path_img = "/home/admin2/Downloads/Shape detection/ShapeDetector raw/PXL_20250925_050204575.jpg"
out_dir  = "/home/admin2/Documents/repos/cct/pwpw/outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = str(Path(out_dir) / ("detected_" + Path(path_img).stem + ".jpg"))

# --- Konfiguracja (do przyszłego grid-searcha) ---
CFG = {
    # preprocessing
    "resize_max_side": 2000,       # skalujemy do krótszego/b. długiego boku – przyspiesza
    "gauss_blur_ksize": 3,         # 0/None = bez
    "clahe_clip_limit": 2.0,       # 0/None = bez
    "clahe_tile_grid": 8,
    # threshold
    "threshold_mode": "otsu",      # "otsu" | "adaptive_mean" | "adaptive_gauss"
    "adaptive_block": 35,          # nieparzyste
    "adaptive_C": 5,
    "invert_binary": True,         # rysunek (ciemne linie) na jasnym tle
    # morfologia
    "morph_close_ksz": 3,          # domykanie cienkich przerw
    "morph_open_ksz": 0,           # odszumianie – 0 = pomiń
    "dilate_ksz": 1,               # 0 = pomiń
    # deskew (opcjonalnie)
    "deskew": False,
    "hough_thresh": 200,
    # filtrowanie konturów
    "min_area": 60,                # minimalne pole konturu w pikselach (po przeskalowaniu)
    "max_area": 1e6,
    "min_perimeter": 30,
    # approx
    "approx_eps_frac": 0.02,       # epsilon = eps_frac * obwód
    # klasyfikacja
    "circle_circularity_min": 0.78, # 4πA/P^2 – im bliżej 1 tym bardziej kołowe
    "circle_hough_enable": False,  # dodatkowe wsparcie przez HoughCircles
    "square_aspect_tol": 0.20,     # |1 - w/h| <= tol → kwadrat
    # ocena/score
    "score_from_circularity_weight": 0.7,
    # rysowanie
    "draw_thickness": 2,
}

# --- narzędzia ---
def resize_keep_aspect(img, max_side):
    if max_side is None: return img, 1.0
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv.resize(img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)
    return img, scale

def deskew_by_hough(gray, cfg):
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLines(edges, 1, np.deg2rad(1), cfg["hough_thresh"])
    if lines is None: return gray, np.eye(2), 0.0
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        ang = (theta - pi/2)  # linie pionowe → 0
        angles.append(ang)
    ang = np.median(angles) * 180/pi
    # jeśli odchyłka mała – nie kręcimy
    if abs(ang) < 0.5: return gray, np.eye(2), 0.0
    (h, w) = gray.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    rot = cv.warpAffine(gray, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return rot, M, ang

def threshold_image(gray, cfg):
    if cfg["gauss_blur_ksize"] and cfg["gauss_blur_ksize"] > 0:
        k = int(cfg["gauss_blur_ksize"]) | 1
        gray = cv.GaussianBlur(gray, (k, k), 0)
    if cfg["clahe_clip_limit"] and cfg["clahe_clip_limit"] > 0:
        clahe = cv.createCLAHE(clipLimit=float(cfg["clahe_clip_limit"]),
                               tileGridSize=(int(cfg["clahe_tile_grid"]), int(cfg["clahe_tile_grid"])))
        gray = clahe.apply(gray)
    mode = cfg["threshold_mode"]
    if mode == "otsu":
        _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    elif mode == "adaptive_mean":
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                  int(cfg["adaptive_block"]) | 1, cfg["adaptive_C"])
    elif mode == "adaptive_gauss":
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                  int(cfg["adaptive_block"]) | 1, cfg["adaptive_C"])
    else:
        raise ValueError("unknown threshold_mode")
    if cfg["invert_binary"]:
        bw = cv.bitwise_not(bw)
    # morfologia
    if cfg["morph_close_ksz"] and cfg["morph_close_ksz"] > 0:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (cfg["morph_close_ksz"], cfg["morph_close_ksz"]))
        bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, k, iterations=1)
    if cfg["morph_open_ksz"] and cfg["morph_open_ksz"] > 0:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (cfg["morph_open_ksz"], cfg["morph_open_ksz"]))
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, k, iterations=1)
    if cfg["dilate_ksz"] and cfg["dilate_ksz"] > 0:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (cfg["dilate_ksz"], cfg["dilate_ksz"]))
        bw = cv.dilate(bw, k, iterations=1)
    return bw

def classify_contour(cnt, cfg):
    area = cv.contourArea(cnt)
    peri = cv.arcLength(cnt, True)
    if peri < cfg["min_perimeter"] or area < cfg["min_area"] or area > cfg["max_area"]:
        return None
    # aproksymacja
    eps = cfg["approx_eps_frac"] * peri
    approx = cv.approxPolyDP(cnt, eps, True)
    # bbox i minRect
    x,y,w,h = cv.boundingRect(approx)
    aspect = w / float(h) if h > 0 else 0
    # kołowość
    circularity = 0 if peri == 0 else (4 * pi * area) / (peri * peri)
    label, score = None, 0.0

    if len(approx) == 3:
        label, score = "triangle", 0.8
    elif len(approx) == 4:
        if abs(1 - aspect) <= cfg["square_aspect_tol"]:
            label, score = "square", 0.85
        else:
            # prostokąty ignorujemy lub możemy zaklasyfikować jako square z niższym score
            label, score = "square", max(0.5, 1 - abs(1 - aspect))
    else:
        if circularity >= cfg["circle_circularity_min"]:
            label = "circle"
            # skalujemy score 0..1 z circularity
            score = min(1.0, (circularity - cfg["circle_circularity_min"]) / (1 - cfg["circle_circularity_min"]) )
        else:
            # czasem cienko narysowane koła mają wiele wierzchołków – wspomóżmy się score z circularity
            if circularity > 0.6:
                label, score = "circle", 0.4

    if label is None:
        return None
    return {
        "label": label,
        "score": float(score),
        "bbox": (int(x), int(y), int(w), int(h)),
        "area": float(area),
        "peri": float(peri),
        "vertices": int(len(approx)),
        "circularity": float(circularity),
    }

def detect_shapes(img_bgr, cfg):
    work, scale = resize_keep_aspect(img_bgr, cfg["resize_max_side"])
    gray = cv.cvtColor(work, cv.COLOR_BGR2GRAY)

    if cfg["deskew"]:
        gray, _, _ = deskew_by_hough(gray, cfg)

    bw = threshold_image(gray, cfg)

    # kontury
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        det = classify_contour(cnt, cfg)
        if det is None:
            continue
        # skala bboxów do oryginału
        x,y,w,h = det["bbox"]
        if scale != 1.0:
            x = int(x / scale); y = int(y / scale)
            w = int(w / scale); h = int(h / scale)
        det["bbox"] = (x,y,w,h)
        detections.append(det)

    # (opcjonalnie) Hough circles jako uzupełnienie
    if cfg["circle_hough_enable"]:
        blurred = cv.medianBlur(gray, 3)
        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp=1.2, minDist=18,
                                  param1=100, param2=18, minRadius=5, maxRadius=60)
        if circles is not None:
            for c in np.round(circles[0, :]).astype("int"):
                cx, cy, r = c
                x, y, w, h = cx - r, cy - r, 2*r, 2*r
                if scale != 1.0:
                    x = int(x / scale); y = int(y / scale)
                    w = int(w / scale); h = int(h / scale)
                detections.append({"label":"circle","score":0.7,"bbox":(x,y,w,h),
                                   "area":float(pi*r*r), "peri":float(2*pi*r),
                                   "vertices":0,"circularity":1.0})
    return detections, bw

# --- uruchomienie ---
img = cv.imread(path_img, cv.IMREAD_COLOR)
assert img is not None, f"Nie mogę wczytać {path_img}"

dets, bw = detect_shapes(img, CFG)

# rysowanie
COL = {"square":(0,255,0), "circle":(255,0,0), "triangle":(0,165,255)}
vis = img.copy()

for d in dets:
    x,y,w,h = d["bbox"]
    label = f'{d["label"]}:{d["score"]:.2f}'
    cv.rectangle(vis, (x,y), (x+w, y+h), COL.get(d["label"], (255,255,255)), CFG["draw_thickness"])
    # tło pod podpis
    (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), (0,0,0), -1)
    cv.putText(vis, label, (x+2, y-4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

# zapis
cv.imwrite(out_path, vis)
print(f"Detekcji: {len(dets)} | Zapisano: {out_path}")

# (opcjonalnie) podgląd binarki – tymczasowo zapisujemy
cv.imwrite(str(Path(out_dir) / ("bw_" + Path(path_img).stem + ".jpg")), bw)
