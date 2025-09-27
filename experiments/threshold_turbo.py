# threshold_turbo.py
from pathlib import Path
import cv2 as cv
import numpy as np
from itertools import product

# --------- CFG ----------
CFG = {
    "img_path": "/home/admin2/Downloads/Shape detection/ShapeDetector raw/PXL_20250925_050125594.jpg",
    "save_dir": "/home/admin2/Documents/repos/pwpw/outputs/thresholded_turbo",

    # Twój stały próg (nie ruszamy)
    "threshold": 200,
    "invert": False,

    # Minimalny zestaw wariantów do przeglądu:
    # promień elementu strukturalnego (px): 0 oznacza "pomijam"
    "close_radii": [1, 2],             # domknięcie przerw
    "open_radii": [0, 1],              # lekkie wygładzenie ząbków
    "min_area_list": [40, 80],         # filtr CC (px)
    # tryby wygładzania konturu
    #   none       – brak dodatkowej operacji
    #   grad1/2    – morfologiczny gradient (cienki obrys)
    #   solid      – wypełnianie dziur (pełne kształty)
    "boundary_modes": ["none", "grad1", "grad2", "solid"],

    # ewentualny wstępny median (0 = brak; nie zmienia progu, tylko ucisza sól/pieprz)
    "median_ksize": 0,                 # 0,3,5...
}
# ------------------------

def se(radius):
    if radius <= 0:
        return None
    k = 2 * radius + 1
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

def area_filter(bw, min_area):
    # 0/255 -> 0/1
    num, labels, stats, _ = cv.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(bw, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep

def fill_holes(bw):
    # wypełnij „dziury” wewnątrz białych obiektów
    h, w = bw.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    inv = 255 - bw
    flood = inv.copy()
    cv.floodFill(flood, mask, (0, 0), 255)
    holes = 255 - flood
    return cv.bitwise_or(bw, holes)

def morph_gradient(bw, rad):
    if rad <= 0:
        return bw
    k = se(rad)
    dil = cv.dilate(bw, k, iterations=1)
    ero = cv.erode(bw, k, iterations=1)
    grad = cv.subtract(dil, ero)
    # sprowadź do czystej binarki (0/255)
    grad[grad > 0] = 255
    return grad

def main():
    p = Path(CFG["img_path"])
    out_dir = Path(CFG["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) wczytanie
    gray = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
    assert gray is not None, f"Nie mogę wczytać {p}"

    # (opcjonalnie) median
    if CFG["median_ksize"] and CFG["median_ksize"] >= 3:
        k = CFG["median_ksize"] | 1
        gray = cv.medianBlur(gray, k)

    # 2) próg – NA SZTYWNO wg Twoich ustaleń
    T = CFG["threshold"]
    _, bw = cv.threshold(gray, T, 255, cv.THRESH_BINARY)  # prosto
    if CFG["invert"]:
        bw = cv.bitwise_not(bw)

    # zapis „gołego” binara (referencja)
    cv.imwrite(str(out_dir / f"{p.stem}_T{T}_inv{int(CFG['invert'])}_rawbin.png"), bw)

    # 3) sweep parametrów morfologii
    for cr, or_, ma, mode in product(CFG["close_radii"], CFG["open_radii"], CFG["min_area_list"], CFG["boundary_modes"]):
        img = bw.copy()

        # domknięcie przerw
        if cr > 0:
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, se(cr), iterations=1)

        # odrzut drobnicy
        if ma and ma > 0:
            img = area_filter(img, ma)

        # lekkie wygładzenie ząbków
        if or_ > 0:
            img = cv.morphologyEx(img, cv.MORPH_OPEN, se(or_), iterations=1)

        # tryb wykończenia
        if mode == "none":
            out = img
        elif mode == "solid":
            out = fill_holes(img)
        elif mode == "grad1":
            out = morph_gradient(img, 1)
        elif mode == "grad2":
            out = morph_gradient(img, 2)
        else:
            out = img

        fname = f"{p.stem}_T{T}_inv{int(CFG['invert'])}_cl{cr}_op{or_}_area{ma}_{mode}.png"
        cv.imwrite(str(out_dir / fname), out)

    print(f"OK. Wyniki w: {out_dir}")

if __name__ == "__main__":
    main()
