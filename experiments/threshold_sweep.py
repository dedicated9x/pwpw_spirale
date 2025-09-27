# threshold_sweep.py
from pathlib import Path
import cv2 as cv
import numpy as np

# ---------------- CFG ----------------
CFG = {
    "img_path": "/home/admin2/Downloads/Shape detection/ShapeDetector raw/PXL_20250925_050125594.jpg",
    "save_dir": "/home/admin2/Documents/repos/pwpw/outputs/thresholded",

    # progi do sprawdzenia (0..255). Możesz podać listę lub wygenerować range.
    "thresholds": list(range(60, 201, 10)),  # 60,70,...,200

    # czy odwrócić (na Twoich danych kontury są ciemniejsze od tła — zwykle tak)
    "invert": False,

    # opcjonalne: podgląd Otsu jako punkt odniesienia (zapisze osobny plik)
    "save_otsu": True,
}
# -------------------------------------

def main():
    p = Path(CFG["img_path"])
    out_dir = Path(CFG["save_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    # wczytanie i szarość
    img = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
    assert img is not None, f"Nie mogę wczytać: {p}"

    # Otsu (opcjonalnie)
    if CFG["save_otsu"]:
        _, bw_otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if CFG["invert"]:
            bw_otsu = cv.bitwise_not(bw_otsu)
        cv.imwrite(str(out_dir / f"{p.stem}_otsu.jpg"), bw_otsu)

    # sweep po zadanych progach
    for T in CFG["thresholds"]:
        _, bw = cv.threshold(img, T, 255, cv.THRESH_BINARY)
        if CFG["invert"]:
            bw = cv.bitwise_not(bw)
        out_path = out_dir / f"{p.stem}_T{T:03d}.jpg"
        cv.imwrite(str(out_path), bw)

    print(f"OK. Zapisałem {len(CFG['thresholds']) + int(CFG['save_otsu'])} plików do: {out_dir}")

if __name__ == "__main__":
    main()
