#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _library.random_search_sacks_points_v3 import detect_points
from pathlib import Path
from tqdm import tqdm

# Ścieżki bazowe
BASE_DIR = Path(__file__).parents[0]
INPUT_DIR = BASE_DIR / "_inputs"
OUTPUT_DIR = BASE_DIR / "_outputs/_spirale/step1_heurystyka"

params = {
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

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu wejściowego: {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lista plików tylko bezpośrednio w katalogu
    files = [f for f in INPUT_DIR.iterdir() if f.is_file()]

    if not files:
        print("Brak plików w katalogu _inputs")
        return

    for f in tqdm(files, desc="Przetwarzanie obrazów"):
        out_path = OUTPUT_DIR / f.name
        try:
            detect_points(str(f), str(out_path), params)
        except Exception as e:
            print(f"[BŁĄD] {f.name}: {e}")

if __name__ == "__main__":
    main()
