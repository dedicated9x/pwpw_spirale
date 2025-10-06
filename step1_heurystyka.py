#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _library.random_search_sacks_points_v3 import detect_points
from pathlib import Path
import argparse

# Domyślne ścieżki (używane jako default w argparse)
DEFAULT_INPUT_IMG = Path(__file__).parents[0] / "_inputs/PXL_20250925_061456317.jpg"
DEFAULT_OUTPUT_IMG = Path(__file__).parents[0] / "_outputs/_spirale/step1_heurystyka/PXL_20250925_061456317.jpg"

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

def parse_args():
    p = argparse.ArgumentParser(
        description="Wykrywa punkty i zapisuje obraz z kropkami."
    )
    p.add_argument(
        "-i", "--input",
        type=Path,
        default=DEFAULT_INPUT_IMG,
        help=f"Ścieżka do obrazu wejściowego (domyślnie: {DEFAULT_INPUT_IMG})"
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT_IMG,
        help=f"Ścieżka do obrazu wyjściowego (domyślnie: {DEFAULT_OUTPUT_IMG})"
    )
    return p.parse_args()

def main():
    args = parse_args()

    input_img: Path = args.input
    output_img: Path = args.output

    if not input_img.exists():
        raise FileNotFoundError(f"Brak obrazu wejściowego: {input_img}")

    # Upewnij się, że katalog wyjścia istnieje
    output_img.parent.mkdir(parents=True, exist_ok=True)

    _ = detect_points(str(input_img), str(output_img), params)

if __name__ == "__main__":
    main()

"""
.../pwpw$ python -m _spirale_src.step1_heurystyka
"""