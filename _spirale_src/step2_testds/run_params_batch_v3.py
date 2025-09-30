#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _spirale_src.step1_rs.random_search_sacks_points_v3 import detect_points
from _spirale_src.step2_testds.list_params import PARAMS

import os
from pathlib import Path
from tqdm import tqdm

# Stałe ścieżki z zadania
# INPUT_IMG = Path("/home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317.jpg")
INPUT_IMG = Path("/home/admin2/Documents/repos/pwpw/_spirale/inputs/reprezentanci/PXL_20250925_064336700.jpg")
OUT_DIR   = Path("/home/admin2/Documents/repos/pwpw/_spirale/outputs/po_reprezentantach_v3_v2/po_mainie")

def main():
    if not INPUT_IMG.exists():
        raise FileNotFoundError(f"Brak obrazu wejściowego: {INPUT_IMG}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # deterministyczna kolejność
    items = sorted(PARAMS.items(), key=lambda kv: kv[0])

    for fname, params in tqdm(items, desc="Batch detect", ncols=100):
        target_path = OUT_DIR / fname
        tmp_path = target_path.with_suffix(".tmp.png")

        # uruchom detekcję -> zapis do pliku tymczasowego
        res = detect_points(str(INPUT_IMG), str(tmp_path), params)
        # przenieś atomowo na docelową nazwę
        try:
            os.replace(tmp_path, target_path)
        except FileNotFoundError:
            # jeśli z jakiegoś powodu plik tmp nie powstał, kontynuuj
            continue

        # (opcjonalnie) możesz sobie odkomentować log:
        # print(f"{fname}: n={res['count']}")

    print(f"[OK] Zapisano wyniki do: {OUT_DIR}")

if __name__ == "__main__":
    main()
