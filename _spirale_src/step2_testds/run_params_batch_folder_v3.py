#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _spirale_src.step1_rs.random_search_sacks_points_v3 import detect_points
from _spirale_src.step2_testds.list_params import PARAMS

import os
from pathlib import Path
from tqdm import tqdm

# Wejście/wyjście
INPUT_DIR = Path("/home/admin2/Documents/repos/pwpw/_spirale/inputs/reprezentanci")
OUT_BASE  = Path("/home/admin2/Documents/repos/pwpw/_spirale/outputs/po_reprezentantach_v3_v2")

# Dozwolone rozszerzenia obrazów
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Brak folderu wejściowego: {INPUT_DIR}")
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(INPUT_DIR))
    if not images:
        raise SystemExit(f"Brak plików graficznych w: {INPUT_DIR}")

    # Deterministycznie po kluczach (nazwach overlayów)
    param_items = sorted(PARAMS.items(), key=lambda kv: kv[0])

    pbar_params = tqdm(param_items, desc="Param sets", ncols=100)
    for overlay_name, params in pbar_params:
        # Nazwa podfolderu = nazwa overlay'a bez .png
        subdir_name = Path(overlay_name).with_suffix("").name
        out_dir = OUT_BASE / subdir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Iter po obrazach wejściowych
        for img_path in tqdm(images, desc=f"{subdir_name}", leave=False, ncols=100):
            # Nazwa wyniku = nazwa wejścia (bez zmian)
            out_path = out_dir / (img_path.stem + ".png")
            tmp_path = out_path.with_suffix(".tmp.png")

            # Detekcja i zapis (atomowy rename)
            try:
                detect_points(str(img_path), str(tmp_path), params)
                os.replace(tmp_path, out_path)
            except Exception as e:
                # Czytelny log, ale nie przerywamy całej partii
                tqdm.write(f"[WARN] {img_path.name} @ {subdir_name}: {e}")
                # Sprzątanie pliku tymczasowego, jeśli został
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

    print(f"[OK] Zapisano wyniki do: {OUT_BASE}")
    print("Struktura: jeden folder na jeden params dict; wewnątrz – overlaye dla wszystkich zdjęć.")

if __name__ == "__main__":
    main()
