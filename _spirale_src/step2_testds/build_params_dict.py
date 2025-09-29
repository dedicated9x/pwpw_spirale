#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_list_params.py
Czyta listę nazw overlayów z:
  /home/admin2/Documents/repos/pwpw/_spirale/_notes/drugi_rs.txt
i generuje plik:
  /home/admin2/Documents/repos/pwpw/_spirale_src/list_params.py

Wyjściowy list_params.py zawiera:
  PARAMS: dict[str, dict]  # mapa nazwa_pliku -> params_dict do detect_points(...)
"""

import re
from pathlib import Path
from typing import Dict, Any, List

SRC_LIST = Path("/_spirale/_notes/drugi_rs.txt")
OUT_PY   = Path("/_spirale_src/step2_testds/list_params.py")

TOKEN_PATTERNS = {
    "clahe_clip":       re.compile(r"^cl(?P<val>[0-9p]+)$"),
    "clahe_grid":       re.compile(r"^cg(?P<val>\d+)$"),
    "gauss_blur":       re.compile(r"^gb(?P<val>\d+)$"),
    "adapt_block":      re.compile(r"^ab(?P<val>\d+)$"),
    "adapt_C":          re.compile(r"^aC(?P<val>-?\d+)$"),
    "morph_open":       re.compile(r"^mo(?P<val>\d+)$"),
    "morph_close":      re.compile(r"^mc(?P<val>\d+)$"),
    "min_area":         re.compile(r"^minA(?P<val>\d+)$"),
    "max_area":         re.compile(r"^maxA(?P<val>\d+)$"),
    "min_circularity":  re.compile(r"^circ(?P<val>[0-9p]+)$"),
    "max_elongation":   re.compile(r"^elong(?P<val>[0-9p]+)$"),
    "dot_radius":       re.compile(r"^dr(?P<val>\d+)$"),
    "n_count":          re.compile(r"^n(?P<val>\d+)$"),  # ignorujemy
}

def _p_to_float(s: str) -> float:
    return float(s.replace('p', '.'))

def _init_params() -> Dict[str, Any]:
    return {
        "clahe_clip": None,
        "clahe_grid": None,
        "gauss_blur": None,
        "method": None,            # 'GAUSS' / 'MEAN'
        "thresh_type": "BIN_INV",  # domyślnie; poprawimy jeśli w nazwie jest inaczej
        "adapt_block": None,
        "adapt_C": None,
        "morph_open": None,
        "morph_close": None,
        "min_area": None,
        "max_area": None,
        "min_circularity": None,
        "max_elongation": None,
        "dot_radius": None,
    }

def parse_params_from_filename(fname: str) -> Dict[str, Any]:
    base = Path(fname).name
    if not base.startswith("overlay_"):
        raise ValueError(f"Nazwa nie wygląda na overlay_: {fname}")

    name_no_prefix = base[len("overlay_"):]
    name_no_ext = name_no_prefix[:-4] if name_no_prefix.endswith(".png") else name_no_prefix
    tokens = name_no_ext.split("_")

    params = _init_params()

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # metoda progowania
        if tok in ("GAUSS", "MEAN"):
            params["method"] = tok
            i += 1
            continue

        # przypadki progowania binarnego:
        #  - "BIN_INV" jako jeden token (rzadziej)
        #  - albo sekwencja "BIN" "_" "INV" jako dwa tokeny (częściej)
        if tok == "BIN_INV":
            params["thresh_type"] = "BIN_INV"
            i += 1
            continue
        if tok == "BIN":
            if i + 1 < len(tokens) and tokens[i + 1] == "INV":
                params["thresh_type"] = "BIN_INV"
                i += 2
                continue
            else:
                params["thresh_type"] = "BIN"
                i += 1
                continue
        if tok == "INV":
            # samotne "INV" — jeśli dotąd mieliśmy BIN, popraw na BIN_INV; w przeciwnym razie ignoruj
            if params.get("thresh_type") == "BIN":
                params["thresh_type"] = "BIN_INV"
            i += 1
            continue

        matched = False
        for key, pat in TOKEN_PATTERNS.items():
            m = pat.match(tok)
            if not m:
                continue
            val = m.group("val")
            matched = True

            if key in ("clahe_clip", "min_circularity", "max_elongation"):
                params[key] = _p_to_float(val)
            elif key in ("clahe_grid", "gauss_blur", "adapt_block", "morph_open",
                         "morph_close", "min_area", "max_area", "dot_radius"):
                params[key] = int(val)
            elif key == "adapt_C":
                params[key] = int(val)
            elif key == "n_count":
                pass  # ignorujemy
            break

        # niepasujące tokeny ignorujemy (np. artefakty)
        i += 1 if matched else 1

    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise ValueError(f"Brak parametrów w {fname}: {missing}")

    return params

def load_names(txt_path: Path) -> List[str]:
    names: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("//", 1)[0].strip()  # utnij komentarz po //
            if not line:
                continue
            names.append(Path(line).name)
    return names

def build_params_map(names: List[str]) -> Dict[str, Dict[str, Any]]:
    return {nm: parse_params_from_filename(nm) for nm in names}

def write_params_py(params_map: Dict[str, Dict[str, Any]], out_py: Path) -> None:
    out_py.parent.mkdir(parents=True, exist_ok=True)
    with open(out_py, "w", encoding="utf-8") as f:
        f.write("# AUTOGENERATED by build_list_params.py — nie edytuj ręcznie.\n")
        f.write("# Mapowanie: nazwa_pliku_overlay -> params_dict do detect_points(...)\n\n")
        f.write("PARAMS = {\n")
        for k in sorted(params_map.keys()):
            v = params_map[k]
            f.write(f"    {repr(k)}: {{\n")
            f.write(f"        'clahe_clip': {v['clahe_clip']},\n")
            f.write(f"        'clahe_grid': {v['clahe_grid']},\n")
            f.write(f"        'gauss_blur': {v['gauss_blur']},\n")
            f.write(f"        'method': {repr(v['method'])},\n")
            f.write(f"        'thresh_type': {repr(v['thresh_type'])},\n")
            f.write(f"        'adapt_block': {v['adapt_block']},\n")
            f.write(f"        'adapt_C': {v['adapt_C']},\n")
            f.write(f"        'morph_open': {v['morph_open']},\n")
            f.write(f"        'morph_close': {v['morph_close']},\n")
            f.write(f"        'min_area': {v['min_area']},\n")
            f.write(f"        'max_area': {v['max_area']},\n")
            f.write(f"        'min_circularity': {v['min_circularity']},\n")
            f.write(f"        'max_elongation': {v['max_elongation']},\n")
            f.write(f"        'dot_radius': {v['dot_radius']},\n")
            f.write("    },\n")
        f.write("}\n\n")
        f.write("__all__ = ['PARAMS']\n")

def main():
    names = load_names(SRC_LIST)
    if not names:
        raise SystemExit(f"Brak nazw w {SRC_LIST}")
    params_map = build_params_map(names)
    write_params_py(params_map, OUT_PY)
    print(f"[OK] Wygenerowano: {OUT_PY}  (pozycje: {len(params_map)})")

if __name__ == "__main__":
    main()
