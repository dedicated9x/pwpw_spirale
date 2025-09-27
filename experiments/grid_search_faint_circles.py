# grid_search_faint_circles.py
from pathlib import Path
import numpy as np
import cv2 as cv
from tqdm import trange
import math

# ===================== CONFIG =====================
CFG = {
    # Folder wyjściowy
    "out_dir": "/home/admin2/Documents/repos/pwpw/outputs/random_search",

    # Rozmiar całego obrazu (do przeglądania)
    "canvas_w": 1800,
    "canvas_h": 1200,
    "bg_level": 240,            # jasność papieru 0..255

    # Ustawienia siatki K × (M x N)
    "K": 8,                     # ile obrazów (wartości 'ink_delta')
    "M": 6,                     # rzędy – stroke_px
    "N": 10,                    # kolumny – blur_sigma

    # Wartości paramów na osiach (równomierne)
    "stroke_px_range": (0.4, 2.6),   # [min,max] px po downsamplingu
    "blur_sigma_range": (0.0, 2.0),  # sigma Gaussa (po downsamplingu)
    "ink_delta_range": (6, 40),      # różnica jasności linii vs tło

    # Kółka
    "circle_diameter": 27,      # piksele w docelowej rozdzielczości
    "cell_size": 120,           # rozmiar komórki siatki (docelowy)
    "supersample": 4,           # render 4× większy, potem downsample
    "aa": True,                 # antyalias na konturze

    # Opisy/rysowanie
    "thickness_outline": 2,      # grubość ramek i osi
    "font_scale": 0.5,
    "font_thickness": 1,
    "jpeg_quality": 92,
    "seed": 1337
}
# ==================================================

def linspace_list(a, b, n):
    if n <= 1: return [float(a)]
    return [float(a + i*(b-a)/(n-1)) for i in range(n)]

def render_circle_patch(diameter, stroke_px, blur_sigma, ink_delta, bg_level, supersample=4, aa=True):
    """
    Renderuje pojedyncze kółko w skali szarości na tle 'bg_level'.
    Zwraca patch (H,W,3) BGR.
    """
    # rozmiary SS
    D = int(round(diameter * supersample))
    pad = int(round(max(6, 3*supersample)))   # margines aby blur się zmieścił
    S = D + 2*pad
    # płótno SS
    base = np.full((S, S), bg_level, dtype=np.uint8)

    # parametry linii
    r = D // 2
    cx = cy = S // 2
    # grubość w SS (umożliwia subpiksele)
    t_ss = max(1, int(round(stroke_px * supersample)))
    color = int(max(0, bg_level - ink_delta))

    line_type = cv.LINE_AA if aa else cv.LINE_8
    cv.circle(base, (cx, cy), r, color, t_ss, line_type)

    # downsample -> finalny patch
    patch = cv.resize(base, (S//supersample, S//supersample), interpolation=cv.INTER_AREA)

    # dodatkowy blur w skali finalnej
    if blur_sigma > 0.01:
        k = int(max(3, 2*int(3*blur_sigma)+1))
        patch = cv.GaussianBlur(patch, (k, k), blur_sigma)

    # na BGR
    return cv.cvtColor(patch, cv.COLOR_GRAY2BGR)

def put_text(img, text, org, scale, thickness=1, color=(0,0,0), bg=True):
    (tw, th), bl = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    if bg:
        cv.rectangle(img, (x-2, y-th-2), (x+tw+2, y+2), (255,255,255), -1)
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)

def draw_axes(canvas, xs, ys, cfg, label_top, label_left):
    h, w = canvas.shape[:2]
    margin_left = 130
    margin_top = 90
    cv.rectangle(canvas, (margin_left-10, margin_top-10), (w-20, h-20), (200,200,200), cfg["thickness_outline"])

    # opisy osi
    put_text(canvas, f"{label_top}", (margin_left, 30), cfg["font_scale"]+0.1, 1)
    put_text(canvas, f"{label_left}", (0, margin_top+20), cfg["font_scale"]+0.1, 1)

    # znaczniki X
    start_x = margin_left
    end_x = w - 20
    y_axis = margin_top - 20
    for j, val in enumerate(xs):
        x = int(start_x + j * (end_x - start_x) / max(1, len(xs)-1))
        put_text(canvas, f"{val:.2f}", (x-20, y_axis), cfg["font_scale"], 1)

    # znaczniki Y
    start_y = margin_top
    end_y = h - 20
    x_axis = margin_left - 120
    for i, val in enumerate(ys):
        y = int(start_y + i * (end_y - start_y) / max(1, len(ys)-1))
        put_text(canvas, f"{val:.2f}", (x_axis, y+5), cfg["font_scale"], 1)

def compose_canvas(cfg, stroke_vals, blur_vals, ink_delta):
    W = cfg["canvas_w"]; H = cfg["canvas_h"]
    canvas = np.full((H, W, 3), cfg["bg_level"], dtype=np.uint8)

    # gdzie rysować siatkę
    margin_left = 130
    margin_top = 90
    margin_right = 20
    margin_bottom = 20

    grid_w = W - margin_left - margin_right
    grid_h = H - margin_top - margin_bottom

    rows = len(stroke_vals)
    cols = len(blur_vals)

    # rozmiar komórki
    cell_w = int(grid_w / cols)
    cell_h = int(grid_h / rows)

    # rysowanie kółek
    for i, stroke_px in enumerate(stroke_vals):
        for j, blur_sigma in enumerate(blur_vals):
            cx = margin_left + j*cell_w + cell_w//2
            cy = margin_top  + i*cell_h + cell_h//2

            patch = render_circle_patch(
                diameter=CFG["circle_diameter"],
                stroke_px=stroke_px,
                blur_sigma=blur_sigma,
                ink_delta=ink_delta,
                bg_level=CFG["bg_level"],
                supersample=CFG["supersample"],
                aa=CFG["aa"]
            )

            ph, pw = patch.shape[:2]
            # wpasuj patch w komórkę (zostaw mały margines)
            scale = min((cell_w-8)/pw, (cell_h-8)/ph)
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            patch = cv.resize(patch, (new_w, new_h), interpolation=cv.INTER_AREA)

            x1 = int(cx - new_w//2)
            y1 = int(cy - new_h//2)
            canvas[y1:y1+new_h, x1:x1+new_w] = patch

    # osie i podpisy
    draw_axes(canvas, blur_vals, stroke_vals, cfg,
              label_top=f"blur_sigma  (ink_delta={ink_delta:.1f})",
              label_left="stroke_px")

    return canvas

def main():
    out_dir = Path(CFG["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    stroke_vals = linspace_list(*CFG["stroke_px_range"], CFG["M"])
    blur_vals   = linspace_list(*CFG["blur_sigma_range"], CFG["N"])
    ink_vals    = linspace_list(*CFG["ink_delta_range"], CFG["K"])

    for k in trange(CFG["K"], desc="Generating grids"):
        ink = ink_vals[k]
        canvas = compose_canvas(CFG, stroke_vals, blur_vals, ink)

        # Nagłówek globalny
        put_text(canvas,
                 f"Grid search – faint circles | diameter={CFG['circle_diameter']}px | K={k+1}/{CFG['K']}",
                 (10, CFG["canvas_h"]-5), CFG["font_scale"]+0.05, 1, (50,50,50), bg=False)

        out_path = out_dir / f"grid_circles_d{CFG['circle_diameter']}_ink{ink:.1f}_K{k+1:02d}.jpg"
        cv.imwrite(str(out_path), canvas, [int(cv.IMWRITE_JPEG_QUALITY), CFG["jpeg_quality"]])

    print(f"OK. Zapisano {CFG['K']} obrazów do: {out_dir}")

if __name__ == "__main__":
    main()
