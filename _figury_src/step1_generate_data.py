import os
import argparse
from pathlib import Path
import math
import random
import numpy as np
import cv2 as cv
from tqdm import trange

# ===================== CONFIG =====================
CFG = {
    # --- rozmiar i liczba obrazów ---
    "img_width": 3072,
    "img_height": 4080,
    "n_images": 500,                 # ile wygenerować

    # --- rozkład obiektów na obraz ---
    "min_shapes_per_img": 30,
    "max_shapes_per_img": 120,
    "shape_classes": ["square", "circle", "triangle"],   # mapowane na YOLO: 0,1,2
    "class_probs": [1/3, 1/3, 1/3],

    # --- rozmiary figur (w pikselach) ---
    # "min_size": 36,                  # ~średnica koła / bok kwadratu / najdłuższy bok trójkąta
    "min_size": 27,
    "max_size": 120,

    # --- styl rysunku ---
    "line_thickness_range": [2, 5],  # grubość konturu
    "anti_alias": True,
    "fill_prob": 0.0,                # 0 = puste środki (jak w danych), >0 = czasem wypełnione

    # --- kolizje/nałożenia ---
    "allow_overlap": True,
    "max_try_place": 80,             # próby umieszczenia jednej figury
    "min_iou_between_shapes": 0.0,   # jeśli allow_overlap=False, wymagany odstęp

    # --- globalne „foto” augmentacje (stosowane na całym obrazie) ---
    "global_rotation_deg": [-2.0, 2.0],       # delikatny przekręt
    "global_shear_x": [-0.02, 0.02],          # ścinanie
    "global_shear_y": [-0.02, 0.02],
    "global_translate_frac": [-0.01, 0.01],   # przesunięcie względem rozmiaru
    "blur_sigma": [0.0, 1.2],                 # Gauss sigma (0 = brak)
    "noise_std": [0.0, 8.0],                  # szum Gaussa w [0..255]
    "paper_gradient_strength": [0.0, 35.0],   # maks. różnica jasności tła
    "vignette_strength": [0.0, 0.25],         # 0..1


    # --- I/O ---
    "out_root": Path(__file__).parents[1] / "_outputs/_figury/step1_generate_data",
    "img_format": "jpg",
    "jpg_quality": 92,

    # --- losowość/reprodukowalność ---
    "seed": 1337,
}
# ===================================================

# mapowanie klas -> id
CLASS2ID = {c:i for i, c in enumerate(CFG["shape_classes"])}

# --- utilsy ---
rng = np.random.default_rng(CFG["seed"])
random.seed(CFG["seed"])

def aa(thickness):
    return cv.LINE_AA if CFG["anti_alias"] else cv.LINE_8

def rand_uniform(a, b):
    return float(rng.uniform(a, b))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def make_background(h, w):
    # gładkie tło + delikatny gradient papieru + winieta
    base = np.full((h, w), 240, dtype=np.uint8)

    # gradient
    g_strength = rand_uniform(*CFG["paper_gradient_strength"])
    if g_strength > 0:
        gx = np.linspace(0, 1, w, dtype=np.float32)
        gy = np.linspace(0, 1, h, dtype=np.float32)
        gxx, gyy = np.meshgrid(gx, gy)
        angle = rand_uniform(0, 2*math.pi)
        grad = (np.cos(angle)*gxx + np.sin(angle)*gyy)
        grad = (grad - grad.min())/(grad.ptp()+1e-6)
        grad = (grad * g_strength).astype(np.uint8)
        base = cv.subtract(base, grad)

    # winieta
    vstr = rand_uniform(*CFG["vignette_strength"])
    if vstr > 0:
        xs = np.linspace(-1, 1, w)
        ys = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv**2 + yv**2)
        vmask = (1 - vstr*(r**2))
        vmask = np.clip(vmask, 0.0, 1.0)
        base = (base.astype(np.float32) * vmask).astype(np.uint8)

    bg = cv.cvtColor(base, cv.COLOR_GRAY2BGR)
    return bg

def iou(a, b):
    # a,b: (x1,y1,x2,y2)
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter/union

def draw_square(img, cx, cy, size, angle_deg, color, thickness, fill=False):
    half = size/2.0
    pts = np.array([[-half,-half],[half,-half],[half,half],[-half,half]], dtype=np.float32)
    ang = np.deg2rad(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang),  np.cos(ang)]], dtype=np.float32)
    pts = (pts @ R.T) + np.array([cx, cy], dtype=np.float32)
    pts_int = pts.astype(np.int32)

    if fill:
        cv.fillPoly(img, [pts_int], color)
        if thickness > 0:
            cv.polylines(img, [pts_int], True, color, thickness, aa(thickness))
    else:
        cv.polylines(img, [pts_int], True, color, thickness, aa(thickness))
    return pts  # zwracamy wierzchołki

def draw_triangle(img, cx, cy, size, angle_deg, color, thickness, fill=False):
    r = size/2.0
    pts = np.array([[0, -r],
                    [r*np.cos(np.deg2rad(210)), -r*np.sin(np.deg2rad(210))],
                    [r*np.cos(np.deg2rad(330)), -r*np.sin(np.deg2rad(330))]], dtype=np.float32)
    ang = np.deg2rad(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang),  np.cos(ang)]], dtype=np.float32)
    pts = (pts @ R.T) + np.array([cx, cy], dtype=np.float32)
    pts_int = pts.astype(np.int32)

    if fill:
        cv.fillPoly(img, [pts_int], color)
        if thickness > 0:
            cv.polylines(img, [pts_int], True, color, thickness, aa(thickness))
    else:
        cv.polylines(img, [pts_int], True, color, thickness, aa(thickness))
    return pts

def draw_circle(img, cx, cy, size, angle_deg_unused, color, thickness, fill=False):
    r = int(size/2.0)
    if fill:
        cv.circle(img, (int(cx), int(cy)), r, color, -1, aa(thickness))
        if thickness > 0:
            cv.circle(img, (int(cx), int(cy)), r, color, thickness, aa(thickness))
    else:
        cv.circle(img, (int(cx), int(cy)), r, color, thickness, aa(thickness))
    # punkty „ramujące” do bboxa
    thetas = np.linspace(0, 2*math.pi, 12)
    pts = np.stack([cx + r*np.cos(thetas), cy + r*np.sin(thetas)], axis=1).astype(np.float32)
    return pts

def polygon_to_aabb(pts, W, H):
    x1 = clamp(float(pts[:,0].min()), 0, W-1)
    y1 = clamp(float(pts[:,1].min()), 0, H-1)
    x2 = clamp(float(pts[:,0].max()), 0, W-1)
    y2 = clamp(float(pts[:,1].max()), 0, H-1)
    return (x1,y1,x2,y2)

def aabb_to_yolo(aabb, W, H, cls_id):
    x1,y1,x2,y2 = aabb
    cx = (x1 + x2)/2.0
    cy = (y1 + y2)/2.0
    w  = (x2 - x1)
    h  = (y2 - y1)
    if w <= 1 or h <= 1:
        return None
    # normalizacja
    return (cls_id, cx/W, cy/H, w/W, h/H)

def global_affine(img, pts_list, W, H):
    # budujemy macierz 2x3: rot + shear + trans
    ang = rand_uniform(*CFG["global_rotation_deg"])
    shx = rand_uniform(*CFG["global_shear_x"])
    shy = rand_uniform(*CFG["global_shear_y"])
    tx  = rand_uniform(*CFG["global_translate_frac"]) * W
    ty  = rand_uniform(*CFG["global_translate_frac"]) * H

    a = math.radians(ang)
    R = np.array([[math.cos(a), -math.sin(a)],
                  [math.sin(a),  math.cos(a)]], dtype=np.float32)
    S = np.array([[1.0, shx],
                  [shy, 1.0]], dtype=np.float32)
    A2 = R @ S
    M = np.hstack([A2, np.array([[tx],[ty]], dtype=np.float32)])

    warped = cv.warpAffine(img, M, (W, H), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    # transformujemy punkty
    new_pts_list = []
    for pts in pts_list:
        if pts is None:
            new_pts_list.append(None); continue
        ones = np.ones((pts.shape[0],1), dtype=np.float32)
        homo = np.hstack([pts, ones])
        tpts = (homo @ M.T)
        new_pts_list.append(tpts.astype(np.float32))
    return warped, new_pts_list

def add_noise_and_blur(img):
    # blur
    sigma = rand_uniform(*CFG["blur_sigma"])
    if sigma > 0.05:
        k = int(max(3, 2*int(3*sigma)+1))
        img = cv.GaussianBlur(img, (k,k), sigma)

    # noise
    std = rand_uniform(*CFG["noise_std"])
    if std > 0.5:
        noise = rng.normal(0, std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img

def ensure_dirs(root):
    root = Path(root)
    (root/"dataset"/"images").mkdir(parents=True, exist_ok=True)
    (root/"dataset"/"labels").mkdir(parents=True, exist_ok=True)
    (root/"previews").mkdir(parents=True, exist_ok=True)
    return root

# --- główna pętla ---
def main():
    W, H = CFG["img_width"], CFG["img_height"]
    out_root = ensure_dirs(CFG["out_root"])
    img_dir  = out_root/"dataset"/"images"
    lbl_dir  = out_root/"dataset"/"labels"
    prev_dir = out_root/"previews"

    for i in trange(CFG["n_images"], desc="Synth"):
        img = make_background(H, W)
        drawn_aabbs = []  # do kontroli IOU, jeśli potrzeba
        yolo_lines = []
        debug_polys = []  # surowe wielokąty przed transformatą

        n_shapes = rng.integers(CFG["min_shapes_per_img"], CFG["max_shapes_per_img"]+1)
        for _ in range(int(n_shapes)):
            # losuj klasę
            cls = random.choices(CFG["shape_classes"], weights=CFG["class_probs"], k=1)[0]
            cls_id = CLASS2ID[cls]

            size = rand_uniform(CFG["min_size"], CFG["max_size"])
            thick = int(rand_uniform(*CFG["line_thickness_range"]))
            fill = (random.random() < CFG["fill_prob"])

            # pozycja – próbujemy położyć tak, by figura w całości mieściła się w kadrze
            placed = False
            for _try in range(CFG["max_try_place"]):
                cx = rand_uniform(size, W - size)
                cy = rand_uniform(size, H - size)
                ang = rand_uniform(0, 360)

                color = (0,0,0)  # czarny „druk”
                if cls == "square":
                    pts = draw_square(img, cx, cy, size, ang, color, thick, fill)
                elif cls == "triangle":
                    pts = draw_triangle(img, cx, cy, size, ang, color, thick, fill)
                else:
                    pts = draw_circle(img, cx, cy, size, ang, color, thick, fill)

                aabb = polygon_to_aabb(pts, W, H)

                # jeśli nie pozwalamy na overlap – sprawdź IoU
                ok = True
                if not CFG["allow_overlap"]:
                    for prev in drawn_aabbs:
                        if iou(aabb, prev) > CFG["min_iou_between_shapes"]:
                            ok = False
                            break
                if ok:
                    drawn_aabbs.append(aabb)
                    debug_polys.append((pts, cls_id))
                    placed = True
                    break
                else:
                    # wycofaj rysowanie nieudanej próby przez nadpisanie danym tłem (kosztowne),
                    # ale łatwiej: rysuj na osobnej warstwie i dopiero scalaj – tu keep it simple.
                    pass
            # jeśli nie udało się położyć – pomijamy figurę

        # globalna transformacja + efekt foto
        img, tpolys = global_affine(img, [p for p,_ in debug_polys], W, H)

        # z tpolys liczymy aabbs i YOLO linię
        for (tpts, cls_id) in zip(tpolys, [c for _,c in debug_polys]):
            if tpts is None:
                continue
            aabb = polygon_to_aabb(tpts, W, H)
            y = aabb_to_yolo(aabb, W, H, cls_id)
            if y is not None:
                yolo_lines.append(f"{y[0]} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f} {y[4]:.6f}")

        img = add_noise_and_blur(img)

        # zapis obrazu + etykiet
        stem = f"synth_{i:06d}"
        img_path = img_dir / f"{stem}.{CFG['img_format']}"
        lbl_path = lbl_dir / f"{stem}.txt"

        if CFG["img_format"].lower() == "jpg":
            cv.imwrite(str(img_path), img, [int(cv.IMWRITE_JPEG_QUALITY), CFG["jpg_quality"]])
        elif CFG["img_format"].lower() in ["png","bmp","tif","tiff"]:
            cv.imwrite(str(img_path), img)
        else:
            raise ValueError("Nieobsługiwany format obrazu")

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # zrobię też podgląd z bboxami
        vis = img.copy()
        for line in yolo_lines:
            cls_id, xc, yc, ww, hh = line.split()
            cls_id = int(cls_id)
            xc = float(xc)*W; yc = float(yc)*H
            ww = float(ww)*W; hh = float(hh)*H
            x1 = int(xc - ww/2); y1 = int(yc - hh/2)
            x2 = int(xc + ww/2); y2 = int(yc + hh/2)
            color = [(0,255,0),(255,0,0),(0,165,255)][cls_id]
            cv.rectangle(vis, (x1,y1), (x2,y2), color, 2, cv.LINE_AA)
            name = CFG["shape_classes"][cls_id]
            cv.putText(vis, name, (x1, max(10, y1-6)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
        prev_path = prev_dir / f"{stem}.{CFG['img_format']}"
        if CFG["img_format"].lower() == "jpg":
            cv.imwrite(str(prev_path), vis, [int(cv.IMWRITE_JPEG_QUALITY), CFG["jpg_quality"]])
        else:
            cv.imwrite(str(prev_path), vis)

    print("Done. Images:", CFG["n_images"])

def parse_args():
    p = argparse.ArgumentParser(description="Generator syntetycznych figur z etykietami YOLO.")
    p.add_argument(
        "-o", "--out-root", "--out_root",
        dest="out_root",
        default=CFG["out_root"],
        help="Katalog wyjściowy (domyślnie: %(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG["out_root"] = args.out_root
    main()

