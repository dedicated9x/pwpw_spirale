# synth_shapes_v2_faint.py
from pathlib import Path
import math, random
import numpy as np
import cv2 as cv
from tqdm import trange

# ===================== CONFIG =====================
CFG = {
    # --- rozmiar i liczba obrazów ---
    "img_width": 3072,
    "img_height": 4080,
    "n_images": 500,

    # --- rozkład obiektów ---
    "min_shapes_per_img": 30,
    "max_shapes_per_img": 120,
    "shape_classes": ["square", "circle", "triangle"],
    "class_probs": [1/3, 1/3, 1/3],

    # --- rozmiary figur (docelowe) ---
    "min_size": 24,
    "max_size": 120,

    # --- logika konturu: tryb normal vs faint (blady) ---
    "faint_prob": 0.25,        # jaki odsetek figur „popsuć”
    # Twój wniosek = skrajności dla faint:
    "faint_stroke_min": 1.72,  # px (po downsample)
    "faint_stroke_max": 2.4,
    "faint_blur_min": 0.0,
    "faint_blur_max": 1.33,    # <= 1.33
    "faint_ink_delta_min": 40, # >= 40 (tło - delta = kolor linii)
    "faint_ink_delta_max": 70,

    # normal (większość ostrych)
    "normal_stroke_min": 2.2,
    "normal_stroke_max": 3.6,
    "normal_blur_min": 0.0,
    "normal_blur_max": 0.25,
    "normal_ink_delta_min": 65,
    "normal_ink_delta_max": 110,

    # --- supersampling aby uzyskać subpikselowy kontur ---
    "supersample": 4,
    "anti_alias": True,

    # --- globalny „papier” ---
    "bg_level": 240,
    "paper_gradient_strength": [5.0, 25.0],
    "vignette_strength": [0.0, 0.18],
    "blur_sigma_global": [0.0, 0.8],
    "noise_std": [0.0, 6.0],
    "jpeg_quality": 92,

    # --- I/O ---
    "out_root": "/home/admin2/Documents/repos/pwpw/inputs",
    "img_format": "jpg",

    # reproducibility
    "seed": 1337,
}
# ==================================================

CLASS2ID = {c:i for i, c in enumerate(CFG["shape_classes"])}
rng = np.random.default_rng(CFG["seed"])
random.seed(CFG["seed"])

def aa(thk): return cv.LINE_AA if CFG["anti_alias"] else cv.LINE_8
def rand(a,b): return float(rng.uniform(a,b))
def clamp(v, lo, hi): return max(lo, min(hi, v))

def make_background(h, w):
    base = np.full((h, w), CFG["bg_level"], dtype=np.uint8)
    # gradient
    g_strength = rand(*CFG["paper_gradient_strength"])
    gx = np.linspace(0,1,w,dtype=np.float32)
    gy = np.linspace(0,1,h,dtype=np.float32)
    gxx,gyy = np.meshgrid(gx,gy)
    ang = rand(0, 2*math.pi)
    grad = (np.cos(ang)*gxx + np.sin(ang)*gyy)
    grad = (grad - grad.min())/(grad.ptp()+1e-6)
    grad = (grad * g_strength).astype(np.uint8)
    base = cv.subtract(base, grad)
    # winieta
    vstr = rand(*CFG["vignette_strength"])
    if vstr>0:
        xs = np.linspace(-1,1,w); ys = np.linspace(-1,1,h)
        xv,yv = np.meshgrid(xs,ys)
        r = np.sqrt(xv**2 + yv**2)
        vmask = np.clip(1 - vstr*(r**2), 0, 1)
        base = (base.astype(np.float32)*vmask).astype(np.uint8)
    return cv.cvtColor(base, cv.COLOR_GRAY2BGR)

def draw_square(img, cx, cy, size, ang, color, thickness, fill=False):
    half = size/2.0
    pts = np.array([[-half,-half],[half,-half],[half,half],[-half,half]], np.float32)
    a = np.deg2rad(ang); R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]], np.float32)
    pts = (pts @ R.T) + np.array([cx,cy], np.float32)
    p = pts.astype(np.int32)
    if fill: cv.fillPoly(img,[p],color)
    cv.polylines(img,[p],True,color,thickness,aa(thickness)); return pts
def draw_triangle(img, cx, cy, size, ang, color, thickness, fill=False):
    r = size/2.0
    pts = np.array([[0,-r],[r*np.cos(np.deg2rad(210)),-r*np.sin(np.deg2rad(210))],[r*np.cos(np.deg2rad(330)),-r*np.sin(np.deg2rad(330))]], np.float32)
    a = np.deg2rad(ang); R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]], np.float32)
    pts = (pts @ R.T) + np.array([cx,cy], np.float32)
    p = pts.astype(np.int32)
    if fill: cv.fillPoly(img,[p],color)
    cv.polylines(img,[p],True,color,thickness,aa(thickness)); return pts
def draw_circle(img, cx, cy, size, ang_unused, color, thickness, fill=False):
    r = int(size/2.0)
    if fill: cv.circle(img,(int(cx),int(cy)),r,color,-1,aa(thickness))
    cv.circle(img,(int(cx),int(cy)),r,color,thickness,aa(thickness))
    thetas = np.linspace(0,2*math.pi,12)
    return np.stack([cx + r*np.cos(thetas), cy + r*np.sin(thetas)],1).astype(np.float32)

def polygon_to_aabb(pts, W, H):
    x1 = clamp(float(pts[:,0].min()), 0, W-1); y1 = clamp(float(pts[:,1].min()), 0, H-1)
    x2 = clamp(float(pts[:,0].max()), 0, W-1); y2 = clamp(float(pts[:,1].max()), 0, H-1)
    return (x1,y1,x2,y2)
def aabb_to_yolo(aabb, W, H, cls_id):
    x1,y1,x2,y2 = aabb; cx=(x1+x2)/2; cy=(y1+y2)/2; w=(x2-x1); h=(y2-y1)
    if w<=1 or h<=1: return None
    return (cls_id, cx/W, cy/H, w/W, h/H)

def render_shape_ss(img, cx, cy, size, ang, is_faint):
    """
    Rysuje figurę w trybie supersampling -> downsample z kontrolą:
      stroke_px, blur_sigma, ink_delta
    """
    # wybór parametrów zgodnie z trybem
    if is_faint:
        stroke_px = rand(CFG["faint_stroke_min"], CFG["faint_stroke_max"])
        blur_sigma = rand(CFG["faint_blur_min"], CFG["faint_blur_max"])
        ink_delta = rand(CFG["faint_ink_delta_min"], CFG["faint_ink_delta_max"])
    else:
        stroke_px = rand(CFG["normal_stroke_min"], CFG["normal_stroke_max"])
        blur_sigma = rand(CFG["normal_blur_min"], CFG["normal_blur_max"])
        ink_delta = rand(CFG["normal_ink_delta_min"], CFG["normal_ink_delta_max"])

    # supersampling canvas
    ss = CFG["supersample"]
    W,H = img.shape[1], img.shape[0]
    S = np.full((H*ss, W*ss), CFG["bg_level"], np.uint8)

    # „docelowe” parametry w SS
    cx_ss, cy_ss = int(cx*ss), int(cy*ss)
    size_ss = int(size*ss)
    thk_ss = max(1, int(round(stroke_px*ss)))
    color = int(max(0, CFG["bg_level"] - ink_delta))

    return S, cx_ss, cy_ss, size_ss, thk_ss, color, blur_sigma

def paste_ss_back(base_bgr, ss_gray):
    ss = CFG["supersample"]
    down = cv.resize(ss_gray, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv.INTER_AREA)
    base = cv.cvtColor(down, cv.COLOR_GRAY2BGR)
    np.copyto(base_bgr, base)

def add_global_noise_and_blur(img):
    sigma = rand(*CFG["blur_sigma_global"])
    if sigma>0.05:
        k = int(max(3, 2*int(3*sigma)+1))
        img[:] = cv.GaussianBlur(img, (k,k), sigma)
    std = rand(*CFG["noise_std"])
    if std>0.5:
        noise = rng.normal(0, std, img.shape).astype(np.float32)
        img[:] = np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)

def ensure_dirs(root):
    root = Path(root)
    (root/"dataset"/"images").mkdir(parents=True, exist_ok=True)
    (root/"dataset"/"labels").mkdir(parents=True, exist_ok=True)
    (root/"previews").mkdir(parents=True, exist_ok=True)
    return root

def main():
    out_root = ensure_dirs(CFG["out_root"])
    img_dir, lbl_dir, prev_dir = out_root/"dataset"/"images", out_root/"dataset"/"labels", out_root/"previews"

    W, H = CFG["img_width"], CFG["img_height"]

    for i in trange(CFG["n_images"], desc="Synth v2"):
        canvas = make_background(H, W)
        yolo_lines = []

        n_shapes = rng.integers(CFG["min_shapes_per_img"], CFG["max_shapes_per_img"]+1)
        # pracujemy na oddzielnej warstwie SS, żeby kontrolować blur/ink/stroke per figura
        ss = CFG["supersample"]
        ss_gray = np.full((H*ss, W*ss), CFG["bg_level"], np.uint8)

        for _ in range(int(n_shapes)):
            cls = random.choices(CFG["shape_classes"], weights=CFG["class_probs"], k=1)[0]
            cls_id = CLASS2ID[cls]
            size = rand(CFG["min_size"], CFG["max_size"])
            cx = rand(size, W-size)
            cy = rand(size, H-size)
            ang = rand(0, 360)
            is_faint = (random.random() < CFG["faint_prob"])

            S, cx_ss, cy_ss, size_ss, thk_ss, color, blur_sigma = render_shape_ss(canvas, cx, cy, size, ang, is_faint)

            # rysowanie w SS na buforze wspólnym
            if cls == "square":
                half = size_ss/2.0
                pts = np.array([[-half,-half],[half,-half],[half,half],[-half,half]], np.float32)
                a = np.deg2rad(ang); R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]], np.float32)
                pts = (pts @ R.T) + np.array([cx_ss, cy_ss], np.float32)
                cv.polylines(ss_gray, [pts.astype(np.int32)], True, color, thk_ss, aa(thk_ss))
                poly = pts
            elif cls == "triangle":
                r = size_ss/2.0
                pts = np.array([[0,-r],[r*np.cos(np.deg2rad(210)),-r*np.sin(np.deg2rad(210))],[r*np.cos(np.deg2rad(330)),-r*np.sin(np.deg2rad(330))]], np.float32)
                a = np.deg2rad(ang); R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]], np.float32)
                pts = (pts @ R.T) + np.array([cx_ss, cy_ss], np.float32)
                cv.polylines(ss_gray, [pts.astype(np.int32)], True, color, thk_ss, aa(thk_ss))
                poly = pts
            else:
                r = int(size_ss/2.0)
                cv.circle(ss_gray, (int(cx_ss), int(cy_ss)), r, color, thk_ss, aa(thk_ss))
                thetas = np.linspace(0,2*math.pi,12)
                poly = np.stack([cx_ss + r*np.cos(thetas), cy_ss + r*np.sin(thetas)],1).astype(np.float32)

            # lokalny blur dla tej figury: wycinamy ROI wokół niej, blurujemy i wklejamy
            if blur_sigma > 0.01:
                x1 = int(max(0, poly[:,0].min() - 6*ss))
                y1 = int(max(0, poly[:,1].min() - 6*ss))
                x2 = int(min(W*ss-1, poly[:,0].max() + 6*ss))
                y2 = int(min(H*ss-1, poly[:,1].max() + 6*ss))
                roi = ss_gray[y1:y2, x1:x2].copy()
                k = int(max(3, 2*int(3*blur_sigma*ss)+1))
                roi_blur = cv.GaussianBlur(roi, (k,k), blur_sigma*ss)
                ss_gray[y1:y2, x1:x2] = roi_blur

            # bbox w skali docelowej
            poly_final = (poly / ss).astype(np.float32)
            aabb = (
                clamp(float(poly_final[:,0].min()),0,W-1),
                clamp(float(poly_final[:,1].min()),0,H-1),
                clamp(float(poly_final[:,0].max()),0,W-1),
                clamp(float(poly_final[:,1].max()),0,H-1),
            )
            y = aabb_to_yolo(aabb, W, H, cls_id)
            if y: yolo_lines.append(f"{y[0]} {y[1]:.6f} {y[2]:.6f} {y[3]:.6f} {y[4]:.6f}")

        # downsample i scalenie z tłem
        down = cv.resize(ss_gray, (W, H), interpolation=cv.INTER_AREA)
        canvas[:] = cv.cvtColor(down, cv.COLOR_GRAY2BGR)

        # dodaj „papierowe” tło przez min-blend (tusze ciemniejsze)
        bg = make_background(H, W)
        canvas = cv.min(canvas, bg)

        # globalne „psucie”
        add_global_noise_and_blur(canvas)

        # zapis
        stem = f"synth_v2_{i:06d}"
        img_path = img_dir / f"{stem}.{CFG['img_format']}"
        lbl_path = lbl_dir / f"{stem}.txt"
        if CFG["img_format"].lower()=="jpg":
            cv.imwrite(str(img_path), canvas, [int(cv.IMWRITE_JPEG_QUALITY), CFG["jpeg_quality"]])
        else:
            cv.imwrite(str(img_path), canvas)
        with open(lbl_path,"w",encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # preview
        vis = canvas.copy()
        for line in yolo_lines:
            cls_id, xc, yc, ww, hh = line.split()
            cls_id = int(cls_id); xc=float(xc)*W; yc=float(yc)*H; ww=float(ww)*W; hh=float(hh)*H
            x1=int(xc-ww/2); y1=int(yc-hh/2); x2=int(xc+ww/2); y2=int(yc+hh/2)
            color = [(0,200,0),(230,160,0),(0,140,255)][cls_id]
            cv.rectangle(vis,(x1,y1),(x2,y2),color,2,cv.LINE_AA)
        prev_path = out_root/"previews"/f"{stem}.{CFG['img_format']}"
        cv.imwrite(str(prev_path), vis)

    print("Done.")
if __name__ == "__main__":
    main()
