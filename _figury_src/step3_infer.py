# predict_one_img.py
from pathlib import Path
import argparse
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# -------- CFG (stałe, nie-ścieżkowe) --------
CFG = {
    "imgsz": 1280,
    "conf": 0.35,
    "iou": 0.55,
    "max_det": 4000,
    "shrink": 0.55,          # zmniejszenie W,H bboxa
    "thickness": 2,
    "font_scale": 0.6,
    "class_names": ["square", "circle", "triangle"],
}
# --------------------------------------------



def shrink_box(x1, y1, x2, y2, shrink, W, H):
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    w = (x2 - x1) * shrink; h = (y2 - y1) * shrink
    nx1 = int(max(0, cx - w/2)); ny1 = int(max(0, cy - h/2))
    nx2 = int(min(W-1, cx + w/2)); ny2 = int(min(H-1, cy + h/2))
    return nx1, ny1, nx2, ny2

def draw_alpha_box(img, pt1, pt2, color_bgr, alpha=0.18, thickness=2):
    overlay = img.copy()
    cv.rectangle(overlay, pt1, pt2, color_bgr, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv.rectangle(img, pt1, pt2, color_bgr, thickness, cv.LINE_AA)

def predict_one(model: YOLO, img_path: Path, out_path: Path):
    res = model.predict(
        source=str(img_path),
        imgsz=CFG["imgsz"],
        conf=CFG["conf"],
        iou=CFG["iou"],
        max_det=CFG["max_det"],
        save=False,
        verbose=False
    )[0]

    img = cv.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Nie wczytam obrazu: {img_path}")
    H, W = img.shape[:2]

    colors = {0:(0,200,0), 1:(230,160,0), 2:(0,140,255)}

    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
            sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2, CFG["shrink"], W, H)
            col = colors.get(k, (255,255,255))
            draw_alpha_box(img, (sx1,sy1), (sx2,sy2), col, alpha=0.18, thickness=CFG["thickness"])
            label = f"{CFG['class_names'][k]} {c:.2f}" if 0 <= k < len(CFG["class_names"]) else f"{k} {c:.2f}"
            (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], 2)
            tx = max(0, sx1); ty = max(th + 4, sy1)
            cv.rectangle(img, (tx, ty - th - 4), (tx + tw + 2, ty + 2), (0,0,0), -1)
            cv.putText(img, label, (tx + 1, ty - 2), cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], (255,255,255), 2, cv.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(out_path), img):
        raise RuntimeError(f"Nie zapisano: {out_path}")
    print(f"Saved: {out_path}")

# Domyślne ścieżki (wykminione z poprzedniego skryptu)
DEFAULT_IMG = str(Path(__file__).parents[1] / "_inputs/PXL_20250925_050125594.jpg")
DEFAULT_OUT_DIR = str(Path(__file__).parents[1] / "_outputs/_figury/step3_infer")
DEFAULT_WEIGHTS = str(Path(__file__).parents[1] / "_outputs/_figury/step2_training/runs/shapes_yolo_n/weights/last.pt")

def parse_args():
    ap = argparse.ArgumentParser(description="Batch->Single: predykcja YOLO na jednym obrazie z własnymi wagami.")
    ap.add_argument("--input-img", type=Path, default=DEFAULT_IMG,
                    help="Ścieżka do wejściowego obrazu.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Katalog wyjściowy (plik będzie miał sufiks _pred.jpg).")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="Ścieżka do pliku wag last.pt.")
    return ap.parse_args()

def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Brak pliku wag: {args.weights}")
    if not args.input_img.exists():
        raise FileNotFoundError(f"Brak wejściowego obrazu: {args.input_img}")

    model = YOLO(str(args.weights))

    out_path = args.output_dir / (args.input_img.stem + "_pred.jpg")
    predict_one(model, args.input_img, out_path)

if __name__ == "__main__":
    main()
