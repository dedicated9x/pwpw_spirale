# predict_small_boxes_batch.py
from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# ---------------- CFG ----------------
CFG = {
    # wagi
    "runs_dir": "/home/admin2/Documents/repos/pwpw/outputs/runs",
    "run_name": "shapes_yolo_n",     # jak w treningu
    "weights": None,                  # można podać pełną ścieżkę do .pt

    # wejście
    "img_dir": "/home/admin2/Downloads/Shape detection/ShapeDetector raw",
    "filenames": [
        "PXL_20250925_050227404.jpg",
        "PXL_20250925_050125594.jpg",
        "PXL_20250925_050134261.jpg",
        "PXL_20250925_050136581.jpg",
        "PXL_20250925_050148269.jpg",
        "PXL_20250925_050322604.jpg",
        "PXL_20250925_050348799.jpg",
        "PXL_20250925_050127789.jpg",
        "PXL_20250925_050147493.jpg",
        "PXL_20250925_050201833.MP.jpg",
    ],

    # wyjście
    "save_dir": "/home/admin2/Documents/repos/pwpw/outputs/test_last",

    # inferencja
    "imgsz": 1280,
    "conf": 0.35,
    "iou": 0.55,
    "max_det": 4000,

    # rysowanie
    "shrink": 0.55,          # zmniejszenie W,H bboxa
    "thickness": 2,
    "font_scale": 0.6,
    "class_names": ["square", "circle", "triangle"],
}
# -------------------------------------

def find_last_weights(runs_dir: Path, run_name: str) -> Path:
    p = runs_dir / run_name / "weights" / "last.pt"
    if p.exists(): return p
    cands = sorted(runs_dir.glob(f"{run_name}*/weights/last.pt"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if cands: return cands[0]
    raise FileNotFoundError(f"Brak wag last.pt w {runs_dir}/**/{run_name}*/weights/")

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
        print(f"[WARN] Nie wczytam: {img_path}")
        return
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
            label = f"{CFG['class_names'][k]} {c:.2f}"
            (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], 2)
            tx = max(0, sx1); ty = max(th + 4, sy1)
            cv.rectangle(img, (tx, ty - th - 4), (tx + tw + 2, ty + 2), (0,0,0), -1)
            cv.putText(img, label, (tx + 1, ty - 2), cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], (255,255,255), 2, cv.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")

def main():
    runs_dir = Path(CFG["runs_dir"])
    weights = Path(CFG["weights"]) if CFG["weights"] else find_last_weights(runs_dir, CFG["run_name"])
    model = YOLO(str(weights))

    img_dir = Path(CFG["img_dir"])
    save_dir = Path(CFG["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)

    for name in CFG["filenames"]:
        in_path = img_dir / name
        out_path = save_dir / (Path(name).stem + "_pred.jpg")
        predict_one(model, in_path, out_path)

if __name__ == "__main__":
    main()
