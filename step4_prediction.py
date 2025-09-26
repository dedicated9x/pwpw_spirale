# predict_small_boxes.py
from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO

# ---------------- CFG ----------------
CFG = {
    # skąd wziąć wagi
    "runs_dir": "/home/admin2/Documents/repos/cct/pwpw/outputs/runs",
    "run_name": "shapes_yolo_n",                   # jak w treningu
    "weights": None,                                # nadpisz ścieżką do .pt jeśli chcesz

    # wejście/wyjście
    "img_path": "/home/admin2/Downloads/Shape detection/ShapeDetector raw/PXL_20250925_050227404.jpg",
    "save_dir": "/home/admin2/Documents/repos/cct/pwpw/outputs/test_last",
    "out_name": "pred_small_boxes.jpg",

    # inferencja
    "imgsz": 1280,          # YOLO letterbox – nie musi być natywne
    # "conf": 0.35,
    "conf": 0.55,
    "iou": 0.55,
    "max_det": 4000,

    # rysowanie (małe bboxy)
    "shrink": 0.55,         # 0.55 = 55% pierwotnego W i H (mniejszy bbox)
    "thickness": 2,
    "font_scale": 0.6,
    "class_names": ["square", "circle", "triangle"],
}
# -------------------------------------

def find_last_weights(runs_dir: Path, run_name: str) -> Path:
    # standardowy layout ultralytics: runs/<run_name>/weights/last.pt
    p = runs_dir / run_name / "weights" / "last.pt"
    if p.exists():
        return p
    # czasami Ultralytics tworzy podkatalogi z numerami – bierz najnowszy
    cands = sorted((runs_dir).glob(f"{run_name}*/weights/last.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    raise FileNotFoundError(f"Nie znalazłem wag last.pt w {runs_dir}/**/{run_name}*/weights/")

def shrink_box(x1, y1, x2, y2, shrink, W, H):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = (x2 - x1) * shrink
    h  = (y2 - y1) * shrink
    nx1 = int(max(0, cx - w/2)); ny1 = int(max(0, cy - h/2))
    nx2 = int(min(W-1, cx + w/2)); ny2 = int(min(H-1, cy + h/2))
    return nx1, ny1, nx2, ny2

def draw_alpha_box(img, pt1, pt2, color_bgr, alpha=0.15, thickness=2):
    # półprzezroczyste wypełnienie + kontur
    overlay = img.copy()
    cv.rectangle(overlay, pt1, pt2, color_bgr, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv.rectangle(img, pt1, pt2, color_bgr, thickness, cv.LINE_AA)

def main():
    runs_dir = Path(CFG["runs_dir"])
    weights = Path(CFG["weights"]) if CFG["weights"] else find_last_weights(runs_dir, CFG["run_name"])
    save_dir = Path(CFG["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    # inferencja bez auto-zapisu, my sami narysujemy
    res = model.predict(
        source=CFG["img_path"],
        imgsz=CFG["imgsz"],
        conf=CFG["conf"],
        iou=CFG["iou"],
        max_det=CFG["max_det"],
        save=False,
        verbose=False
    )[0]

    # wczytaj oryginalny obraz (rysujemy w natywnej skali)
    img = cv.imread(CFG["img_path"])
    assert img is not None, f"Nie mogę wczytać {CFG['img_path']}"
    H, W = img.shape[:2]

    # kolory (BGR)
    colors = {
        0: (0, 200, 0),       # square – zielony
        1: (230, 160, 0),     # circle – niebieskawy (BGR)
        2: (0, 140, 255),     # triangle – pomarańcz
    }

    # pętla po detekcjach
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        out_path = str(save_dir / CFG["out_name"])
        cv.imwrite(out_path, img)
        print(f"Brak detekcji. Zapisano {out_path}")
        return

    xyxy = boxes.xyxy.cpu().numpy()     # [N,4]
    conf = boxes.conf.cpu().numpy()     # [N]
    cls  = boxes.cls.cpu().numpy().astype(int)

    # rysowanie „skurczonych” bboxów i dyskretnych etykiet
    for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
        sx1, sy1, sx2, sy2 = shrink_box(x1, y1, x2, y2, CFG["shrink"], W, H)
        col = colors.get(k, (255,255,255))
        draw_alpha_box(img, (sx1,sy1), (sx2,sy2), col, alpha=0.18, thickness=CFG["thickness"])

        label = f"{CFG['class_names'][k]} {c:.2f}"
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], 2)
        tx = max(0, sx1); ty = max(th + 4, sy1)
        cv.rectangle(img, (tx, ty - th - 4), (tx + tw + 2, ty + 2), (0,0,0), -1)
        cv.putText(img, label, (tx + 1, ty - 2), cv.FONT_HERSHEY_SIMPLEX, CFG["font_scale"], (255,255,255), 2, cv.LINE_AA)

    out_path = str(save_dir / CFG["out_name"])
    cv.imwrite(out_path, img)
    print(f"Zapisano: {out_path}\nWagi: {weights}")

if __name__ == "__main__":
    main()
