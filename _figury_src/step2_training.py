from pathlib import Path
import random
import argparse
from ultralytics import YOLO

# ===================== CONFIG =====================
CFG = {
    # --- ścieżki (domyślne) ---
    "root_data": str(Path(__file__).parents[1] / "_outputs/_figury/step1_generate_data"),
    "run_root":  str(Path(__file__).parents[1] / "_outputs/_figury/step2_training"),

    # --- model ---
    "model_name": "yolov8n.pt",

    # --- dane / split ---
    "classes": ["square", "circle", "triangle"],
    "val_frac": 0.12,
    "seed": 1337,

    # --- trening ---
    "epochs": 25,
    "batch": 16,
    "imgsz": 1024,
    "lr0": 0.01,
    "device": 0,     # -1 = CPU, 0 = pierwsze GPU
    "workers": 4,
    "patience": 50,
    "conf_pred": 0.25,  # (niewykorzystywane, ale zostawiam jeśli zechcesz dodać predykcje)
}
# ==================================================

def make_split_txts(root_data: Path, val_frac: float, seed: int):
    """Tworzy listy train.txt/val.txt wskazujące obrazy. Labels YOLO muszą leżeć obok w dataset/labels."""
    img_dir = root_data / "dataset" / "images"
    lbl_dir = root_data / "dataset" / "labels"
    assert img_dir.is_dir() and lbl_dir.is_dir(), f"Brak {img_dir} lub {lbl_dir}"

    imgs = sorted([p for p in img_dir.glob("*.jpg")] +
                  [p for p in img_dir.glob("*.png")] +
                  [p for p in img_dir.glob("*.jpeg")] +
                  [p for p in img_dir.glob("*.bmp")] +
                  [p for p in img_dir.glob("*.tif")] +
                  [p for p in img_dir.glob("*.tiff")])

    # filtr: tylko te, które mają label
    imgs = [p for p in imgs if (lbl_dir / (p.stem + ".txt")).exists()]
    assert imgs, "Nie znaleziono obrazów z etykietami"

    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac))
    val = imgs[:n_val]
    train = imgs[n_val:]

    split_dir = root_data / "dataset" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train))
    val_txt.write_text("\n".join(str(p) for p in val))
    return train_txt, val_txt

def write_data_yaml(run_root: Path, train_txt: Path, val_txt: Path, classes):
    data_yaml = run_root / "shapes_data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        f"names:\n" + "".join([f"  {i}: {n}\n" for i, n in enumerate(classes)])
    )
    data_yaml.write_text(content)
    return data_yaml

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO on generated shapes dataset.")
    p.add_argument("--root-data", type=Path, default=Path(CFG["root_data"]),
                   help="Katalog z datasetem: oczekuje dataset/{images,labels}. Domyślnie z CFG.")
    p.add_argument("--run-root", type=Path, default=Path(CFG["run_root"]),
                   help="Katalog wyjściowy na runs/ i shapes_data.yaml. Domyślnie z CFG.")
    return p.parse_args()

def main(root_data: Path, run_root: Path):
    run_root.mkdir(parents=True, exist_ok=True)

    # 1) split + yaml
    train_txt, val_txt = make_split_txts(root_data, CFG["val_frac"], CFG["seed"])
    data_yaml = write_data_yaml(run_root, train_txt, val_txt, CFG["classes"])

    # 2) model
    model = YOLO(CFG["model_name"])

    # 3) trening
    run_dir = run_root / "runs"
    run_name = "shapes_yolo_n"

    model.train(
        data=str(data_yaml),
        imgsz=CFG["imgsz"],
        epochs=CFG["epochs"],
        batch=CFG["batch"],
        device=CFG["device"],
        workers=CFG["workers"],
        lr0=CFG["lr0"],
        project=str(run_dir),
        name=run_name,
        exist_ok=True,
        patience=CFG["patience"],
        verbose=True,
    )

if __name__ == "__main__":
    args = parse_args()
    main(root_data=args.root_data, run_root=args.run_root)
