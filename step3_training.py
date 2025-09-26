# path_img = "/home/admin2/Downloads/Shape detection/ShapeDetector raw/PXL_20250925_050227404.jpg"

# train_shapes_yolo.py
from pathlib import Path
import random
from ultralytics import YOLO
import os

# ===================== CONFIG =====================
CFG = {
    # --- ścieżki ---
    "root_data": "/home/admin2/Documents/repos/cct/pwpw/inputs",  # gdzie generator zrobił dataset/
    "run_root":  "/home/admin2/Documents/repos/cct/pwpw/outputs", # tu zapisze modele/metryki
    "test_img":  "/home/admin2/Downloads/Shape detection/ShapeDetector raw/PXL_20250925_050227404.jpg",
    "test_out":  "/home/admin2/Documents/repos/cct/pwpw/outputs/test",

    # --- model ---
    # podmień na 'yolov11n.pt' jeżeli wolisz najnowszy – API to samo
    "model_name": "yolov8n.pt",

    # --- dane / split ---
    "classes": ["square", "circle", "triangle"],  # 0/1/2 – zgodnie z generatorem
    "val_frac": 0.12,         # część walidacyjna
    "seed": 1337,

    # --- trening ---
    "epochs": 25,
    "batch": 16,
    "imgsz": 1024,            # YOLO robi letterbox; nie musi być 3072x4080
    "lr0": 0.01,
    "device": 0,              # -1 = CPU, 0 = pierwsze GPU
    "workers": 4,
    "patience": 50,           # early stop (duże żeby nie przerywać)
    "conf_pred": 0.25,        # próg do predykcji na test_img
}
# ==================================================

def make_split_txts(root_data: Path, val_frac: float, seed: int):
    """
    Tworzy listy train.txt/val.txt wskazujące obrazy. Labels YOLO muszą leżeć obok w dataset/labels.
    """
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
    # YOLOv8 akceptuje ścieżki do .txt z listą obrazów
    content = (
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        f"names:\n" + "".join([f"  {i}: {n}\n" for i, n in enumerate(classes)])
    )
    data_yaml.write_text(content)
    return data_yaml

# w CFG dodaj:
# "save_period": 1,

def main():
    root_data = Path(CFG["root_data"])
    run_root  = Path(CFG["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    test_out = Path(CFG["test_out"])
    test_out.mkdir(parents=True, exist_ok=True)

    # 1) split + yaml
    train_txt, val_txt = make_split_txts(root_data, CFG["val_frac"], CFG["seed"])
    data_yaml = write_data_yaml(run_root, train_txt, val_txt, CFG["classes"])

    # 2) model
    model = YOLO(CFG["model_name"])

    # 3) trenowanie W JEDNYM WYWOŁANIU i zapis checkpointów co epokę
    run_dir = run_root / "runs"
    run_name = "shapes_yolo_n"  # albo zrób z nazwy modelu

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
        save_period=CFG.get("save_period", 1)  # <<--- kluczowe
    )

    # 4) po treningu: znajdź katalog runu i przejdź po wagach epoch*.pt
    #    (w tej wersji API trener zostaje podpięty do obiektu model)
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", run_dir / run_name))
    weights_dir = save_dir / "weights"
    assert weights_dir.exists(), f"Nie znaleziono katalogu wag: {weights_dir}"

    # zbierz checkpointy per epoka
    def epoch_num(p):
        stem = p.stem  # 'epoch001'
        return int(''.join(ch for ch in stem if ch.isdigit()) or -1)

    ckpts = sorted(weights_dir.glob("epoch*.pt"), key=epoch_num)
    if not ckpts:
        # fallback: jeśli ktoś wyłączył save_period – przynajmniej zrób na last/best
        ckpts = [p for p in [weights_dir / "last.pt", weights_dir / "best.pt"] if p.exists()]

    # 5) predykcje na stałym obrazie dla KAŻDEGO checkpointu
    for ck in ckpts:
        ep = epoch_num(ck)
        out_name = f"epoch_{ep:03d}" if ep >= 0 else ck.stem
        m = YOLO(str(ck))
        m.predict(
            source=CFG["test_img"],
            conf=CFG["conf_pred"],
            imgsz=CFG["imgsz"],
            save=True,
            project=str(test_out),
            name=out_name,
            exist_ok=True,
            device=CFG["device"],
            verbose=False
        )

    print(f"OK. Run: {save_dir}")
    print(f"Checkpointy: {len(ckpts)}  |  Overlays: {CFG['test_out']}/*")


if __name__ == "__main__":
    main()