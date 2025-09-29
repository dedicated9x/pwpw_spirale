# cvat_autolabel_yolo.py (version-agnostic)
from pathlib import Path
import io, zipfile, tempfile
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from cvat_sdk import make_client

# ---------------- CFG ----------------
CFG = {
    "cvat_url": "http://localhost:8080",
    "cvat_user": "pbrysch",
    "cvat_pass": "password",
    "task_id": 1,  # ID taska z URL

    # YOLO
    "runs_dir": "/home/admin2/Documents/repos/cct/pwpw/outputs/runs",
    "run_name": "shapes_yolo_n",
    "weights": None,  # pełna ścieżka do .pt, jeśli chcesz nadpisać

    # kolejność = kolejność etykiet w Tasku CVAT
    "class_names": ["square", "circle", "triangle"],

    # inferencja
    "imgsz": 1280,
    "conf": 0.35,
    "iou": 0.55,
    "max_det": 4000,

    # czyszczenie istniejących adnotacji
    "clear_existing": True,
}
# -------------------------------------


def find_last_weights(runs_dir: Path, run_name: str) -> Path:
    p = runs_dir / run_name / "weights" / "last.pt"
    if p.exists():
        return p
    cands = sorted(
        runs_dir.glob(f"{run_name}*/weights/last.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0]
    raise FileNotFoundError(f"Brak wag last.pt w {runs_dir}/**/{run_name}*/weights/")


def yolo_lines(res, W, H):
    """Lista linii YOLO 'cls xc yc w h' (znormalizowane)."""
    out = []
    b = res.boxes
    if b is None or len(b) == 0:
        return out
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), k in zip(xyxy, cls):
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue
        xc = x1 + w / 2
        yc = y1 + h / 2
        out.append(f"{k} {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}")
    return out


def iter_frames_meta(task):
    """
    Zwraca iterator (fid:int, stem:str).
    * fid – indeks klatki/linii danych, zgodny z task.get_frame(fid)
    * stem – nazwa pliku bez rozszerzenia (jeśli brak: fid z zerami)
    Odporne na różne wersje SDK.
    """
    meta = task.get_meta()
    frames = getattr(meta, "frames", None) or []
    size = getattr(meta, "size", None)

    if frames:
        for i, fr in enumerate(frames):
            # nazwa
            name = getattr(fr, "name", None)
            if name is None:
                # próba przez dict
                try:
                    d = fr.to_dict()
                except Exception:
                    d = getattr(fr, "received_data", {}) or {}
                name = d.get("name", None)
            stem = Path(name).stem if isinstance(name, str) else f"{i:06d}"

            # id/indeks (często i tak == i)
            fid = None
            for k in ("index", "frame", "id", "order"):
                v = getattr(fr, k, None)
                if isinstance(v, (int, np.integer)):
                    fid = int(v); break
            if fid is None:
                # fallback: użyj kolejności na liście
                fid = i
            yield fid, stem
        return

    # brak listy frames – iteruj po rozmiarze
    if isinstance(size, (int, np.integer)) and size > 0:
        for fid in range(int(size)):
            yield fid, f"{fid:06d}"
        return

    # ostateczny fallback: spróbuj pobierać kolejne ramki aż do pierwszej porażki
    fid = 0
    while True:
        try:
            _ = task.get_frame(fid)
        except Exception:
            break
        yield fid, f"{fid:06d}"
        fid += 1


def main():
    weights = (
        Path(CFG["weights"])
        if CFG["weights"]
        else find_last_weights(Path(CFG["runs_dir"]), CFG["run_name"])
    )
    model = YOLO(str(weights))

    with make_client(CFG["cvat_url"]) as client:
        client.login((CFG["cvat_user"], CFG["cvat_pass"]))
        task = client.tasks.retrieve(CFG["task_id"])

        # wyczyść istniejące adnotacje (obsłuż różne API)
        if CFG["clear_existing"]:
            try:
                task.remove_annotations()
            except Exception:
                try:
                    # starsze API
                    client.tasks.delete_annotations(task.id)
                except Exception as e:
                    print("[WARN] Nie udało się wyczyścić adnotacji:", e)

        # przygotuj ZIP YOLO 1.1 w pamięci
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # wymagany plik klas
            zf.writestr("obj.names", "\n".join(CFG["class_names"]) + "\n")

            for fid, stem in iter_frames_meta(task):
                # pobierz obraz
                try:
                    img_bytes = task.get_frame(fid)
                except Exception as e:
                    print(f"[WARN] Nie mogę pobrać frame {fid}: {e}")
                    zf.writestr(f"{stem}.txt", "")
                    continue

                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv.imdecode(arr, cv.IMREAD_COLOR)
                if img is None:
                    zf.writestr(f"{stem}.txt", "")
                    continue

                H, W = img.shape[:2]
                res = model.predict(
                    source=img,
                    imgsz=CFG["imgsz"],
                    conf=CFG["conf"],
                    iou=CFG["iou"],
                    max_det=CFG["max_det"],
                    verbose=False,
                )[0]
                lines = yolo_lines(res, W, H)
                zf.writestr(f"{stem}.txt", "\n".join(lines))

        # zapisz ZIP tymczasowo i zaimportuj do CVAT
        mem_zip.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(mem_zip.read())
            tmp_path = Path(tmp.name)

        client.tasks.import_annotations(task.id, "YOLO 1.1", tmp_path)
        print(f"OK: wgrano adnotacje do Taska {task.id}")
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
