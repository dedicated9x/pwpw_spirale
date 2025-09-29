import os
from pathlib import Path
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
# from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates as img_coords

# ============ KONFIG ============
CFG = {
    "runs_dir": "/home/admin2/Documents/repos/cct/pwpw/outputs/runs",
    "run_name": "shapes_yolo_n",  # jak w treningu
    "weights": None,              # pełna ścieżka .pt jeśli chcesz nadpisać
    "save_root": "/home/admin2/Documents/repos/pwpw/outputs/dataset_annotated",
    "class_names": ["square", "circle", "triangle"],
    "colors": {0: (0, 200, 0), 1: (230, 160, 0), 2: (0, 140, 255)},
    "default_conf": 0.35,
    "default_iou": 0.55,
    "imgsz": 1280,
}
# =================================

st.set_page_config(page_title="Shape Detector – Annotator", layout="wide")

def find_last_weights(runs_dir: Path, run_name: str) -> Path:
    p = runs_dir / run_name / "weights" / "last.pt"
    if p.exists():
        return p
    cands = sorted(runs_dir.glob(f"{run_name}*/weights/last.pt"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    raise FileNotFoundError(f"Nie znalazłem last.pt w {runs_dir}/**/{run_name}*/weights/")

def pil_draw_boxes(img: Image.Image, boxes, labels, colors, thickness=3, alpha=64):
    """boxes w pikselach: [x1,y1,x2,y2], labels: (cls_id, score|None)"""
    vis = img.convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for (x1,y1,x2,y2), (cls_id, score) in zip(boxes, labels):
        col = colors.get(cls_id, (255,255,255))
        fill = (*col, alpha)
        odraw.rectangle([x1,y1,x2,y2], fill=fill, outline=col+(255,), width=thickness)
        text = CFG["class_names"][cls_id] + (f" {score:.2f}" if score is not None else "")
        tw, th = odraw.textlength(text, font=font), 12
        odraw.rectangle([x1, y1-th-4, x1+tw+6, y1], fill=(0,0,0,200))
        odraw.text((x1+3, y1-th-2), text, font=font, fill=(255,255,255,255))
    return Image.alpha_composite(vis, overlay).convert("RGB")

def yolo_xyxy_to_norm_xywh(x1,y1,x2,y2,W,H):
    xc = (x1+x2)/2.0; yc = (y1+y2)/2.0
    w = (x2-x1); h = (y2-y1)
    return (xc/W, yc/H, w/W, h/H)

def ensure_dirs(root):
    root = Path(root)
    (root/"images").mkdir(parents=True, exist_ok=True)
    (root/"labels").mkdir(parents=True, exist_ok=True)
    return root

# ---------- SIDEBAR ----------
st.sidebar.header("Ustawienia")
conf = st.sidebar.slider("conf", 0.0, 1.0, CFG["default_conf"], 0.01)
iou  = st.sidebar.slider("iou",  0.1, 0.95, CFG["default_iou"], 0.01)
imgsz = st.sidebar.selectbox("imgsz (inferencja)", [640, 960, 1024, 1280, 1536], index=3)

# Wagi
try:
    weights = Path(CFG["weights"]) if CFG["weights"] else find_last_weights(Path(CFG["runs_dir"]), CFG["run_name"])
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()
st.sidebar.success(f"Wagi: {weights}")

# ---------- MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    return YOLO(weights_path)
model = load_model(str(weights))

# ---------- INPUT ----------
st.title("Shape Detector – Annotator (Streamlit)")
up = st.file_uploader("Wgraj obraz (JPG/PNG)", type=["jpg","jpeg","png"])

if "state" not in st.session_state:
    st.session_state.state = {
        "img": None,     # PIL
        "img_path": None,
        "W": None, "H": None,
        "boxes": [],     # list[(x1,y1,x2,y2)]
        "labels": [],    # list[(cls_id, score|None)]
    }

state = st.session_state.state

col_left, col_right = st.columns([2,1])

with col_left:
    if up:
        img = Image.open(up).convert("RGB")
        state["img"] = img
        state["img_path"] = up.name
        state["W"], state["H"] = img.size

    if state["img"] is None:
        st.info("Wgraj obraz po lewej.")
        st.stop()

    # ----- PREDICT -----
    if st.button("Uruchom predykcję YOLO", type="primary"):
        res = model.predict(
            source=np.array(state["img"]),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=4000,
            save=False,
            verbose=False
        )[0]
        boxes = res.boxes
        state["boxes"], state["labels"] = [], []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs= boxes.conf.cpu().numpy()
            clses= boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), c, k in zip(xyxy, confs, clses):
                # Clamp do rozmiaru obrazu
                x1 = int(max(0, min(state["W"]-1, x1))); y1 = int(max(0, min(state["H"]-1, y1)))
                x2 = int(max(0, min(state["W"]-1, x2))); y2 = int(max(0, min(state["H"]-1, y2)))
                state["boxes"].append([x1,y1,x2,y2])
                state["labels"].append([int(k), float(c)])
        st.success(f"Detekcje: {len(state['boxes'])}")

    # ----- PODGLĄD -----
    vis = pil_draw_boxes(state["img"], state["boxes"], state["labels"], CFG["colors"], thickness=3, alpha=70)
    st.image(vis, caption="Podgląd z bboxami (kliknij 'Detekcja', aby odświeżyć)")

with col_right:
    st.subheader("Adnotacje")

    # ====== USUWANIE (FP) ======
    if state["boxes"]:
        idx_to_remove = st.multiselect(
            "Zaznacz bboxy do usunięcia (FP)",
            options=list(range(len(state["boxes"]))),
            format_func=lambda i: f"{i}: {CFG['class_names'][state['labels'][i][0]]} "
                                  f"({state['labels'][i][1]:.2f if state['labels'][i][1] is not None else 'manual'})"
        )
        if st.button("Usuń zaznaczone"):
            keep = [i for i in range(len(state["boxes"])) if i not in set(idx_to_remove)]
            state["boxes"]  = [state["boxes"][i] for i in keep]
            state["labels"] = [state["labels"][i] for i in keep]

    st.markdown("---")

    # ====== DODAWANIE (FN) NA DWA KLIKI ======
    st.markdown("**Dodaj bbox (FN):** kliknij dwa punkty na obrazie: lewy-górny, potem prawy-dolny.")

    # stan pomocniczy
    if "click_a" not in state: state["click_a"] = None  # (x,y) pierwszego kliku
    add_cls = st.selectbox("Klasa nowego bboxa", list(range(len(CFG["class_names"]))),
                           format_func=lambda i: CFG["class_names"][i], key="add_cls")

    # pokazujemy ten sam obraz co w lewym panelu, bez skalowania – żeby współrzędne były 1:1
    coords = img_coords(state["img"], key="click_add", width=state["W"])  # width=W gwarantuje 1:1

    if coords is not None:
        x = int(np.clip(coords["x"], 0, state["W"]-1))
        y = int(np.clip(coords["y"], 0, state["H"]-1))
        if state["click_a"] is None:
            state["click_a"] = (x, y)
            st.info(f"Pierwszy punkt: {state['click_a']}. Kliknij drugi (prawy-dolny).")
        else:
            x1, y1 = state["click_a"]
            x2, y2 = x, y
            # uporządkuj współrzędne
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            if x2 > x1 and y2 > y1:
                state["boxes"].append([x1, y1, x2, y2])
                state["labels"].append([int(add_cls), None])  # manual
                st.success(f"Dodano bbox: {(x1,y1,x2,y2)} klasa={CFG['class_names'][int(add_cls)]}")
            else:
                st.warning("Narysuj prostokąt o niezerowym rozmiarze.")
            state["click_a"] = None  # reset sekwencji

    if st.button("Reset punktu startowego"):
        state["click_a"] = None

    st.markdown("---")
    # ====== ZAPIS ======
    save_root = ensure_dirs(CFG["save_root"])
    default_name = Path(state["img_path"]).stem if state.get("img_path") else "annotated_image"
    save_name = st.text_input("Nazwa pliku (bez rozszerzenia)", value=default_name)

    if st.button("Zapisz obraz + YOLO labels"):
        img_out = save_root/"images"/f"{save_name}.jpg"
        state["img"].save(img_out, quality=95)
        W,H = state["W"], state["H"]
        lines = []
        for (x1,y1,x2,y2), (cls_id, _) in zip(state["boxes"], state["labels"]):
            xc,yc,w,h = yolo_xyxy_to_norm_xywh(x1,y1,x2,y2,W,H)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (save_root/"labels"/f"{save_name}.txt").write_text("\n".join(lines))
        st.success(f"Zapisano:\n- {img_out}\n- {save_root/'labels'/(save_name + '.txt')}")
        st.toast("Zapis OK", icon="✅")