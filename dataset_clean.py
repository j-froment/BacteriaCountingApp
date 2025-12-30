from pathlib import Path
import json
import random
import shutil
import cv2

# ---- paths ----
CLEAN_DIR = Path("raw") / "AGAR_dataset" / "dataset_clean"
LISTS_DIR = Path("raw") / "AGAR_dataset" / "training_lists"

# Use the split you trained on earlier (lower-resolution lists)
TRAIN_LIST = LISTS_DIR / "lower_resolution_train.txt"
VAL_LIST   = LISTS_DIR / "lower_resolution_val.txt"

YOLO_DIR = Path("yolo_colonies")  # output dataset folder
IMG_TRAIN = YOLO_DIR / "images" / "train"
IMG_VAL   = YOLO_DIR / "images" / "val"
LBL_TRAIN = YOLO_DIR / "labels" / "train"
LBL_VAL   = YOLO_DIR / "labels" / "val"

for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    p.mkdir(parents=True, exist_ok=True)


def read_ids(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    if txt.startswith("["):
        return [str(x) for x in json.loads(txt)]
    ids = []
    for line in txt.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(Path(s).stem)
    return ids


def yolo_line_from_box(x, y, w, h, W, H):
    # json boxes are top-left (x,y) with width/height (you confirmed overlay looks correct)
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    bw = w / W
    bh = h / H
    # clamp to [0,1] just in case
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    # single class: 0 = colony
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_one(stem: str, out_img: Path, out_lbl: Path) -> bool:
    img_path = CLEAN_DIR / f"{stem}.jpg"
    js_path  = CLEAN_DIR / f"{stem}.json"
    if not img_path.exists() or not js_path.exists():
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        return False
    H, W = img.shape[:2]

    d = json.loads(js_path.read_text(encoding="utf-8"))
    labels = d.get("labels", [])
    cn = d.get("colonies_number", None)

    # only keep truly labeled examples
    if cn is None or int(cn) < 0:
        return False

    # convert boxes
    lines = []
    for lab in labels:
        x = float(lab["x"])
        y = float(lab["y"])
        w = float(lab["width"])
        h = float(lab["height"])
        lines.append(yolo_line_from_box(x, y, w, h, W, H))

    # copy image + write labels
    shutil.copy2(img_path, out_img / img_path.name)
    (out_lbl / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return True


def main():
    train_ids = read_ids(TRAIN_LIST)
    val_ids   = read_ids(VAL_LIST)

    kept_train = 0
    kept_val = 0

    for stem in train_ids:
        if convert_one(stem, IMG_TRAIN, LBL_TRAIN):
            kept_train += 1
    for stem in val_ids:
        if convert_one(stem, IMG_VAL, LBL_VAL):
            kept_val += 1

    # write data.yaml
    data_yaml = YOLO_DIR / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {YOLO_DIR.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: colony",
        ]),
        encoding="utf-8",
    )

    print("DONE")
    print("Kept train:", kept_train)
    print("Kept val  :", kept_val)
    print("Wrote:", data_yaml)


if __name__ == "__main__":
    main()
