from pathlib import Path
import json, shutil, cv2

DATA_DIR = Path("raw") / "AGAR_dataset" / "dataset"
OUT_DIR = Path("out")

def safe_load_json(p: Path):
    try:
        if p.stat().st_size == 0:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def image_size(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h

def json_to_yolo_lines(ann: dict, img_w: int, img_h: int):
    """
    TEMP placeholder — we will fix this once you confirm JSON schema
    """
    lines = []
    for obj in ann.get("annotations", []):
        bbox = obj.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = map(float, bbox)
            xc = (x + w/2) / img_w
            yc = (y + h/2) / img_h
            ww = w / img_w
            hh = h / img_h
            lines.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    return lines

def main():
    out_img = OUT_DIR / "images_all"
    out_lbl = OUT_DIR / "labels_all"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    made, bad_json, missing_img = 0, 0, 0

    for js in DATA_DIR.glob("*.json"):
        img = DATA_DIR / f"{js.stem}.jpg"
        if not img.exists():
            missing_img += 1
            continue

        # ---------- ADD THIS ----------
        out_img_file = out_img / img.name
        out_lbl_file = out_lbl / f"{js.stem}.txt"

        if out_img_file.exists() and out_lbl_file.exists():
            continue
        # ---------- END ADD ----------

        ann = safe_load_json(js)
        if ann is None:
            bad_json += 1
            continue

        size = image_size(img)
        if size is None:
            continue
        w, h = size

        shutil.copy2(img, out_img_file)
        lines = json_to_yolo_lines(ann, w, h)
        out_lbl_file.write_text("\n".join(lines))

        made += 1
        if made % 1000 == 0:
            print(f"Converted {made} files...")



    print("DONE")
    print("Converted pairs:", made)
    print("Bad JSON:", bad_json)
    print("Missing image:", missing_img)

if __name__ == "__main__":
    main()
