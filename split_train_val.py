from pathlib import Path
import json, shutil

DATASET_DIR = Path("raw") / "AGAR_dataset"
LISTS_DIR   = DATASET_DIR / "training_lists"

OUT_DIR = Path("out")
IM_ALL = OUT_DIR / "images_all"
LB_ALL = OUT_DIR / "labels_all"

# choose which lists you want
TRAIN_LIST = LISTS_DIR / "lower_resolution_train.txt"
VAL_LIST   = LISTS_DIR / "lower_resolution_val.txt"

def read_ids(p: Path) -> set[str]:
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()

    # If file is JSON/Python-list style: ["1","2",...]
    if txt.startswith("["):
        try:
            arr = json.loads(txt)
            return {str(x).strip().strip('"').strip("'") for x in arr}
        except Exception:
            # fallback: try to extract digits
            pass

    # Otherwise assume one item per line
    ids = set()
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.add(Path(s).stem)
    return ids

def ensure_dirs():
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_split(ids: set[str], split: str):
    made = 0
    missing = []

    for id_ in ids:
        img = IM_ALL / f"{id_}.jpg"
        lbl = LB_ALL / f"{id_}.txt"

        if not img.exists() or not lbl.exists():
            missing.append(id_)
            continue

        shutil.copy2(img, OUT_DIR / "images" / split / img.name)
        shutil.copy2(lbl, OUT_DIR / "labels" / split / lbl.name)
        made += 1

    (OUT_DIR / f"missing_in_{split}.txt").write_text("\n".join(missing), encoding="utf-8")
    print(f"{split}: copied {made}, missing {len(missing)} (see out/missing_in_{split}.txt)")

def main():
    ensure_dirs()

    train_ids = read_ids(TRAIN_LIST)
    val_ids   = read_ids(VAL_LIST)

    # avoid overlap
    overlap = train_ids & val_ids
    if overlap:
        print(f"WARNING: {len(overlap)} overlap ids; removing from val")
        val_ids -= overlap

    copy_split(train_ids, "train")
    copy_split(val_ids, "val")

if __name__ == "__main__":
    main()
