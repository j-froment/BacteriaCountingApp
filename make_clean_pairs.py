from pathlib import Path
import json
import shutil

SRC_DIR = Path("raw") / "AGAR_dataset" / "dataset"
DST_DIR = Path("raw") / "AGAR_dataset" / "dataset_clean"
DST_DIR.mkdir(parents=True, exist_ok=True)

def safe_load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    jpgs = sorted(SRC_DIR.glob("*.jpg"))
    jsons = sorted(SRC_DIR.glob("*.json"))

    # Map: sample_id -> json path (from inside json)
    id_to_json = {}
    bad_json = 0
    for js in jsons:
        d = safe_load(js)
        if d is None:
            bad_json += 1
            continue
        sid = d.get("sample_id", None)
        if sid is None:
            continue
        id_to_json[str(sid)] = js

    kept = 0
    skipped_missing_json = 0
    skipped_invalid_count = 0
    skipped_count_mismatch = 0

    for img in jpgs:
        stem = img.stem
        js = id_to_json.get(stem, None)
        if js is None:
            skipped_missing_json += 1
            continue

        d = safe_load(js)
        if d is None:
            continue

        cn = d.get("colonies_number", None)
        if cn is None or int(cn) < 0:
            skipped_invalid_count += 1
            continue

        labels = d.get("labels", None)
        if isinstance(labels, list) and int(cn) != len(labels) and len(labels) > 0:
            # optional strict check; this catches weird inconsistencies
            skipped_count_mismatch += 1
            continue

        # Copy image and its correct JSON into clean folder using the same stem
        shutil.copy2(img, DST_DIR / f"{stem}.jpg")
        shutil.copy2(js, DST_DIR / f"{stem}.json")
        kept += 1

        if kept % 2000 == 0:
            print(f"Kept {kept} clean pairs...")

    print("\nDONE")
    print("Kept clean pairs:", kept)
    print("Bad json files:", bad_json)
    print("Skipped (no json for image id):", skipped_missing_json)
    print("Skipped (colonies_number < 0 or missing):", skipped_invalid_count)
    print("Skipped (count mismatch):", skipped_count_mismatch)
    print("Clean dataset folder:", DST_DIR)

if __name__ == "__main__":
    main()
