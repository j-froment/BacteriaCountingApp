from pathlib import Path
import json
from collections import Counter

PAIRS_DIR = Path("raw") / "AGAR_dataset" / "dataset"

def safe_load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    jpgs = sorted(PAIRS_DIR.glob("*.jpg"))
    jsons = sorted(PAIRS_DIR.glob("*.json"))

    jpg_stems = {p.stem for p in jpgs}
    json_stems = {p.stem for p in jsons}

    print("Images:", len(jpgs))
    print("JSONs :", len(jsons))
    print("Matched pairs:", len(jpg_stems & json_stems))
    print("Missing JSON for images:", len(jpg_stems - json_stems))
    print("Missing image for JSONs:", len(json_stems - jpg_stems))

    # Deep checks on matched pairs
    problems = []
    bg_counts = Counter()
    cls_counts = Counter()

    n = 0
    for stem in sorted(jpg_stems & json_stems):
        n += 1
        js_path = PAIRS_DIR / f"{stem}.json"
        d = safe_load(js_path)
        if d is None:
            problems.append((stem, "bad_json"))
            continue

        # 1) sample_id matches filename stem?
        sid = d.get("sample_id", None)
        if sid is not None and str(sid) != str(stem):
            problems.append((stem, f"sample_id_mismatch json={sid} file={stem}"))

        # 2) colonies_number exists
        cn = d.get("colonies_number", None)
        if cn is None:
            problems.append((stem, "missing_colonies_number"))

        # 3) labels length matches colonies_number (if labels exist)
        labels = d.get("labels", None)
        if isinstance(labels, list) and cn is not None:
            if len(labels) != int(cn):
                problems.append((stem, f"count_mismatch colonies_number={cn} len(labels)={len(labels)}"))

        # just for summary
        bg_counts[str(d.get("background", "NA"))] += 1
        for c in (d.get("classes") or []):
            cls_counts[str(c)] += 1

        if n % 2000 == 0:
            print(f"Checked {n} pairs...")

    print("\n--- Summary ---")
    print("Background distribution:", dict(bg_counts))
    print("Top classes:", cls_counts.most_common(10))

    print("\n--- Problems (first 50) ---")
    for p in problems[:50]:
        print(p)

    print("\nTotal problems found:", len(problems))
    if len(problems) == 0:
        print("✅ Pairing looks consistent (no obvious image↔json mismatch).")
    else:
        print("⚠️ Issues found. If sample_id mismatches exist, pairing could be wrong.")

if __name__ == "__main__":
    main()
