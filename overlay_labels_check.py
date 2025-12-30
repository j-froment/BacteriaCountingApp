from pathlib import Path
import json
import cv2

PAIRS_DIR = Path("raw") / "AGAR_dataset" / "dataset"

def main():
    sid = input("Enter an image id (stem), e.g. 1805: ").strip()
    img_path = PAIRS_DIR / f"{sid}.jpg"
    js_path  = PAIRS_DIR / f"{sid}.json"

    if not img_path.exists() or not js_path.exists():
        print("Missing file(s):", img_path, js_path)
        return

    d = json.loads(js_path.read_text(encoding="utf-8"))
    img = cv2.imread(str(img_path))
    if img is None:
        print("Could not read image")
        return

    labels = d.get("labels", [])
    cn = d.get("colonies_number", None)

    # draw boxes
    for lab in labels:
        x = int(lab["x"])
        y = int(lab["y"])
        w = int(lab["width"])
        h = int(lab["height"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"id={sid} colonies_number={cn} len(labels)={len(labels)} bg={d.get('background')}"
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Overlay check (press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
