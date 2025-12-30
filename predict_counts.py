from pathlib import Path
import json
import time
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ---------------- Paths ----------------
DATASET_DIR = Path("raw") / "AGAR_dataset"
PAIRS_DIR   = DATASET_DIR / "dataset"
LISTS_DIR   = DATASET_DIR / "training_lists"
RUNS_DIR    = Path("runs") / "count"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RUNS_DIR / "best_count_model.pt"

# ---------------- Settings ----------------
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Choose which images to predict on:
#   "val" = recommended first (faster, meaningful)
#   "all" = every jpg in dataset (slow)
PRED_SPLIT = "val"

# Choose which condition split lists you used for training:
TRAIN_LIST = LISTS_DIR / "lower_resolution_train.txt"
VAL_LIST   = LISTS_DIR / "lower_resolution_val.txt"

# Progress reporting
PRINT_EVERY = 250

# Output CSV
OUT_CSV = RUNS_DIR / f"predicted_counts_{PRED_SPLIT}.csv"


def read_ids(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    if txt.startswith("["):
        arr = json.loads(txt)
        return [str(x) for x in arr]
    ids = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(Path(s).stem)
    return ids


def resize_pad_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = img2
    return out


def preprocess_image(img_path: Path) -> torch.Tensor | None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = resize_pad_square(img, IMG_SIZE)

    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def make_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def sanitize_count(x: float) -> int:
    # no negative colonies; round to integer
    return int(round(max(0.0, x)))


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = make_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def get_stems() -> list[str]:
    if PRED_SPLIT == "val":
        stems = read_ids(VAL_LIST)
    elif PRED_SPLIT == "train":
        stems = read_ids(TRAIN_LIST)
    elif PRED_SPLIT == "all":
        stems = [p.stem for p in PAIRS_DIR.glob("*.jpg")]
    else:
        raise ValueError("PRED_SPLIT must be 'val', 'train', or 'all'")
    return stems


def read_already_done(csv_path: Path) -> set[str]:
    done = set()
    if not csv_path.exists():
        return done
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row:
                    continue
                done.add(row[0])
    except Exception:
        # if file is corrupted, ignore resume and overwrite later
        return set()
    return done


@torch.no_grad()
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}")

    stems = get_stems()
    if not stems:
        raise RuntimeError("No stems found. Check PRED_SPLIT and list files.")

    done = read_already_done(OUT_CSV)
    if done:
        stems = [s for s in stems if s not in done]
        print(f"Resuming: {len(done)} already in CSV, {len(stems)} remaining.")
    else:
        print(f"Starting fresh: {len(stems)} images.")

    model = load_model()
    print("Device:", DEVICE)
    print("Writing to:", OUT_CSV)

    write_header = not OUT_CSV.exists()

    start_time = time.time()
    last_time = start_time
    processed = 0

    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["sample_id", "predicted_count"])

        for stem in stems:
            img_path = PAIRS_DIR / f"{stem}.jpg"
            x = preprocess_image(img_path)
            if x is None:
                continue

            x = x.to(DEVICE)
            pred = float(model(x).item())
            count = sanitize_count(pred)

            w.writerow([stem, count])
            processed += 1

            if processed % PRINT_EVERY == 0:
                now = time.time()
                dt = now - last_time
                total_dt = now - start_time
                ips = PRINT_EVERY / dt if dt > 0 else float("inf")
                print(f"{processed} done | {ips:.2f} img/s | elapsed {total_dt/60:.1f} min")
                last_time = now

    total = time.time() - start_time
    print(f"DONE. Wrote predictions for {processed} images in {total/60:.2f} min.")
    print("CSV:", OUT_CSV)


if __name__ == "__main__":
    main()
