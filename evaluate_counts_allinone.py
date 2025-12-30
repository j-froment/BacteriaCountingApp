from pathlib import Path
import json
import csv
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
DATASET_DIR = Path("raw") / "AGAR_dataset"
PAIRS_DIR   = DATASET_DIR / "dataset_clean"
LISTS_DIR   = DATASET_DIR / "training_lists"

RUNS_DIR    = Path("runs") / "count"
MODEL_PATH  = RUNS_DIR / "best_count_model.pt"

# -------- Choose evaluation split --------
# "val" is what you should report
EVAL_SPLIT = "val"

TRAIN_LIST = LISTS_DIR / "lower_resolution_train.txt"
VAL_LIST   = LISTS_DIR / "lower_resolution_val.txt"

OUT_CSV     = RUNS_DIR / f"eval_{EVAL_SPLIT}_pred_vs_true.csv"
SCATTER_PNG = RUNS_DIR / f"scatter_{EVAL_SPLIT}.png"
RESID_PNG   = RUNS_DIR / f"residuals_{EVAL_SPLIT}.png"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Utilities ----------------
def read_ids(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    if txt.startswith("["):
        return [str(x) for x in json.loads(txt)]
    ids = []
    for line in txt.splitlines():
        s = line.strip()
        if s:
            ids.append(Path(s).stem)
    return ids


def load_true_count(stem: str) -> float | None:
    js = PAIRS_DIR / f"{stem}.json"
    if not js.exists():
        return None
    try:
        d = json.loads(js.read_text(encoding="utf-8"))
    except Exception:
        return None
    return float(d.get("colonies_number")) if "colonies_number" in d else None


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


def preprocess_image(path: Path) -> torch.Tensor | None:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = resize_pad_square(img, IMG_SIZE)

    x = torch.from_numpy(img).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def make_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot


# ---------------- Main ----------------
@torch.no_grad()
def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = make_model().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if EVAL_SPLIT == "val":
        stems = read_ids(VAL_LIST)
    elif EVAL_SPLIT == "train":
        stems = read_ids(TRAIN_LIST)
    else:
        stems = [p.stem for p in PAIRS_DIR.glob("*.jpg")]

    y_true, y_pred = [], []
    rows = []

    for stem in stems:
        img_path = PAIRS_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue

        gt = load_true_count(stem)
        if gt is None:
            continue

        x = preprocess_image(img_path)
        if x is None:
            continue

        pred = float(model(x.to(DEVICE)).item())
        pred = max(0.0, pred)

        y_true.append(gt)
        y_pred.append(pred)
        rows.append((stem, gt, pred, int(round(pred))))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"Evaluated {len(y_true)} samples on '{EVAL_SPLIT}' split")
    print(f"MAE  : {mae(y_true, y_pred):.3f}")
    print(f"RMSE : {rmse(y_true, y_pred):.3f}")
    print(f"R²   : {r2(y_true, y_pred):.3f}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "true_count", "pred_count_float", "pred_count_rounded"])
        w.writerows(rows)

    print("Wrote CSV:", OUT_CSV)

    # Scatter plot
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True colony count")
    plt.ylabel("Predicted colony count")
    plt.title(f"Pred vs True ({EVAL_SPLIT})")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=200)
    plt.close()

    # Residuals
    plt.figure()
    plt.scatter(y_true, y_pred - y_true, s=10)
    plt.axhline(0)
    plt.xlabel("True colony count")
    plt.ylabel("Residual (pred - true)")
    plt.title(f"Residuals ({EVAL_SPLIT})")
    plt.tight_layout()
    plt.savefig(RESID_PNG, dpi=200)
    plt.close()

    print("Saved plots:", SCATTER_PNG, "and", RESID_PNG)


if __name__ == "__main__":
    main()
