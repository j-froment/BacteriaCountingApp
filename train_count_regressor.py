from pathlib import Path
import json
import random
import math
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# -------- Paths (match your layout) --------
DATASET_DIR = Path("raw") / "AGAR_dataset"
PAIRS_DIR   = DATASET_DIR / "dataset_clean"
LISTS_DIR   = DATASET_DIR / "training_lists"
RUNS_DIR    = Path("runs") / "count"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Choose a condition split to train on:
TRAIN_LIST = LISTS_DIR / "lower_resolution_train.txt"
VAL_LIST   = LISTS_DIR / "lower_resolution_val.txt"

IMAGE_EXT = ".jpg"
JSON_EXT  = ".json"

# -------- Training settings --------
IMG_SIZE = 224              # ResNet default; fast on CPU
BATCH = 16                  # if slow or memory issues -> 8
EPOCHS = 10                 # start small; increase later
LR = 1e-3
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_ids(path: Path) -> list[str]:
    """
    Your lists seem to be JSON-list formatted like ["16078", "16831", ...]
    but this also supports one-id-per-line files.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
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


def load_count(stem: str) -> float | None:
    js = PAIRS_DIR / f"{stem}{JSON_EXT}"
    if not js.exists():
        return None
    try:
        d = json.loads(js.read_text(encoding="utf-8"))
    except Exception:
        return None
    # AGAR key from your inspection:
    if "colonies_number" not in d:
        return None
    return float(d["colonies_number"])


def load_image_rgb(stem: str) -> np.ndarray | None:
    img_path = PAIRS_DIR / f"{stem}{IMAGE_EXT}"
    if not img_path.exists():
        return None
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def resize_pad_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect, pad to square, then resize to (size,size)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    out[y0:y0+nh, x0:x0+nw] = img2
    return out


class AgarCountDataset(Dataset):
    def __init__(self, ids: list[str]):
        self.samples: list[tuple[str, float]] = []
        missing = 0
        for stem in ids:
            y = load_count(stem)
            img_path = PAIRS_DIR / f"{stem}{IMAGE_EXT}"
            js_path = PAIRS_DIR / f"{stem}{JSON_EXT}"
            if y is None or not img_path.exists() or not js_path.exists():
                missing += 1
                continue
            self.samples.append((stem, y))
        if missing:
            print(f"[Dataset] Skipped {missing} ids due to missing files or count")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem, y = self.samples[idx]
        img = load_image_rgb(stem)
        if img is None:
            # should be rare; fallback
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = resize_pad_square(img, IMG_SIZE)

        # normalize to float tensor
        x = torch.from_numpy(img).float() / 255.0           # (H,W,C)
        x = x.permute(2, 0, 1).contiguous()                 # (C,H,W)

        # ImageNet normalization (good default for ResNet transfer learning)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        x = (x - mean) / std

        y = torch.tensor([y], dtype=torch.float32)
        return x, y, stem


def make_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 1)  # regression head
    return m


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    for x, y, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += float(loss.item()) * x.size(0)
        total_mae += float(torch.abs(pred - y).mean().item()) * x.size(0)
    n = max(1, len(loader.dataset))
    return total_loss / n, total_mae / n


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_ids = read_ids(TRAIN_LIST)
    val_ids   = read_ids(VAL_LIST)

    # prevent overlap
    val_set = set(val_ids)
    train_ids = [i for i in train_ids]
    val_ids = [i for i in val_ids if i not in set(train_ids)]

    ds_train = AgarCountDataset(train_ids)
    ds_val   = AgarCountDataset(val_ids)

    print("Train samples:", len(ds_train))
    print("Val samples:  ", len(ds_val))
    print("Device:", DEVICE)

    train_loader = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=0)

    model = make_model().to(DEVICE)

    # MAE is often easier to interpret for "count" tasks, but MSE trains smoothly.
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = math.inf
    best_path = RUNS_DIR / "best_count_model.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        va_loss, va_mae = eval_one_epoch(model, val_loader, loss_fn)
        print(f"Epoch {epoch}/{EPOCHS} | train_mse={tr_loss:.4f} | val_mse={va_loss:.4f} | val_mae={va_mae:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model_state": model.state_dict(),
                "img_size": IMG_SIZE
            }, best_path)
            print("  saved:", best_path)

    print("DONE. Best saved at:", best_path)


if __name__ == "__main__":
    main()
