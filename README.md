# Bacteria Counting App (YOLO)

Python tool that counts bacterial colonies in a plate image using a custom-trained YOLO model.

## What’s in this repo
- `predict_single_image.py` — select an image + ROI, run detection, output count + annotated image

## What’s NOT included (by design)
- Trained model weights (`.pt`)
- Training images/labels

These are excluded due to file size and data ownership constraints.

## Run (requires weights)
1) Place your weights here:
`models/yolo_colonies_best.pt`

2) Install dependencies:
```bash
pip install ultralytics opencv-python numpy
