from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

#trained model hashtag yolo 
MODEL_PATH = Path("models/yolo_colonies_best.pt")  

IMGSZ = 1024
CONF = 0.20
IOU  = 0.30
MAX_DET = 4000
AUGMENT = True

#window stuff
MAX_W, MAX_H = 1200, 800


def pick_image() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    fp = filedialog.askopenfilename(
        title="Pick a plate image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return fp if fp else None


def scale_for_preview(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def manual_roi_select(img_bgr: np.ndarray):
    """
    User drags rectangle around plate. (The popup you like.)
    Returns (x, y, w, h) in original image coords, or None.
    """
    show = scale_for_preview(img_bgr)
    scale = show.shape[1] / img_bgr.shape[1]

    roi = cv2.selectROI("Draw ROI around the plate, press ENTER", show, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Draw ROI around the plate, press ENTER")

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None

    inv = 1.0 / scale
    X = int(x * inv)
    Y = int(y * inv)
    W = int(w * inv)
    H = int(h * inv)

    H0, W0 = img_bgr.shape[:2]
    X = max(0, min(X, W0 - 1))
    Y = max(0, min(Y, H0 - 1))
    W = max(1, min(W, W0 - X))
    H = max(1, min(H, H0 - Y))
    return (X, Y, W, H)


def point_in_box(px, py, b):
    x1, y1, x2, y2 = b
    return x1 <= px <= x2 and y1 <= py <= y2


def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return x1, y1, x2, y2


def draw_text_with_bg(img, lines, font_scale, thickness, pad=10):
    """
    Draw a black bar across top and put readable text on it.
    Auto sizes bar height based on text.
    """
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    line_h = max(h for (w, h) in sizes) if sizes else 20
    bar_h = pad * 2 + line_h * len(lines) + int(0.35 * line_h) * (len(lines) - 1)
    bar_h = min(bar_h, H // 3)
    cv2.rectangle(img, (0, 0), (W, bar_h), (0, 0, 0), -1)

    y = pad + line_h
    for s in lines:
        
        cv2.putText(img, s, (pad, y), font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(img, s, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h + int(0.35 * line_h)
    return bar_h


def edit_boxes_fullres(img_bgr: np.ndarray, boxes_xyxy: list[tuple[float, float, float, float]]):
    """
      - Click a box to toggle ON/OFF
      - A = add box mode (drag mouse to add)
      - R = reset to original
      - + / - = bigger/smaller text
      - ENTER = accept
      - ESC = cancel
    """
    boxes = boxes_xyxy[:]
    orig_boxes = boxes_xyxy[:]
    active = [True] * len(boxes)

    win = "Edit detections"

    add_mode = False
    dragging = False
    sx = sy = ex = ey = 0


    base_font_scale = 0.9
    font_scale = base_font_scale
    thickness = 2

    H0, W0 = img_bgr.shape[:2]

    def mouse(event, x, y, *_):
        nonlocal add_mode, dragging, sx, sy, ex, ey

        if add_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                sx, sy = x, y
                ex, ey = x, y
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                ex, ey = x, y
            elif event == cv2.EVENT_LBUTTONUP and dragging:
                dragging = False
                ex, ey = x, y
                x1, y1, x2, y2 = clamp_box(sx, sy, ex, ey, W0, H0)
                if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    active.append(True)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                for i in range(len(boxes) - 1, -1, -1):
                    if point_in_box(x, y, boxes[i]):
                        active[i] = not active[i]
                        break

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # good size don't change
    start_w, start_h = 1400, 900
    cv2.resizeWindow(win, start_w, start_h)

    cv2.setMouseCallback(win, mouse)

    while True:
        draw = img_bgr.copy()

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = (0, 255, 0) if active[i] else (0, 0, 255)
            cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if add_mode:
            hint_color = (0, 255, 255)
            if dragging:
                x1, y1, x2, y2 = clamp_box(sx, sy, ex, ey, W0, H0)
                cv2.rectangle(draw, (x1, y1), (x2, y2), hint_color, 2)

        
        font_scale = max(0.9, min(font_scale, 2.2))
        thickness = 2 if font_scale < 1.6 else 3

        lines = [
            ("ADD MODE: drag to add box | Click boxes toggles OFF/ON"
             if add_mode else
             "Click=toggle | A=add mode | R=reset | +/- text | ENTER=done | ESC=cancel"),
            f"Count(active): {sum(active)}   Total boxes: {len(active)}"
        ]
        draw_text_with_bg(draw, lines, font_scale, thickness, pad=12)

        cv2.imshow(win, draw)
        k = cv2.waitKey(20) & 0xFF

        if k in (13, 10):  # ENTER
            cv2.destroyAllWindows()
            return boxes, active

        if k == 27:  # ESC
            cv2.destroyAllWindows()
            return None, None

        if k in (ord("a"), ord("A")):
            add_mode = not add_mode
            dragging = False

        if k in (ord("r"), ord("R")):
            boxes = orig_boxes[:]
            active = [True] * len(boxes)
            add_mode = False
            dragging = False

        # text size changing for accessibility
        if k in (ord("+"), ord("=")):
            font_scale = min(2.2, font_scale + 0.1)
        if k in (ord("-"), ord("_")):
            font_scale = max(0.8, font_scale - 0.1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    img_path = pick_image()
    if not img_path:
        print("No image selected. Exiting.")
        return

    im_orig = cv2.imread(img_path)
    if im_orig is None:
        raise RuntimeError(f"Could not read image: {img_path}")


    roi = manual_roi_select(im_orig)
    if roi is None:
        print("No ROI selected. Exiting.")
        return
    x, y, w, h = roi
    crop = im_orig[y:y+h, x:x+w].copy()

    # MODEL PREDICT
    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=crop,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        augment=AUGMENT,
        agnostic_nms=True,
        save=False,
        verbose=False
    )
    r0 = results[0]

   
    boxes = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes.xyxy.cpu().numpy().tolist():
            x1, y1, x2, y2 = map(float, b)
            boxes.append((x1 + x, y1 + y, x2 + x, y2 + y))

    print("\nYOLO raw detections:", len(boxes))

  
    edited_boxes, active = edit_boxes_fullres(im_orig, boxes)
    if edited_boxes is None:
        print("Canceled edit. Using raw detections.")
        edited_boxes, active = boxes, [True] * len(boxes)

    kept = [(b, a) for b, a in zip(edited_boxes, active) if a]
    final_count = len(kept)

    out_img = im_orig.copy()
    for (x1, y1, x2, y2), _ in kept:
        cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.rectangle(out_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    stem = Path(img_path).stem
    out_path = out_dir / f"{stem}_final.jpg"
    cv2.imwrite(str(out_path), out_img)

    print("\n------------------------------")
    print("Image:", img_path)
    print("FINAL colony count:", final_count)
    print("Saved annotated image:", out_path)
    print("------------------------------\n")

    # FINAL DISPLAY STUFF
    win = f"FINAL ({final_count})"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1400, 900)     
    cv2.imshow(win, out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#;)