# Bacteria Colony Counting App

Python-based tool for counting bacterial colonies in plate images using a custom-trained YOLO model.

## Overview
This project detects and counts dense bacterial colonies from a single image of a culture plate.  
The workflow allows a user to select an image, define a region of interest (ROI), and automatically
compute a final colony count while saving an annotated output image.

## Features
- Single-image colony counting
- ROI-based detection for improved accuracy
- Custom-trained YOLO model tuned for small, dense colonies
- Annotated image output with bounding boxes
- Configurable confidence and NMS thresholds

## Repository Contents
- `predict_single_image.py` — main script for single-image colony counting
- Supporting scripts for dataset cleaning, evaluation, and training

## Model Weights
Trained model weights are **not stored directly in the repository**.

Download the trained YOLO model from the **Releases** page:
- `yolo_colonies_best.pt`

After downloading, place the file in:
