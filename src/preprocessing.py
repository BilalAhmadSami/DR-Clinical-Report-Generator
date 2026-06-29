"""Training-matched fundus preprocessing.

These functions reproduce the exact pipeline used to train the grading model
(auto fundus crop -> pad to square -> resize to 384 -> CLAHE on the LAB
L-channel), so inference sees the same image distribution as training.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.config import (
    IMAGE_SIZE, CROP_MODE, CROP_THRESHOLD, CROP_MARGIN,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
)


def auto_crop_fundus(image_bgr, threshold=CROP_THRESHOLD, margin=CROP_MARGIN):
    """Detect the visible fundus region and crop around it (removes black border)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray_blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_bgr
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    img_h, img_w = image_bgr.shape[:2]
    margin_px = int(max(w, h) * margin)
    x1, y1 = max(0, x - margin_px), max(0, y - margin_px)
    x2, y2 = min(img_w, x + w + margin_px), min(img_h, y + h + margin_px)
    cropped = image_bgr[y1:y2, x1:x2]
    return cropped if cropped.size else image_bgr


def pad_to_square(image_bgr, fill_value=(0, 0, 0)):
    """Pad to a square so resizing does not stretch the round fundus into an oval."""
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(image_bgr, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=fill_value)


def apply_clahe_bgr(image_bgr, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID):
    """Apply CLAHE to the L channel in LAB space (better colour preservation than RGB)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
    l_clahe = clahe.apply(l_channel)
    merged = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_fundus(pil_image: Image.Image) -> Image.Image:
    """Full training pipeline: returns a CLAHE'd, cropped, 384x384 RGB PIL image."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cropped = auto_crop_fundus(bgr) if CROP_MODE == "auto" else bgr
    square = pad_to_square(cropped)
    resized = cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    clahe = apply_clahe_bgr(resized)

    out_rgb = cv2.cvtColor(clahe, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)
