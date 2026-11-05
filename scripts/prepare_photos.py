#!/usr/bin/env python3
"""Prepare product photos for eBay listings.

For every input image:
  1. Apply the camera orientation (EXIF), so the picture is upright.
  2. Find the product against the desk and crop to it with a margin,
     dropping laptops, cables and other clutter near the edges.
  3. Correct white balance and brightness from the background colour,
     and lift contrast slightly.
  4. Resize to at most 1600 px on the long side.
  5. Save as JPEG without ANY metadata (no location, device or timestamp).

Usage:
  python3 prepare_photos.py OUTPUT_DIR IMAGE [IMAGE ...]
  python3 prepare_photos.py OUTPUT_DIR IMAGE --no-crop     # keep the full frame
  python3 prepare_photos.py OUTPUT_DIR IMAGE --rotate 90   # turn clockwise
  python3 prepare_photos.py OUTPUT_DIR IMAGE --box 0.2,0.1,0.9,1.0
      # manual crop as fractions of width/height (left, top, right, bottom),
      # for cluttered shots where the automatic crop keeps too much

HEIC input needs: pip install pillow-heif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # iPhone photos
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

MAX_SIDE = 1600
WORK_SIDE = 640  # analysis resolution
MARGIN = 0.12  # crop margin around the product, relative to its size
TARGET_BG = 222  # background brightness after correction (0..255)
MIN_GAMMA = 0.82  # strongest allowed brightness lift (1.0 = none)
WB_STRENGTH = 0.5  # 0 = keep original colour cast, 1 = fully neutral background
SATURATION = 1.08  # mild saturation boost after the lift


def load_upright(path: Path) -> Image.Image:
    """Open an image and rotate it according to its EXIF orientation."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def background_colour(rgb: np.ndarray) -> np.ndarray:
    """Estimate the desk colour from the brightest, least saturated pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    mask = (sat < np.percentile(sat, 40)) & (val > np.percentile(val, 50))
    if mask.sum() < 100:
        mask = np.ones(sat.shape, bool)
    return np.median(rgb[mask], axis=0)


def product_box(rgb: np.ndarray, bg: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the product in a small working image."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    # Soft shadows differ from the desk mainly in brightness: weight colour more.
    dist_colour = np.linalg.norm(lab[..., 1:] - bg_lab[1:], axis=2)
    score = 0.45 * dist + 0.55 * dist_colour * 2.2
    mask = (score > max(18.0, np.percentile(score, 60))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    cx, cy = w / 2, h / 2
    diag = (w * w + h * h) ** 0.5
    best, best_score = None, -1.0
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < 0.01 * w * h:
            continue
        touches = (x <= 1) + (y <= 1) + (x + bw >= w - 1) + (y + bh >= h - 1)
        centre_dist = ((centroids[i][0] - cx) ** 2 + (centroids[i][1] - cy) ** 2) ** 0.5 / diag
        # Prefer large blobs near the centre; penalise clutter cut off by the frame.
        s = area * (1.0 - centre_dist) ** 2 / (1 + 0.6 * touches)
        if s > best_score:
            best, best_score = i, s
    if best is None:
        return None

    # Every other clear blob that is not cut off by the frame belongs to the
    # product too (several straps, cables, attachments); clutter at the edges
    # (laptops, phones, lamps) is dropped.
    keep = labels == best
    for i in range(1, count):
        if i == best:
            continue
        ix, iy, iw, ih, area = stats[i]
        if area < 0.003 * w * h:
            continue
        touches = (ix <= 1) + (iy <= 1) + (ix + iw >= w - 1) + (iy + ih >= h - 1)
        if touches == 0:
            keep |= labels == i
    ys, xs = np.nonzero(keep)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_product(img: Image.Image) -> Image.Image:
    """Crop to the product with a margin; return the original if nothing is found."""
    scale = WORK_SIDE / max(img.size)
    small = np.asarray(img.resize((round(img.width * scale), round(img.height * scale))))
    bg = background_colour(small)
    box = product_box(small, bg)
    if box is None:
        return img
    x0, y0, x1, y1 = (v / scale for v in box)
    bw, bh = x1 - x0, y1 - y0
    # Ignore boxes that cover almost the whole frame or are implausibly small.
    if bw * bh > 0.85 * img.width * img.height or bw * bh < 0.02 * img.width * img.height:
        return img
    mx, my = bw * MARGIN, bh * MARGIN
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(img.width, x1 + mx), min(img.height, y1 + my)
    # Keep a pleasant aspect ratio between 3:4 and 4:3 by widening the short side.
    w, h = x1 - x0, y1 - y0
    if w / h < 0.75:
        extra = (0.75 * h - w) / 2
        x0, x1 = max(0, x0 - extra), min(img.width, x1 + extra)
    elif w / h > 4 / 3:
        extra = (0.75 * w - h) / 2
        y0, y1 = max(0, y0 - extra), min(img.height, y1 + extra)
    return img.crop((round(x0), round(y0), round(x1), round(y1)))


def correct_colour(img: Image.Image) -> Image.Image:
    """Gentle correction: half-way white balance, a modest brightness lift and a
    small saturation boost. Deliberately conservative so colours stay natural and
    dark products keep their depth instead of turning grey."""
    arr = np.asarray(img).astype(np.float32)
    small = cv2.resize(arr, (0, 0), fx=WORK_SIDE / max(img.size), fy=WORK_SIDE / max(img.size))
    bg = background_colour(small.astype(np.uint8)).astype(np.float32)
    # White balance: move the desk only half-way towards neutral grey.
    gains = 1 + WB_STRENGTH * (bg.mean() / np.maximum(bg, 1) - 1)
    arr *= gains
    # Brightness: lift the desk towards TARGET_BG, but never by much.
    level = max(1.0, float(np.clip(bg.mean() * gains.mean(), 1, 254)))
    gamma = np.log(TARGET_BG / 255) / np.log(level / 255)
    gamma = float(np.clip(gamma, MIN_GAMMA, 1.0))
    arr = 255 * (np.clip(arr, 0, 255) / 255) ** gamma
    # Restore the saturation a brightness lift takes away.
    hsv = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * SATURATION, 0, 255)
    return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


def save_clean(img: Image.Image, out: Path) -> None:
    """Resize and save without EXIF, GPS, ICC or any other metadata."""
    img = img.copy()
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    clean = Image.new("RGB", img.size)
    clean.paste(img)
    clean.save(out, "JPEG", quality=88, optimize=True, progressive=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--no-crop", action="store_true", help="keep the full frame")
    parser.add_argument(
        "--box",
        help="manual crop left,top,right,bottom as fractions 0..1 (applied to every image given)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="rotate clockwise after cropping, for products photographed lying sideways",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        img = load_upright(path)
        if args.box:
            left, top, right, bottom = (float(v) for v in args.box.split(","))
            img = img.crop(
                (round(left * img.width), round(top * img.height), round(right * img.width), round(bottom * img.height))
            )
        elif not args.no_crop:
            img = crop_to_product(img)
        if args.rotate:
            img = img.rotate(-args.rotate, expand=True)
        img = correct_colour(img)
        out = args.output_dir / f"{path.stem}.jpg"
        save_clean(img, out)
        print(f"{path.name} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
