"""
preprocess_images.py
====================
Module 2 of the blueprint: image inspection and preprocessing.

What it does (all with OpenCV):
  1. Reads an image and reports its real dimensions (height, width, channels).
  2. Optionally resizes it to a chosen size and saves a *copy* (original is
     never modified -- good scientific practice).
  3. Fails gracefully on an invalid/missing file instead of crashing.

Run it from the project root, e.g.:
    python preprocessing/preprocess_images.py input_images/sample.jpg --resize 640 640
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

# Make the project root importable so we can reuse src/common.py helpers.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import read_image  # noqa: E402


def inspect_image(path: str) -> dict:
    """Read an image and return its basic properties."""
    img = read_image(path)          # raises a clear error if unreadable
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    info = {
        "path": path,
        "width": w,
        "height": h,
        "channels": channels,
        "dtype": str(img.dtype),
    }
    return info, img


def resize_and_save(img, out_path: str, width: int, height: int) -> str:
    """Resize a copy of the image and save it. Returns the output path."""
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, resized)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Inspect and optionally resize an image.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),
                        help="resize to width height and save a copy")
    parser.add_argument("--outdir", default="results/images",
                        help="where to save the resized copy")
    args = parser.parse_args()

    try:
        info, img = inspect_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        # Graceful failure -- exactly the "handle an invalid file" checklist item.
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("Image inspected successfully:")
    for k, v in info.items():
        print(f"  {k:10s}: {v}")

    if args.resize:
        w, h = args.resize
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(args.outdir, f"{base}_resized_{w}x{h}.jpg")
        saved = resize_and_save(img, out_path, w, h)
        print(f"Saved resized copy ({w}x{h}) to: {saved}")
        print("  (Original file was left unchanged.)")


if __name__ == "__main__":
    main()
