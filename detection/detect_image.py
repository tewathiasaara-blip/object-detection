"""
detect_image.py
===============
Module 4 of the blueprint: run YOLO object detection on one or more images.

For each image it:
  - runs the pretrained YOLO model,
  - draws bounding boxes + labels + confidence scores,
  - saves the annotated image into results/images/,
  - appends every detection (class, confidence, box) to reports/detections.csv.

All outputs are REAL model outputs. Nothing is invented.

Examples:
    python detection/detect_image.py input_images/sample.jpg
    python detection/detect_image.py input_images/*.jpg --conf 0.4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import (  # noqa: E402
    load_model, pick_device, run_on_image, read_image,
    detections_to_rows, DEFAULT_WEIGHTS, DEFAULT_CONF, DEFAULT_IMGSZ,
)

RESULTS_DIR = "results/images"
REPORTS_CSV = "reports/detections.csv"


def expand_inputs(patterns):
    """Turn a list of paths/globs into an actual list of files."""
    files = []
    for p in patterns:
        matched = glob.glob(p)
        if matched:
            files.extend(sorted(matched))
        elif os.path.exists(p):
            files.append(p)
        else:
            print(f"[WARN] No file matched: {p}")
    return files


def detect_images(image_paths, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF,
                 imgsz=DEFAULT_IMGSZ):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORTS_CSV), exist_ok=True)

    device = pick_device()
    print(f"Using device: {device}")
    print(f"Loading model: {weights} ...")
    model = load_model(weights)

    all_rows = []
    for path in image_paths:
        try:
            img = read_image(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {e}")
            continue

        result = run_on_image(model, img, source_name=os.path.basename(path),
                            conf=conf, imgsz=imgsz, device=device)

        # Save the annotated image.
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(RESULTS_DIR, f"{base}_detected.jpg")
        cv2.imwrite(out_path, result.annotated_image)

        # Print a short, honest summary.
        print(f"\n{path}")
        print(f"  detections: {result.count()}  "
              f"| inference: {result.inference_ms:.1f} ms "
              f"| total: {result.total_ms:.1f} ms")
        for d in result.detections:
            print(f"    - {d.class_name:15s} conf={d.confidence:.2f}")
        print(f"  saved -> {out_path}")

        # Collect rows for CSV, adding per-image timing columns.
        for row in detections_to_rows(result.detections):
            row["inference_ms"] = round(result.inference_ms, 2)
            row["total_ms"] = round(result.total_ms, 2)
            row["conf_threshold"] = conf
            row["imgsz"] = imgsz
            row["device"] = device
            row["weights"] = weights
            all_rows.append(row)

    # Write/append the CSV of all detections.
    if all_rows:
        df = pd.DataFrame(all_rows)
        write_header = not os.path.exists(REPORTS_CSV)
        df.to_csv(REPORTS_CSV, mode="a", header=write_header, index=False)
        print(f"\nAppended {len(all_rows)} detection rows to {REPORTS_CSV}")
    else:
        print("\nNo detections were produced for the given images.")

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="YOLO object detection on images.")
    parser.add_argument("images", nargs="+", help="image paths or globs")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="confidence threshold (0..1)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = parser.parse_args()

    files = expand_inputs(args.images)
    if not files:
        print("[ERROR] No valid input images found.")
        sys.exit(1)

    detect_images(files, weights=args.weights, conf=args.conf, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
