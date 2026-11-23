"""
run_experiments.py
==================
Module 6 of the blueprint: run the SAME model under different visual conditions
and log genuine observations so they can be compared fairly.

Conditions (from the blueprint):
    1. normal          -> your baseline image, unchanged
    2. low_light       -> baseline darkened by a documented factor
    3. blur            -> baseline blurred with a documented kernel
    4. multiple_objects-> a separate image you provide (crowded scene)
    5. different_sizes -> a separate image you provide (near + far objects)

Conditions 2 and 3 are DERIVED from the baseline using a known transformation.
This is honest and reproducible: we state exactly how the image was altered.
Conditions 4 and 5 depend on picture content, so you supply those images.

For fairness, every condition uses the SAME weights, confidence threshold and
image size. Only the input image changes.

Output:
    - annotated images in results/images/
    - the derived (low_light/blur) inputs in experiments/derived/
    - a row per condition appended to experiments/experiment_log.csv

Example:
    python experiments/run_experiments.py \
        --baseline input_images/street.jpg \
        --multiple input_images/crowd.jpg \
        --sizes input_images/near_far.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import (  # noqa: E402
    load_model, pick_device, run_on_image, read_image,
    DEFAULT_WEIGHTS, DEFAULT_CONF, DEFAULT_IMGSZ,
)

DERIVED_DIR = "experiments/derived"
RESULTS_DIR = "results/images"
LOG_CSV = "experiments/experiment_log.csv"

# Documented transformation settings (record these in your report).
LOW_LIGHT_FACTOR = 0.35          # multiply pixel brightness by this (<1 = darker)
BLUR_KERNEL = (15, 15)           # Gaussian blur kernel size (odd numbers)


def make_low_light(img: np.ndarray, factor: float = LOW_LIGHT_FACTOR) -> np.ndarray:
    """Darken an image by scaling brightness. factor<1 makes it darker."""
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def make_blur(img: np.ndarray, kernel=BLUR_KERNEL) -> np.ndarray:
    """Apply Gaussian blur with a documented kernel size."""
    return cv2.GaussianBlur(img, kernel, 0)


def _mean_conf(detections) -> float:
    """Average confidence of detections (descriptive only, NOT accuracy)."""
    if not detections:
        return 0.0
    return float(np.mean([d.confidence for d in detections]))


def build_conditions(baseline, multiple, sizes):
    """
    Return a list of (condition_name, image_array, note) tuples.
    Derives low_light and blur from the baseline and saves those inputs.
    """
    os.makedirs(DERIVED_DIR, exist_ok=True)
    conditions = []

    base_img = read_image(baseline)
    base_name = os.path.splitext(os.path.basename(baseline))[0]
    conditions.append(("normal", base_img, f"baseline: {baseline}"))

    low = make_low_light(base_img)
    low_path = os.path.join(DERIVED_DIR, f"{base_name}_lowlight.jpg")
    cv2.imwrite(low_path, low)
    conditions.append(("low_light", low,
                    f"derived from baseline, brightness x{LOW_LIGHT_FACTOR}"))

    blurred = make_blur(base_img)
    blur_path = os.path.join(DERIVED_DIR, f"{base_name}_blur.jpg")
    cv2.imwrite(blur_path, blurred)
    conditions.append(("blur", blurred,
                    f"derived from baseline, GaussianBlur{BLUR_KERNEL}"))

    if multiple:
        conditions.append(("multiple_objects", read_image(multiple),
                        f"user image: {multiple}"))
    if sizes:
        conditions.append(("different_sizes", read_image(sizes),
                        f"user image: {sizes}"))

    return conditions


def run_experiments(baseline, multiple=None, sizes=None,
                    weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = pick_device()
    print(f"Using device: {device} | conf={conf} | imgsz={imgsz} | weights={weights}")
    model = load_model(weights)

    conditions = build_conditions(baseline, multiple, sizes)
    rows = []

    for name, img, note in conditions:
        result = run_on_image(model, img, source_name=name,
                            conf=conf, imgsz=imgsz, device=device)

        out_path = os.path.join(RESULTS_DIR, f"experiment_{name}_detected.jpg")
        cv2.imwrite(out_path, result.annotated_image)

        # Tally detections per class for a readable summary.
        classes = sorted({d.class_name for d in result.detections})
        row = {
            "condition": name,
            "note": note,
            "num_detections": result.count(),
            "mean_confidence": round(_mean_conf(result.detections), 3),
            "classes_detected": ", ".join(classes) if classes else "(none)",
            "inference_ms": round(result.inference_ms, 2),
            "total_ms": round(result.total_ms, 2),
            "conf_threshold": conf,
            "imgsz": imgsz,
            "device": device,
            "weights": weights,
            "annotated_output": out_path,
        }
        rows.append(row)

        print(f"\n[{name}]  {note}")
        print(f"  detections={row['num_detections']} "
              f"mean_conf={row['mean_confidence']} "
              f"inference={row['inference_ms']}ms")
        print(f"  classes: {row['classes_detected']}")
        print(f"  saved -> {out_path}")

    df = pd.DataFrame(rows)
    df.to_csv(LOG_CSV, index=False)
    print(f"\nExperiment log written to: {LOG_CSV}")
    print("Remember: 'mean_confidence' is a descriptive statistic, NOT accuracy.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run controlled detection experiments.")
    parser.add_argument("--baseline", required=True,
                        help="baseline image (normal lighting, common objects)")
    parser.add_argument("--multiple", default=None,
                        help="optional: crowded/multiple-objects image")
    parser.add_argument("--sizes", default=None,
                        help="optional: image with near + far (different-size) objects")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = parser.parse_args()

    run_experiments(args.baseline, multiple=args.multiple, sizes=args.sizes,
                    weights=args.weights, conf=args.conf, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
