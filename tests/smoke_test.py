"""
smoke_test.py
=============
A quick end-to-end check that the environment and detection code work.
It uses the sample image bundled inside the Ultralytics package (no download),
so it can verify the pipeline even before you add your own images.

Run:
    python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("1) Checking PyTorch and device...")
    import torch
    print(f"   torch version      : {torch.__version__}")
    print(f"   MPS available      : {torch.backends.mps.is_available()}")

    from src.common import pick_device, load_model, run_on_image, read_image
    device = pick_device()
    print(f"   selected device    : {device}")

    print("2) Locating a bundled sample image from Ultralytics...")
    import ultralytics
    pkg_dir = os.path.dirname(ultralytics.__file__)
    candidates = [
        os.path.join(pkg_dir, "assets", "bus.jpg"),
        os.path.join(pkg_dir, "assets", "zidane.jpg"),
    ]
    sample = next((c for c in candidates if os.path.exists(c)), None)
    if sample is None:
        print("   No bundled asset found; place an image in input_images/ and")
        print("   run: python detection/detect_image.py input_images/<your>.jpg")
        return 1
    print(f"   using sample       : {sample}")

    print("3) Loading model and running detection (first run downloads weights)...")
    model = load_model("yolo11n.pt")
    img = read_image(sample)
    result = run_on_image(model, img, source_name="smoke_test", device=device)

    print(f"   detections         : {result.count()}")
    for d in result.detections:
        print(f"     - {d.class_name:12s} conf={d.confidence:.2f}")
    print(f"   inference time     : {result.inference_ms:.1f} ms")

    # Save proof output.
    import cv2
    os.makedirs("results/images", exist_ok=True)
    out = "results/images/smoke_test_detected.jpg"
    cv2.imwrite(out, result.annotated_image)
    print(f"   saved annotated    : {out}")

    ok = result.count() > 0
    print("\nSMOKE TEST:", "PASS ✅" if ok else "RAN, but 0 detections (check image)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
