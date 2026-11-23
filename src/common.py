"""
common.py
=========
Shared helper functions used by every part of the project.

Keeping these in one place means the image detector, the video detector and the
experiment runner all behave identically (same model, same device, same record
format). This is good software practice and makes the project easy to explain in
a viva: "all detection goes through one function."

Nothing here invents results. Every number returned (confidence, inference time)
comes directly from the model and from Python's own timer.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

# These heavy imports (torch, ultralytics, cv2) are only needed when we actually
# run detection, so we import them lazily inside functions where helpful. At the
# top level we keep it simple and readable.
import cv2  # OpenCV: reading/writing images and video
import numpy as np


# ---------------------------------------------------------------------------
# Default settings for the whole project.
# Changing a value here changes it everywhere, which keeps experiments fair.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = "yolo11n.pt"   # YOLO11 "nano": smallest, fastest Ultralytics model
DEFAULT_CONF = 0.25              # confidence threshold: ignore detections below this
DEFAULT_IMGSZ = 640              # image size the model runs at (square)


def pick_device() -> str:
    """
    Decide which hardware to run the model on.

    - "mps"  = Apple's Metal Performance Shaders = the GPU inside the M1 chip.
    - "cpu"  = fallback that always works.

    On a MacBook Air M1 there is no NVIDIA/CUDA GPU, so MPS is the fast path.
    We check that PyTorch was built with MPS support AND that it is available.
    """
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        # If torch is missing or MPS check fails for any reason, fall back safely.
        pass
    return "cpu"


def load_model(weights: str = DEFAULT_WEIGHTS):
    """
    Load a pretrained YOLO model.

    The first time this runs, Ultralytics downloads the weights file
    (e.g. yolo11n.pt, a few megabytes) and caches it. After that it loads
    instantly from disk. We are NOT training anything here — we load weights that
    were already trained on the public COCO dataset (80 common object classes).
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    return model


@dataclass
class Detection:
    """One detected object. A plain record so it is easy to save to CSV."""
    source: str          # which image/frame this came from
    class_id: int        # numeric class id from the model
    class_name: str      # human-readable label, e.g. "person"
    confidence: float    # model's confidence for THIS detection (0..1)
    x1: float            # bounding-box corners (pixels)
    y1: float
    x2: float
    y2: float


@dataclass
class DetectionResult:
    """Everything produced by running the model on ONE image/frame."""
    source: str
    annotated_image: np.ndarray                 # BGR image with boxes drawn
    detections: List[Detection] = field(default_factory=list)
    # Timings reported by Ultralytics, in milliseconds, for this single input:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        """Total model time for this input (pre + inference + post), in ms."""
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def count(self) -> int:
        return len(self.detections)


def run_on_image(model, image, source_name: str,
                 conf: float = DEFAULT_CONF,
                 imgsz: int = DEFAULT_IMGSZ,
                 device: str | None = None) -> DetectionResult:
    """
    Run the YOLO model on a single image (a NumPy array already loaded by OpenCV)
    and return a tidy DetectionResult.

    We pass the array (not a path) so this same function works for both still
    images and individual video frames.
    """
    if device is None:
        device = pick_device()

    # verbose=False keeps the console clean; Ultralytics still records timings.
    results = model.predict(source=image, conf=conf, imgsz=imgsz,
                            device=device, verbose=False)
    r = results[0]  # predict returns a list; one image -> one result

    # Build our own list of Detection records from the model's boxes.
    detections: List[Detection] = []
    names = r.names  # dict: {class_id: "class_name"}
    if r.boxes is not None and len(r.boxes) > 0:
        # Move tensors to CPU and convert to plain Python numbers.
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for cid, cf, box in zip(cls_ids, confs, xyxy):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(Detection(
                source=source_name,
                class_id=int(cid),
                class_name=str(names.get(int(cid), str(cid))),
                confidence=float(cf),
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

    # r.plot() draws the boxes+labels and returns a BGR image (ready for cv2).
    annotated = r.plot()

    # r.speed is a dict like {'preprocess':.., 'inference':.., 'postprocess':..} in ms.
    speed = getattr(r, "speed", {}) or {}
    return DetectionResult(
        source=source_name,
        annotated_image=annotated,
        detections=detections,
        preprocess_ms=float(speed.get("preprocess", 0.0)),
        inference_ms=float(speed.get("inference", 0.0)),
        postprocess_ms=float(speed.get("postprocess", 0.0)),
    )


def read_image(path: str) -> np.ndarray:
    """
    Read an image from disk with OpenCV, raising a clear error if it fails.
    OpenCV returns None (not an exception) for an unreadable/missing file, which
    is a classic beginner trap, so we check for it explicitly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(
            f"OpenCV could not read '{path}'. Is it a valid image file?"
        )
    return img


def detections_to_rows(detections: List[Detection]) -> List[Dict[str, Any]]:
    """Convert Detection objects to dicts for pandas/CSV export."""
    return [asdict(d) for d in detections]
