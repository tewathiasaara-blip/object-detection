# Real-Time Object Detection and Classification (Edge AI Demonstration)

An independent academic demonstration of deep-learning object detection using a
**pretrained YOLO model** on a MacBook Air M1. It takes images and prerecorded
videos, detects and labels common objects, draws bounding boxes with confidence
scores, and records genuine measurements for analysis.

> **Integrity note.** This project uses *pretrained* public weights (YOLO11,
> trained on the COCO dataset of 80 common classes). No model was trained from
> scratch. All accuracy observations, timings, and screenshots in the report must
> come from *your own* runs of this code. The code never invents results.

---

## 1. What it does

| Capability | Script | Output |
|---|---|---|
| Inspect & resize an image | `preprocessing/preprocess_images.py` | image info + resized copy |
| Detect objects in images | `detection/detect_image.py` | annotated images + `reports/detections.csv` |
| Detect objects in a video | `detection/detect_video.py` | annotated video + timing stats |
| Controlled experiments | `experiments/run_experiments.py` | `experiments/experiment_log.csv` |
| Graphs from results | `graphs/make_graphs.py` | PNG charts in `graphs/` |
| Web interface (optional) | `app.py` (Streamlit) | browser UI |
| One entry point | `main.py` | menu / sub-commands |

## 2. Technology choices (and why)

- **Python 3.13** in a virtual environment — stable, well-supported wheels for
  PyTorch and Ultralytics on Apple Silicon (Python 3.14 was too new at build time).
- **Ultralytics YOLO11-nano (`yolo11n.pt`)** — the smallest, fastest current YOLO
  model; ideal for CPU/Apple-GPU inference on an M1.
- **Apple MPS** (Metal Performance Shaders) — uses the M1 GPU; falls back to CPU
  automatically if unavailable.
- **OpenCV** for all image/video reading, writing and preprocessing.
- **Pandas + Matplotlib** for honest result tables and graphs.
- **Streamlit** for an optional, polished demo interface.

## 3. Setup (reproducible)

```bash
# 1. Create and activate a virtual environment (Python 3.13)
python3.13 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

The first detection run downloads the YOLO weights file automatically (a few MB).

## 4. How to run

```bash
# Activate the environment first:
source .venv/bin/activate

# Inspect / resize an image
python preprocessing/preprocess_images.py input_images/sample.jpg --resize 640 640

# Detect objects in one or more images
python detection/detect_image.py input_images/sample.jpg

# Detect objects in a video (use --stride 2 to speed up long videos)
python detection/detect_video.py input_videos/sample.mp4 --stride 2

# Run the controlled experiments (low-light & blur are derived from baseline)
python experiments/run_experiments.py --baseline input_images/sample.jpg \
    --multiple input_images/crowd.jpg --sizes input_images/near_far.jpg

# Build graphs from the experiment log
python graphs/make_graphs.py

# Optional web UI
streamlit run app.py
```

Or just use the menu:

```bash
python main.py
```

## 5. Experiments (blueprint conditions)

The experiment runner keeps **weights, confidence threshold and image size
constant** and changes only the input, so comparisons are fair:

| Condition | Source |
|---|---|
| `normal` | your baseline image |
| `low_light` | baseline darkened (brightness × 0.35, documented) |
| `blur` | baseline blurred (GaussianBlur 15×15, documented) |
| `multiple_objects` | a crowded-scene image you provide |
| `different_sizes` | an image with near + far objects you provide |

`mean_confidence` in the log is a **descriptive statistic, not accuracy**.
True precision/recall/mAP require labelled ground truth (not included by default).

## 6. Getting input data (use only permitted sources)

- Personally captured photos/videos from your own phone (recommended — no licence
  issues), or
- Public, openly-licensed images. Record the source and licence of each file in
  `dataset/notes_sources.txt`.

## 7. Project structure

```
preprocessing/   image inspection & resizing
detection/       image & video detectors
experiments/     controlled-condition runner + logs
graphs/          chart generator + PNGs
results/         annotated images and videos (generated)
reports/         detections.csv (generated)
src/common.py    shared model/device/detection helpers
app.py           Streamlit interface
main.py          single entry point
```

## 8. Honest limitations

- Detects only the 80 COCO classes the model already knows.
- Performance varies with lighting, blur, occlusion, and object size.
- M1 (MPS) performance differs from an NVIDIA GPU; measured speeds are specific to
  this machine and setup.
- Confidence is the model's own score, not real-world correctness.
```
