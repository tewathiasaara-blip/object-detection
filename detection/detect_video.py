"""
detect_video.py
===============
Module 5 of the blueprint: run YOLO detection on a prerecorded video file.

It reads the video frame by frame, runs detection on each processed frame,
draws the boxes, and writes an annotated output video to results/videos/.
It also reports honest processing statistics (frames processed, average
inference time per frame, and the resulting processing FPS).

IMPORTANT (integrity): the "FPS" reported here is how fast THIS program
processed the file on your machine. Do not describe it as live real-time
camera performance unless you actually measured that separately.

Example:
    python detection/detect_video.py input_videos/sample.mp4
    python detection/detect_video.py input_videos/sample.mp4 --stride 2 --conf 0.3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import (  # noqa: E402
    load_model, pick_device, run_on_image,
    DEFAULT_WEIGHTS, DEFAULT_CONF, DEFAULT_IMGSZ,
)

RESULTS_DIR = "results/videos"


def detect_video(video_path, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF,
                 imgsz=DEFAULT_IMGSZ, stride=1, max_frames=None):
    """
    stride=1 processes every frame; stride=2 processes every 2nd frame, etc.
    On an M1, a higher stride gives a smoother/faster demo on long videos.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = pick_device()
    print(f"Using device: {device}")
    print(f"Loading model: {weights} ...")
    model = load_model(weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] OpenCV could not open the video: {video_path}")
        sys.exit(1)

    # Read source video properties.
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video: {width}x{height}, {in_fps:.1f} fps, {total} frames")

    # Prepare the output writer. Output fps is scaled by stride so playback
    # speed roughly matches the original.
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(RESULTS_DIR, f"{base}_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = max(1.0, in_fps / stride)
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

    frame_idx = 0
    processed = 0
    total_inf_ms = 0.0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video

            if frame_idx % stride == 0:
                result = run_on_image(model, frame,
                                    source_name=f"{base}_frame{frame_idx}",
                                    conf=conf, imgsz=imgsz, device=device)
                annotated = result.annotated_image
                # The model may output a slightly different size; enforce writer size.
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))
                writer.write(annotated)
                total_inf_ms += result.inference_ms
                processed += 1

                if processed % 20 == 0:
                    print(f"  processed {processed} frames...")

                if max_frames and processed >= max_frames:
                    print(f"  reached max_frames={max_frames}, stopping early.")
                    break

            frame_idx += 1
    finally:
        # Always release resources, even if something goes wrong.
        cap.release()
        writer.release()

    elapsed = time.time() - t_start
    avg_inf = (total_inf_ms / processed) if processed else 0.0
    proc_fps = (processed / elapsed) if elapsed > 0 else 0.0

    print("\n--- Video processing summary (genuine measurements) ---")
    print(f"  frames processed     : {processed}")
    print(f"  wall-clock time      : {elapsed:.1f} s")
    print(f"  avg inference / frame: {avg_inf:.1f} ms")
    print(f"  processing speed     : {proc_fps:.2f} FPS (this program, this machine)")
    print(f"  saved annotated video: {out_path}")

    return {
        "frames_processed": processed,
        "elapsed_s": round(elapsed, 2),
        "avg_inference_ms": round(avg_inf, 2),
        "processing_fps": round(proc_fps, 2),
        "output": out_path,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO object detection on a video file.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--stride", type=int, default=1,
                        help="process every Nth frame (speeds up long videos)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many processed frames (for a quick demo)")
    args = parser.parse_args()

    detect_video(args.video, weights=args.weights, conf=args.conf,
                 imgsz=args.imgsz, stride=args.stride, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
