"""
make_graphs.py
==============
Module 7 of the blueprint: create graphs from the ACTUAL experiment log.

It reads experiments/experiment_log.csv (produced by run_experiments.py) and
makes two simple, honest bar charts:
    1. number of detections per condition
    2. average inference time (ms) per condition

Charts are only as truthful as the CSV they read. Run the experiments first.

Example:
    python graphs/make_graphs.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")           # no GUI needed; save straight to file
import matplotlib.pyplot as plt
import pandas as pd

LOG_CSV = "experiments/experiment_log.csv"
OUT_DIR = "graphs"


def main():
    if not os.path.exists(LOG_CSV):
        print(f"[ERROR] {LOG_CSV} not found. Run experiments/run_experiments.py first.")
        sys.exit(1)

    df = pd.read_csv(LOG_CSV)
    if df.empty:
        print("[ERROR] Experiment log is empty.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Chart 1: detections per condition.
    plt.figure(figsize=(8, 5))
    plt.bar(df["condition"], df["num_detections"], color="#2a6fb0")
    plt.title("Number of detections per condition")
    plt.xlabel("Condition")
    plt.ylabel("Detections")
    plt.xticks(rotation=20)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "detections_per_condition.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Saved: {p1}")

    # Chart 2: inference time per condition.
    plt.figure(figsize=(8, 5))
    plt.bar(df["condition"], df["inference_ms"], color="#c0622a")
    plt.title("Average inference time per condition")
    plt.xlabel("Condition")
    plt.ylabel("Inference time (ms)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "inference_time_per_condition.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Saved: {p2}")

    print("\nGraphs created from your real experiment_log.csv.")


if __name__ == "__main__":
    main()
