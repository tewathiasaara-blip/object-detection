"""
main.py
=======
A single friendly entry point for the whole project. It wraps the individual
modules so you (or an examiner) can run everything without remembering each
command. The underlying modules can still be run directly if preferred.

Usage:
    python main.py                      # interactive menu
    python main.py image  input_images/sample.jpg
    python main.py video  input_videos/sample.mp4
    python main.py experiments --baseline input_images/street.jpg
    python main.py graphs
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)


def run_image(args):
    from detection.detect_image import detect_images, expand_inputs
    files = expand_inputs(args or [])
    if not files:
        print("Provide at least one image path, e.g.:")
        print("  python main.py image input_images/sample.jpg")
        return
    detect_images(files)


def run_video(args):
    from detection.detect_video import detect_video
    if not args:
        print("Provide a video path, e.g.:")
        print("  python main.py video input_videos/sample.mp4")
        return
    detect_video(args[0])


def run_experiments(args):
    from experiments.run_experiments import run_experiments as _run
    # very small arg parse for the menu path
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--multiple", default=None)
    p.add_argument("--sizes", default=None)
    ns = p.parse_args(args)
    _run(ns.baseline, multiple=ns.multiple, sizes=ns.sizes)


def run_graphs(_args):
    import runpy
    runpy.run_path(os.path.join(HERE, "graphs", "make_graphs.py"), run_name="__main__")


def interactive_menu():
    print("\n=== Object Detection Project ===")
    print("1. Detect objects in an image")
    print("2. Detect objects in a video")
    print("3. Run controlled experiments")
    print("4. Make graphs from experiment log")
    print("5. Quit")
    choice = input("Choose (1-5): ").strip()
    if choice == "1":
        path = input("Image path: ").strip()
        run_image([path])
    elif choice == "2":
        path = input("Video path: ").strip()
        run_video([path])
    elif choice == "3":
        base = input("Baseline image path: ").strip()
        run_experiments(["--baseline", base])
    elif choice == "4":
        run_graphs([])
    else:
        print("Bye.")


def main():
    if len(sys.argv) == 1:
        interactive_menu()
        return

    command = sys.argv[1]
    rest = sys.argv[2:]
    dispatch = {
        "image": run_image,
        "video": run_video,
        "experiments": run_experiments,
        "graphs": run_graphs,
    }
    if command in dispatch:
        dispatch[command](rest)
    else:
        print(f"Unknown command: {command}")
        print("Commands: image | video | experiments | graphs (or no args for menu)")


if __name__ == "__main__":
    main()
