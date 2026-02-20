#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import shutil
from multiprocessing import Pool, cpu_count
import re

BASE_DIR = Path("/home/alex/Physi/PhysiCell/data")

# Check converter
if shutil.which("rsvg-convert"):
    converter = "rsvg-convert"
elif shutil.which("inkscape"):
    converter = "inkscape"
else:
    raise RuntimeError("Install rsvg-convert or inkscape to convert SVG to PNG.")


def get_snapshot_number(f: Path):
    """Extract the number from snapshot filenames for proper ordering."""
    match = re.search(r"snapshot0*(\d+)\.svg", f.name)
    return int(match.group(1)) if match else 0


def process_run(run_dir: Path):
    """Convert snapshots in a run folder to video and cleanup."""
    svg_files = list(run_dir.glob("snapshot*.svg"))
    if not svg_files:
        return

    # Sort numerically
    svg_files.sort(key=get_snapshot_number)

    tmp_dir = run_dir / "tmp_png"
    tmp_dir.mkdir(exist_ok=True)

    # Convert SVGs to PNG
    for i, svg_file in enumerate(svg_files):
        png_file = tmp_dir / f"frame_{i:05d}.png"
        if converter == "rsvg-convert":
            subprocess.run(
                ["rsvg-convert", "-o", str(png_file), str(svg_file)], check=True
            )
        else:
            subprocess.run(
                ["inkscape", str(svg_file), "--export-filename", str(png_file)],
                check=True,
            )

    # Build video
    video_file = run_dir / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "10",
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_file),
        ],
        check=True,
    )

    # Cleanup
    for svg_file in svg_files:
        svg_file.unlink()
    for png_file in tmp_dir.glob("*.png"):
        png_file.unlink()
    tmp_dir.rmdir()

    print(f"Processed {run_dir}: video created at {video_file}")


def find_all_runs(base_dir: Path):
    """Recursively find all run_* folders under train/test/episodes for async folders."""
    all_runs = []
    for async_folder in base_dir.iterdir():
        if (
            "async_sac_tip_train_test" not in async_folder.name
            or not async_folder.is_dir()
        ):
            continue

        # Look for all env*/train/test/episodes/run_*
        for env_dir in async_folder.glob("env*"):
            for data_type in ["train", "test"]:
                episodes_dir = env_dir / data_type / "episodes"
                if not episodes_dir.exists():
                    continue
                run_dirs = [d for d in episodes_dir.glob("run_*") if d.is_dir()]
                all_runs.extend(run_dirs)
    return all_runs


def main():
    runs = find_all_runs(BASE_DIR)
    print(f"Found {len(runs)} runs to process.")

    # Multiprocessing
    with Pool(cpu_count()) as pool:
        pool.map(process_run, runs)

    print("All done!")


if __name__ == "__main__":
    main()
