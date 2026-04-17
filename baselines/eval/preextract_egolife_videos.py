#!/usr/bin/env python3
"""
Pre-extract frames from EgoLife clips and compose per-question mp4 videos.

Two-phase approach:
  Phase 1: Extract middle frame from each 30s clip once, save to disk.
  Phase 2: For each question, select frames and compose into 1fps mp4.

Usage:
    python baselines/eval/preextract_egolife_videos.py \
        --num-frames 64 \
        --output-dir output/metadata/baselines/egolife_videos_64f
"""

import json
import os
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "EgoLife"
QA_PATH = DATA_ROOT / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
VIDEO_ROOT = DATA_ROOT / "A1_JAKE"


def parse_timestamp(date: str, time_str: str) -> int:
    day_num = int(date.replace("DAY", ""))
    return day_num * 100000000 + int(time_str)


def _extract_one(args_tuple):
    """Extract middle frame from a single clip. Module-level for multiprocessing."""
    ts, clip_path, cache_dir, max_size = args_tuple
    import decord as _decord
    _decord.bridge.set_bridge("native")
    frame_out = Path(cache_dir) / f"{ts}.jpg"
    if frame_out.exists():
        return ts, str(frame_out)
    try:
        vr = _decord.VideoReader(str(clip_path))
        if len(vr) == 0:
            return ts, None
        mid = len(vr) // 2
        frame = vr[mid].asnumpy()
        img = Image.fromarray(frame)
        w, h = img.size
        scale = min(max_size / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size
        if w % 2:
            img = img.crop((0, 0, w - 1, h))
        if h % 2:
            img = img.crop((0, 0, img.size[0], h - 1))
        img.save(str(frame_out), quality=90)
        return ts, str(frame_out)
    except Exception:
        return ts, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--max-frame-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--qa-path", type=str, default=None)
    args = parser.parse_args()

    import decord
    decord.bridge.set_bridge("native")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_cache_dir = out_dir / "frame_cache"
    frames_cache_dir.mkdir(exist_ok=True)

    # =========================================================
    # Phase 1: Extract middle frame from every 30s clip (once)
    # =========================================================
    print("Phase 1: Extracting frames from all clips...")

    # Build list of all clips with timestamps
    all_clips = []  # (timestamp, clip_path)
    for day_dir in sorted(VIDEO_ROOT.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith("DAY"):
            continue
        day_num = int(day_dir.name.replace("DAY", ""))
        for mp4 in sorted(day_dir.glob("*.mp4")):
            ts_str = mp4.stem.split("_")[-1]
            clip_ts = day_num * 100000000 + int(ts_str)
            all_clips.append((clip_ts, mp4))
    all_clips.sort(key=lambda x: x[0])
    print(f"  Total clips: {len(all_clips)}")

    # Extract frames (skip existing)
    ts_to_frame = {}  # timestamp -> frame_path
    to_extract = [(ts, p) for ts, p in all_clips
                  if not (frames_cache_dir / f"{ts}.jpg").exists()]
    print(f"  Already cached: {len(all_clips) - len(to_extract)}, to extract: {len(to_extract)}")

    from multiprocessing import Pool, cpu_count
    num_workers = min(cpu_count(), 16)
    print(f"  Extracting with {num_workers} workers...")
    extract_args = [(ts, p, frames_cache_dir, args.max_frame_size) for ts, p in to_extract]
    with Pool(num_workers) as pool:
        for ts, fp in tqdm(pool.imap_unordered(_extract_one, extract_args),
                           total=len(extract_args), desc="Extracting frames"):
            pass

    # Build full timestamp -> frame_path mapping
    for ts, _ in all_clips:
        fp = frames_cache_dir / f"{ts}.jpg"
        if fp.exists():
            ts_to_frame[ts] = str(fp)
    print(f"  Cached frames: {len(ts_to_frame)}")

    # =========================================================
    # Phase 2: Compose per-question mp4 videos
    # =========================================================
    print("\nPhase 2: Composing per-question videos...")

    qa_path = args.qa_path or str(QA_PATH)
    with open(qa_path) as f:
        questions = json.load(f)

    # Sorted timestamps for binary search
    sorted_ts = sorted(ts_to_frame.keys())

    manifest = {}
    for q in tqdm(questions, desc="Composing videos"):
        qid = str(q["ID"])
        video_path = out_dir / f"q{qid}.mp4"

        if video_path.exists():
            manifest[qid] = {"video_path": str(video_path)}
            continue

        query_ts = parse_timestamp(q["query_time"]["date"], q["query_time"]["time"])

        # Find all clips before query_time
        before = [ts for ts in sorted_ts if ts <= query_ts]
        if not before:
            continue

        # Uniform sample
        n = len(before)
        if n <= args.num_frames:
            selected_ts = before
        else:
            indices = np.linspace(0, n - 1, args.num_frames, dtype=int)
            selected_ts = [before[i] for i in indices]

        # Get frame paths
        frame_paths = [ts_to_frame[ts] for ts in selected_ts if ts in ts_to_frame]
        if not frame_paths:
            continue

        # Compose mp4 with ffmpeg concat
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for fp in frame_paths:
                f.write(f"file '{fp}'\nduration 1\n")
            list_file = f.name

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", "1",
            str(video_path),
        ], capture_output=True)
        os.unlink(list_file)

        if video_path.exists():
            manifest[qid] = {"video_path": str(video_path)}

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {len(manifest)} videos")


if __name__ == "__main__":
    main()
