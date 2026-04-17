#!/usr/bin/env python3
"""
Pre-extract uniformly sampled frames for MM-Lifelong baseline evaluation.

For each video, uniformly samples N frames across the entire duration.
Each question then uses the same set of frames from its video(s).

Usage:
    python baselines/eval/preextract_mml_frames.py \
        --num-frames 64 \
        --output-dir output/metadata/baselines/mml_frames_64f

Output:
    output_dir/
        video_{id}/
            frame_000.jpg ... frame_063.jpg
        manifest.json   # {question_index: [frame_paths]}
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "MM-Lifelong" / "month"
QA_PATH = DATA_ROOT / "val.json"

# Module-level globals for multiprocessing
_OUT_DIR = None
_NUM_FRAMES = None
_MAX_FRAME_SIZE = None


def _extract_video(vid):
    """Extract frames from a single video. Module-level for multiprocessing."""
    import decord as _decord
    _decord.bridge.set_bridge("native")

    video_path = DATA_ROOT / f"{vid}.mp4"
    if not video_path.exists():
        return vid, []

    vid_dir = Path(_OUT_DIR) / f"video_{vid}"
    vid_dir.mkdir(exist_ok=True)

    existing = sorted(vid_dir.glob("frame_*.jpg"))
    if len(existing) >= _NUM_FRAMES:
        return vid, [str(p) for p in existing[:_NUM_FRAMES]]

    vr = _decord.VideoReader(str(video_path))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, _NUM_FRAMES, dtype=int)
    batch = vr.get_batch(indices).asnumpy()

    paths = []
    for i, frame_np in enumerate(batch):
        img = Image.fromarray(frame_np)
        w, h = img.size
        scale = min(_MAX_FRAME_SIZE / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        frame_path = vid_dir / f"frame_{i:04d}.jpg"
        img.save(str(frame_path), quality=80)
        paths.append(str(frame_path))

    del vr, batch
    return vid, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--max-frame-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    import decord
    decord.bridge.set_bridge("native")

    with open(QA_PATH) as f:
        questions = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find all videos referenced by questions
    video_ids = set()
    for q in questions:
        for ci in q.get("clue_interval", []):
            video_ids.add(ci["video_id"])
    video_ids = sorted(video_ids, key=int)
    print(f"Videos to process: {len(video_ids)}")

    # Step 2: Uniform sample frames from each video (parallel)
    global _OUT_DIR, _NUM_FRAMES, _MAX_FRAME_SIZE
    _OUT_DIR = str(out_dir)
    _NUM_FRAMES = args.num_frames
    _MAX_FRAME_SIZE = args.max_frame_size

    from multiprocessing import Pool, cpu_count
    num_workers = min(cpu_count(), len(video_ids))
    print(f"Extracting with {num_workers} parallel workers...")

    video_frames = {}
    with Pool(num_workers) as pool:
        for vid, paths in tqdm(pool.imap_unordered(_extract_video, video_ids),
                               total=len(video_ids), desc="Extracting"):
            video_frames[vid] = paths
            if paths:
                print(f"  video {vid}: {len(paths)} frames")

    # Step 3: Build manifest — map each question to its video's frames
    # Uniform sampling: each question gets ALL frames from its video(s)
    manifest = {}
    for qi, q in enumerate(questions):
        q_frames = []
        vids_used = set()
        for ci in q.get("clue_interval", []):
            vid = ci["video_id"]
            if vid not in vids_used and vid in video_frames:
                q_frames.extend(video_frames[vid])
                vids_used.add(vid)

        # If question spans multiple videos, trim to num_frames
        if len(q_frames) > args.num_frames:
            indices = np.linspace(0, len(q_frames) - 1, args.num_frames, dtype=int)
            q_frames = [q_frames[i] for i in indices]

        manifest[qi] = q_frames

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    total_frames = sum(len(v) for v in manifest.values())
    print(f"\nDone: {len(manifest)} questions, {total_frames} total frame refs")
    print(f"Unique video frames: {sum(len(v) for v in video_frames.values())}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
