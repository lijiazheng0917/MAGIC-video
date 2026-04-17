#!/usr/bin/env python3
"""
InternVideo2.5-Chat-8B baseline evaluation on EgoLifeQA / EgoR1 / MM-Lifelong.

Based on: https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B

Usage:
    python baselines/eval/eval_internvideo.py \
        --benchmark egolife --num-frames 512 \
        --frame-cache output/metadata/baselines/egolife_videos_64f/frame_cache \
        --output-dir output/eval/baselines/egolife_internvideo_512f
"""

import os
import json
import re
import argparse
import subprocess
import tempfile
import numpy as np
from typing import Optional, List
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QA_PATH = PROJECT_ROOT / "data" / "EgoLife" / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
MML_QA_PATH = PROJECT_ROOT / "data" / "MM-Lifelong" / "month" / "val.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    if re.fullmatch(r"[A-Da-d]", stripped):
        return stripped.upper()
    m = re.match(r"\(?([A-Da-d])(?:[\.\)\:,]|\s|$)", stripped)
    if m:
        return m.group(1).upper()
    for pat in [
        r'"answer"\s*:\s*"([A-Da-d])"',
        r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-Da-d])\)?",
        r"(?:final\s+)?answer\s*:\s*\(?([A-Da-d])\)?",
        r"(?:I\s+(?:would\s+)?(?:choose|pick|select|go\s+with))\s+\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s+is\s+(?:the\s+)?(?:correct|best|most\s+supported)\b",
    ]:
        m = re.search(pat, stripped, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    all_letters = list(re.finditer(r"\b([A-Da-d])\b", stripped))
    if all_letters:
        cutoff = int(len(stripped) * 0.8)
        tail = [m for m in all_letters if m.start() >= cutoff]
        if tail:
            return tail[-1].group(1).upper()
    return None


# ---------------------------------------------------------------------------
# InternVideo2.5 video processing (from HuggingFace model card)
# ---------------------------------------------------------------------------
def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = (1, 1)
    best_diff = float("inf")
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
            best_diff = diff
            best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_video(video_path, num_segments=128, max_num=1, input_size=448):
    """Load video and return pixel_values + num_patches_list."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    frame_indices = np.linspace(0, max_frame, num_segments, dtype=int)

    transform = build_transform(input_size)
    pixel_values_list, num_patches_list = [], []

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in img_tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------
def parse_timestamp(date: str, time_str: str) -> int:
    day_num = int(date.replace("DAY", ""))
    return day_num * 100000000 + int(time_str)


def get_sorted_timestamps(frame_cache_dir: Path) -> List[int]:
    timestamps = []
    for jpg in frame_cache_dir.glob("*.jpg"):
        try:
            timestamps.append(int(jpg.stem))
        except ValueError:
            pass
    timestamps.sort()
    return timestamps


def compose_temp_video(frame_paths: List[str]) -> Optional[str]:
    """Compose frame JPEGs into a temp 1fps mp4 (no resize, model handles it)."""
    if not frame_paths:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    list_file = tmp.name + ".txt"
    with open(list_file, "w") as f:
        for fp in frame_paths:
            f.write(f"file '{os.path.abspath(fp)}'\nduration 1\n")

    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "1",
        tmp.name,
    ], capture_output=True)

    os.unlink(list_file)

    if result.returncode == 0 and os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
        return tmp.name
    else:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVideo2_5_Chat_8B")
    parser.add_argument("--num-frames", type=int, default=512)
    parser.add_argument("--input-size", type=int, default=448,
                        help="Vision input size per frame (448=default, 224=compact)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-ids", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--benchmark", type=str, default="egolife",
                        choices=["egolife", "mmlifelong"])
    parser.add_argument("--frame-cache", type=str, default=None)
    parser.add_argument("--qa-path", type=str, default=None)
    parser.add_argument("--mml-frames-dir", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).half().cuda().to(torch.bfloat16)
    model.eval()
    print("Model loaded")

    generation_config = dict(
        do_sample=True,
        temperature=0.7,
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
    )

    # Load data
    if args.benchmark == "egolife":
        frame_cache_dir = Path(args.frame_cache)
        all_timestamps = get_sorted_timestamps(frame_cache_dir)
        print(f"Frame cache: {len(all_timestamps)} frames")

        qa_path = args.qa_path or str(QA_PATH)
        with open(qa_path) as f:
            all_questions = json.load(f)
        if args.eval_ids:
            target_ids = set(args.eval_ids.split(","))
            questions = [(str(q["ID"]), q) for q in all_questions if str(q["ID"]) in target_ids]
        else:
            questions = [(str(q["ID"]), q) for q in all_questions]

    elif args.benchmark == "mmlifelong":
        with open(MML_QA_PATH) as f:
            all_questions = json.load(f)
        mml_manifest = {}
        if args.mml_frames_dir:
            mp = os.path.join(args.mml_frames_dir, "manifest.json")
            if os.path.exists(mp):
                with open(mp) as f:
                    mml_manifest = json.load(f)
                print(f"MML frames: {len(mml_manifest)} questions")
        if args.eval_ids:
            target_ids = set(args.eval_ids.split(","))
            questions = [(str(i), all_questions[i]) for i in range(len(all_questions)) if str(i) in target_ids]
        else:
            questions = [(str(i), all_questions[i]) for i in range(len(all_questions))]

    print(f"Benchmark: {args.benchmark}, Questions: {len(questions)}, Frames: {args.num_frames}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")

    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for r in json.load(f):
                existing[str(r["id"])] = r
        print(f"Resuming: {len(existing)} existing")

    results = list(existing.values())

    for qid, q in tqdm(questions, desc="Evaluating"):
        if qid in existing:
            continue

        # Get frame paths
        if args.benchmark == "egolife":
            query_ts = parse_timestamp(q["query_time"]["date"], q["query_time"]["time"])
            before = [ts for ts in all_timestamps if ts <= query_ts]
            if not before:
                continue
            n = len(before)
            if n <= args.num_frames:
                selected = before
            else:
                indices = np.linspace(0, n - 1, args.num_frames, dtype=int)
                selected = [before[i] for i in indices]
            frame_paths = [str(frame_cache_dir / f"{ts}.jpg") for ts in selected]
            frame_paths = [p for p in frame_paths if os.path.exists(p)]
        else:
            all_fp = mml_manifest.get(str(qid), [])
            if len(all_fp) <= args.num_frames:
                frame_paths = all_fp
            else:
                indices = np.linspace(0, len(all_fp) - 1, args.num_frames, dtype=int)
                frame_paths = [all_fp[i] for i in indices]

        if not frame_paths:
            continue

        # Compose temp video
        video_path = compose_temp_video(frame_paths)
        if not video_path:
            print(f"  Warning: failed to compose video for Q{qid}")
            continue

        try:
            # Load video with InternVideo2.5's processing
            pixel_values, num_patches_list = load_video(
                video_path, num_segments=args.num_frames, max_num=1,
                input_size=args.input_size,
            )
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

            # Build prompt
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

            if args.benchmark == "egolife":
                choices = {"A": q["choice_a"], "B": q["choice_b"], "C": q["choice_c"], "D": q["choice_d"]}
                question_text = (
                    f"You are watching a first-person egocentric video recorded over several days.\n\n"
                    f"Question: {q['question']}\n"
                    f"A. {choices['A']}\nB. {choices['B']}\nC. {choices['C']}\nD. {choices['D']}\n\n"
                    f"Choose the best answer (A/B/C/D). State your answer as a single letter."
                )
            else:
                question_text = (
                    f"You are watching livestream broadcast videos.\n\n"
                    f"Question: {q['question']}\n\n"
                    f"Answer the question concisely based on what you observe in the video."
                )

            full_question = video_prefix + question_text

            with torch.no_grad():
                response, _ = model.chat(
                    tokenizer, pixel_values, full_question,
                    generation_config, num_patches_list=num_patches_list,
                    history=None, return_history=True,
                )
        except Exception as e:
            print(f"  Error on Q{qid}: {e}")
            response = ""
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

        # Build result
        if args.benchmark == "egolife":
            predicted = extract_choice_letter(response)
            correct = predicted == q["answer"] if predicted else False
            result = {"id": qid, "type": q["type"], "question": q["question"],
                      "gold": q["answer"], "predicted": predicted, "correct": correct,
                      "response": response, "num_frames": len(frame_paths)}
        else:
            result = {"id": qid, "question_type": q["question_type"], "question": q["question"],
                      "gold": q["answer"], "gold_answer": q["answer"],
                      "response": response, "num_frames": len(frame_paths)}

        results.append(result)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Stats
    results.sort(key=lambda r: int(r["id"]))
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    if args.benchmark == "egolife":
        correct = sum(1 for r in results if r.get("correct"))
        no_ans = sum(1 for r in results if r.get("predicted") is None)
        print(f"\nResults: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"No answer: {no_ans}")
        from collections import defaultdict
        by_type = defaultdict(lambda: {"c": 0, "t": 0})
        for r in results:
            by_type[r["type"]]["t"] += 1
            if r.get("correct"):
                by_type[r["type"]]["c"] += 1
        for t, v in sorted(by_type.items()):
            print(f"  {t:20s}: {v['c']:3d}/{v['t']:3d} = {v['c']/v['t']*100:.1f}%")
    else:
        empty = sum(1 for r in results if not r.get("response"))
        print(f"\nResults: {total} questions, {empty} empty responses")
        print("Run rejudge to get scores.")


if __name__ == "__main__":
    main()
