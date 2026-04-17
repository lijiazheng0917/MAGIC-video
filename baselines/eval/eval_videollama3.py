#!/usr/bin/env python3
"""
VideoLLaMA3-7B baseline evaluation on EgoLifeQA / EgoR1 / MM-Lifelong.

Uses transformers directly with video mode input.
For each question: sample frames -> compose temp mp4 -> feed as video.

Usage:
    # EgoLife
    python baselines/eval/eval_videollama3.py \
        --benchmark egolife --num-frames 128 \
        --frame-cache output/metadata/baselines/egolife_videos_64f/frame_cache \
        --output-dir output/eval/baselines/egolife_videollama3_128f

    # MM-Lifelong
    python baselines/eval/eval_videollama3.py \
        --benchmark mmlifelong --num-frames 128 \
        --mml-frames-dir output/metadata/baselines/mml_frames_512f \
        --output-dir output/eval/baselines/mmlifelong_videollama3_128f
"""

import os
import sys
import json
import re
import argparse
import subprocess
import tempfile
import numpy as np
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "EgoLife"
QA_PATH = DATA_ROOT / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
MML_QA_PATH = PROJECT_ROOT / "data" / "MM-Lifelong" / "month" / "val.json"


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
# Frame / video utilities
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


def compose_temp_video(frame_paths: List[str], resize: int = 224) -> Optional[str]:
    """Compose frame JPEGs into a temp 1fps mp4."""
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
        "-vf", f"scale={resize}:{resize}",
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


def select_frame_paths_egolife(all_timestamps, frame_cache_dir, query_ts, num_frames):
    """Select uniform frames before query_time from EgoLife frame cache."""
    before = [ts for ts in all_timestamps if ts <= query_ts]
    if not before:
        return []
    n = len(before)
    if n <= num_frames:
        selected = before
    else:
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
        selected = [before[i] for i in indices]
    paths = [str(frame_cache_dir / f"{ts}.jpg") for ts in selected]
    return [p for p in paths if os.path.exists(p)]


def select_frame_paths_mml(mml_manifest, qid, num_frames):
    """Select uniform frames from MML pre-extracted frames."""
    all_paths = mml_manifest.get(str(qid), [])
    if not all_paths:
        return []
    if len(all_paths) <= num_frames:
        return all_paths
    indices = np.linspace(0, len(all_paths) - 1, num_frames, dtype=int)
    return [all_paths[i] for i in indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B")
    parser.add_argument("--num-frames", type=int, default=128)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-ids", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--benchmark", type=str, default="egolife",
                        choices=["egolife", "mmlifelong"])
    # EgoLife args
    parser.add_argument("--frame-cache", type=str, default=None)
    parser.add_argument("--qa-path", type=str, default=None)
    # MML args
    parser.add_argument("--mml-frames-dir", type=str, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    print("Model loaded")

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
            frame_paths = select_frame_paths_egolife(all_timestamps, frame_cache_dir, query_ts, args.num_frames)
        else:
            frame_paths = select_frame_paths_mml(mml_manifest, qid, args.num_frames)

        if not frame_paths:
            continue

        # Compose temp video
        video_path = compose_temp_video(frame_paths)
        if not video_path:
            print(f"  Warning: failed to compose video for Q{qid}")
            continue

        # Build prompt
        if args.benchmark == "egolife":
            choices = {"A": q["choice_a"], "B": q["choice_b"], "C": q["choice_c"], "D": q["choice_d"]}
            prompt = (
                f"You are watching a first-person egocentric video recorded over several days.\n\n"
                f"Question: {q['question']}\n"
                f"A. {choices['A']}\nB. {choices['B']}\nC. {choices['C']}\nD. {choices['D']}\n\n"
                f"Choose the best answer (A/B/C/D). State your answer as a single letter."
            )
        else:
            prompt = (
                f"You are watching livestream broadcast videos.\n\n"
                f"Question: {q['question']}\n\n"
                f"Answer the question concisely based on what you observe in the video."
            )

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": args.num_frames}},
                {"type": "text", "text": prompt},
            ]},
        ]

        try:
            inputs = processor(conversation=conversation, return_tensors="pt")
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7)
            input_len = inputs["input_ids"].shape[1]
            if output_ids.shape[1] > input_len:
                response = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
            else:
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
