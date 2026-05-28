#!/usr/bin/env python3
"""
VLM baseline evaluation on EgoLifeQA via OpenRouter API.

Usage:
    python baselines/eval_vlm_api.py \
        --model qwen/qwen3.5-9b \
        --num-frames 32 \
        --output-dir output/eval/baselines/qwen35_9b_32f \
        --eval-ids 1,2,3

Requires: OPENROUTER_API_KEY environment variable.
"""

import os
import sys
import json
import re
import base64
import argparse
import time
from io import BytesIO
from typing import Optional, Dict, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("pip install pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # WorldMM_jz/
DATA_ROOT = PROJECT_ROOT / "data" / "EgoLife"
QA_PATH = DATA_ROOT / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
VIDEO_ROOT = DATA_ROOT / "A1_JAKE"
CAPTION_PATH = DATA_ROOT / "EgoLifeCap" / "A1_JAKE" / "A1_JAKE_30sec.json"
SUBJECT = "A1_JAKE"


# ---------------------------------------------------------------------------
# Answer extraction (copied from eval/eval.py for independence)
# ---------------------------------------------------------------------------
def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()

    # 1. Single letter
    if re.fullmatch(r"[A-Da-d]", stripped):
        return stripped.upper()

    # 2. Starts with letter pattern
    m = re.match(r"\(?([A-Da-d])(?:[\.\)\:,]|\s|$)", stripped)
    if m:
        return m.group(1).upper()

    # 3. Explicit answer phrases
    for pat in [
        r'"answer"\s*:\s*"([A-Da-d])"',  # JSON format: "answer": "C"
        r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-Da-d])\)?",
        r"(?:final\s+)?answer\s*:\s*\(?([A-Da-d])\)?",
        r"(?:I\s+(?:would\s+)?(?:choose|pick|select|go\s+with))\s+\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s+is\s+(?:the\s+)?(?:correct|best|most\s+supported)\b",
    ]:
        m = re.search(pat, stripped, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 4. Last standalone letter in tail 20%
    all_letters = list(re.finditer(r"\b([A-Da-d])\b", stripped))
    if all_letters:
        cutoff = int(len(stripped) * 0.8)
        tail = [m for m in all_letters if m.start() >= cutoff]
        if tail:
            return tail[-1].group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Timestamp / video utilities
# ---------------------------------------------------------------------------
def parse_timestamp(date: str, time_str: str) -> int:
    """Convert date+time to comparable integer DHHMMSSFF."""
    day_num = int(date.replace("DAY", ""))
    return day_num * 100000000 + int(time_str)


def get_all_clips_before(query_ts: int) -> List[Path]:
    """Get all 30s video clips before query_time, sorted by timestamp."""
    clips = []
    for day_dir in sorted(VIDEO_ROOT.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith("DAY"):
            continue
        day_num = int(day_dir.name.replace("DAY", ""))
        for mp4 in sorted(day_dir.glob("*.mp4")):
            # Parse timestamp from filename: DAY1_A1_JAKE_11094208.mp4
            ts_str = mp4.stem.split("_")[-1]
            clip_ts = day_num * 100000000 + int(ts_str)
            if clip_ts <= query_ts:
                clips.append((clip_ts, mp4))
    clips.sort(key=lambda x: x[0])
    return [c[1] for c in clips]


def extract_middle_frame(video_path: Path, max_size: int = 512) -> Optional[str]:
    """Extract middle frame from a 30s clip, return as base64 JPEG."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(video_path))
        if len(vr) == 0:
            return None
        mid = len(vr) // 2
        frame = vr[mid].asnumpy()
        img = Image.fromarray(frame)
        # Resize to save tokens
        w, h = img.size
        scale = min(max_size / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  Warning: failed to extract frame from {video_path.name}: {e}")
        return None


def sample_frames_uniform(query_ts: int, num_frames: int, max_size: int = 512) -> List[Dict]:
    """Uniformly sample frames from all clips before query_time.

    Returns list of image_url content dicts (one per frame).
    """
    clips = get_all_clips_before(query_ts)
    if not clips:
        return []

    # Uniform sampling
    n = len(clips)
    if n <= num_frames:
        selected = clips
    else:
        import numpy as np
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
        selected = [clips[i] for i in indices]

    frames = []
    for clip_path in selected:
        b64 = extract_middle_frame(clip_path, max_size=max_size)
        if b64:
            frames.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            })
    return frames


def get_captions_for_clips(query_ts: int, num_frames: int) -> str:
    """Get captions that correspond to the same clips selected by sample_frames_uniform."""
    clips_with_ts = []
    for day_dir in sorted(VIDEO_ROOT.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith("DAY"):
            continue
        day_num = int(day_dir.name.replace("DAY", ""))
        for mp4 in sorted(day_dir.glob("*.mp4")):
            ts_str = mp4.stem.split("_")[-1]
            clip_ts = day_num * 100000000 + int(ts_str)
            if clip_ts <= query_ts:
                clips_with_ts.append(clip_ts)
    clips_with_ts.sort()

    if not clips_with_ts:
        return ""

    import numpy as np
    n = len(clips_with_ts)
    if n <= num_frames:
        selected_ts = set(clips_with_ts)
    else:
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
        selected_ts = set(clips_with_ts[i] for i in indices)

    # Match captions by timestamp
    all_caps = _load_captions()
    lines = []
    for c in all_caps:
        if c["_ts"] in selected_ts:
            lines.append(f"[{c['date']} {c['start_time']}-{c['end_time']}] {c['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Caption utilities
# ---------------------------------------------------------------------------
_captions_cache = None


def _load_captions() -> List[Dict]:
    global _captions_cache
    if _captions_cache is None:
        with open(CAPTION_PATH) as f:
            _captions_cache = json.load(f)
        # Pre-compute timestamp for sorting/filtering
        for c in _captions_cache:
            day_num = int(c["date"].replace("DAY", ""))
            c["_ts"] = day_num * 100000000 + int(c["start_time"])
        _captions_cache.sort(key=lambda x: x["_ts"])
    return _captions_cache


def sample_captions_uniform(
    query_ts: int, num_captions: int, max_input_tokens: int = 80000
) -> str:
    """Uniformly sample captions before query_time, return as text.

    Samples num_captions uniformly, then truncates by token budget if needed.
    """
    all_caps = _load_captions()
    before = [c for c in all_caps if c["_ts"] <= query_ts]
    if not before:
        return ""

    import numpy as np
    n = len(before)
    target = min(n, num_captions)

    if n <= target:
        selected = before
    else:
        indices = np.linspace(0, n - 1, target, dtype=int)
        selected = [before[i] for i in indices]

    # Build lines and truncate by token budget
    lines = []
    total_chars = 0
    for c in selected:
        line = f"[{c['date']} {c['start_time']}-{c['end_time']}] {c['text']}"
        total_chars += len(line) + 1  # +1 for newline
        if total_chars / 3.2 > max_input_tokens:
            break
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_vlm(
    client: OpenAI,
    model: str,
    question: str,
    choices: Dict[str, str],
    frames: List[Dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    extra_body: Optional[Dict] = None,
    caption_text: str = "",
) -> str:
    """Call VLM via API with frames and/or captions + question."""
    is_qwen35 = "qwen3.5" in model.lower() or "Qwen3.5" in model

    # Build context description
    has_frames = len(frames) > 0
    has_captions = len(caption_text) > 0

    if has_frames and has_captions:
        context = (
            f"You are watching a first-person egocentric video recorded over several days. "
            f"Below are {len(frames)} uniformly sampled frames and their corresponding text descriptions.\n\n"
            f"=== Text Descriptions ===\n{caption_text}\n\n"
        )
    elif has_frames:
        context = (
            f"You are watching a first-person egocentric video recorded over several days. "
            f"Below are {len(frames)} uniformly sampled frames from the video in chronological order.\n\n"
        )
    else:
        context = (
            f"You are reading text descriptions of a first-person egocentric video recorded over several days. "
            f"Below are uniformly sampled descriptions in chronological order.\n\n"
            f"=== Text Descriptions ===\n{caption_text}\n\n"
        )

    if is_qwen35:
        prompt = (
            f"{context}"
            f"Question: {question}\n"
            f"A. {choices['A']}\n"
            f"B. {choices['B']}\n"
            f"C. {choices['C']}\n"
            f"D. {choices['D']}\n\n"
            f"Analyze briefly, then you MUST end your response with exactly:\n"
            f'"answer": "X"\n'
            f"where X is A, B, C, or D."
        )
    else:
        prompt = (
            f"{context}"
            f"Question: {question}\n"
            f"A. {choices['A']}\n"
            f"B. {choices['B']}\n"
            f"C. {choices['C']}\n"
            f"D. {choices['D']}\n\n"
            f"Choose the best answer (A/B/C/D). Think briefly, then state your answer as a single letter."
        )

    content = frames + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            # Strip thinking tags if present
            if "<think>" in text:
                # Extract content after </think>
                parts = text.split("</think>")
                if len(parts) > 1:
                    text = parts[-1].strip()
            return text
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VLM baseline eval on EgoLifeQA")
    parser.add_argument("--model", type=str, default="qwen/qwen3.5-9b",
                        help="OpenRouter model ID")
    parser.add_argument("--num-frames", type=int, default=32,
                        help="Number of frames to uniformly sample")
    parser.add_argument("--max-frame-size", type=int, default=512,
                        help="Max frame dimension (pixels)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--eval-ids", type=str, default=None,
                        help="Comma-separated question IDs to evaluate (default: all)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel API calls")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--api-base", type=str,
                        default="https://openrouter.ai/api/v1",
                        help="API base URL")
    parser.add_argument("--mode", type=str, default="frames",
                        choices=["frames", "caption", "frames+caption"],
                        help="Input mode: frames only, caption only, or both")
    parser.add_argument("--num-captions", type=int, default=512,
                        help="Number of captions to uniformly sample (default: 512)")
    parser.add_argument("--qa-path", type=str, default=None,
                        help="Override QA JSON path (for EgoR1 etc.)")
    args = parser.parse_args()

    # Setup client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(base_url=args.api_base, api_key=api_key)

    # Load questions
    qa_path = args.qa_path or str(QA_PATH)
    with open(qa_path) as f:
        all_questions = json.load(f)

    # Filter by eval-ids
    if args.eval_ids:
        target_ids = set(args.eval_ids.split(","))
        questions = [q for q in all_questions if str(q["ID"]) in target_ids]
    else:
        questions = all_questions

    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Questions: {len(questions)}")
    print(f"Frames per question: {args.num_frames}")

    # Output setup
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")

    # Load existing results for resume
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for r in json.load(f):
                existing[str(r["id"])] = r
        print(f"Resuming: {len(existing)} existing results")

    def process_question(q):
        qid = str(q["ID"])
        if qid in existing:
            return existing[qid]

        query_ts = parse_timestamp(q["query_time"]["date"], q["query_time"]["time"])
        choices = {
            "A": q["choice_a"],
            "B": q["choice_b"],
            "C": q["choice_c"],
            "D": q["choice_d"],
        }

        # Build input based on mode
        frames = []
        caption_text = ""
        if args.mode in ("frames", "frames+caption"):
            frames = sample_frames_uniform(query_ts, args.num_frames, args.max_frame_size)
        if args.mode == "caption":
            caption_text = sample_captions_uniform(query_ts, args.num_captions)
        elif args.mode == "frames+caption":
            caption_text = get_captions_for_clips(query_ts, args.num_frames)

        # Call API (use Qwen3.5 recommended params if applicable)
        is_qwen35 = "qwen3.5" in args.model.lower() or "Qwen3.5" in args.model
        if is_qwen35:
            temp = 0.7
            max_tok = 512
            extra = {
                "top_k": 20,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            temp = args.temperature
            max_tok = args.max_tokens
            extra = None

        response = call_vlm(
            client, args.model, q["question"], choices, frames,
            temperature=temp, max_tokens=max_tok, extra_body=extra,
            caption_text=caption_text,
        )

        predicted = extract_choice_letter(response)
        correct = predicted == q["answer"] if predicted else False

        return {
            "id": qid,
            "type": q["type"],
            "question": q["question"],
            "gold": q["answer"],
            "predicted": predicted,
            "correct": correct,
            "response": response,
            "num_frames": len(frames),
            "mode": args.mode,
        }

    # Run evaluation
    results = list(existing.values())
    remaining = [q for q in questions if str(q["ID"]) not in existing]

    if remaining:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(process_question, q): q for q in remaining}
            for future in tqdm(as_completed(futures), total=len(remaining), desc="Evaluating"):
                result = future.result()
                results.append(result)
                # Save incrementally
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    # Final stats
    results.sort(key=lambda r: int(r["id"]))
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_answer = sum(1 for r in results if r["predicted"] is None)
    print(f"\n{'='*50}")
    print(f"Results: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"No answer extracted: {no_answer}")

    # Per-type breakdown
    from collections import defaultdict
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_type[r["type"]]["total"] += 1
        if r["correct"]:
            by_type[r["type"]]["correct"] += 1

    print(f"\nPer type:")
    for t, v in sorted(by_type.items()):
        acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0
        print(f"  {t:20s}: {v['correct']:3d}/{v['total']:3d} = {acc:.1f}%")

    # Save summary
    summary = {
        "model": args.model,
        "num_frames": args.num_frames,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "no_answer": no_answer,
        "per_type": dict(by_type),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
