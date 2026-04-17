#!/usr/bin/env python3
"""
VLM baseline evaluation on MM-Lifelong via OpenRouter API.

Usage:
    python baselines/eval_vlm_mmlifelong.py \
        --model qwen/qwen3.5-flash-02-23 \
        --judge-model qwen/qwen3.5-flash-02-23 \
        --num-frames 32 \
        --output-dir output/eval/baselines/mmlifelong_flash_32f \
        --eval-indices 0,1,2,3,4

Requires: OPENROUTER_API_KEY environment variable.
"""

import os
import sys
import json
import re
import base64
import argparse
import time
import math
from io import BytesIO
from typing import Optional, Dict, List, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
DATA_ROOT = PROJECT_ROOT / "data" / "MM-Lifelong" / "month"
QA_PATH = DATA_ROOT / "val.json"
CAPTION_DIR = PROJECT_ROOT / "output" / "metadata" / "mmlifelong"


# ---------------------------------------------------------------------------
# LLM-as-Judge (from MM-Lifelong paper, same as eval_mmlifelong.py)
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct ground truth answer, and the candidate answer will be provided to you.

You must FIRST provide a brief analysis explaining the semantic similarity between the groundtruth and the candidate answer.

THEN, on a new line, output the final score.

Scoring criteria:

- 0: No similarity.
  The candidate answer is completely irrelevant, contradictory, or does not address the question at all.

- 1: Very low similarity.
  The candidate answer mentions a related topic or keyword, but fails to answer the question
  and does not convey the main meaning of the groundtruth.

- 2: Low similarity.
  The candidate answer addresses the question in a limited way, capturing some minor aspects,
  but misses or misrepresents the core idea or key facts of the groundtruth.

- 3: Moderate similarity.
  The candidate answer captures the main idea of the groundtruth,
  but omits several important details or includes noticeable inaccuracies.

- 4: High similarity.
  The candidate answer correctly captures the main idea and most key details of the groundtruth,
  with only minor omissions, simplifications, or non-critical inaccuracies.

- 5: Complete similarity.
  The candidate answer is semantically equivalent to the groundtruth,
  covering all essential information with no meaningful omissions or errors.

Special Rules:

- Hallucination-sensitive questions:
  Score 5 only if all required items are correct;
  if any item is incorrect, missing, or hallucinated, score 0 (no partial credit).

- Time-duration questions:
  Allow errors within the range defined by the question; answers outside the range should receive score 0.

Output format (strictly follow):
Analysis:
<your analysis>

Final Score:
<an integer from 0 to 5>

Question: {question}
Ground truth answer: {gold_answer}
Candidate answer: {prediction}
Your response:"""


def judge_score(client: OpenAI, judge_model: str, question: str, prediction: str, gold_answer: str) -> dict:
    """Score a prediction using LLM-as-judge via API."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        prediction=prediction or "(no answer)",
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            response = resp.choices[0].message.content or ""
            break
        except Exception as e:
            print(f"  Judge error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            response = ""

    # Parse score
    raw_score = 0
    m = re.search(r"Final Score:\s*(\d)", response)
    if m:
        raw_score = int(m.group(1))
    else:
        digits = re.findall(r"\b([0-5])\b", response)
        if digits:
            raw_score = int(digits[-1])

    # Smoothing per MM-Lifelong paper
    if raw_score >= 4:
        smoothed = 1.0
    elif raw_score == 3:
        smoothed = 0.5
    else:
        smoothed = 0.0

    return {
        "raw_score": raw_score,
        "smoothed_score": smoothed,
        "judge_response": response.strip(),
    }


# ---------------------------------------------------------------------------
# Caption utilities
# ---------------------------------------------------------------------------
_mml_captions_cache: Dict[str, List[Dict]] = {}


def _load_mml_captions(video_id: str) -> List[Dict]:
    """Load 30sec captions for a MM-Lifelong video."""
    if video_id not in _mml_captions_cache:
        cap_path = CAPTION_DIR / video_id / "merge" / "captions" / f"{video_id}_30sec.json"
        if cap_path.exists():
            with open(cap_path) as f:
                _mml_captions_cache[video_id] = json.load(f)
        else:
            _mml_captions_cache[video_id] = []
    return _mml_captions_cache[video_id]


def sample_captions_for_question(
    question_data: dict, num_captions: int = 512, max_input_tokens: int = 80000
) -> str:
    """Uniformly sample captions from the question's video(s), truncate by token budget."""
    clue_intervals = question_data.get("clue_interval", [])
    if not clue_intervals:
        return ""

    # Collect all captions from referenced videos
    all_caps = []
    seen_vids = set()
    for ci in clue_intervals:
        vid = ci["video_id"]
        if vid not in seen_vids:
            all_caps.extend(_load_mml_captions(vid))
            seen_vids.add(vid)

    if not all_caps:
        return ""

    import numpy as np
    n = len(all_caps)
    target = min(n, num_captions)

    if n <= target:
        selected = all_caps
    else:
        indices = np.linspace(0, n - 1, target, dtype=int)
        selected = [all_caps[i] for i in indices]

    # Build lines and truncate by token budget
    lines = []
    total_chars = 0
    for c in selected:
        line = f"[{c.get('start_sec', 0):.0f}s-{c.get('end_sec', 0):.0f}s] {c['text']}"
        total_chars += len(line) + 1
        if total_chars / 3.2 > max_input_tokens:
            break
        lines.append(line)


def get_captions_for_frames(question_data: dict, num_frames: int) -> str:
    """Get captions corresponding to uniformly sampled frame positions.

    Matches the same uniform sampling used for frame extraction:
    N frames uniformly from the video → find the caption covering each frame's timestamp.
    """
    clue_intervals = question_data.get("clue_interval", [])
    if not clue_intervals:
        return ""

    import numpy as np

    lines = []
    seen_vids = set()
    for ci in clue_intervals:
        vid = ci["video_id"]
        if vid in seen_vids:
            continue
        seen_vids.add(vid)

        all_caps = _load_mml_captions(vid)
        if not all_caps:
            continue

        # Video duration from last caption
        total_duration = max(c.get("end_sec", 0) for c in all_caps)
        if total_duration <= 0:
            continue

        # Same uniform positions as frame extraction
        frame_secs = np.linspace(0, total_duration, num_frames)

        # For each frame position, find the matching caption
        for sec in frame_secs:
            for c in all_caps:
                if c.get("start_sec", 0) <= sec < c.get("end_sec", 0):
                    lines.append(f"[{c['start_sec']:.0f}s-{c['end_sec']:.0f}s] {c['text']}")
                    break

    return "\n".join(lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def _frame_to_b64(frame_np, max_size: int = 512) -> str:
    """Convert numpy frame to base64 JPEG."""
    img = Image.fromarray(frame_np)
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


from functools import lru_cache

@lru_cache(maxsize=2)
def _get_video_reader(video_path: str):
    """Open a VideoReader with LRU cache (keep last 2 videos in memory)."""
    import decord
    decord.bridge.set_bridge("native")
    return decord.VideoReader(video_path)


def sample_frames_for_question(question_data: dict, num_frames: int, max_size: int = 512) -> List[Dict]:
    """Sample frames from the videos referenced by the question's clue_intervals.

    Optimized: groups frame requests by video, opens each video only once.
    """
    clue_intervals = question_data.get("clue_interval", [])
    if not clue_intervals:
        return []

    import numpy as np

    # Collect all (video_path, seconds_list) grouped by video
    all_points = []  # (video_path, start, end)
    for ci in clue_intervals:
        vid = ci["video_id"]
        video_path = DATA_ROOT / f"{vid}.mp4"
        if not video_path.exists():
            print(f"  Warning: video {video_path} not found")
            continue
        for interval in ci.get("intervals", []):
            start_sec, end_sec = interval[0], interval[1]
            all_points.append((str(video_path), start_sec, end_sec))

    if not all_points:
        return []

    total_duration = sum(end - start for _, start, end in all_points)
    if total_duration <= 0:
        return []

    # Compute sample seconds per interval, group by video
    from collections import defaultdict
    video_secs = defaultdict(list)  # video_path -> [sec, sec, ...]
    sec_order = []  # maintain global order: (video_path, sec)

    for video_path, start_sec, end_sec in all_points:
        duration = end_sec - start_sec
        n = max(1, int(round(num_frames * duration / total_duration)))
        secs = np.linspace(start_sec, end_sec, n)
        for sec in secs:
            video_secs[video_path].append(sec)
            sec_order.append((video_path, sec))

    # Batch extract: open each video once, get all frames
    video_frames = {}  # (video_path, sec) -> b64
    for video_path, secs in video_secs.items():
        try:
            vr = _get_video_reader(video_path)
            fps = vr.get_avg_fps()
            frame_indices = [min(int(s * fps), len(vr) - 1) for s in secs]
            batch = vr.get_batch(frame_indices).asnumpy()
            for sec, frame_np in zip(secs, batch):
                video_frames[(video_path, sec)] = _frame_to_b64(frame_np, max_size)
        except Exception as e:
            print(f"  Warning: failed to extract frames from {Path(video_path).name}: {e}")

    # Assemble in order
    frames = []
    for video_path, sec in sec_order:
        b64 = video_frames.get((video_path, sec))
        if b64:
            frames.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    return frames


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_vlm(
    client: OpenAI,
    model: str,
    question: str,
    frames: List[Dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    extra_body: Optional[Dict] = None,
    caption_text: str = "",
) -> str:
    """Call VLM via API with frames and/or captions + open-ended question."""
    has_frames = len(frames) > 0
    has_captions = len(caption_text) > 0

    if has_frames and has_captions:
        context = (
            f"You are watching livestream broadcast videos. "
            f"Below are {len(frames)} frames and text descriptions from the relevant segments.\n\n"
            f"=== Text Descriptions ===\n{caption_text}\n\n"
        )
    elif has_frames:
        context = (
            f"You are watching livestream broadcast videos. "
            f"Below are {len(frames)} frames sampled from the relevant video segments.\n\n"
        )
    else:
        context = (
            f"You are reading text descriptions of livestream broadcast videos.\n\n"
            f"=== Text Descriptions ===\n{caption_text}\n\n"
        )

    prompt = (
        f"{context}"
        f"Question: {question}\n\n"
        f"Answer the question concisely based on the provided information."
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
            if "<think>" in text:
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
    parser = argparse.ArgumentParser(description="VLM baseline eval on MM-Lifelong")
    parser.add_argument("--model", type=str, default="qwen/qwen3.5-flash-02-23")
    parser.add_argument("--judge-model", type=str, default="qwen/qwen3.5-flash-02-23")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-frame-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-indices", type=str, default=None,
                        help="Comma-separated question indices (0-based)")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--skip-judge", action="store_true",
                        help="Only run VLM inference, skip judge scoring")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Pre-extracted frames dir (with manifest.json)")
    parser.add_argument("--mode", type=str, default="frames",
                        choices=["frames", "caption", "frames+caption"],
                        help="Input mode")
    parser.add_argument("--num-captions", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort level for thinking models (GPT-5, Gemini 3)")
    parser.add_argument("--api-base", type=str,
                        default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-base", type=str, default=None,
                        help="Separate API base for judge (default: same as --api-base)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(base_url=args.api_base, api_key=api_key)
    judge_api_base = args.judge_api_base or "https://openrouter.ai/api/v1"
    judge_client = OpenAI(base_url=judge_api_base, api_key=api_key)

    # Load pre-extracted frames manifest if available
    _frames_manifest = None
    if args.frames_dir:
        manifest_path = os.path.join(args.frames_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                _frames_manifest = json.load(f)
            print(f"Using pre-extracted frames from {args.frames_dir} ({len(_frames_manifest)} questions)")

    # Load questions
    with open(QA_PATH) as f:
        all_questions = json.load(f)

    if args.eval_indices:
        indices = [int(x) for x in args.eval_indices.split(",")]
        questions = [(i, all_questions[i]) for i in indices]
    else:
        questions = list(enumerate(all_questions))

    print(f"Model: {args.model}")
    print(f"Judge: {args.judge_model}")
    print(f"Questions: {len(questions)}")
    print(f"Frames per question: {args.num_frames}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")

    # Load existing for resume
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for r in json.load(f):
                existing[r["index"]] = r
        print(f"Resuming: {len(existing)} existing results")

    def process_question(idx_and_q):
        idx, q = idx_and_q
        if idx in existing:
            return existing[idx]

        # Build input based on mode
        frames = []
        caption_text = ""
        if args.mode in ("frames", "frames+caption"):
            if args.frames_dir and _frames_manifest is not None:
                frame_paths = _frames_manifest.get(str(idx), [])
                for fp in frame_paths:
                    with open(fp, "rb") as img_f:
                        b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    frames.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
            else:
                frames = sample_frames_for_question(q, args.num_frames, args.max_frame_size)
        if args.mode == "caption":
            caption_text = sample_captions_for_question(q, args.num_captions)
        elif args.mode == "frames+caption":
            caption_text = get_captions_for_frames(q, args.num_frames)

        # Call VLM (Qwen3.5 non-thinking mode)
        is_qwen35 = "qwen3.5" in args.model.lower() or "Qwen3.5" in args.model
        if is_qwen35:
            temp = 0.7
            max_tok = 512
            extra = {"top_k": 20, "presence_penalty": 1.5,
                     "chat_template_kwargs": {"enable_thinking": False}}
        else:
            temp = args.temperature
            max_tok = args.max_tokens
            extra = None

        if args.reasoning_effort:
            if extra is None:
                extra = {}
            extra["reasoning"] = {"effort": args.reasoning_effort}

        response = call_vlm(
            client, args.model, q["question"], frames,
            temperature=temp, max_tokens=max_tok, extra_body=extra,
            caption_text=caption_text,
        )

        result = {
            "index": idx,
            "question_type": q["question_type"],
            "question": q["question"],
            "gold": q["answer"],
            "response": response,
            "num_frames": len(frames),
        }

        # Judge (skip if --skip-judge)
        if not args.skip_judge:
            judge_result = judge_score(
                judge_client, args.judge_model, q["question"], response, q["answer"]
            )
            result.update({
                "raw_score": judge_result["raw_score"],
                "smoothed_score": judge_result["smoothed_score"],
                "judge_response": judge_result["judge_response"],
            })

        return result

    # Run
    results = list(existing.values())
    remaining = [(i, q) for i, q in questions if i not in existing]

    if remaining:
        for idx_q in tqdm(remaining, desc="Evaluating"):
            result = process_question(idx_q)
            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Stats
    results.sort(key=lambda r: r["index"])
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    has_judge = "raw_score" in results[0] if results else False

    print(f"\n{'='*50}")
    print(f"Results ({total} questions):")

    if has_judge:
        avg_raw = sum(r["raw_score"] for r in results) / total if total else 0
        avg_smoothed = sum(r["smoothed_score"] for r in results) / total if total else 0
        print(f"  Raw score:      {avg_raw:.3f}")
        print(f"  Smoothed score: {avg_smoothed:.3f}")

        from collections import defaultdict
        by_type = defaultdict(lambda: {"raw": 0, "smoothed": 0.0, "total": 0})
        for r in results:
            by_type[r["question_type"]]["total"] += 1
            by_type[r["question_type"]]["raw"] += r["raw_score"]
            by_type[r["question_type"]]["smoothed"] += r["smoothed_score"]

        print(f"\nPer type:")
        for t, v in sorted(by_type.items()):
            raw_avg = v["raw"] / v["total"]
            sm_avg = v["smoothed"] / v["total"]
            print(f"  {t:25s}: raw={raw_avg:.2f} smoothed={sm_avg:.3f} (n={v['total']})")
    else:
        print("  (judge skipped, run rejudge separately)")

    summary = {
        "model": args.model,
        "judge_model": args.judge_model,
        "num_frames": args.num_frames,
        "total": total,
    }
    if has_judge:
        summary["avg_raw_score"] = avg_raw
        summary["avg_smoothed_score"] = avg_smoothed
        summary["per_type"] = dict(by_type)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
