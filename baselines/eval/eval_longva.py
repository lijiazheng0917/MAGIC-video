#!/usr/bin/env python3
"""
LongVA-7B baseline evaluation on EgoLifeQA / EgoR1 / MM-Lifelong.

Based on: https://huggingface.co/lmms-lab/LongVA-7B

Usage:
    python baselines/eval/eval_longva.py \
        --benchmark egolife --num-frames 512 \
        --frame-cache output/metadata/baselines/egolife_videos_64f/frame_cache \
        --output-dir output/eval/baselines/egolife_longva_512f
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QA_PATH = PROJECT_ROOT / "data" / "EgoLife" / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
MML_QA_PATH = PROJECT_ROOT / "data" / "MM-Lifelong" / "month" / "val.json"


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
        "-i", list_file, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1",
        tmp.name,
    ], capture_output=True)
    os.unlink(list_file)
    if result.returncode == 0 and os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
        return tmp.name
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
    return None


def load_video_frames(video_path, num_frames):
    """Load video and return list of PIL images."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lmms-lab/LongVA-7B")
    parser.add_argument("--num-frames", type=int, default=512)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-ids", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--benchmark", type=str, default="egolife",
                        choices=["egolife", "mmlifelong"])
    parser.add_argument("--frame-cache", type=str, default=None)
    parser.add_argument("--qa-path", type=str, default=None)
    parser.add_argument("--mml-frames-dir", type=str, default=None)
    args = parser.parse_args()

    from longva.model.builder import load_pretrained_model

    print(f"Loading model: {args.model}")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model, None, "llava_qwen",
        device_map="auto", torch_dtype="bfloat16",
    )
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

        # Build prompt
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

        try:
            # Load video frames
            pil_frames = load_video_frames(video_path, args.num_frames)

            # Process frames
            video_tensor = image_processor.preprocess(pil_frames, return_tensors="pt")["pixel_values"]
            video_tensor = video_tensor.to(model.device, dtype=torch.float16)

            # Use model.chat if available, otherwise manual
            if hasattr(model, 'chat'):
                with torch.no_grad():
                    response = model.chat(
                        video_path=video_path,
                        tokenizer=tokenizer,
                        user_prompt=question_text,
                        max_num_frames=args.num_frames,
                        generation_config=dict(do_sample=True, temperature=0.7, max_new_tokens=args.max_new_tokens),
                    )
                    if isinstance(response, tuple):
                        response = response[0]
            else:
                # Manual generation with video tokens
                from longva.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
                from longva.conversation import conv_templates
                from longva.mm_utils import tokenizer_image_token
                import copy

                conv = copy.deepcopy(conv_templates["qwen_1_5"])
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids, images=[video_tensor], modalities=["video"],
                        do_sample=True, temperature=0.7, max_new_tokens=args.max_new_tokens,
                    )
                response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        except Exception as e:
            import traceback
            print(f"  Error on Q{qid}: {e}")
            traceback.print_exc()
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


if __name__ == "__main__":
    main()
