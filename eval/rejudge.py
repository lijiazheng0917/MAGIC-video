#!/usr/bin/env python3
"""
Re-judge existing evaluation results with a different judge model.
Only re-scores, does not re-run retrieval or answer generation.

Usage:
    python eval/rejudge.py \
        --input output/eval/mmlifelong/chain_100q/.../results.json \
        --output output/eval/mmlifelong/chain_100q_rejudge/.../results.json \
        --judge-model qwen/qwen3.5-flash-02-23 \
        --parallel 8
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Same judge prompt as eval_mmlifelong.py
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
Candidate answer: {candidate_answer}
Your response:"""


def call_judge(client, model, question, gold_answer, candidate_answer, max_retries=3):
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        candidate_answer=candidate_answer,
    )
    for attempt in range(max_retries):
        raw = None
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.0,
                timeout=120,
            )
            raw = resp.choices[0].message.content
            if not raw:
                continue
            clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            # Parse "Final Score:\n<digit>" format
            m = re.search(r"Final Score:\s*(\d)", clean)
            if m:
                score = int(m.group(1))
                analysis = clean
                return score, analysis, raw
            # Fallback: look for any digit
            digits = re.findall(r"\b([0-5])\b", clean)
            if digits:
                return int(digits[-1]), clean, raw
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Judge error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return 0, "Judge failed", ""


def smooth_score(raw_score):
    if raw_score <= 2:
        return 0.0
    elif raw_score == 3:
        return 0.5
    else:
        return 1.0


def main():
    parser = argparse.ArgumentParser(description="Re-judge evaluation results")
    parser.add_argument("--input", required=True, help="Input results JSON file")
    parser.add_argument("--output", default=None, help="Output re-judged JSON file (defaults to --input when --retry-failed)")
    parser.add_argument("--judge-model", default="openai/gpt-5")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only re-judge items where judge_response == 'Judge failed'")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per item (default 3, use more with --retry-failed)")
    args = parser.parse_args()

    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if args.output is None:
        if args.retry_failed:
            args.output = args.input  # in-place update
        else:
            sys.exit("--output is required (unless --retry-failed)")

    data = json.load(open(args.input))
    # Only re-judge items that have a response
    to_judge = [x for x in data if x.get("response") and x["response"] != "Error"]
    if args.retry_failed:
        to_judge = [x for x in to_judge if x.get("judge_response") == "Judge failed"]
        logger.info(f"Loaded {len(data)} items, {len(to_judge)} FAILED items to retry with {args.judge_model}")
    else:
        logger.info(f"Loaded {len(data)} items, {len(to_judge)} to re-judge with {args.judge_model}")

    results = list(data)  # copy
    def _get_idx(x):
        return x.get("index", x.get("id"))
    def _get_gold(x):
        return x.get("gold_answer", x.get("gold"))

    idx_map = {_get_idx(x): i for i, x in enumerate(results)}

    def judge_one(item):
        score, analysis, raw = call_judge(
            client, args.judge_model,
            item["question"], _get_gold(item), item["response"],
            max_retries=args.max_retries,
        )
        return _get_idx(item), score, analysis, raw

    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(judge_one, x): x for x in to_judge}
        for future in as_completed(futures):
            idx, score, analysis, raw = future.result()
            i = idx_map[idx]
            results[i]["raw_score"] = score
            results[i]["smoothed_score"] = smooth_score(score)
            results[i]["judge_response"] = analysis
            results[i]["judge_model"] = args.judge_model
            done += 1
            if done % 10 == 0:
                logger.info(f"  {done}/{len(to_judge)} judged")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    scored = [x for x in results if x.get("raw_score") is not None]
    avg_raw = sum(x["raw_score"] for x in scored) / len(scored) if scored else 0
    avg_sm = sum(x.get("smoothed_score") or 0 for x in scored) / len(scored) if scored else 0
    logger.info(f"Done: {len(scored)} scored, avg_raw={avg_raw:.3f}, avg_smoothed={avg_sm:.3f}")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
