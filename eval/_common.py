"""Shared helpers used by EgoLifeQA / Ego-R1 / MM-Lifelong evaluation scripts."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# LLM-as-Judge prompt for MM-Lifelong (paper Appendix C.3). Used by eval_mmlifelong
# and rejudge. Format keys: {question}, {gold_answer}, {prediction}.
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


def load_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def resolve_semantic_results_path(subject: str, retriever_model: str, respond_model: str) -> str:
    """Resolve semantic consolidation file without hardcoded model names."""
    semantic_dir = os.path.join("output", "metadata", "semantic_memory", subject)
    candidates: List[str] = []

    translation_model = os.getenv("TRANSLATION_MODEL")
    if translation_model:
        candidates.append(
            os.path.join(
                semantic_dir,
                f"semantic_consolidation_results_{_safe_model_name(translation_model)}.json",
            )
        )

    for model_name in (retriever_model, respond_model):
        candidates.append(
            os.path.join(
                semantic_dir,
                f"semantic_consolidation_results_{_safe_model_name(model_name)}.json",
            )
        )

    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Using semantic results: {path}")
            return path

    pattern = os.path.join(semantic_dir, "semantic_consolidation_results_*.json")
    fallback_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if fallback_files:
        logger.warning(
            "No exact semantic file match for TRANSLATION_MODEL/retriever/respond model. "
            f"Falling back to latest file: {fallback_files[0]}"
        )
        return fallback_files[0]

    raise FileNotFoundError(
        "No semantic consolidation file found. Expected one of: "
        + ", ".join(candidates)
        + f" or any file matching {pattern}"
    )


def normalize(text: str) -> str:
    if not text:
        return ""
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract A/B/C/D from a prediction, handling clean and verbose outputs."""
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
        tail_letters = [m for m in all_letters if m.start() >= cutoff]
        if tail_letters:
            return tail_letters[-1].group(1).upper()

    return None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])

    if pred_norm == gold_candidate:
        return True

    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True

    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}"),
    ]
    if pred_norm in full_patterns:
        return True

    return False


def find_30s_segment(target_timestamp: int, segments_30s: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Find the 30s segment containing the target timestamp; (0, 0) if none."""
    for segment in segments_30s:
        date = segment.get("date", "")
        start_time_raw = segment.get("start_time", 0)
        end_time_raw = segment.get("end_time", 0)

        day = date.replace("DAY", "").replace("Day", "") if isinstance(date, str) else str(date)

        if isinstance(start_time_raw, str):
            start_time = int(day + start_time_raw.zfill(8))
        elif isinstance(start_time_raw, int):
            start_time = int(day + str(start_time_raw).zfill(8))
        else:
            continue

        if isinstance(end_time_raw, str):
            end_time = int(day + end_time_raw.zfill(8))
        elif isinstance(end_time_raw, int):
            end_time = int(day + str(end_time_raw).zfill(8))
        else:
            continue

        if start_time <= target_timestamp <= end_time:
            return (start_time, end_time)

    return (0, 0)


def parse_target_time(row: Dict[str, Any], segments_30s: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    target_time_list: List[Tuple[int, int]] = []

    if "time" in row["target_time"] and row["target_time"]["time"]:
        time_str = row["target_time"]["time"]
        time_str_upper = time_str.upper()

        if "DAY" in time_str_upper:
            parts = re.split(r"DAY|Day", time_str, maxsplit=1)
            if len(parts) == 2:
                start_time_str = parts[0]
                day_and_end = parts[1].split("_")
                if len(day_and_end) == 2:
                    end_day = day_and_end[0]
                    end_time_str = day_and_end[1]
                    start_day = row["target_time"]["date"].replace("DAY", "").replace("Day", "")

                    start_time = int(start_day + start_time_str.zfill(8))
                    end_time = int(end_day + end_time_str.zfill(8))
                    target_time_list.append((start_time, end_time))
        else:
            day = row["target_time"]["date"].replace("DAY", "").replace("Day", "")
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)

    elif "time_list" in row["target_time"] and row["target_time"]["time_list"]:
        day = row["target_time"]["date"].replace("DAY", "").replace("Day", "")
        for time_str in row["target_time"]["time_list"]:
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)

    return target_time_list
