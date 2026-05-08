#!/usr/bin/env python3
"""
Ego-R1 Bench evaluation script using WorldMM.

Ego-R1 Bench (arXiv 2506.13654) uses the same EgoLife video data and
the same 4-option multiple-choice format as EgoLifeQA, but provides
50 separate QA pairs (25 manual + 25 gemini-generated).

This script intentionally mirrors eval.py so it can share the same
WorldMemory pipeline; only the data-loading and reporting differ.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult

from _common import (
    load_json,
    extract_choice_letter,
    evaluate_prediction,
    resolve_semantic_results_path,
    find_30s_segment,
    parse_target_time,
)


# ---------------------------------------------------------------------------
# Ego-R1 specific helpers
# ---------------------------------------------------------------------------

def load_egor1_data(
    bench_dir: str,
    subject: str,
    splits: List[str],
) -> List[Dict[str, Any]]:
    """Load Ego-R1 Bench QA data for a subject.

    Args:
        bench_dir: Root of Ego-R1-Bench (contains manual-benchmark/ and gemini-benchmark/).
        subject: e.g. "A1_JAKE".
        splits: List of splits to load, e.g. ["manual", "gemini"] or ["manual"].

    Returns:
        Combined list of QA entries, each tagged with a ``_split`` field.
    """
    all_rows: List[Dict[str, Any]] = []
    for split in splits:
        path = os.path.join(bench_dir, f"{split}-benchmark", f"{subject}.json")
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue
        rows = load_json(path)
        for r in rows:
            r["_split"] = split
        all_rows.extend(rows)
        logger.info(f"Loaded {len(rows)} {split} questions for {subject}")
    return all_rows


def parse_query_time(row: Dict[str, Any]) -> int:
    """Convert Ego-R1 query_time dict to integer timestamp (DHHMMSSFF).

    Some entries have a range format like "12041900-12003000";
    we take the later (larger) endpoint as the query moment.
    """
    qt = row["query_time"]
    day = qt["date"].replace("DAY", "").replace("Day", "")
    time_str = qt["time"]
    if "-" in time_str:
        parts = time_str.split("-")
        candidates = [int(day + p.zfill(8)) for p in parts]
        return max(candidates)
    return int(day + time_str.zfill(8))


def _parse_time_value(day: str, time_str: str) -> List[int]:
    """Parse a single time value that may be a plain timestamp or a dash-range.

    Returns a list of 1 or 2 integer timestamps.
    Examples:
        "12332505"            -> [612332505]
        "12332505-12542517"   -> [612332505, 612542517]
        "111153417DAY1_11181201" -> handled by caller (DAY format)
    """
    s = str(time_str)
    if "DAY" not in s.upper() and "-" in s:
        return [int(day + p.zfill(8)) for p in s.split("-")]
    return [int(day + s.zfill(8))]


def safe_parse_target_time(
    row: Dict[str, Any],
    segments_30s: List[Dict[str, Any]],
) -> List[tuple]:
    """Robust target_time parser for Ego-R1 Bench.

    Handles all formats found in the dataset:
      - Plain timestamp: {"date":"DAY1", "time":"20250513"}
      - Dash range:      {"date":"DAY6", "time":"12332505-12542517"}
      - DAY range:       {"date":"DAY1", "time":"111153417DAY1_11181201"}
      - time_list with any of the above
    Falls back to eval.parse_target_time for DAY-range format.
    """
    tt = row.get("target_time")
    if not tt:
        return []

    time_val = tt.get("time")
    time_list_val = tt.get("time_list")

    # If it contains "DAY" in the time string, delegate to eval.py's parser
    if time_val and "DAY" in str(time_val).upper():
        try:
            return parse_target_time(row, segments_30s)
        except (ValueError, KeyError):
            return []

    day_str = tt.get("date", "")
    day = day_str.replace("DAY", "").replace("Day", "") if isinstance(day_str, str) else str(day_str)

    results = []

    if time_val:
        try:
            timestamps = _parse_time_value(day, time_val)
            if len(timestamps) == 2:
                results.append((min(timestamps), max(timestamps)))
            else:
                seg = find_30s_segment(timestamps[0], segments_30s)
                if seg != (0, 0):
                    results.append(seg)
        except (ValueError, KeyError):
            pass

    elif time_list_val:
        for t in time_list_val:
            try:
                timestamps = _parse_time_value(day, t)
                if len(timestamps) == 2:
                    results.append((min(timestamps), max(timestamps)))
                else:
                    seg = find_30s_segment(timestamps[0], segments_30s)
                    if seg != (0, 0):
                        results.append(seg)
            except (ValueError, KeyError):
                continue

    return results


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ego-R1 Bench Evaluation with WorldMM")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID (e.g. A1_JAKE)")
    parser.add_argument("--bench-dir", type=str, default="data/Ego-R1-Bench",
                        help="Path to Ego-R1-Bench root directory")
    parser.add_argument("--retriever-model", type=str,
                        default=os.getenv("RETRIEVER_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--respond-model", type=str,
                        default=os.getenv("RESPOND_MODEL", "qwen/qwen3.5-flash-02-23"))
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--episodic-top-k", type=int, default=3)
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--visual-top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife",
                        help="EgoLife data directory (captions, etc.)")
    parser.add_argument("--cache-dir", type=str, default=".cache/episodic_memory")
    parser.add_argument("--retrieval-backend", type=str, default="unified_graph",
                        choices=["unified_graph", "independent"])
    parser.add_argument("--graph-path", type=str, default=None)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--eval-ids", type=str, default=None,
                        help="Comma-separated list of IDs or ranges to evaluate, e.g. '70,71' or '70-78'")
    parser.add_argument("--chain-mode", type=str, default="facts",
                        help="Chain injection mode: 'facts' (default, paper config), or pass empty string to disable.")
    parser.add_argument("--topic-chain-facts-path", type=str,
                        default="output/metadata/topic_chains/A1_JAKE/topic_chains.json")
    parser.add_argument("--event-chain-path", type=str,
                        default="output/metadata/event_chains/A1_JAKE/step3_enriched_chains.json")
    parser.add_argument("--parallel", type=int, default=8,
                        help="Number of parallel workers for evaluation (default 8)")
    parser.add_argument("--chain-min-hits", type=int, default=1)
    parser.add_argument("--chain-max-topics", type=int, default=3)
    parser.add_argument("--chain-max-events", type=int, default=1,
                        help="Max event chains to inject per round (default 1)")
    parser.add_argument("--chain-topic-sim", type=float, default=0.7)
    parser.add_argument("--chain-event-sim", type=float, default=0.7,
                        help="Event-chain embedding similarity threshold (default 0.7)")
    parser.add_argument("--chain-event-min-hits", type=int, default=1,
                        help="Minimum event-chain step hits required for candidacy")
    parser.add_argument("--chain-event-granularities", type=str,
                        default="30sec,3min",
                        help="Comma-separated episode granularities used to "
                             "detect event-chain step hits (default '30sec,3min'). "
                             "Add '10min' / '1h' to loosen coarse filter.")
    args = parser.parse_args()

    splits = ["manual", "gemini"]

    # Resolve graph path default
    if args.retrieval_backend == "unified_graph" and args.graph_path is None:
        args.graph_path = os.path.join(
            "output", "metadata", "unified_graph", args.subject, "graph.pkl"
        )

    # ---- Initialize models ----
    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    retriever_llm_model = LLMModel(model_name=args.retriever_model, provider="openrouter")
    respond_llm_model = LLMModel(model_name=args.respond_model, provider="openrouter")
    prompt_template_manager = PromptTemplateManager()

    # ---- Initialize WorldMemory ----
    logger.info(f"Initializing WorldMemory (backend={args.retrieval_backend})...")
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
        retrieval_backend=args.retrieval_backend,
        graph_path=args.graph_path,
        cache_dir=args.cache_dir,
        chain_mode=args.chain_mode,
        topic_chain_facts_path=args.topic_chain_facts_path,
        event_chain_path=args.event_chain_path,
    )

    if args.retrieval_backend == "unified_graph":
        world_memory.bm25_weight = args.bm25_weight
        world_memory.damping = args.damping
        world_memory.chain_min_hits = args.chain_min_hits
        world_memory.chain_max_topics = args.chain_max_topics
        world_memory.chain_max_events = args.chain_max_events
        world_memory.chain_topic_sim = args.chain_topic_sim
        world_memory.chain_event_sim = args.chain_event_sim
        world_memory.chain_event_min_hits = args.chain_event_min_hits
        world_memory.chain_event_granularities = tuple(
            g.strip() for g in args.chain_event_granularities.split(",") if g.strip()
        )

    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    # ---- Load EgoLife base data (same as eval.py) ----
    logger.info("Loading EgoLife base data...")
    subject = args.subject
    data_dir = args.data_dir

    episodic_caption_dir = os.path.join(data_dir, f"EgoLifeCap/{subject}")
    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {
        g: os.path.join(episodic_caption_dir, f"{subject}_{g}.json")
        for g in granularities
    }
    episodic_captions_30sec = load_json(episodic_caption_files["30sec"])

    semantic_path = resolve_semantic_results_path(
        subject=subject,
        retriever_model=args.retriever_model,
        respond_model=args.respond_model,
    )
    semantic_results = load_json(semantic_path)

    visual_path = os.path.join(f"output/metadata/visual_memory/{subject}/visual_embeddings.pkl")

    logger.info("Loading data into WorldMemory...")
    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    world_memory.load_semantic_triples(data=semantic_results)
    world_memory.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)

    # ---- Load Ego-R1 Bench questions ----
    eval_data = load_egor1_data(args.bench_dir, subject, splits)
    if not eval_data:
        logger.error(f"No Ego-R1 data found for {subject} in {args.bench_dir}")
        return

    # ---- Output path ----
    split_tag = "_".join(splits)
    strategy_suffix = ""
    if args.retrieval_backend == "unified_graph":
        parts = []
        if args.bm25_weight > 0:
            parts.append(f"bm25{args.bm25_weight}" if args.bm25_weight != 0.3 else "bm25")
        if args.damping != 0.85:
            parts.append(f"damp{args.damping}")
        if args.chain_mode:
            parts.append(f"cm_{args.chain_mode}")
        if parts:
            strategy_suffix = "_" + "_".join(parts)

    output_path = os.path.join(
        args.output_dir,
        f"{_safe_model_name(args.retriever_model)}_{_safe_model_name(args.respond_model)}",
        f"egor1_eval_{subject}_{split_tag}{strategy_suffix}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---- Parse --eval-ids ----
    eval_id_set = None
    if args.eval_ids:
        eval_id_set = set()
        for part in args.eval_ids.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                eval_id_set.update(range(int(lo), int(hi) + 1))
            else:
                eval_id_set.add(int(part))
        logger.info(f"--eval-ids specified: evaluating {len(eval_id_set)} IDs")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_ids_{args.eval_ids.replace(',', '_').replace('-', '_')}{ext}"

    # ---- Resume ----
    # Use (ID, split) as unique key since IDs can overlap across manual/gemini splits.
    def _key(r):
        return (r["ID"], r.get("split", r.get("_split", "unknown")))

    results: List[Dict[str, Any]] = []
    done_keys: set = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            loaded = json.load(f)
        results = [r for r in loaded if r.get("response") != "Error"]
        done_keys = {_key(r) for r in results}
        if eval_id_set is not None:
            results = [r for r in results if int(r["ID"]) not in eval_id_set]
            done_keys = {k for k in done_keys if int(k[0]) not in eval_id_set}
            logger.info(f"--eval-ids mode: will re-run {len(eval_id_set)} IDs (kept {len(results)} other results)")
        else:
            logger.info(f"Resuming: {len(done_keys)} done, {len(eval_data) - len(done_keys)} remaining.")

    evaluate_true = sum(int(r["evaluate"]) for r in results)

    # ---- Filter remaining ----
    if eval_id_set is not None:
        remaining = [row for row in eval_data if int(row["ID"]) in eval_id_set]
    else:
        remaining = [row for row in eval_data if _key(row) not in done_keys]
    logger.info(f"Starting evaluation on {len(remaining)} samples (total {len(eval_data)})...")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_workers = args.parallel
    save_lock = threading.Lock()

    def _process_one(row, wm):
        ID = row["ID"]
        query_type = row["type"]
        question = row["question"]
        answer = row["answer"]
        split = row.get("_split", "unknown")

        choices = {}
        for key, label in [("choice_a", "A"), ("choice_b", "B"), ("choice_c", "C"), ("choice_d", "D")]:
            if key in row and row[key]:
                choices[label] = row[key]

        query_time = parse_query_time(row)
        target_time_list = safe_parse_target_time(row, episodic_captions_30sec)

        qa_result: Optional[QAResult] = None
        try:
            qa_result = wm.answer(
                query=question,
                choices=choices,
                until_time=query_time,
            )
            response = qa_result.answer
        except Exception as e:
            logger.error(f"Error processing ID {ID}: {e}")
            response = "Error"

        evaluate_val = evaluate_prediction(response, answer, choices)

        return {
            "ID": ID,
            "split": split,
            "type": query_type,
            "question": question,
            "choices": choices,
            "answer": answer,
            "response": response,
            "round_history": qa_result.round_history if qa_result else [],
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "evaluate": evaluate_val,
            "query_time": query_time,
            "target_time": target_time_list,
            "keywords": row.get("keywords", ""),
        }

    def _create_worker_memory():
        wm = WorldMemory(
            embedding_model=embedding_model,
            retriever_llm_model=LLMModel(model_name=args.retriever_model, provider="openrouter"),
            respond_llm_model=LLMModel(model_name=args.respond_model, provider="openrouter"),
            prompt_template_manager=prompt_template_manager,
            max_rounds=args.max_rounds,
            max_errors=args.max_errors,
            retrieval_backend=args.retrieval_backend,
            graph_path=args.graph_path,
            cache_dir=args.cache_dir,
            chain_mode=args.chain_mode,
            topic_chain_facts_path=args.topic_chain_facts_path,
            event_chain_path=args.event_chain_path,
        )
        if args.retrieval_backend == "unified_graph":
            wm.bm25_weight = args.bm25_weight
            wm.damping = args.damping
            wm.chain_min_hits = args.chain_min_hits
            wm.chain_max_topics = args.chain_max_topics
            wm.chain_max_events = args.chain_max_events
            wm.chain_topic_sim = args.chain_topic_sim
            wm.chain_event_sim = args.chain_event_sim
            wm.chain_event_min_hits = args.chain_event_min_hits
            wm.chain_event_granularities = tuple(
                g.strip() for g in args.chain_event_granularities.split(",") if g.strip()
            )
        wm.set_retrieval_top_k(
            episodic=args.episodic_top_k,
            semantic=args.semantic_top_k,
            visual=args.visual_top_k,
        )
        wm.load_episodic_captions(caption_files=episodic_caption_files)
        wm.load_semantic_triples(data=semantic_results)
        wm.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)
        return wm

    if num_workers <= 1:
        for row in tqdm(remaining, total=len(remaining)):
            result_entry = _process_one(row, world_memory)
            evaluate_true += int(result_entry["evaluate"])
            results.append(result_entry)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(
                f"ID {result_entry['ID']} Answer: {result_entry['response']}, "
                f"Gold: {result_entry['answer']}, Correct: {result_entry['evaluate']} "
                f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true / len(results):.4f}"
            )
    else:
        logger.info(f"Creating {num_workers} parallel workers...")
        worker_memories = [_create_worker_memory() for _ in range(num_workers)]
        worker_locks = [threading.Lock() for _ in range(num_workers)]
        logger.info(f"All {num_workers} workers ready.")

        pbar = tqdm(total=len(remaining))

        def _worker_fn(row, worker_idx):
            idx = worker_idx % num_workers
            with worker_locks[idx]:
                return _process_one(row, worker_memories[idx])

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_worker_fn, row, i): row
                for i, row in enumerate(remaining)
            }
            for future in as_completed(futures):
                result_entry = future.result()
                with save_lock:
                    evaluate_true += int(result_entry["evaluate"])
                    results.append(result_entry)
                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=4)
                    pbar.update(1)
                    logger.info(
                        f"ID {result_entry['ID']} Answer: {result_entry['response']}, "
                        f"Gold: {result_entry['answer']}, Correct: {result_entry['evaluate']} "
                        f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true / len(results):.4f}"
                    )
        pbar.close()

    # ---- Save final results ----
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    # ---- Summary ----
    total = len(results)
    accuracy = evaluate_true / total if total else 0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Ego-R1 Bench Evaluation Complete")
    logger.info(f"Subject: {subject}  |  Splits: {splits}")
    logger.info(f"Total: {total}  |  Correct: {evaluate_true}  |  Accuracy: {accuracy:.4f}")

    # Per-type breakdown
    type_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        t = r["type"]
        if t not in type_stats:
            type_stats[t] = {"total": 0, "correct": 0}
        type_stats[t]["total"] += 1
        type_stats[t]["correct"] += int(r["evaluate"])

    logger.info("--- Per-type breakdown ---")
    for t, s in sorted(type_stats.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        logger.info(f"  {t:20s}: {s['correct']}/{s['total']} = {acc:.4f}")

    # Per-split breakdown
    split_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        sp = r.get("split", "unknown")
        if sp not in split_stats:
            split_stats[sp] = {"total": 0, "correct": 0}
        split_stats[sp]["total"] += 1
        split_stats[sp]["correct"] += int(r["evaluate"])

    logger.info("--- Per-split breakdown ---")
    for sp, s in sorted(split_stats.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        logger.info(f"  {sp:20s}: {s['correct']}/{s['total']} = {acc:.4f}")

    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'=' * 60}")

    world_memory.cleanup()


if __name__ == "__main__":
    main()
