#!/usr/bin/env python3
"""
EgoLifeQA evaluation script using WorldMM unified memory system.
"""

import os
import json
import re
import argparse
import glob
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult, transform_timestamp


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def resolve_semantic_results_path(subject: str, retriever_model: str, respond_model: str) -> str:
    """Resolve semantic consolidation file without hardcoded model names."""
    semantic_dir = os.path.join("output", "metadata", "semantic_memory", subject)
    candidates = []

    # 1) Prefer explicit translation model if provided.
    translation_model = os.getenv("TRANSLATION_MODEL")
    if translation_model:
        candidates.append(
            os.path.join(
                semantic_dir,
                f"semantic_consolidation_results_{_safe_model_name(translation_model)}.json",
            )
        )

    # 2) Then try the eval models.
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

    # 3) Fallback to newest consolidation file.
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
    """Normalize text for comparison."""
    if not text:
        return ""
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    """Extracts A/B/C/D from a prediction, handling both clean and verbose outputs.

    Strategy (first match wins):
      1. If the entire text (stripped) is a single letter A-D → return it.
      2. If the text starts with a letter pattern like "(C)" or "B." → return it.
      3. Search for explicit answer phrases: "the answer is (C)", "Answer: B", etc.
      4. Look for the *last* standalone A-D near the end of the text (models often
         put the final answer at the very end after reasoning).
    """
    if not text:
        return None

    stripped = text.strip()

    # 1. Entire text is a single letter
    if re.fullmatch(r"[A-Da-d]", stripped):
        return stripped.upper()

    # 2. Starts with a letter pattern like "A", "A.", "(A)", "A) ..."
    #    Require the letter to be followed by punctuation, whitespace, or end-of-string
    #    to avoid matching words like "Based", "Alice", etc.
    m = re.match(r"\(?([A-Da-d])(?:[\.\)\:,]|\s|$)", stripped)
    if m:
        return m.group(1).upper()

    # 3. Explicit answer phrases (case-insensitive)
    for pat in [
        r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-Da-d])\)?",
        r"(?:final\s+)?answer\s*:\s*\(?([A-Da-d])\)?",
        r"(?:I\s+(?:would\s+)?(?:choose|pick|select|go\s+with))\s+\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\s+is\s+(?:the\s+)?(?:correct|best|most\s+supported)\b",
    ]:
        m = re.search(pat, stripped, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 4. Last standalone A-D in the text (word boundary on both sides)
    all_letters = list(re.finditer(r"\b([A-Da-d])\b", stripped))
    if all_letters:
        # Prefer letters in the last 20% of the text
        cutoff = int(len(stripped) * 0.8)
        tail_letters = [m for m in all_letters if m.start() >= cutoff]
        if tail_letters:
            return tail_letters[-1].group(1).upper()

    return None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    """
    Evaluate if prediction matches the gold answer.
    
    Args:
        prediction: Model's prediction
        gold_letter: Correct answer letter (e.g., 'A', 'B', 'C', 'D')
        choices: Dict of answer choices
        
    Returns:
        True if prediction is correct
    """
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])

    if pred_norm == gold_candidate:
        return True

    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True

    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}")
    ]
    if pred_norm in full_patterns:
        return True

    return False


def find_30s_segment(target_timestamp: int, segments_30s: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Find the 30s segment that contains the target timestamp.
    
    Args:
        target_timestamp: Target timestamp as integer (format: day + time.zfill(8))
        segments_30s: List of 30s segments
    
    Returns:
        Tuple of (start_time, end_time) for the matching segment, or (0, 0) if not found
    """
    for segment in segments_30s:
        date = segment.get('date', '')
        start_time_raw = segment.get('start_time', 0)
        end_time_raw = segment.get('end_time', 0)
        
        day = date.replace('DAY', '').replace('Day', '') if isinstance(date, str) else str(date)
        
        # Format times
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
        
        # Check if target timestamp falls within this segment
        if start_time <= target_timestamp <= end_time:
            return (start_time, end_time)
    
    return (0, 0)


def parse_target_time(row: Dict[str, Any], segments_30s: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Parse target time from row data.
    
    Args:
        row: QA row data
        segments_30s: List of 30s segments for finding time ranges
        
    Returns:
        List of (start_time, end_time) tuples
    """
    target_time_list = []
    
    if "time" in row['target_time'] and row['target_time']["time"]:
        time_str = row['target_time']["time"]
        time_str_upper = time_str.upper()
        
        if "DAY" in time_str_upper:
            # Parse range format: "11153417DAY1_11181201"
            parts = re.split(r'DAY|Day', time_str, maxsplit=1)
            if len(parts) == 2:
                start_time_str = parts[0]
                day_and_end = parts[1].split("_")
                if len(day_and_end) == 2:
                    end_day = day_and_end[0]
                    end_time_str = day_and_end[1]
                    start_day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
                    
                    start_time = int(start_day + start_time_str.zfill(8))
                    end_time = int(end_day + end_time_str.zfill(8))
                    target_time_list.append((start_time, end_time))
        else:
            # Single timestamp - find its 30s segment
            day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    elif "time_list" in row['target_time'] and row['target_time']["time_list"]:
        # Multiple timestamps
        day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
        for time_str in row['target_time']["time_list"]:
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    return target_time_list


def main():
    parser = argparse.ArgumentParser(description="EgoLifeQA Evaluation with WorldMM")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID")
    parser.add_argument("--retriever-model", type=str, default=os.getenv("RETRIEVER_MODEL", "openai/gpt-oss-120b"), help="LLM model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default=os.getenv("RESPOND_MODEL", "qwen/qwen3.5-flash-02-23"), help="LLM model for iterative reasoning and generating answers")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3, help="Top-k for episodic retrieval")
    parser.add_argument("--semantic-top-k", type=int, default=10, help="Top-k for semantic retrieval")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Top-k for visual retrieval")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory")
    parser.add_argument("--cache-dir", type=str, default=".cache/episodic_memory", help="HippoRAG cache directory (use different dirs for parallel runs)")
    parser.add_argument(
        "--retrieval-backend", type=str, default="unified_graph",
        choices=["unified_graph", "independent"],
        help="Retrieval backend: 'unified_graph' (default, ours) or 'independent' (baseline three-way retrieval)."
    )
    parser.add_argument(
        "--graph-path", type=str, default=None,
        help="Path to serialised MultimodalGraph pkl (required for unified_graph backend)"
    )
    parser.add_argument(
        "--bm25-weight", type=float, default=0.3,
        help="BM25 seed weight for unified_graph backend (0 to disable, default 0.3)."
    )
    parser.add_argument(
        "--damping", type=float, default=0.85,
        help="PPR damping factor (default 0.85, use 0.5 for concentrated diffusion)."
    )
    parser.add_argument(
        "--eval-ids", type=str, default=None,
        help="Comma-separated list of IDs or ranges to evaluate, e.g. '70,71,72' or '70-78' or '70-78,100,120-130'. "
             "If set, only these IDs are evaluated (ignoring resume state)."
    )
    parser.add_argument(
        "--parallel", type=int, default=8,
        help="Number of parallel workers for evaluation (default 8). "
             "Each worker gets its own WorldMemory instance. "
             "Only effective with unified_graph backend."
    )
    parser.add_argument(
        "--chain-mode", type=str, default="facts",
        help="Chain injection mode: 'facts' (default, paper config), or pass empty string to disable."
    )
    parser.add_argument(
        "--topic-chain-facts-path", type=str,
        default="output/metadata/topic_chains/A1_JAKE/topic_chains.json",
        help="Path to topic_chains.json (entity lifecycle facts)."
    )
    parser.add_argument(
        "--storyline-path", type=str,
        default="output/metadata/storylines/A1_JAKE/step3_enriched_chains.json",
        help="Path to step3_enriched_chains.json (cross-time storyline chains)."
    )
    parser.add_argument("--chain-min-hits", type=int, default=1,
                        help="Topic chain coarse filter: min episode hits (default 1)")
    parser.add_argument("--chain-max-topics", type=int, default=3,
                        help="Max topic chains to inject per round (default 3)")
    parser.add_argument("--chain-max-events", type=int, default=2,
                        help="Max storyline chains to inject per round (default 2)")
    parser.add_argument("--chain-topic-sim", type=float, default=0.5,
                        help="Topic chain embedding similarity threshold (default 0.5)")
    parser.add_argument("--chain-storyline-sim", type=float, default=0.5,
                        help="Storyline embedding similarity threshold (default 0.5)")
    parser.add_argument("--chain-keyword-bypass-topk", action="store_true", default=True,
                        help="Keyword-matched chains bypass topk limit (default True for EgoLife)")
    args = parser.parse_args()

    # Resolve graph path default for unified_graph backend
    if args.retrieval_backend == "unified_graph" and args.graph_path is None:
        args.graph_path = os.path.join(
            "output", "metadata", "unified_graph", args.subject, "graph.pkl"
        )

    # Initialize models
    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    # Qwen3.5 recommended sampling params for non-thinking reasoning tasks
    _qwen_reasoning_kwargs = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    retriever_llm_model = LLMModel(
        model_name=args.retriever_model,
        provider="openrouter",
    )
    respond_llm_model = LLMModel(
        model_name=args.respond_model,
        provider="openrouter",
        **_qwen_reasoning_kwargs,
    )
    prompt_template_manager = PromptTemplateManager()

    # Initialize WorldMemory
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
        storyline_path=args.storyline_path,
    )

    # Set parameters (unified_graph backend)
    if args.retrieval_backend == "unified_graph":
        world_memory.bm25_weight = args.bm25_weight
        world_memory.damping = args.damping
        world_memory.chain_min_hits = args.chain_min_hits
        world_memory.chain_max_topics = args.chain_max_topics
        world_memory.chain_max_events = args.chain_max_events
        world_memory.chain_topic_sim = args.chain_topic_sim
        world_memory.chain_storyline_sim = args.chain_storyline_sim
        world_memory.chain_keyword_bypass_topk = args.chain_keyword_bypass_topk
        logger.info(
            f"BM25 weight: {args.bm25_weight}, damping: {args.damping}"
        )

    # Set retrieval top-k
    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    # Load data
    logger.info("Loading data...")
    subject = args.subject
    data_dir = args.data_dir
    
    eval_data_path = os.path.join(data_dir, f"EgoLifeQA/EgoLifeQA_{subject}.json")
    eval_data = load_json(eval_data_path)
    
    # Load episodic captions for all granularities (multiscale memory)
    episodic_caption_dir = os.path.join(data_dir, f"EgoLifeCap/{subject}")
    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {
        g: os.path.join(episodic_caption_dir, f"{subject}_{g}.json")
        for g in granularities
    }
    # Load 30sec captions separately for target time parsing
    episodic_captions_30sec = load_json(episodic_caption_files["30sec"])
    
    # Load semantic results
    semantic_path = resolve_semantic_results_path(
        subject=subject,
        retriever_model=args.retriever_model,
        respond_model=args.respond_model,
    )
    semantic_results = load_json(semantic_path)
    
    # Load visual embeddings
    visual_path = os.path.join(f"output/metadata/visual_memory/{subject}/visual_embeddings.pkl")
    
    # Load data into WorldMemory
    logger.info("Loading data into WorldMemory...")
    
    # Load episodic captions for all granularities
    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    
    # Load semantic triples
    world_memory.load_semantic_triples(data=semantic_results)
    
    # Load visual embeddings
    world_memory.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)

    # Determine output path early (needed for resume)
    # Include tags in filename to separate experiments
    strategy_suffix = ""
    if args.retrieval_backend == "unified_graph":
        parts = []
        if args.bm25_weight > 0:
            parts.append("bm25")
        if args.damping != 0.85:
            parts.append(f"damp{args.damping}")
        if args.chain_mode:
            parts.append(f"cm_{args.chain_mode}")
        if parts:
            strategy_suffix = "_" + "_".join(parts)
    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}",
        f"egolife_eval_{subject}{strategy_suffix}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Parse --eval-ids if provided
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
        logger.info(f"--eval-ids specified: evaluating {len(eval_id_set)} IDs: {sorted(eval_id_set)}")
        # Write to a separate file so we don't overwrite the main results
        base, ext = os.path.splitext(output_path)
        ids_tag = args.eval_ids.replace(',', '_').replace('-', '_')
        if len(ids_tag) > 120:
            # Truncate long ID lists to avoid filesystem name-too-long errors
            import hashlib
            ids_hash = hashlib.md5(ids_tag.encode()).hexdigest()[:8]
            ids_tag = f"{len(eval_id_set)}q_{ids_hash}"
        output_path = f"{base}_ids_{ids_tag}{ext}"
        logger.info(f"Results will be saved to: {output_path}")

    # Resume: load existing results if output file already exists
    # When --eval-ids is specified, always re-run (overwrite previous results for those IDs).
    results = []
    done_ids: set = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        # Only count successfully answered items (not Error placeholders)
        results = [r for r in loaded if r.get("response") != "Error"]
        done_ids = {r["ID"] for r in results}
        if eval_id_set is not None:
            # When --eval-ids is set, discard previous results for those IDs so they get re-run
            results = [r for r in results if int(r["ID"]) not in eval_id_set]
            done_ids = {id_ for id_ in done_ids if int(id_) not in eval_id_set}
            logger.info(f"--eval-ids mode: will re-run {len(eval_id_set)} IDs (kept {len(results)} other results)")
        else:
            logger.info(f"Resuming: {len(done_ids)} already done, {len(eval_data) - len(done_ids)} remaining.")

    evaluate_true = sum(int(r["evaluate"]) for r in results)

    # Evaluation loop — normalize IDs to int for consistent comparison
    if eval_id_set is not None:
        remaining = [row for row in eval_data if int(row["ID"]) in eval_id_set]
    else:
        remaining = [row for row in eval_data if row["ID"] not in done_ids]
    logger.info(f"Starting evaluation on {len(remaining)} samples (total {len(eval_data)})...")

    # ------------------------------------------------------------------
    # Helper: process a single question (used by both serial and parallel)
    # ------------------------------------------------------------------
    def _process_one(row, wm):
        """Process one QA row and return the result dict."""
        ID = row['ID']
        query_type = row['type']
        question = row['question']
        answer_gold = row['answer']

        choices = {}
        for key, label in [('choice_a', 'A'), ('choice_b', 'B'), ('choice_c', 'C'), ('choice_d', 'D')]:
            if key in row and row[key]:
                choices[label] = row[key]

        query_time = int(row['query_time']["date"][-1] + row['query_time']["time"].zfill(8))
        target_time_list = parse_target_time(row, episodic_captions_30sec)

        logger.info(f"Processing ID {ID}: {question[:50]}...")

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

        evaluate = evaluate_prediction(response, answer_gold, choices)

        result_entry = {
            "ID": ID,
            "type": query_type,
            "question": question,
            "choices": choices,
            "answer": answer_gold,
            "response": response,
            "round_history": qa_result.round_history if qa_result else [],
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "evaluate": evaluate,
            "query_time": query_time,
            "target_time": target_time_list,
        }
        return result_entry

    # ------------------------------------------------------------------
    # Dispatch: serial (1 worker) or parallel (N workers)
    # ------------------------------------------------------------------
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_workers = args.parallel
    save_lock = threading.Lock()

    def _create_worker_memory():
        """Create an independent WorldMemory for a parallel worker.

        Shares the same embedding_model (GPU model, thread-safe for encode)
        but creates independent LLM clients and graph instances.
        """
        wm = WorldMemory(
            embedding_model=embedding_model,  # shared across all workers
            retriever_llm_model=LLMModel(model_name=args.retriever_model, provider="openrouter"),
            respond_llm_model=LLMModel(model_name=args.respond_model, provider="openrouter", **_qwen_reasoning_kwargs),
            prompt_template_manager=prompt_template_manager,
            max_rounds=args.max_rounds,
            max_errors=args.max_errors,
            retrieval_backend=args.retrieval_backend,
            graph_path=args.graph_path,
            cache_dir=args.cache_dir,
            chain_mode=args.chain_mode,
            topic_chain_facts_path=args.topic_chain_facts_path,
            storyline_path=args.storyline_path,
        )
        if args.retrieval_backend == "unified_graph":
            wm.bm25_weight = args.bm25_weight
            wm.damping = args.damping
            wm.chain_min_hits = args.chain_min_hits
            wm.chain_max_topics = args.chain_max_topics
            wm.chain_max_events = args.chain_max_events
            wm.chain_topic_sim = args.chain_topic_sim
            wm.chain_storyline_sim = args.chain_storyline_sim
            wm.chain_keyword_bypass_topk = args.chain_keyword_bypass_topk
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
        # Serial mode (original behavior)
        for row in tqdm(remaining, total=len(remaining)):
            result_entry = _process_one(row, world_memory)
            evaluate_true += int(result_entry["evaluate"])
            results.append(result_entry)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)

            logger.info(
                f"ID {result_entry['ID']} Answer: {result_entry['response']}, "
                f"Gold: {result_entry['answer']}, Correct: {result_entry['evaluate']} "
                f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true/len(results):.4f}"
            )
    else:
        # Parallel mode
        logger.info(f"Creating {num_workers} parallel workers...")
        worker_memories = [_create_worker_memory() for _ in range(num_workers)]
        worker_locks = [threading.Lock() for _ in range(num_workers)]
        logger.info(f"All {num_workers} workers ready.")

        pbar = tqdm(total=len(remaining))

        def _worker_fn(row, worker_idx):
            idx = worker_idx % num_workers
            with worker_locks[idx]:
                wm = worker_memories[idx]
                return _process_one(row, wm)

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

                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=4)

                    logger.info(
                        f"ID {result_entry['ID']} Answer: {result_entry['response']}, "
                        f"Gold: {result_entry['answer']}, Correct: {result_entry['evaluate']} "
                        f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true/len(results):.4f}"
                    )
                    pbar.update(1)

        pbar.close()
        # Cleanup worker memories
        for wm in worker_memories:
            wm.cleanup()

    # Save final results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary
    final_accuracy = evaluate_true / len(results) if results else 0
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Complete")
    logger.info(f"Subject: {subject}")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Correct: {evaluate_true}")
    logger.info(f"Accuracy: {final_accuracy:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*50}")

    # Cleanup
    world_memory.cleanup()


if __name__ == "__main__":
    main()
