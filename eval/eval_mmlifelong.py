#!/usr/bin/env python3
"""
MM-Lifelong (month-level) evaluation script.

Evaluates WorldMM on MM-Lifelong livestream broadcasts.
For each broadcast:
  1. Load preprocessed data (captions, semantic, visual, unified graph)
  2. Answer all questions for that broadcast (open-ended, free-form)
  3. Score with LLM-as-judge
  4. Reset memory and move to the next broadcast

Key differences from eval_lvbench.py:
  - Open-ended answers (NOT multiple choice A/B/C/D)
  - LLM-as-judge evaluation (0-5 scale, smoothed to 0/0.5/1)
  - Questions grouped by broadcast (video_id in clue_interval)
  - 4 granularity levels: 30sec, 3min, 10min, 1h
  - Phase 1: only single-broadcast questions (92.3% of val set)
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult


def _log_memory(label: str):
    """Log current CPU RAM and GPU memory usage."""
    import psutil
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    msg = f"[MEM] {label}: CPU RSS={rss_gb:.1f}GB"
    try:
        import torch
        if torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            msg += f", GPU alloc={alloc_gb:.1f}GB, reserved={reserved_gb:.1f}GB"
    except ImportError:
        pass
    logger.info(msg)


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluation (from MM-Lifelong paper Appendix C.3)
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


def llm_judge_score(
    question: str,
    prediction: str,
    gold_answer: str,
    judge_llm: LLMModel,
) -> dict:
    """Score a prediction using LLM-as-judge.

    Returns dict with:
      - raw_score (int 0-5)
      - smoothed_score (float 0.0/0.5/1.0)
      - judge_response (str, full judge output)
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        prediction=prediction or "(no answer)",
    )

    try:
        response = judge_llm.generate(prompt)
    except Exception as e:
        logger.error(f"Judge LLM failed: {e}")
        return {"raw_score": 0, "smoothed_score": 0.0, "judge_response": f"Error: {e}"}

    if not response:
        return {"raw_score": 0, "smoothed_score": 0.0, "judge_response": "Empty response"}

    # Parse score from response
    raw_score = 0
    m = re.search(r"Final Score:\s*(\d)", response)
    if m:
        raw_score = int(m.group(1))
    else:
        # Fallback: look for any digit at end
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
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def load_broadcast_data(
    world_memory: WorldMemory,
    broadcast_id: str,
    metadata_dir: str,
    caption_mode: str = "asr",
) -> bool:
    """Load all preprocessed data for a single MM-Lifelong broadcast."""
    base = os.path.join(metadata_dir, "mmlifelong", broadcast_id)
    mode_dir = os.path.join(base, caption_mode)
    cap_dir = os.path.join(mode_dir, "captions")
    sem_dir = os.path.join(mode_dir, "semantic_memory")
    vis_dir = os.path.join(base, "visual_memory")  # shared

    # Episodic captions (4 granularities)
    cap_30sec_path = os.path.join(cap_dir, f"{broadcast_id}_30sec.json")
    if not os.path.exists(cap_30sec_path):
        logger.error(f"Missing 30sec captions: {cap_30sec_path}")
        return False

    caption_files = {"30sec": cap_30sec_path}
    for gran in ["3min", "10min", "1h"]:
        p = os.path.join(cap_dir, f"{broadcast_id}_{gran}.json")
        if os.path.exists(p):
            caption_files[gran] = p

    world_memory.load_episodic_captions(caption_files=caption_files)

    # Semantic triples
    sem_path = os.path.join(sem_dir, "semantic_consolidation_results.json")
    if os.path.exists(sem_path):
        world_memory.load_semantic_triples(file_path=sem_path)
    else:
        logger.warning(f"No semantic results for broadcast {broadcast_id}")

    # Visual clips + embeddings
    vis_emb_path = os.path.join(vis_dir, "visual_embeddings.pkl")
    captions_30sec = load_json(cap_30sec_path)
    if os.path.exists(vis_emb_path):
        world_memory.load_visual_clips(
            embeddings_path=vis_emb_path,
            clips_data=captions_30sec,
        )
    else:
        logger.warning(f"No visual embeddings for broadcast {broadcast_id}")
        world_memory.load_visual_clips(clips_data=captions_30sec)

    # Unified graph
    if world_memory.retrieval_backend == "unified_graph":
        graph_path = os.path.join(mode_dir, "unified_graph", "graph.pkl")
        if os.path.exists(graph_path):
            from worldmm.memory.unified import UnifiedMemory
            chain_mode = getattr(world_memory, '_chain_mode', None)
            um = UnifiedMemory(world_memory.embedding_model, graph_path,
                               chain_mode=chain_mode)
            # Load chain facts if available
            topic_path = os.path.join(mode_dir, "topic_chains.json")
            story_path = os.path.join(mode_dir, "storyline_step3_enriched.json")
            um.load(
                topic_chain_facts_path=topic_path if os.path.exists(topic_path) else None,
                storyline_path=story_path if os.path.exists(story_path) else None,
            )
            world_memory.unified_memory = um
        else:
            logger.error(f"Missing graph for broadcast {broadcast_id}: {graph_path}")
            return False

    return True


# ---------------------------------------------------------------------------
# Question filtering and grouping
# ---------------------------------------------------------------------------

def get_broadcast_ids(question: dict) -> set:
    """Extract the set of broadcast IDs referenced by a question."""
    ids = set()
    for clue in question.get("clue_interval", []):
        ids.add(clue["video_id"])
    return ids


def filter_single_broadcast_questions(
    questions: List[dict],
    available_broadcasts: set,
) -> List[dict]:
    """Filter to questions that reference a single available broadcast."""
    filtered = []
    for q in questions:
        bids = get_broadcast_ids(q)
        if len(bids) == 1 and bids.issubset(available_broadcasts):
            filtered.append(q)
    return filtered


def filter_evaluable_questions(
    questions: List[dict],
    available_broadcasts: set,
) -> List[dict]:
    """Filter to questions whose referenced broadcasts are ALL available."""
    filtered = []
    for q in questions:
        bids = get_broadcast_ids(q)
        if bids and bids.issubset(available_broadcasts):
            filtered.append(q)
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MM-Lifelong Evaluation")
    parser.add_argument("--metadata-path", default="data/MM-Lifelong/month/val.json",
                        help="Path to MM-Lifelong val.json")
    parser.add_argument("--metadata-dir", default="output/metadata",
                        help="Root directory for preprocessed outputs")
    parser.add_argument("--broadcast-ids", default=None,
                        help="Comma-separated broadcast IDs to evaluate (default: auto-detect from metadata-dir)")
    parser.add_argument("--caption-mode", default="merge", choices=["asr", "vlm", "merge"],
                        help="Caption mode: asr, vlm, merge (default: merge).")
    parser.add_argument("--retriever-model", default=os.getenv("RETRIEVER_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--respond-model", default=os.getenv("RESPOND_MODEL", "qwen/qwen3.5-flash-02-23"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "openai/gpt-5"),
                        help="LLM for judge scoring (default: openai/gpt-5)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge scoring; save results without scores for later rejudge")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--episodic-top-k", type=int, default=3)
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--visual-top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="output/eval/mmlifelong")
    parser.add_argument("--cache-dir", default=".cache/episodic_memory_mmlifelong")
    parser.add_argument("--retrieval-backend", default="unified_graph",
                        choices=["unified_graph", "independent"])
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--parallel", type=int, default=8,
                        help="Number of questions to evaluate in parallel per broadcast (default: 8)")
    parser.add_argument("--eval-indices", type=str, default=None,
                        help="Comma-separated question indices to evaluate (e.g. '30,53,72'). "
                             "If not set, all questions are evaluated.")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="If set, shuffle broadcast order and per-broadcast question order "
                             "with this seed. Resume-safe (done_indices are tracked globally).")
    parser.add_argument("--chain-mode", type=str, default="facts",
                        help="Chain injection mode: 'facts' (default, paper config), or pass empty string to disable.")
    parser.add_argument("--chain-min-hits", type=int, default=4)
    parser.add_argument("--chain-max-topics", type=int, default=2)
    parser.add_argument("--chain-max-events", type=int, default=2)
    parser.add_argument("--chain-topic-sim", type=float, default=0.7)
    parser.add_argument("--chain-storyline-sim", type=float, default=0.7)
    parser.add_argument("--chain-storyline-min-hits", type=int, default=1,
                        help="Minimum storyline step hits required for candidacy (default 1).")
    parser.add_argument("--chain-storyline-granularities", type=str,
                        default="30sec,3min",
                        help="Comma-separated episode granularities used to "
                             "detect storyline step hits (default '30sec,3min'). "
                             "Add '10min' / '1h' to loosen coarse filter.")
    args = parser.parse_args()

    # Load val questions
    all_questions = load_json(args.metadata_path)
    logger.info(f"Loaded {len(all_questions)} questions from {args.metadata_path}")

    # Determine available broadcasts
    if args.broadcast_ids:
        available = set(args.broadcast_ids.split(","))
    else:
        mmlifelong_dir = os.path.join(args.metadata_dir, "mmlifelong")
        if os.path.isdir(mmlifelong_dir):
            available = {d for d in os.listdir(mmlifelong_dir)
                         if os.path.isdir(os.path.join(mmlifelong_dir, d))}
        else:
            available = set()
    logger.info(f"Available broadcasts: {sorted(available)}")

    # Filter to evaluable questions (all referenced broadcasts available)
    questions = filter_evaluable_questions(all_questions, available)
    single_bcast = [q for q in questions if len(get_broadcast_ids(q)) == 1]
    multi_bcast = [q for q in questions if len(get_broadcast_ids(q)) > 1]
    logger.info(
        f"Evaluable questions: {len(questions)}/{len(all_questions)} "
        f"(single-broadcast: {len(single_bcast)}, cross-broadcast: {len(multi_bcast)})"
    )

    if not questions:
        logger.error("No evaluable questions found. Check broadcast data.")
        return

    # Group single-broadcast questions by broadcast
    broadcast_questions: Dict[str, List[dict]] = {}
    for q in single_bcast:
        bid = list(get_broadcast_ids(q))[0]
        broadcast_questions.setdefault(bid, []).append(q)

    # Initialize models
    logger.info("Initializing models...")
    _log_memory("before model init")
    embedding_model = EmbeddingModel()
    embedding_model.load_model("vision")  # Pre-load VLM2Vec for visual retrieval
    _log_memory("after VLM2Vec load")
    # Qwen3.5 recommended sampling params for non-thinking reasoning tasks
    _qwen_reasoning_kwargs = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    retriever_llm = LLMModel(model_name=args.retriever_model, provider="openrouter")
    respond_llm = LLMModel(model_name=args.respond_model, provider="openrouter", **_qwen_reasoning_kwargs)
    judge_llm = LLMModel(model_name=args.judge_model, provider="openrouter")
    prompt_mgr = PromptTemplateManager()

    # Output path
    backend_suffix = "_unified" if args.retrieval_backend == "unified_graph" else ""
    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('/', '_')}_{args.respond_model.replace('/', '_')}",
        f"mmlifelong_eval{backend_suffix}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume support
    results = []
    done_indices = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            loaded = json.load(f)
        results = [r for r in loaded if r.get("response") != "Error"]
        done_indices = {r["index"] for r in results}
        logger.info(f"Resuming: {len(done_indices)} questions already done")

    total_score = sum(r.get("smoothed_score") or 0.0 for r in results)

    # Parse --eval-indices if provided
    eval_index_set = None
    if args.eval_indices:
        eval_index_set = set(int(x) for x in args.eval_indices.split(","))
        logger.info(f"--eval-indices: evaluating {len(eval_index_set)} specific questions")

    # Evaluation loop: per-broadcast
    bcast_items = sorted(broadcast_questions.items())
    if args.shuffle_seed is not None:
        import random as _rand
        _rng = _rand.Random(args.shuffle_seed)
        _rng.shuffle(bcast_items)
        logger.info(f"--shuffle-seed={args.shuffle_seed}: broadcast order shuffled")
    for bid, bid_questions in tqdm(bcast_items, desc="Broadcasts"):
        remaining = [q for q in bid_questions if q["index"] not in done_indices]
        if eval_index_set is not None:
            remaining = [q for q in remaining if q["index"] in eval_index_set]
        if args.shuffle_seed is not None:
            import random as _rand
            _rand.Random(args.shuffle_seed + hash(bid) % (2**31)).shuffle(remaining)
        if not remaining:
            logger.info(f"Broadcast {bid}: all questions done, skipping")
            continue

        _log_memory(f"before broadcast {bid}")
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing broadcast {bid} ({len(remaining)} questions)")
        logger.info(f"{'='*50}")

        # Resolve per-video chain paths
        chain_dir = os.path.join(args.metadata_dir, "mmlifelong")
        vid_chain_dir = os.path.join(chain_dir, bid, args.caption_mode)
        topic_chain_path = os.path.join(vid_chain_dir, "topic_chains.json")
        storyline_path = os.path.join(vid_chain_dir, "storyline_step3_enriched.json")
        if args.chain_mode and not os.path.exists(topic_chain_path):
            topic_chain_path = None
        if args.chain_mode and not os.path.exists(storyline_path):
            storyline_path = None

        world_memory = WorldMemory(
            embedding_model=embedding_model,
            retriever_llm_model=retriever_llm,
            respond_llm_model=respond_llm,
            prompt_template_manager=prompt_mgr,
            episodic_granularities=["30sec", "3min", "10min", "1h"],
            max_rounds=args.max_rounds,
            max_errors=args.max_errors,
            retrieval_backend="independent",
            cache_dir=os.path.join(args.cache_dir, bid),
            chain_mode=args.chain_mode,
            topic_chain_facts_path=topic_chain_path if args.chain_mode else None,
            storyline_path=storyline_path if args.chain_mode else None,
        )

        # Set QA template for open-ended answers
        world_memory.qa_template_name = "qa_mmlifelong"

        world_memory.set_retrieval_top_k(
            episodic=args.episodic_top_k,
            semantic=args.semantic_top_k,
            visual=args.visual_top_k,
        )
        if args.retrieval_backend == "unified_graph":
            world_memory.retrieval_backend = args.retrieval_backend
            world_memory.bm25_weight = args.bm25_weight
            world_memory.damping = args.damping
            world_memory.reasoning_version = "mmlifelong"
            world_memory.chain_min_hits = args.chain_min_hits
            world_memory.chain_max_topics = args.chain_max_topics
            world_memory.chain_max_events = args.chain_max_events
            world_memory.chain_topic_sim = args.chain_topic_sim
            world_memory.chain_storyline_sim = args.chain_storyline_sim
            world_memory.chain_storyline_min_hits = args.chain_storyline_min_hits
            world_memory.chain_storyline_granularities = tuple(
                g.strip() for g in args.chain_storyline_granularities.split(",") if g.strip()
            )

        _log_memory(f"after WorldMemory init (broadcast {bid})")

        if not load_broadcast_data(world_memory, bid, args.metadata_dir, args.caption_mode):
            logger.error(f"Failed to load data for broadcast {bid}, skipping")
            for q in remaining:
                results.append({
                    "index": q["index"],
                    "broadcast_id": bid,
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "response": "Error",
                    "raw_score": 0,
                    "smoothed_score": 0.0,
                    "question_type": q.get("question_type", ""),
                    "temporal_certificate": q.get("temporal_certificate", ""),
                })
            continue

        _log_memory(f"after load data (broadcast {bid})")

        def _eval_one_question(q):
            """Evaluate a single question. Thread-safe: only reads shared state."""
            qidx = q["index"]
            question = q["question"]
            gold_answer = q["answer"]

            logger.info(f"  Q {qidx}: {question[:80]}...")

            qa_result: Optional[QAResult] = None
            try:
                qa_result = world_memory.answer(
                    query=question,
                    choices=None,  # open-ended, no multiple choice
                    until_time=float("inf"),
                )
                response = qa_result.answer
            except Exception as e:
                logger.error(f"  Error on Q{qidx}: {e}")
                response = "Error"

            # LLM-as-judge scoring
            if args.skip_judge:
                judge_result = {"raw_score": None, "smoothed_score": None, "judge_response": "skipped"}
            elif response and response != "Error":
                judge_result = llm_judge_score(question, response, gold_answer, judge_llm)
            else:
                judge_result = {"raw_score": 0, "smoothed_score": 0.0, "judge_response": "N/A"}

            return {
                "index": qidx,
                "broadcast_id": bid,
                "question": question,
                "gold_answer": gold_answer,
                "response": response,
                "raw_score": judge_result["raw_score"],
                "smoothed_score": judge_result["smoothed_score"],
                "judge_response": judge_result["judge_response"],
                "round_history": qa_result.round_history if qa_result else [],
                "num_rounds": qa_result.num_rounds if qa_result else 0,
                "question_type": q.get("question_type", ""),
                "temporal_certificate": q.get("temporal_certificate", ""),
            }

        import threading
        _save_lock = threading.Lock()

        if args.parallel <= 1:
            # Sequential (original behavior)
            for q in remaining:
                result_entry = _eval_one_question(q)
                total_score += result_entry["smoothed_score"] or 0
                results.append(result_entry)

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

                total_done = len(results)
                avg_score = total_score / total_done if total_done else 0
                logger.info(
                    f"  Q{result_entry['index']} → Score: {result_entry['raw_score']}/5 "
                    f"(smoothed: {result_entry['smoothed_score']}) "
                    f"// Avg: {avg_score:.4f} ({total_done} done)"
                )
        else:
            # Parallel evaluation
            from concurrent.futures import ThreadPoolExecutor, as_completed
            logger.info(f"  Evaluating {len(remaining)} questions with {args.parallel} parallel workers")
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(_eval_one_question, q): q for q in remaining}
                for future in as_completed(futures):
                    result_entry = future.result()
                    with _save_lock:
                        total_score += result_entry["smoothed_score"] or 0
                        results.append(result_entry)

                        with open(output_path, "w") as f:
                            json.dump(results, f, indent=4, ensure_ascii=False)

                        total_done = len(results)
                        avg_score = total_score / total_done if total_done else 0
                    logger.info(
                        f"  Q{result_entry['index']} → Score: {result_entry['raw_score']}/5 "
                        f"(smoothed: {result_entry['smoothed_score']}) "
                        f"// Avg: {avg_score:.4f} ({total_done} done)"
                    )

        world_memory.cleanup()
        _log_memory(f"after cleanup (broadcast {bid})")

    # ---- Cross-broadcast questions ----
    cross_remaining = [q for q in multi_bcast if q["index"] not in done_indices]
    if eval_index_set is not None:
        cross_remaining = [q for q in cross_remaining if q["index"] in eval_index_set]
    if cross_remaining and args.retrieval_backend == "unified_graph":
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {len(cross_remaining)} cross-broadcast questions")
        logger.info(f"{'='*50}")

        # Group by broadcast-set to reuse loaded graphs
        from collections import defaultdict as _dd
        bset_questions: Dict[frozenset, List[dict]] = _dd(list)
        for q in cross_remaining:
            bset_questions[frozenset(get_broadcast_ids(q))].append(q)

        for bset, bset_qs in bset_questions.items():
            bids_str = ",".join(sorted(bset))
            logger.info(f"  Broadcast set: {bids_str} ({len(bset_qs)} questions)")
            _log_memory(f"before cross-broadcast {bids_str}")

            # Create WorldMemory with multi-graph unified memory
            # Chain data is merged per-broadcast in MultiGraphUnifiedMemory.add()
            world_memory = WorldMemory(
                embedding_model=embedding_model,
                retriever_llm_model=retriever_llm,
                respond_llm_model=respond_llm,
                prompt_template_manager=prompt_mgr,
                episodic_granularities=["30sec", "3min", "10min", "1h"],
                max_rounds=args.max_rounds,
                max_errors=args.max_errors,
                retrieval_backend="independent",
                cache_dir=os.path.join(args.cache_dir, "cross_" + "_".join(sorted(bset))),
                chain_mode=args.chain_mode,
            )
            world_memory.qa_template_name = "qa_mmlifelong"
            world_memory.retrieval_backend = args.retrieval_backend
            world_memory.bm25_weight = args.bm25_weight
            world_memory.damping = args.damping
            world_memory.reasoning_version = "mmlifelong"
            world_memory.chain_min_hits = args.chain_min_hits
            world_memory.chain_max_topics = args.chain_max_topics
            world_memory.chain_max_events = args.chain_max_events
            world_memory.chain_topic_sim = args.chain_topic_sim
            world_memory.chain_storyline_sim = args.chain_storyline_sim
            world_memory.chain_storyline_min_hits = args.chain_storyline_min_hits
            world_memory.chain_storyline_granularities = tuple(
                g.strip() for g in args.chain_storyline_granularities.split(",") if g.strip()
            )
            world_memory.set_retrieval_top_k(
                episodic=args.episodic_top_k,
                semantic=args.semantic_top_k,
                visual=args.visual_top_k,
            )

            # Load all broadcasts in this set, build multi-graph
            from worldmm.memory.unified import UnifiedMemory
            from worldmm.memory.unified.memory import MultiGraphUnifiedMemory
            multi_mem = MultiGraphUnifiedMemory()
            load_ok = True
            for bid in sorted(bset):
                # Load episodic/semantic/visual into world_memory (accumulates)
                if not load_broadcast_data(world_memory, bid, args.metadata_dir, args.caption_mode):
                    logger.error(f"Failed to load broadcast {bid} for cross-broadcast question")
                    load_ok = False
                    break
                # Grab the unified memory that was just loaded and add to multi-graph
                if world_memory.unified_memory is not None:
                    multi_mem.add(bid, world_memory.unified_memory)
                    world_memory.unified_memory = None  # detach so next load doesn't overwrite

            if not load_ok:
                for q in bset_qs:
                    results.append({
                        "index": q["index"],
                        "broadcast_id": list(bset),
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "response": "Error",
                        "raw_score": 0,
                        "smoothed_score": 0.0,
                        "question_type": q.get("question_type", ""),
                        "temporal_certificate": q.get("temporal_certificate", ""),
                    })
                continue

            # Set the multi-graph as the unified memory
            world_memory.unified_memory = multi_mem

            for q in bset_qs:
                qidx = q["index"]
                question = q["question"]
                gold_answer = q["answer"]

                logger.info(f"  Q {qidx} (cross: {bids_str}): {question[:80]}...")

                qa_result: Optional[QAResult] = None
                try:
                    qa_result = world_memory.answer(
                        query=question,
                        choices=None,
                        until_time=float("inf"),
                    )
                    response = qa_result.answer
                except Exception as e:
                    logger.error(f"  Error on Q{qidx}: {e}")
                    response = "Error"

                if args.skip_judge:
                    judge_result = {"raw_score": None, "smoothed_score": None, "judge_response": "skipped"}
                elif response and response != "Error":
                    judge_result = llm_judge_score(question, response, gold_answer, judge_llm)
                else:
                    judge_result = {"raw_score": 0, "smoothed_score": 0.0, "judge_response": "N/A"}

                total_score += judge_result["smoothed_score"] or 0

                result_entry = {
                    "index": qidx,
                    "broadcast_id": list(bset),
                    "question": question,
                    "gold_answer": gold_answer,
                    "response": response,
                    "raw_score": judge_result["raw_score"],
                    "smoothed_score": judge_result["smoothed_score"],
                    "judge_response": judge_result["judge_response"],
                    "round_history": qa_result.round_history if qa_result else [],
                    "num_rounds": qa_result.num_rounds if qa_result else 0,
                    "question_type": q.get("question_type", ""),
                    "temporal_certificate": q.get("temporal_certificate", ""),
                }
                results.append(result_entry)

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

                total_done = len(results)
                avg_score = total_score / total_done if total_done else 0
                logger.info(
                    f"  Q{qidx} → Score: {judge_result['raw_score']}/5 "
                    f"(smoothed: {judge_result['smoothed_score']}) "
                    f"// Avg: {avg_score:.4f} ({total_done} done)"
                )

            world_memory.cleanup()
            _log_memory(f"after cleanup (cross {bids_str})")

    elif cross_remaining:
        logger.warning(
            f"Skipping {len(cross_remaining)} cross-broadcast questions "
            f"(only supported with unified_graph backend)"
        )

    # Final summary
    total_done = len(results)
    avg_score = total_score / total_done if total_done else 0
    logger.info(f"\n{'='*50}")
    logger.info(f"MM-Lifelong Evaluation Complete")
    logger.info(f"Total questions: {total_done}")
    logger.info(f"Average smoothed score: {avg_score:.4f}")
    logger.info(f"Results: {output_path}")

    # Per question_type breakdown
    type_stats: Dict[str, Dict[str, float]] = {}
    for r in results:
        qt = r.get("question_type", "Unknown") or "Unknown"
        if qt not in type_stats:
            type_stats[qt] = {"score_sum": 0.0, "total": 0}
        type_stats[qt]["total"] += 1
        type_stats[qt]["score_sum"] += r.get("smoothed_score") or 0.0

    logger.info("Per question_type accuracy:")
    for qt, s in sorted(type_stats.items()):
        acc = s["score_sum"] / s["total"] if s["total"] else 0
        logger.info(f"  {qt}: {s['score_sum']:.1f}/{s['total']} = {acc:.4f}")

    # Per temporal_certificate breakdown
    tc_stats: Dict[str, Dict[str, float]] = {}
    for r in results:
        tc = r.get("temporal_certificate", "Unknown") or "Unknown"
        if tc not in tc_stats:
            tc_stats[tc] = {"score_sum": 0.0, "total": 0}
        tc_stats[tc]["total"] += 1
        tc_stats[tc]["score_sum"] += r.get("smoothed_score") or 0.0

    logger.info("Per temporal_certificate accuracy:")
    for tc, s in sorted(tc_stats.items()):
        acc = s["score_sum"] / s["total"] if s["total"] else 0
        logger.info(f"  {tc}: {s['score_sum']:.1f}/{s['total']} = {acc:.4f}")

    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
