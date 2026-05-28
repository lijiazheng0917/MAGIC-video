#!/usr/bin/env python3
"""
EgoLifeQA evaluation script using WorldMM unified memory system.
"""

import argparse
import json
import logging
import os
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult

from _common import (
    load_json,
    resolve_semantic_results_path,
    evaluate_prediction,
    parse_target_time,
)


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
        "--event-chain-path", type=str,
        default="output/metadata/event_chains/A1_JAKE/step3_enriched_chains.json",
        help="Path to step3_enriched_chains.json (cross-time event chains)."
    )
    parser.add_argument("--chain-min-hits", type=int, default=1,
                        help="Topic chain coarse filter: min episode hits (default 1)")
    parser.add_argument("--chain-max-topics", type=int, default=3,
                        help="Max topic chains to inject per round (default 3)")
    parser.add_argument("--chain-max-events", type=int, default=1,
                        help="Max event chains to inject per round (default 1)")
    parser.add_argument("--chain-topic-sim", type=float, default=0.7,
                        help="Topic chain embedding similarity threshold (default 0.7)")
    parser.add_argument("--chain-event-sim", type=float, default=0.7,
                        help="Event-chain embedding similarity threshold (default 0.7)")
    parser.add_argument("--chain-event-min-hits", type=int, default=1,
                        help="Minimum event-chain step hits required for candidacy (default 1)")
    parser.add_argument("--chain-event-granularities", type=str,
                        default="30sec,3min",
                        help="Comma-separated episode granularities used to "
                             "detect event-chain step hits (default '30sec,3min'). "
                             "Add '10min' / '1h' to loosen coarse filter.")
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
        event_chain_path=args.event_chain_path,
    )

    # Set parameters (unified_graph backend)
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
