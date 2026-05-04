#!/usr/bin/env python3
"""
Offline script to build and serialise the unified multimodal graph.

Reads from already-computed preprocessing outputs:
  - data/EgoLife/EgoLifeCap/{PERSON}/A1_JAKE_{30sec,3min,10min,1h}.json
  - output/metadata/episodic_memory/{PERSON}/openie_results_*.json
  - output/metadata/semantic_memory/{PERSON}/semantic_consolidation_results_*.json
  - output/metadata/visual_memory/{PERSON}/visual_embeddings.pkl

Writes:
  - output/metadata/unified_graph/{PERSON}/graph.pkl

Run once before evaluation with --retrieval-backend unified_graph.
"""

import argparse
import glob
import json
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_latest(pattern: str) -> str:
    """Return the most recently modified file matching the glob pattern."""
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No file found matching: {pattern}")
    return files[0]


def _resolve_egolife_paths(args, person):
    """Resolve all input/output paths for EgoLife dataset."""
    cap_dir  = os.path.join(args.data_dir, "EgoLifeCap", person)
    ep_dir   = os.path.join(args.output_dir, "episodic_memory",  person)
    sem_dir  = os.path.join(args.output_dir, "semantic_memory",  person)
    vis_dir  = os.path.join(args.output_dir, "visual_memory",    person)
    out_dir  = os.path.join(args.output_dir, "unified_graph",    person)

    caption_files = {
        gran: os.path.join(cap_dir, f"{person}_{gran}.json")
        for gran in ["30sec", "3min", "10min", "1h"]
    }
    for gran, path in caption_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Caption file not found: {path}")

    return {
        "caption_files":  caption_files,
        "openie_path":    _find_latest(os.path.join(ep_dir, "openie_results_*.json")),
        "sem_path":       _find_latest(os.path.join(sem_dir, "semantic_consolidation_results_*.json")),
        "vis_emb_path":   os.path.join(vis_dir, "visual_embeddings.pkl"),
        "out_dir":        out_dir,
        "granularities":  ["30sec", "3min", "10min", "1h"],
    }


def _resolve_videomme_paths(args, video_id):
    """Resolve all input/output paths for Video-MME dataset.

    Uses --caption-mode to select mode-specific subdirectory.
    """
    base = os.path.join(args.output_dir, "videomme", video_id)
    mode = getattr(args, 'caption_mode', 'srt')
    mode_dir = os.path.join(base, mode)
    cap_dir = os.path.join(mode_dir, "captions")
    out_dir = os.path.join(mode_dir, "unified_graph")

    granularities = ["30sec", "3min"]
    caption_files = {}
    for gran in granularities:
        path = os.path.join(cap_dir, f"{video_id}_{gran}.json")
        if os.path.exists(path):
            caption_files[gran] = path

    if not caption_files:
        raise FileNotFoundError(f"No caption files found in {cap_dir}")

    return {
        "caption_files":  caption_files,
        "openie_path":    os.path.join(mode_dir, "episodic_memory", "openie_results.json"),
        "sem_path":       os.path.join(mode_dir, "semantic_memory", "semantic_consolidation_results.json"),
        "vis_emb_path":   os.path.join(base, "visual_memory", "visual_embeddings.pkl"),
        "out_dir":        out_dir,
        "granularities":  granularities,
    }


def _resolve_lvbench_paths(args, video_id):
    """Resolve all input/output paths for LVBench dataset.

    Uses --caption-mode to select mode-specific subdirectory for caption-dependent
    files (captions, episodic, semantic, graph), while visual embeddings are shared.
    """
    base = os.path.join(args.output_dir, "lvbench", video_id)
    mode = getattr(args, 'caption_mode', 'asr')
    mode_dir = os.path.join(base, mode)
    cap_dir = os.path.join(mode_dir, "captions")
    out_dir = os.path.join(mode_dir, "unified_graph")

    granularities = ["30sec", "3min"]
    caption_files = {}
    for gran in granularities:
        path = os.path.join(cap_dir, f"{video_id}_{gran}.json")
        if os.path.exists(path):
            caption_files[gran] = path

    if not caption_files:
        raise FileNotFoundError(f"No caption files found in {cap_dir}")

    return {
        "caption_files":  caption_files,
        "openie_path":    os.path.join(mode_dir, "episodic_memory", "openie_results.json"),
        "sem_path":       os.path.join(mode_dir, "semantic_memory", "semantic_consolidation_results.json"),
        "vis_emb_path":   os.path.join(base, "visual_memory", "visual_embeddings.pkl"),
        "out_dir":        out_dir,
        "granularities":  granularities,
    }


def _resolve_mmlifelong_paths(args, video_id):
    """Resolve all input/output paths for MM-Lifelong dataset."""
    base = os.path.join(args.output_dir, "mmlifelong", video_id)
    mode = getattr(args, 'caption_mode', 'asr')
    mode_dir = os.path.join(base, mode)
    cap_dir = os.path.join(mode_dir, "captions")
    out_dir = os.path.join(mode_dir, "unified_graph")

    granularities = ["30sec", "3min", "10min", "1h"]
    caption_files = {}
    for gran in granularities:
        path = os.path.join(cap_dir, f"{video_id}_{gran}.json")
        if os.path.exists(path):
            caption_files[gran] = path

    if not caption_files:
        raise FileNotFoundError(f"No caption files found in {cap_dir}")

    return {
        "caption_files":  caption_files,
        "openie_path":    os.path.join(mode_dir, "episodic_memory", "openie_results.json"),
        "sem_path":       os.path.join(mode_dir, "semantic_memory", "semantic_consolidation_results.json"),
        "vis_emb_path":   os.path.join(base, "visual_memory", "visual_embeddings.pkl"),
        "out_dir":        out_dir,
        "granularities":  granularities,
    }


def main():
    parser = argparse.ArgumentParser(description="Build unified multimodal graph")
    parser.add_argument("--subject", default="A1_JAKE", help="Subject ID (EgoLife)")
    parser.add_argument("--dataset", default="egolife", choices=["egolife", "videomme", "lvbench", "mmlifelong"],
                        help="Dataset: 'egolife' (default), 'videomme', or 'lvbench'")
    parser.add_argument("--video-id", default=None,
                        help="Video ID. Required when --dataset=videomme or lvbench")
    parser.add_argument("--caption-mode", default="merge", choices=["asr", "vlm", "merge"],
                        help="Caption mode (determines subdirectory under metadata; default: merge).")
    parser.add_argument("--data-dir",   default="data/EgoLife",    help="Data root")
    parser.add_argument("--output-dir", default="output/metadata", help="Output root")
    parser.add_argument("--embedding-model",
                        default="Qwen/Qwen3-Embedding-4B",
                        help="Text embedding model name")
    parser.add_argument("--embedding-device", default="cuda",
                        help="Device for embedding computation")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for text embedding computation")
    args = parser.parse_args()

    # ------------------------------------------------------------------ resolve paths
    if args.dataset == "videomme":
        if not args.video_id:
            parser.error("--video-id is required when --dataset=videomme")
        person = "narrator"
        paths = _resolve_videomme_paths(args, args.video_id)
        label = f"Video-MME/{args.video_id}"
    elif args.dataset == "lvbench":
        if not args.video_id:
            parser.error("--video-id is required when --dataset=lvbench")
        person = "narrator"
        paths = _resolve_lvbench_paths(args, args.video_id)
        label = f"LVBench/{args.video_id}"
    elif args.dataset == "mmlifelong":
        if not args.video_id:
            parser.error("--video-id is required when --dataset=mmlifelong")
        person = "ishowspeed"
        paths = _resolve_mmlifelong_paths(args, args.video_id)
        label = f"MM-Lifelong/{args.video_id}"
    else:
        person = args.subject
        paths = _resolve_egolife_paths(args, person)
        label = f"EgoLife/{person}"

    out_dir  = paths["out_dir"]
    out_path = os.path.join(out_dir, "graph.pkl")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Dataset: {label}")
    logger.info(f"NER/openie file : {paths['openie_path']}")
    logger.info(f"Semantic file   : {paths['sem_path']}")
    logger.info(f"Visual emb file : {paths['vis_emb_path']}")

    # ------------------------------------------------------------------ load 30sec captions
    caption_30sec_path = paths["caption_files"].get("30sec")
    if caption_30sec_path:
        logger.info("Loading 30sec captions…")
        with open(caption_30sec_path) as f:
            caption_30sec = json.load(f)
    else:
        caption_30sec = []
        logger.warning("No 30sec caption file found; visual/entity nodes will be empty.")

    # ------------------------------------------------------------------ load visual embeddings
    vis_emb_path = paths["vis_emb_path"]
    if os.path.exists(vis_emb_path):
        logger.info("Loading visual embeddings…")
        with open(vis_emb_path, "rb") as f:
            visual_embeddings = pickle.load(f)
        logger.info(f"  {len(visual_embeddings)} embeddings loaded.")
    else:
        visual_embeddings = {}
        logger.warning(f"Visual embeddings not found: {vis_emb_path}")

    # ------------------------------------------------------------------ build graph
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from worldmm.memory.unified.graph import MultimodalGraph

    graph = MultimodalGraph()

    logger.info("Step 1/4  Adding episode nodes…")
    graph.add_episode_nodes(paths["caption_files"])
    logger.info(f"  Stats: {graph.stats()}")

    logger.info("Step 2/4  Adding visual clip nodes…")
    graph.add_visual_nodes(caption_30sec, visual_embeddings)
    logger.info(f"  Stats: {graph.stats()}")

    logger.info("Step 3/4  Adding entity nodes…")
    if os.path.exists(paths["openie_path"]):
        graph.add_entity_nodes(paths["openie_path"], caption_30sec, person=person)
    else:
        logger.warning(f"OpenIE file not found: {paths['openie_path']}; skipping entity nodes.")
    logger.info(f"  Stats: {graph.stats()}")

    logger.info("Step 4/4  Adding semantic nodes…")
    if os.path.exists(paths["sem_path"]):
        graph.add_semantic_nodes(paths["sem_path"], person=person)
    else:
        logger.warning(f"Semantic file not found: {paths['sem_path']}; skipping semantic nodes.")
    logger.info(f"  Stats: {graph.stats()}")

    # ------------------------------------------------------------------ compute text embeddings
    logger.info("Computing text embeddings (this may take a while)…")
    from worldmm.embedding import EmbeddingModel
    embedding_model = EmbeddingModel(
        text_model_name=args.embedding_model,
        device=args.embedding_device,
    )
    embedding_model.load_model("text")
    graph.compute_text_embeddings(embedding_model, batch_size=args.batch_size)

    # ------------------------------------------------------------------ save
    graph.save(out_path)
    logger.info(f"\nGraph saved to: {out_path}")
    logger.info(f"Final stats: {graph.stats()}")


if __name__ == "__main__":
    main()
