#!/usr/bin/env python3
"""
MM-Lifelong (month-level) preprocessing pipeline.

For a single broadcast, runs the complete preprocessing:
  1. Whisper ASR  (mp4 → SRT transcript, faster-whisper large-v3-turbo)
  2. Build 30sec captions from ASR  (align to 30sec bins + LLM rewrite to third-person)
  3. Multi-level aggregation  (30sec → 3min → 10min → 1h, via LLM)
  4. OpenIE/NER extraction  (from 30sec captions)
  5. Semantic extraction + consolidation
  6. Visual embedding extraction  (VLM2Vec)

Output directory:
  output/metadata/mmlifelong/{video_id}/
    whisper/{video_id}_whisper.srt
    whisper/{video_id}_whisper.json          (per-segment ASR output)
    captions/{video_id}_asr_30sec.json       (rewritten third-person ASR captions)
    captions/{video_id}_vlm_30sec.json       (raw VLM captions)
    {mode}/captions/{video_id}_30sec.json    (final captions for this mode)
    {mode}/captions/{video_id}_3min.json
    {mode}/captions/{video_id}_10min.json
    {mode}/captions/{video_id}_1h.json
    {mode}/episodic_memory/openie_results.json
    {mode}/semantic_memory/semantic_extraction_results.json
    {mode}/semantic_memory/semantic_consolidation_results.json
    visual_memory/visual_embeddings.pkl
"""

import argparse
import json
import logging
import math
import os
import pickle
import sys
from typing import Any, Dict, List

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from preprocess.common import (
    make_llm,
    get_video_duration,
    generate_captions_30sec,
    aggregate_captions,
    run_whisper_asr,
    align_asr_to_segments,
    merge_visual_and_asr,
    extract_openie,
    extract_and_consolidate_semantics,
    extract_visual_embeddings,
    run_videos,
)


# ---------------------------------------------------------------------------
# MM-Lifelong-specific: ASR rewrite to third-person
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """You are a professional video captioner. Your task is to rewrite raw speech transcripts from a livestream into clean, third-person video captions.

# Background
- The streamer is **IShowSpeed** (also known as "Speed"), a popular American YouTuber/streamer.
- These are IRL (In Real Life) livestream broadcasts spanning Feb 28 – Apr 20, 2025 (~51 days).
- He travels across multiple cities and countries during this period.

# Travel itinerary and locations
- **China**: Shanghai (Yu Garden, basketball court), Chongqing (light rail, river cruise, barbershop, parks), Changsha (shopping centers), Shenzhen, Beijing (Forbidden City, Great Wall), Nanjing
- **Mongolia**: visited museums, fan interactions
- **Hong Kong**: Victoria Harbour, gaming areas
- **London, UK**: 1V1 football challenge
- **Shaolin**: hiking at the foot of the mountain (warm-up, running, duck-walking, frog-jumping)

# Recurring people
- **Jackson Wang**: played basketball with IShowSpeed in Shanghai
- **Coco**: another creator whose China trip video IShowSpeed watched on Discord
- **Rosso, Alessa, Olivia**: companions/followers in Monster Hunter gameplay
- "Uncle in floral/colorful shirt": recurring fan encountered in multiple cities

# Common activities
- Street interactions with fans, trying local food, visiting landmarks
- Singing (especially on subways across different cities)
- Gaming (FRAGPUNK, Monster Hunter, Split or Steal)
- Dancing, talent shows, sports (football, basketball, tennis)
- Reacting to content on Discord, indoor chatting

# Common ASR Errors to Fix
- "Tongue King" / "Tongue King China" / "Tong King" → "Chongqing"
- "Gongzo" / "Guanzhou" → "Guangzhou"
- "Jia Kang" / "Jia Kang Gua" → likely a misheard Chinese name or phrase
- "Cyber City" (referring to Chongqing) → "Chongqing"
- "Chang sha" / "Chansha" → "Changsha"
- "Shenzen" → "Shenzhen"
- "Yu garden" / "You garden" → "Yu Garden (Shanghai)"

# Guidelines
1. Rewrite in **third person** ("IShowSpeed does X" not "I do X").
2. Filter out repetitive chat interaction noise ("spam the W's", "everybody subscribe", "like the stream").
3. Preserve meaningful content: locations, people met, events, actions, things said.
4. Fix obvious ASR errors, especially Chinese/Mongolian place names and person names.
5. If the transcript is purely filler/noise with no meaningful content, output a single short sentence summarizing the mood or setting.
6. Keep output concise: 1-3 sentences.

# Output
Output ONLY the rewritten caption. No JSON, no explanations."""


def rewrite_asr_captions(
    captions_30sec: List[Dict[str, Any]],
    llm_model,
    parallel: int = 8,
) -> List[Dict[str, Any]]:
    """Rewrite raw ASR 30sec captions into clean third-person descriptions."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _rewrite_one(idx: int, cap: Dict[str, Any]) -> str:
        text = cap.get("text", "").strip()
        if not text:
            return ""

        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Raw transcript [{cap['start_sec']:.0f}s - {cap['end_sec']:.0f}s]:\n{text}"},
        ]
        try:
            result = llm_model.generate(messages)
            return result.strip() if result else text
        except Exception as e:
            logger.warning(f"Rewrite failed for segment {idx}: {e}")
            return text

    rewritten = [None] * len(captions_30sec)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(_rewrite_one, i, cap): i
            for i, cap in enumerate(captions_30sec)
        }
        with tqdm(total=len(captions_30sec), desc="Rewriting ASR→caption") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                rewritten[idx] = future.result()
                pbar.update(1)

    result = []
    for i, cap in enumerate(captions_30sec):
        result.append({
            "start_sec": cap["start_sec"],
            "end_sec": cap["end_sec"],
            "text": rewritten[i] or "",
            "video_path": cap["video_path"],
        })
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_one_video(video_id: str, args) -> bool:
    """Process a single broadcast. Returns True on success."""
    video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return False

    base_dir = os.path.join(args.output_dir, "mmlifelong", video_id)
    mode = args.caption_mode

    # Shared dirs
    whisper_dir = os.path.join(base_dir, "whisper")
    vis_dir = os.path.join(base_dir, "visual_memory")

    # Mode-specific dirs
    mode_dir = os.path.join(base_dir, mode)
    cap_dir = os.path.join(mode_dir, "captions")
    ep_dir = os.path.join(mode_dir, "episodic_memory")
    sem_dir = os.path.join(mode_dir, "semantic_memory")

    for d in [whisper_dir, vis_dir, cap_dir, ep_dir, sem_dir]:
        os.makedirs(d, exist_ok=True)

    # Shared paths
    asr_srt_path = os.path.join(whisper_dir, f"{video_id}_whisper.srt")
    asr_json_path = os.path.join(whisper_dir, f"{video_id}_whisper.json")
    vis_emb_path = os.path.join(vis_dir, "visual_embeddings.pkl")

    # Mode-specific paths
    cap_30sec_path = os.path.join(cap_dir, f"{video_id}_30sec.json")
    cap_3min_path = os.path.join(cap_dir, f"{video_id}_3min.json")
    cap_10min_path = os.path.join(cap_dir, f"{video_id}_10min.json")
    cap_1h_path = os.path.join(cap_dir, f"{video_id}_1h.json")
    openie_path = os.path.join(ep_dir, "openie_results.json")
    sem_extract_path = os.path.join(sem_dir, "semantic_extraction_results.json")
    sem_consol_path = os.path.join(sem_dir, "semantic_consolidation_results.json")

    try:
        duration = get_video_duration(video_path)
        logger.info(f"[{video_id}] Duration: {duration:.1f}s ({duration/3600:.1f}h)")

        # Shared paths for raw outputs
        shared_cap_dir = os.path.join(base_dir, "captions")
        os.makedirs(shared_cap_dir, exist_ok=True)
        asr_30sec_path = os.path.join(shared_cap_dir, f"{video_id}_asr_30sec.json")
        vlm_30sec_path = os.path.join(shared_cap_dir, f"{video_id}_vlm_30sec.json")

        # Step 1: Whisper ASR (needed for 'asr' and 'merge' modes)
        if mode in ("asr", "merge") and not args.skip_whisper:
            if not os.path.exists(asr_json_path):
                logger.info(f"[{video_id}] Step 1a: Whisper ASR")
                asr_segments = run_whisper_asr(
                    video_path, asr_srt_path, asr_json_path,
                    model_size=args.whisper_model,
                    device=args.whisper_device,
                )
            else:
                with open(asr_json_path) as f:
                    asr_segments = json.load(f)
                logger.info(f"[{video_id}] Step 1a: Loaded existing ASR: {len(asr_segments)} segments")
        elif mode in ("asr", "merge"):
            if os.path.exists(asr_json_path):
                with open(asr_json_path) as f:
                    asr_segments = json.load(f)
            else:
                asr_segments = []

        # Step 2: Build 30sec captions
        if not args.skip_captions:
            num_segments = math.ceil(duration / args.segment_sec)
            llm = make_llm(args.llm_model, args.llm_base_url)

            # ASR path: align + rewrite to third-person
            if mode in ("asr", "merge"):
                if not os.path.exists(asr_30sec_path):
                    logger.info(f"[{video_id}] Step 2a: Building ASR 30sec captions")
                    asr_aligned = align_asr_to_segments(
                        asr_segments, args.segment_sec, duration,
                    )
                    raw_captions = []
                    for i in range(num_segments):
                        start_sec = i * args.segment_sec
                        end_sec = min((i + 1) * args.segment_sec, duration)
                        raw_captions.append({
                            "start_sec": start_sec, "end_sec": end_sec,
                            "text": asr_aligned.get(i, ""), "video_path": video_path,
                        })
                    # Rewrite to third-person
                    logger.info(f"[{video_id}] Step 2a: Rewriting ASR to third-person")
                    asr_captions = rewrite_asr_captions(raw_captions, llm, parallel=args.parallel)
                    with open(asr_30sec_path, "w") as f:
                        json.dump(asr_captions, f, indent=2)
                else:
                    logger.info(f"[{video_id}] Step 2a: Loading existing ASR captions")
                    with open(asr_30sec_path) as f:
                        asr_captions = json.load(f)

            # VLM path
            if mode in ("vlm", "merge"):
                if not os.path.exists(vlm_30sec_path):
                    logger.info(f"[{video_id}] Step 2b: Generating VLM captions")
                    from worldmm.llm.openrouter import OpenRouterModel
                    vlm = OpenRouterModel(model_name=args.vlm_model)
                    vlm_captions = generate_captions_30sec(
                        video_path, video_id, vlm,
                        segment_sec=args.segment_sec, parallel=args.parallel,
                    )
                    with open(vlm_30sec_path, "w") as f:
                        json.dump(vlm_captions, f, indent=2)
                else:
                    logger.info(f"[{video_id}] Step 2b: Loading existing VLM captions")
                    with open(vlm_30sec_path) as f:
                        vlm_captions = json.load(f)

            # Final 30sec based on mode
            if mode == "asr":
                captions_30sec = asr_captions
            elif mode == "vlm":
                captions_30sec = vlm_captions
            elif mode == "merge":
                logger.info(f"[{video_id}] Step 2c: Merging ASR + VLM")
                asr_aligned_dict = {
                    i: cap["text"] for i, cap in enumerate(asr_captions) if cap["text"].strip()
                }
                captions_30sec = merge_visual_and_asr(
                    vlm_captions, asr_aligned_dict, llm, parallel=args.parallel,
                )

            with open(cap_30sec_path, "w") as f:
                json.dump(captions_30sec, f, indent=2)
            logger.info(f"[{video_id}] 30sec captions done ({len(captions_30sec)} segments, mode={mode})")

            # Multi-level aggregation
            logger.info(f"[{video_id}] Step 3a: Aggregating to 3min")
            captions_3min = aggregate_captions(
                captions_30sec, llm, group_size=6, level_name="3min",
                parallel=args.parallel,
            )
            with open(cap_3min_path, "w") as f:
                json.dump(captions_3min, f, indent=2)

            logger.info(f"[{video_id}] Step 3b: Aggregating to 10min")
            captions_10min = aggregate_captions(
                captions_3min, llm, group_size=3, level_name="10min",
                parallel=args.parallel,
            )
            with open(cap_10min_path, "w") as f:
                json.dump(captions_10min, f, indent=2)

            logger.info(f"[{video_id}] Step 3c: Aggregating to 1h")
            captions_1h = aggregate_captions(
                captions_10min, llm, group_size=6, level_name="1h",
                parallel=args.parallel,
            )
            with open(cap_1h_path, "w") as f:
                json.dump(captions_1h, f, indent=2)

            logger.info(
                f"[{video_id}] Captions done: "
                f"30sec={len(captions_30sec)}, 3min={len(captions_3min)}, "
                f"10min={len(captions_10min)}, 1h={len(captions_1h)}"
            )
        else:
            if os.path.exists(cap_30sec_path):
                with open(cap_30sec_path) as f:
                    captions_30sec = json.load(f)
            elif not args.skip_visual:
                # Generate segment list from video duration for visual embedding
                # (no need to wait for caption generation)
                logger.info(f"[{video_id}] No captions yet, generating segment list from duration for visual")
                num_segments = math.ceil(duration / args.segment_sec)
                captions_30sec = [
                    {"start_sec": i * args.segment_sec,
                     "end_sec": min((i + 1) * args.segment_sec, duration),
                     "text": "", "video_path": video_path}
                    for i in range(num_segments)
                ]
            else:
                logger.info(f"[{video_id}] No 30sec captions yet, skipping downstream steps")
                return True

        # Step 4: OpenIE/NER
        if not args.skip_openie:
            logger.info(f"[{video_id}] Step 4: OpenIE/NER extraction")
            llm = make_llm(args.llm_model, args.llm_base_url)
            openie_data = extract_openie(captions_30sec, llm, combined=args.combined_openie)
            with open(openie_path, "w") as f:
                json.dump(openie_data, f, indent=2)
        else:
            if os.path.exists(openie_path):
                with open(openie_path) as f:
                    openie_data = json.load(f)
            else:
                openie_data = {"ner_results": {}, "triple_results": {}}

        # Step 5: Semantic
        if not args.skip_semantic:
            logger.info(f"[{video_id}] Step 5: Semantic extraction + consolidation")
            llm = make_llm(args.llm_model, args.llm_base_url)
            extraction_results, consolidation_results = extract_and_consolidate_semantics(
                openie_data, llm, parallel=args.parallel,
            )
            with open(sem_extract_path, "w") as f:
                json.dump(extraction_results, f, indent=2)
            with open(sem_consol_path, "w") as f:
                json.dump(consolidation_results, f, indent=2)

        # Step 6: Visual embeddings
        if not args.skip_visual:
            logger.info(f"[{video_id}] Step 6: Visual embedding extraction")
            from worldmm.embedding import EmbeddingModel
            emb_model = EmbeddingModel()
            emb_model.load_model("vision")
            vis_embeddings = extract_visual_embeddings(
                captions_30sec, emb_model,
                num_frames=args.num_frames,
            )
            with open(vis_emb_path, "wb") as f:
                pickle.dump(vis_embeddings, f)

        logger.info(f"[{video_id}] Done")
        return True

    except Exception as e:
        logger.error(f"[{video_id}] Failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess an MM-Lifelong broadcast")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-id", help="Single broadcast ID (e.g., '14')")
    group.add_argument("--video-ids", help="Comma-separated broadcast IDs (e.g., '14,18,19')")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of videos to process in parallel (default: 1)")
    parser.add_argument("--video-dir", default="data/MM-Lifelong/month",
                        help="Directory containing video files")
    parser.add_argument("--output-dir", default="output/metadata",
                        help="Output root directory")

    # Model arguments
    parser.add_argument("--vlm-model", default="qwen/qwen3.5-flash-02-23",
                        help="OpenRouter VLM model for caption generation")
    parser.add_argument("--llm-model", default="openai/gpt-oss-120b",
                        help="LLM model for merge/aggregation/NER/semantic")
    parser.add_argument("--llm-base-url", default=None,
                        help="Custom LLM API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--whisper-model", default="large-v3-turbo",
                        help="Whisper model size (default: large-v3-turbo)")
    parser.add_argument("--whisper-device", default="cuda",
                        help="Device for Whisper (default: cuda)")

    # Processing arguments
    parser.add_argument("--segment-sec", type=float, default=30.0,
                        help="Segment duration in seconds (default: 30)")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Frames per segment for visual embedding (default: 16)")
    parser.add_argument("--parallel", type=int, default=8,
                        help="Parallel workers for VLM/merge calls (default: 8)")

    parser.add_argument("--caption-mode", default="merge", choices=["asr", "vlm", "merge"],
                        help="Caption mode: asr (whisper+rewrite), vlm (VLM visual), merge (asr+vlm) (default: merge)")

    # Skip flags
    parser.add_argument("--combined-openie", action="store_true", default=True,
                        help="Use single LLM call for NER+Triple extraction (saves ~50%% API calls)")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-captions", action="store_true")
    parser.add_argument("--skip-openie", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()

    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = [v.strip() for v in args.video_ids.split(",")]

    run_videos(video_ids, process_one_video, args, workers=args.workers)


if __name__ == "__main__":
    main()
