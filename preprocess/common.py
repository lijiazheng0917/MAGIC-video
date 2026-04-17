#!/usr/bin/env python3
"""
Shared preprocessing utilities for all benchmarks (VideoMME, LVBench, MM-Lifelong).

Contains functions for:
  - LLM/VLM creation
  - Video utilities (ffmpeg, duration, clip encoding)
  - VLM caption generation (30sec segments, parallel)
  - Caption aggregation (parameterized group size + level name)
  - Whisper ASR + alignment
  - ASR + VLM merge
  - OpenIE/NER extraction
  - Semantic extraction + consolidation (parallel)
  - Visual embedding extraction
"""

import json
import logging
import math
import os
import re as _re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM / VLM creation
# ---------------------------------------------------------------------------

def make_llm(model_name: str, base_url: Optional[str] = None):
    """Create an LLM instance. If base_url is set, connect to a local vLLM server
    (with thinking disabled); otherwise use OpenRouter."""
    if base_url:
        from worldmm.llm.openrouter import OpenRouterModel
        from openai import OpenAI
        model = OpenRouterModel(model_name=model_name)
        model.client = OpenAI(api_key="dummy", base_url=base_url)
        model.model_name = model_name
        _orig_generate = model.generate
        def _generate_no_think(prompt, **kwargs):
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
            return _orig_generate(prompt, **kwargs)
        model.generate = _generate_no_think
        logger.info(f"LLM: {model_name} via {base_url} (thinking=off)")
        return model
    else:
        from worldmm.llm import LLMModel
        logger.info(f"LLM: {model_name} via OpenRouter")
        return LLMModel(model_name=model_name, provider="openrouter")


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def get_ffmpeg_path() -> str:
    """Get ffmpeg binary path."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil
        path = shutil.which("ffmpeg")
        if path:
            return path
        raise RuntimeError("ffmpeg not found. Install imageio-ffmpeg or add ffmpeg to PATH.")


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffmpeg (works with any codec)."""
    import subprocess
    ffmpeg = get_ffmpeg_path()
    result = subprocess.run(
        [ffmpeg, "-i", video_path],
        capture_output=True, text=True, timeout=30,
    )
    m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if m:
        h, mi, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return h * 3600 + mi * 60 + s + cs / 100.0
    raise RuntimeError(f"Could not parse duration from {video_path}: {result.stderr[:200]}")


def encode_video_clip_to_data_url(clip_path: str) -> str:
    """Encode a video clip file to a base64 data URL."""
    import base64
    with open(clip_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


# ---------------------------------------------------------------------------
# VLM caption generation (30sec segments, video clip-based, parallel)
# ---------------------------------------------------------------------------

def caption_one_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    vlm_model,
    ffmpeg: str,
) -> str:
    """Cut one segment and call VLM with video clip. Returns caption text."""
    import subprocess
    import tempfile

    seg_duration = end_sec - start_sec
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            ffmpeg,
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(seg_duration),
            "-vf", "scale=-2:360",
            "-c:v", "libx264", "-crf", "32", "-preset", "ultrafast",
            "-an",
            "-y", tmp_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            return ""

        video_data_url = encode_video_clip_to_data_url(tmp_path)

        content = [
            {
                "type": "text",
                "text": (
                    "Describe what happens in this video clip in 2-3 sentences. "
                    "Focus on actions, objects, people, and locations. "
                    "Be specific and factual."
                ),
            },
            {
                "type": "video_url",
                "video_url": {"url": video_data_url},
            },
        ]

        messages = [{"role": "user", "content": content}]
        text = vlm_model.generate(messages)
        return text.strip() if text else ""

    except Exception as e:
        logger.warning(f"Caption failed for {start_sec:.0f}-{end_sec:.0f}s: {e}")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def generate_captions_30sec(
    video_path: str,
    video_id: str,
    vlm_model,
    segment_sec: float = 30.0,
    parallel: int = 8,
) -> List[Dict[str, Any]]:
    """Generate captions for each 30sec segment of a video (video clip-based, parallel)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ffmpeg = get_ffmpeg_path()
    duration = get_video_duration(video_path)
    num_segments = math.ceil(duration / segment_sec)
    logger.info(f"Video {video_id}: {duration:.1f}s duration, {num_segments} segments, {parallel} workers")

    # Build segment list
    segments = []
    for i in range(num_segments):
        start_sec = i * segment_sec
        end_sec = min((i + 1) * segment_sec, duration)
        segments.append((i, start_sec, end_sec))

    # Parallel VLM calls
    results = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(caption_one_segment, video_path, s, e, vlm_model, ffmpeg): idx
            for idx, s, e in segments
        }
        with tqdm(total=len(segments), desc=f"Captioning {video_id}") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                pbar.update(1)

    # Assemble in order
    captions = []
    for idx, start_sec, end_sec in segments:
        captions.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": results.get(idx, ""),
            "video_path": video_path,
        })

    return captions


# ---------------------------------------------------------------------------
# Caption aggregation (parameterized, parallel)
# ---------------------------------------------------------------------------

def aggregate_captions(
    captions: List[Dict[str, Any]],
    llm_model,
    group_size: int,
    level_name: str,
    parallel: int = 8,
) -> List[Dict[str, Any]]:
    """Aggregate captions by grouping `group_size` consecutive entries."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    num_groups = math.ceil(len(captions) / group_size)

    # Build groups
    groups = []
    for g in range(num_groups):
        group = captions[g * group_size : (g + 1) * group_size]
        if group:
            groups.append((g, group))

    def _aggregate_one(g: int, group: List[Dict]) -> Dict[str, Any]:
        start_sec = group[0]["start_sec"]
        end_sec = group[-1]["end_sec"]
        video_path = group[0]["video_path"]

        segments_text = "\n".join(
            f"[{c['start_sec']:.0f}s - {c['end_sec']:.0f}s]: {c['text']}"
            for c in group if c["text"]
        )

        if not segments_text.strip():
            return {"start_sec": start_sec, "end_sec": end_sec, "text": "", "video_path": video_path}

        prompt = (
            f"Summarize the following video segment descriptions into a coherent "
            f"3-5 sentence summary. Preserve key events, people, and locations:\n\n"
            f"{segments_text}\n\n"
            f"Summary:"
        )

        try:
            text = llm_model.generate(prompt)
            if not text:
                text = ""
        except Exception as e:
            logger.warning(f"{level_name} aggregation failed for group {g}: {e}")
            text = " ".join(c["text"] for c in group if c["text"])

        return {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": text.strip() if text else "",
            "video_path": video_path,
        }

    # Parallel API calls
    results = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_aggregate_one, g, group): g for g, group in groups}
        with tqdm(total=len(groups), desc=f"Aggregating {level_name}") as pbar:
            for future in as_completed(futures):
                g = futures[future]
                results[g] = future.result()
                pbar.update(1)

    # Return in order
    return [results[g] for g, _ in groups]


# ---------------------------------------------------------------------------
# Whisper ASR
# ---------------------------------------------------------------------------

def run_whisper_asr(
    video_path: str,
    output_srt_path: str,
    output_json_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
) -> List[Dict[str, Any]]:
    """Run faster-whisper on video, output SRT and segment JSON.

    Returns list of {start_sec, end_sec, text} dicts.
    """
    from faster_whisper import WhisperModel

    logger.info(f"Loading Whisper model: {model_size} on {device}")
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info(f"Transcribing: {video_path}")
    segments, info = whisper_model.transcribe(
        video_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    logger.info(f"Detected language: {info.language} (prob={info.language_probability:.2f})")

    results = []
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        entry = {
            "start_sec": round(seg.start, 3),
            "end_sec": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        results.append(entry)

        def _fmt_srt_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int((sec % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        srt_lines.append(str(i))
        srt_lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")
        srt_lines.append(seg.text.strip())
        srt_lines.append("")

    os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Whisper done: {len(results)} segments, saved to {output_srt_path}")
    return results


def align_asr_to_segments(
    asr_segments: List[Dict[str, Any]],
    segment_sec: float,
    total_duration: float,
) -> Dict[int, str]:
    """Align ASR segments to fixed-length time bins.

    Returns dict mapping segment_index → concatenated transcript text.
    """
    num_segments = math.ceil(total_duration / segment_sec)
    bins: Dict[int, List[str]] = {i: [] for i in range(num_segments)}

    for seg in asr_segments:
        mid = (seg["start_sec"] + seg["end_sec"]) / 2.0
        idx = min(int(mid // segment_sec), num_segments - 1)
        text = seg["text"].strip()
        if text:
            bins[idx].append(text)

    return {idx: " ".join(texts) for idx, texts in bins.items() if texts}


# ---------------------------------------------------------------------------
# ASR + VLM merge
# ---------------------------------------------------------------------------

MERGE_SYSTEM_PROMPT = """You are an expert video captioner. Your task is to merge a visual description and a speech transcript into a single, coherent caption.

# Input
You will receive:
- **Visual**: A description of what is visually happening in the video segment.
- **Transcript**: What is being said/spoken during the same segment.

# Guidelines
1. Combine both sources into one fluent paragraph.
2. Keep the visual description as the primary narrative.
3. Integrate speech content naturally (e.g., "The narrator explains '...' while the camera shows").
4. If transcript is empty, return the visual description as-is.
5. If visual description is empty but transcript exists, describe the speech.
6. Be concise: 2-4 sentences.

# Output
Output ONLY the merged caption text. No JSON, no explanations."""


def merge_visual_and_asr(
    visual_captions: List[Dict[str, Any]],
    asr_aligned: Dict[int, str],
    llm_model,
    parallel: int = 8,
) -> List[Dict[str, Any]]:
    """Merge visual captions with aligned ASR transcript using LLM."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _merge_one(idx: int, cap: Dict[str, Any]) -> str:
        visual_text = cap.get("text", "").strip()
        asr_text = asr_aligned.get(idx, "").strip()

        if not asr_text:
            return visual_text
        if not visual_text:
            return f"[Speech] {asr_text}"

        prompt = f"Visual: {visual_text}\nTranscript: {asr_text}"
        messages = [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            result = llm_model.generate(messages)
            return result.strip() if result else visual_text
        except Exception as e:
            logger.warning(f"Merge failed for segment {idx}: {e}")
            return f"{visual_text} [Speech: {asr_text}]"

    merged = [None] * len(visual_captions)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(_merge_one, i, cap): i
            for i, cap in enumerate(visual_captions)
        }
        with tqdm(total=len(visual_captions), desc="Merging visual+ASR") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                merged[idx] = future.result()
                pbar.update(1)

    result = []
    for i, cap in enumerate(visual_captions):
        result.append({
            "start_sec": cap["start_sec"],
            "end_sec": cap["end_sec"],
            "text": merged[i] or cap.get("text", ""),
            "video_path": cap["video_path"],
        })
    return result


# ---------------------------------------------------------------------------
# OpenIE / NER extraction
# ---------------------------------------------------------------------------

def extract_openie(
    captions_30sec: List[Dict[str, Any]],
    llm_model,
    combined: bool = False,
) -> Dict[str, Any]:
    """Run OpenIE + NER on 30sec captions using batch_openie.

    Args:
        combined: If True, use a single LLM call per chunk for both NER and
            triple extraction (saves ~50% API calls). If False (default), use
            the original two-call approach.
    """
    from worldmm.memory.episodic.openie import OpenIE

    openie = OpenIE(llm_model)
    texts = [cap["text"] for cap in captions_30sec if cap.get("text", "").strip()]

    if not texts:
        return {"ner_results": {}, "triple_results": {}}

    mode_str = "combined" if combined else "separate"
    logger.info(f"Running OpenIE/NER on {len(texts)} caption segments ({mode_str} mode)...")
    ner_results, triple_results = openie.batch_openie(texts, output_dir=".", combined=combined)

    return {
        "ner_results": ner_results,
        "triple_results": triple_results,
    }


# ---------------------------------------------------------------------------
# Semantic extraction + consolidation (parallel)
# ---------------------------------------------------------------------------

def normalize_subject(subj: str) -> str:
    """Normalize subject name for grouping."""
    return subj.strip().lower().replace("'", "").replace("-", " ")


def batch_consolidate_group(triples: List[List[str]], llm_model) -> List[List[str]]:
    """Consolidate a group of triples sharing the same subject via a single LLM call.

    Sends all triples for one subject to the LLM and asks it to deduplicate/merge.
    Returns the consolidated list of triples.
    """
    if len(triples) <= 1:
        return triples

    formatted = "\n".join(
        f"{i}. [{t[0]}, {t[1]}, {t[2]}]" for i, t in enumerate(triples)
    )

    prompt = (
        f"Below are semantic triples about the same subject. "
        f"Merge duplicates and remove redundant ones. Keep distinct facts.\n\n"
        f"{formatted}\n\n"
        f"Return ONLY a JSON array of merged triples, each as [subject, predicate, object]. "
        f"No explanation."
    )

    try:
        response = llm_model.generate(prompt)
        if not response:
            return triples

        match = _re.search(r'\[.*\]', response, _re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            result = [t for t in parsed if isinstance(t, list) and len(t) >= 3]
            if result:
                return result
    except Exception as e:
        logger.warning(f"Batch consolidation failed: {e}")

    return triples


def extract_and_consolidate_semantics(
    openie_data: Dict[str, Any],
    llm_model,
    parallel: int = 8,
) -> tuple:
    """Run semantic extraction and consolidation (parallel API calls).

    Consolidation uses batch-by-subject strategy: group triples by subject,
    then send each group to LLM in one call.
    """
    from worldmm.memory.semantic.semantic_extraction import SemanticExtraction
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    extractor = SemanticExtraction(llm_model)

    triple_results = openie_data.get("triple_results", {})

    # Parallel semantic extraction
    def _extract_one(chunk_id, triples):
        result = extractor.semantic_extraction(chunk_id, triples)
        return chunk_id, result

    extraction_results = {}
    all_semantic_triples = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(_extract_one, cid, triples): cid
            for cid, triples in triple_results.items()
        }
        with tqdm(total=len(futures), desc="Semantic extraction") as pbar:
            for future in as_completed(futures):
                chunk_id, result = future.result()
                extraction_results[chunk_id] = {
                    "semantic_triples": result.semantic_triples,
                    "episodic_evidence": result.episodic_evidence,
                }
                all_semantic_triples.extend(result.semantic_triples)
                pbar.update(1)

    # Consolidation: group by normalized subject, then batch-merge per group
    consolidated = {}
    if all_semantic_triples:
        subject_groups = defaultdict(list)
        for triple in all_semantic_triples:
            if len(triple) < 3:
                continue
            key = normalize_subject(triple[0])
            subject_groups[key].append(triple)

        logger.info(
            f"Consolidating {len(all_semantic_triples)} triples "
            f"across {len(subject_groups)} subject groups"
        )

        # Parallel consolidation
        def _consolidate_one(subj_key, group):
            return subj_key, batch_consolidate_group(group, llm_model)

        consolidated_triples = []
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_consolidate_one, sk, grp): sk
                for sk, grp in subject_groups.items()
            }
            with tqdm(total=len(futures), desc="Semantic consolidation") as pbar:
                for future in as_completed(futures):
                    _, merged = future.result()
                    consolidated_triples.extend(merged)
                    pbar.update(1)

        consolidated["0"] = {
            "consolidated_semantic_triples": consolidated_triples,
        }
        logger.info(
            f"Consolidated: {len(all_semantic_triples)} → {len(consolidated_triples)} triples"
        )

    return extraction_results, consolidated


# ---------------------------------------------------------------------------
# Visual embedding extraction
# ---------------------------------------------------------------------------

def _cut_one_clip(args_tuple):
    """Cut a single clip from a video using ffmpeg. Returns (emb_key, tmp_path) or (emb_key, None)."""
    import subprocess
    import tempfile

    ffmpeg, video_path, start_sec, end_sec, emb_key = args_tuple
    duration = end_sec - start_sec
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            ffmpeg,
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
            "-an",
            "-y", tmp_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            return (emb_key, tmp_path)
    except Exception:
        pass
    return (emb_key, None)


def extract_visual_embeddings(
    captions_30sec: List[Dict[str, Any]],
    embedding_model,
    num_frames: int = 16,
    cut_workers: int = 4,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Extract visual embeddings for each 30sec segment.

    Optimized: parallel ffmpeg cutting + batched GPU inference.
    """
    from concurrent.futures import ThreadPoolExecutor

    ffmpeg = get_ffmpeg_path()
    embeddings = {}

    # Build work list
    work = []
    for cap in captions_30sec:
        video_path = cap.get("video_path")
        start_sec = cap.get("start_sec", 0)
        end_sec = cap.get("end_sec", 0)
        if not video_path:
            continue
        emb_key = f"{video_path}:{start_sec}-{end_sec}"
        work.append((ffmpeg, video_path, start_sec, end_sec, emb_key))

    # Process in batches: cut clips in parallel, then run GPU inference
    for batch_start in tqdm(range(0, len(work), batch_size), desc="Visual embeddings",
                            total=(len(work) + batch_size - 1) // batch_size):
        batch = work[batch_start:batch_start + batch_size]

        # Parallel ffmpeg cutting
        with ThreadPoolExecutor(max_workers=cut_workers) as pool:
            cut_results = list(pool.map(_cut_one_clip, batch))

        # Collect successful clips
        keys_and_paths = [(k, p) for k, p in cut_results if p is not None]
        if not keys_and_paths:
            continue

        # GPU inference one by one (model internal limitation)
        for emb_key, tmp_path in keys_and_paths:
            try:
                emb = embedding_model.encode_video(
                    [tmp_path],
                    nframes=num_frames,
                )
                if emb is not None and len(emb) > 0:
                    embeddings[emb_key] = emb[0]
            except Exception as e:
                logger.warning(f"Visual embedding failed for {emb_key}: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return embeddings


# ---------------------------------------------------------------------------
# Multi-video parallel runner (shared main loop logic)
# ---------------------------------------------------------------------------

def run_videos(video_ids: List[str], process_fn, args, workers: int = 1):
    """Run process_fn on each video, optionally in parallel.

    Exits with code 1 if any video fails, so SLURM afterok dependencies work.
    """
    failed = []
    if len(video_ids) == 1 or workers <= 1:
        for vid in video_ids:
            ok = process_fn(vid, args)
            if not ok:
                failed.append(vid)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info(f"Processing {len(video_ids)} videos with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_fn, vid, args): vid for vid in video_ids}
            for future in as_completed(futures):
                vid = futures[future]
                try:
                    ok = future.result()
                    logger.info(f"[{vid}] {'OK' if ok else 'FAILED'}")
                    if not ok:
                        failed.append(vid)
                except Exception as e:
                    logger.error(f"[{vid}] Exception: {e}")
                    failed.append(vid)

    if failed:
        logger.error(f"Failed videos: {failed}")
        sys.exit(1)
