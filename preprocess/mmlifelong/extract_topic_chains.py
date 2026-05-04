#!/usr/bin/env python3
"""
Extract topic (entity-lifecycle) chains for MM-Lifelong broadcasts.

For each video, finds entities appearing in 2+ 30sec captions,
then asks LLM to extract key facts about each entity.

Usage:
    python preprocess/mmlifelong/extract_topic_chains.py --video-id 4
    python preprocess/mmlifelong/extract_topic_chains.py --video-id all
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a precise video fact extractor. Output valid JSON only."

EXTRACT_PROMPT = """In a livestream video by IShowSpeed, the entity "{entity}" appears in multiple segments at different times.

Below are the 30-second episode captions where "{entity}" is mentioned:

{episodes_text}

Extract facts about "{entity}" that would help answer questions about the livestream, such as:
- WHAT happened with "{entity}"? (specific actions, events, outcomes)
- WHERE did it happen? (city, venue, location — important for travel streams)
- WHO was involved? (IShowSpeed, fans, specific people by name/description)
- WHAT changed? (state changes, before/after)

Focus on:
1. **Specific events**: what exactly happened, not vague descriptions
2. **Location context**: which city or place this occurred in
3. **People involved**: IShowSpeed, specific fans, celebrities, staff
4. **Concrete details**: song names, food names, game names, scores, prices

Always refer to the streamer as "IShowSpeed". Never use "he", "they", "the man".

Skip trivial mentions where "{entity}" appears in passing without meaningful action.

Good facts:
- "IShowSpeed sang {entity} while riding the bus in Chongqing"
- "A fan wearing a {entity} jersey approached IShowSpeed on the street in Shanghai"
- "IShowSpeed tried {entity} flavor ice cream at the dessert shop in Chengdu"

Bad facts (skip these):
- "IShowSpeed looked at {entity}" (no meaningful action)
- "{entity} was visible in the background" (no action)

Output a JSON array (empty array [] if no meaningful facts):
[
  {{"time": "01:30:00", "fact": "descriptive fact about {entity}"}},
  {{"time": "02:15:00", "fact": "another fact"}}
]"""


def format_sec_to_hms(sec):
    """Convert float seconds to HH:MM:SS string."""
    total = int(sec)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def call_llm(client, model, prompt, system=SYSTEM_PROMPT, max_tokens=4096, max_retries=3):
    raw = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                time.sleep(1)
                continue
            clean = re.sub(r'```json\s*|```\s*', '', raw)
            clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            if raw:
                for sc, ec in [('{', '}'), ('[', ']')]:
                    start = raw.find(sc)
                    if start >= 0:
                        depth = 0
                        for i in range(start, len(raw)):
                            if raw[i] == sc: depth += 1
                            elif raw[i] == ec:
                                depth -= 1
                                if depth == 0:
                                    try: return json.loads(raw[start:i+1])
                                    except json.JSONDecodeError: break
            if attempt < max_retries - 1:
                time.sleep(1)
            logger.warning(f"JSON parse error (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="4", help="Video ID or 'all'")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--metadata-dir", default="output/metadata/mmlifelong")
    parser.add_argument("--caption-mode", default="merge")
    parser.add_argument("--min-mentions", type=int, default=2,
                        help="Minimum number of captions mentioning an entity")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Determine which videos to process
    if args.video_id == "all":
        video_ids = sorted([d for d in os.listdir(args.metadata_dir)
                           if os.path.isdir(os.path.join(args.metadata_dir, d))])
    else:
        video_ids = [args.video_id]

    # Setup LLM
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    for vid in video_ids:
        logger.info(f"=== Processing video {vid} ===")

        # Paths
        cap_path = os.path.join(args.metadata_dir, vid, args.caption_mode,
                                "captions", f"{vid}_30sec.json")
        sem_path = os.path.join(args.metadata_dir, vid, args.caption_mode,
                                "semantic_memory", "semantic_extraction_results.json")
        out_dir = os.path.join(args.metadata_dir, vid, args.caption_mode)
        out_path = os.path.join(out_dir, "topic_chains.json")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(cap_path):
            logger.warning(f"No captions at {cap_path}, skipping")
            continue

        # Load 30sec captions
        captions = json.load(open(cap_path))
        logger.info(f"Loaded {len(captions)} captions")

        # Step 1: Extract entity candidates from semantic triples, then LLM-filter.
        all_entities = set()
        if os.path.exists(sem_path):
            sem_data = json.load(open(sem_path))
            skip_words = {
                'ishowspeed', 'speed', 'streamer', 'man', 'person', 'people', 'fan', 'fans',
                'he', 'she', 'they', 'i', 'we', 'someone', 'everyone', 'narrator',
                'video', 'stream', 'chat', 'livestream', 'broadcast',
                'viewer', 'viewers', 'audience', 'subscriber', 'subscribers',
                'true', 'false', 'yes', 'no', 'something', 'anything', 'nothing',
                'time', 'moment', 'scene', 'clip', 'frame',
            }
            for chunk_data in sem_data.values():
                if isinstance(chunk_data, dict):
                    for triple in chunk_data.get("semantic_triples", []):
                        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                            for ent in [triple[0].lower().strip(), triple[2].lower().strip()]:
                                if len(ent) >= 2 and ent not in skip_words:
                                    all_entities.add(ent)
            logger.info(f"Entities from semantic triples: {len(all_entities)}")

        entity_mentions = defaultdict(list)
        for ent in all_entities:
            pat = re.compile(r'\b' + re.escape(ent) + r's?\b', re.IGNORECASE)
            for i, cap in enumerate(captions):
                if pat.search(cap.get('text', '')):
                    entity_mentions[ent].append(i)

        candidates = {ent: indices for ent, indices in entity_mentions.items()
                      if len(indices) >= args.min_mentions}
        logger.info(f"Candidates (>={args.min_mentions} mentions): {len(candidates)}")

        # LLM filter
        FILTER_PROMPT = """Below is a list of entities extracted from a livestream video.
Remove the noise entities and keep only meaningful ones.

Remove:
- UI/layout elements (bottom-right, split-screen, grid, inset)
- Body parts (hands, face, mouth, eyes)
- Visual descriptions (brightly lit room, purple background)
- Generic actions/emotions (change, talks, smiling, shock)
- Quoted phrases or full sentences
- Vague words (this, that, stuff, thing)
- Generic person descriptions (young man, another person, bearded man)
- Stream interface elements (viewer count, chat log, donation alert)

Keep:
- Named people, places, countries, cities
- Specific objects, instruments, food, animals
- Games, sports, platforms, apps
- Clothing items that identify specific people

Entity list:
{entity_list}

Return a JSON array of entity names to KEEP:
["entity1", "entity2", ...]"""

        if candidates:
            all_cands = sorted(candidates.keys())
            kept_entities = set()
            BATCH = 100
            for i in range(0, len(all_cands), BATCH):
                batch = all_cands[i:i+BATCH]
                entity_list = "\n".join(f"- {ent} ({len(candidates[ent])} mentions)" for ent in batch)
                prompt = FILTER_PROMPT.format(entity_list=entity_list)
                result = call_llm(client, args.model, prompt, max_tokens=2000)
                if isinstance(result, list):
                    for ent in result:
                        if isinstance(ent, str) and ent.lower() in candidates:
                            kept_entities.add(ent.lower())
                    logger.info(f"  LLM filter batch {i//BATCH+1}: {len(batch)} → {len([e for e in result if isinstance(e, str) and e.lower() in candidates])} kept")
                else:
                    logger.warning(f"  LLM filter batch {i//BATCH+1} failed, keeping all")
                    kept_entities.update(e.lower() for e in batch)

            before = len(candidates)
            candidates = {ent: indices for ent, indices in candidates.items() if ent in kept_entities}
            logger.info(f"LLM filter: {before} → {len(candidates)} entities")

        # Resume support
        existing = {}
        if args.resume and os.path.exists(out_path):
            existing = json.load(open(out_path))
            logger.info(f"Resuming: {len(existing)} entities already processed")

        todo = [(ent, indices) for ent, indices in candidates.items() if ent not in existing]
        logger.info(f"To process: {len(todo)} entities")

        CHUNK_SIZE = 50
        results = dict(existing)
        lock = threading.Lock()
        processed = [0]
        useful = [0]

        def process_one(ent, cap_indices):
            # Collect captions mentioning this entity
            pat = re.compile(r'\b' + re.escape(ent) + r's?\b', re.IGNORECASE)
            lines = []
            for cap in captions:
                if pat.search(cap.get('text', '')):
                    ts = format_sec_to_hms(cap.get('start_sec', 0))
                    lines.append(f"[{ts}] {cap.get('text', '')}")

            if not lines:
                return ent, {"entity": ent, "has_lifecycle": False, "facts": []}

            all_facts = []
            for i in range(0, len(lines), CHUNK_SIZE):
                chunk = lines[i:i + CHUNK_SIZE]
                episodes_text = "\n\n".join(chunk)
                prompt = EXTRACT_PROMPT.format(entity=ent, episodes_text=episodes_text)
                result = call_llm(client, args.model, prompt)
                if isinstance(result, list):
                    all_facts.extend([f for f in result if isinstance(f, dict) and f.get("fact")])

            if all_facts:
                return ent, {"entity": ent, "has_lifecycle": True, "facts": all_facts}
            return ent, {"entity": ent, "has_lifecycle": False, "facts": []}

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one, ent, indices): ent
                       for ent, indices in todo}
            for future in as_completed(futures):
                try:
                    ent, result = future.result()
                    with lock:
                        results[ent] = result
                        processed[0] += 1
                        if result.get("has_lifecycle"):
                            useful[0] += 1
                            n_facts = len(result.get("facts", []))
                            logger.info(f"  [{processed[0]}/{len(todo)}] {ent}: {n_facts} facts")
                        else:
                            if processed[0] % 10 == 0:
                                logger.info(f"  [{processed[0]}/{len(todo)}] progress... "
                                            f"({useful[0]} useful so far)")
                        if processed[0] % 10 == 0:
                            with open(out_path, "w") as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Error: {e}")

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        total_useful = sum(1 for r in results.values() if r.get("has_lifecycle"))
        total_facts = sum(len(r.get("facts", [])) for r in results.values() if r.get("has_lifecycle"))
        logger.info(f"Video {vid}: {len(results)} entities, {total_useful} with lifecycle, {total_facts} facts")
        logger.info(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
