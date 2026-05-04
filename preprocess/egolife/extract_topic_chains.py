#!/usr/bin/env python3
"""
Extract topic (entity-lifecycle) chains from semantic triples + 30sec captions.

For each entity appearing in 2+ timestamps, collect the relevant 30sec captions
and ask LLM to determine if the entity has a meaningful lifecycle. If yes,
extract person-action-time facts.

Usage:
    python preprocess/egolife/extract_topic_chains.py --subject A1_JAKE
    python preprocess/egolife/extract_topic_chains.py --subject A1_JAKE --min-ts 3
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

EXTRACT_PROMPT = """In a first-person video, the entity "{entity}" appears in multiple episodes at different times.
The wearer is A1_JAKE (also referred to as "I"), living with housemates: Shure, Katrina, Tasha, Lucia, Alice, and sometimes visitors (Nicous, Choiszt, Jack, Violet, Lucy).

Below are the 30-second episode captions where "{entity}" is mentioned:

{episodes_text}

Extract facts about "{entity}" that help answer these types of questions:

1. **Who & When** (EntityLog): Who used/touched/moved "{entity}" first? Whose "{entity}" is this? Where was "{entity}" before?
2. **What happened** (EventRecall): When was "{entity}" first/last mentioned or discussed? What happened with "{entity}" last time?
3. **Relationships** (RelationMap): Who helped whom with "{entity}"? Who gave/passed "{entity}" to whom? Who was together when using "{entity}"?
4. **Plans & Decisions** (TaskMaster): Who suggested/planned to buy/make/use "{entity}"? What was decided about "{entity}"?
5. **Habits & Preferences** (HabitInsight): Who always/usually/often uses "{entity}"? Who likes/dislikes "{entity}"? What does someone typically do with "{entity}"?

Rules:
- Always use specific person names (A1_JAKE, Shure, Katrina, Tasha, Lucia, Alice, Nicous, Choiszt, Jack). NEVER use "I", "he", "she", "they", "we".
- Include the WHERE (which room, location) when relevant.
- Record preferences and habits ("Tasha refused the {entity}", "Shure always drinks {entity} in the morning").
- Record plans and discussions ("They decided to order {entity} for lunch", "A1_JAKE suggested buying {entity}").
- Skip trivial mentions where "{entity}" appears in background without meaningful action.

Good facts:
- "Shure handed the {entity} to A1_JAKE in the kitchen"
- "Lucia and A1_JAKE set up the {entity} together in the living room"
- "Katrina said she doesn't like {entity}"
- "A1_JAKE suggested ordering {entity} for dinner, everyone agreed"
- "Tasha was the first person to use the {entity} this morning"

Bad facts (skip these):
- "A1_JAKE picked up the {entity}" (no meaningful interaction)
- "The {entity} was on the table" (static description, no action)

Output a JSON array (empty array [] if no meaningful facts):
[
  {{"time": "DAY1 17:45", "fact": "descriptive fact with person names, location, and action"}},
  {{"time": "DAY2 11:00", "fact": "another fact"}}
]"""


def format_timestamp(ts_str, date=""):
    """Convert timestamp like 17450000 to 'DAY1 17:45'."""
    try:
        s = str(ts_str).zfill(8)
        h, m, sec = int(s[0:2]), int(s[2:4]), int(s[4:6])
        prefix = f"{date} " if date else ""
        return f"{prefix}{h:02d}:{m:02d}"
    except (ValueError, IndexError):
        return f"{date} {ts_str}"


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
            logger.warning(f"JSON parse error for '{prompt[:50]}' (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="A1_JAKE")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--data-dir", default="data/EgoLife")
    parser.add_argument("--output-dir", default="output/metadata/topic_chains")
    parser.add_argument("--whitelist", default="",
                        help="Path to entity whitelist JSON. If set, skip semantic triple extraction.")
    parser.add_argument("--min-ts", type=int, default=2,
                        help="Minimum number of timestamps for an entity to be considered")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of entities to process (0=all)")
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, args.subject)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "topic_chains.json")

    # Load semantic triples
    sem_path = os.path.join(
        "output/metadata/semantic_memory", args.subject,
        "semantic_extraction_results_openai_gpt-oss-120b.json")
    sem = json.load(open(sem_path))
    triples = sem.get("semantic_triples", {})

    # Load 30sec captions
    cap_path = os.path.join(args.data_dir,
        f"EgoLifeCap/{args.subject}/{args.subject}_30sec.json")
    captions = json.load(open(cap_path))
    # Build lookup: support both exact match and nearest-match for timestamps.
    # Semantic triples use DHHMMSSFF (e.g. 111393000), 30sec captions use
    # HHMMSSFF (e.g. 11393000). Also triples may have second-level offsets
    # (e.g. 119463000 = 19:46:30) that don't exactly match caption boundaries
    # (e.g. 19460000 = 19:46:00).
    cap_by_ts = {}
    cap_list_by_day = {}  # day_num → sorted list of (ts_int, caption)
    for c in captions:
        ts = c['start_time']
        date = c.get('date', '')
        cap_by_ts[ts] = c
        # Add with day prefix
        if date:
            import re as _re
            m = _re.match(r'DAY(\d+)', date)
            if m:
                day_num = int(m.group(1))
                full_ts = m.group(1) + ts
                cap_by_ts[full_ts] = c
                if day_num not in cap_list_by_day:
                    cap_list_by_day[day_num] = []
                cap_list_by_day[day_num].append((int(full_ts), c))
    # Sort each day's captions for binary search
    for day_num in cap_list_by_day:
        cap_list_by_day[day_num].sort()

    def find_nearest_caption(ts_str):
        """Find the caption whose start_time is closest to ts_str."""
        # Try exact match first
        if ts_str in cap_by_ts:
            return cap_by_ts[ts_str]
        # Try nearest match by day
        try:
            ts_int = int(ts_str)
            day_num = ts_int // 100000000
            if day_num in cap_list_by_day:
                import bisect
                day_list = cap_list_by_day[day_num]
                idx = bisect.bisect_left(day_list, (ts_int,))
                candidates = []
                if idx < len(day_list):
                    candidates.append(day_list[idx])
                if idx > 0:
                    candidates.append(day_list[idx - 1])
                if candidates:
                    best = min(candidates, key=lambda x: abs(x[0] - ts_int))
                    # Only match if within 60 seconds (6000 in HHMMSSFF)
                    if abs(best[0] - ts_int) <= 6000:
                        return best[1]
        except (ValueError, TypeError):
            pass
        return None

    # Find entity candidates
    if args.whitelist and os.path.exists(args.whitelist):
        # Use pre-curated whitelist
        whitelist = json.load(open(args.whitelist))
        candidates = {ent: [] for ent in whitelist}
        logger.info(f"Using whitelist: {len(candidates)} entities from {args.whitelist}")
    else:
        # Extract from semantic triples
        person_names = {'i', 'me', 'my', 'we', 'us', 'everyone', 'narrator', 'person', 'people',
                        'shure', 'katrina', 'tasha', 'lucia', 'alice', 'nicous', 'choiszt',
                        'violet', 'jack', 'liu ting', 'huixin', 'annie', 'zitang',
                        'true', 'false', 'yes', 'no', 'he', 'she', 'they', 'it', 'someone',
                        'something', 'anything', 'nothing', 'things', 'stuff'}

        entity_ts = defaultdict(set)
        for ts, triple_list in triples.items():
            for t in triple_list:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    for ent in [t[0].lower().strip(), t[2].lower().strip()]:
                        if ent not in person_names and len(ent) >= 2:
                            entity_ts[ent].add(ts)

        candidates = {ent: sorted(ts_set) for ent, ts_set in entity_ts.items()
                      if len(ts_set) >= args.min_ts}
        logger.info(f"Entity candidates (>={args.min_ts} timestamps): {len(candidates)}")

    # Resume support
    existing = {}
    if args.resume and os.path.exists(out_path):
        existing = json.load(open(out_path))
        logger.info(f"Resuming: {len(existing)} entities already processed")

    todo = [(ent, ts_list) for ent, ts_list in candidates.items() if ent not in existing]
    if args.limit > 0:
        todo = todo[:args.limit]
    logger.info(f"To process: {len(todo)} entities")

    # Setup LLM
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    results = dict(existing)
    lock = threading.Lock()
    processed = 0
    useful = 0

    CHUNK_SIZE = 50  # max captions per LLM call

    def process_one(ent, ts_list):
        # Search for entity keyword directly in 30sec caption text
        lines = []
        for c in captions:
            if ent.lower() in c.get('text', '').lower():
                date = c.get('date', '')
                td = format_timestamp(c['start_time'], date)
                lines.append(f"[{td}] {c.get('text', '')}")

        if not lines:
            return ent, {"entity": ent, "has_lifecycle": False, "facts": []}

        # Split into chunks and process each separately
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
        futures = {executor.submit(process_one, ent, ts_list): ent
                   for ent, ts_list in todo}
        for future in as_completed(futures):
            try:
                ent, result = future.result()
                with lock:
                    results[ent] = result
                    processed += 1
                    if result.get("has_lifecycle"):
                        useful += 1
                        n_facts = len(result.get("facts", []))
                        logger.info(f"  [{processed}/{len(todo)}] {ent}: {n_facts} facts ✓")
                    else:
                        if processed % 10 == 0:
                            logger.info(f"  [{processed}/{len(todo)}] progress... "
                                        f"({useful} useful so far)")
                    # Incremental save every 50
                    if processed % 10 == 0:
                        with open(out_path, "w") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Error: {e}")

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_useful = sum(1 for r in results.values() if r.get("has_lifecycle"))
    total_facts = sum(len(r.get("facts", [])) for r in results.values() if r.get("has_lifecycle"))
    logger.info(f"Done! {len(results)} entities processed, "
                f"{total_useful} with lifecycle, {total_facts} total facts")
    logger.info(f"Saved to: {out_path}")

    # Print useful entities
    print(f"\n=== Useful entities ({total_useful}) ===")
    for ent, r in sorted(results.items()):
        if r.get("has_lifecycle"):
            facts = r.get("facts", [])
            print(f"  {ent} ({len(facts)} facts)")
            for f in facts[:3]:
                print(f"    [{f.get('time','')}] {f.get('fact','')}")
            if len(facts) > 3:
                print(f"    ... ({len(facts)} total)")


if __name__ == "__main__":
    main()
