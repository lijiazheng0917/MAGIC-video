#!/usr/bin/env python3
"""
Extract storylines (activity chains) for MM-Lifelong broadcasts.

For each video, extracts major activities from 10min captions,
then identifies recurring or related activities within the broadcast.

Usage:
    python preprocess/mmlifelong/extract_storylines.py --video-id 4
    python preprocess/mmlifelong/extract_storylines.py --video-id all
"""

import json, os, re, sys, argparse, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def call_llm(client, model, prompt, max_tokens=4000, max_retries=3):
    for attempt in range(max_retries):
        raw = None
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise video analyzer. Always output valid JSON only. No markdown, no tables, no explanations."},
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
                start = raw.find('[')
                if start >= 0:
                    depth = 0
                    for i in range(start, len(raw)):
                        if raw[i] == '[': depth += 1
                        elif raw[i] == ']':
                            depth -= 1
                            if depth == 0:
                                try: return json.loads(raw[start:i+1])
                                except json.JSONDecodeError: break
            logger.warning(f"JSON parse error (attempt {attempt+1}), raw[:300]={raw[:300] if raw else 'None'}")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


def format_sec_to_hms(sec):
    """Convert float seconds to HH:MM:SS."""
    total = int(float(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


STEP1_PROMPT = """You are analyzing a livestream video by IShowSpeed. Below are 10-minute summary segments.

Create a detailed chronological event log. For EVERY distinct scene or activity, extract:
- name: specific name (e.g., "Eating hot pot at Haidilao restaurant", "Basketball 1v1 with fan near Oriental Pearl Tower")
- time_range: "HH:MM:SS-HH:MM:SS"
- location: specific place name (e.g., "Chengdu night market", "wrestling arena", "in the car")
- summary: 2-3 sentences. Must include:
  * Specific actions (NOT "interacted with fans" — instead "high-fived 3 fans, signed a jersey, took selfies")
  * Named objects (food names, song names, game names, gift items)
  * Outcomes and reactions
- key_entities: specific objects, food, songs, games mentioned

BAD: "IShowSpeed explored the area and met fans."
GOOD: "IShowSpeed ate spicy duck tongue and Sichuan peppercorn chicken at the hot pot restaurant. He struggled with chopsticks and drank three glasses of water from the spiciness."

Extract 5-10 events from these segments. Focus on significant moments (receiving a gift, singing a song, playing a game, visiting a venue), not trivial transitions.

Segments:
{segments}
"""

STEP2_PROMPT = """Below is a chronological event log from one IShowSpeed livestream broadcast.

Identify RECURRING THEMES — things that happen 2+ times in this broadcast. Examples:
- Singing the same song at different moments
- Fans giving gifts (list each gift separately)
- Eating different foods at different places
- Playing the same game multiple rounds
- Getting in and out of a car/bus
- Fan encounters with specific items (e.g., Messi jerseys)
- Doing push-ups, backflips, or physical challenges
- Taking photos with fans at different locations

For each recurring theme, list EVERY occurrence as a separate step with:
- start_time, end_time: exact time range
- description: WHAT specifically happened THIS time (not a generic summary)
  * Always say "IShowSpeed" (never "he/they/the streamer")
  * Include specific details that distinguish this occurrence from others

Also identify KEY VENUE SEQUENCES — visiting a major location with multiple sub-events:
- e.g., "Shaolin Temple visit" with steps: arrived → watched kung fu demo → tried martial arts → took photos → left by bus

Return JSON:
[
  {{
    "name": "theme or venue name",
    "type": "recurring" or "venue_sequence",
    "steps": [
      {{"start_time": "HH:MM:SS", "end_time": "HH:MM:SS", "description": "specific description"}}
    ],
    "key_entities": ["entity1", "entity2"]
  }}
]

Return [] if nothing qualifies. Be thorough — don't miss recurring events.

Events:
{activities}
"""

STEP3_PROMPT = """Storyline: "{chain_name}"
Time window: {start_time} - {end_time}

Below are 30-second captions from this time window. Extract details ONLY about "{chain_name}".

Captions:
{captions_text}

Write 2-4 sentences with maximum specificity:
- Count things precisely (how many times, how many items)
- Name every person, object, food, song, game involved
- Describe the exact sequence of actions
- Note the outcome or result

BAD: "IShowSpeed sang a song in the car."
GOOD: "IShowSpeed sang 'Sunshine Rainbow Little White Horse' twice while sitting in the back seat of the limousine. The driver and two staff members clapped along. He forgot the lyrics midway through the second attempt."

Output: {{"description": "your detailed description"}}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="4", help="Video ID or 'all'")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--metadata-dir", default="output/metadata/mmlifelong")
    parser.add_argument("--caption-mode", default="merge")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.video_id == "all":
        video_ids = sorted([d for d in os.listdir(args.metadata_dir)
                           if os.path.isdir(os.path.join(args.metadata_dir, d))])
    else:
        video_ids = [args.video_id]

    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    for vid in video_ids:
        logger.info(f"=== Processing video {vid} ===")

        mode_dir = os.path.join(args.metadata_dir, vid, args.caption_mode)
        cap_10min_path = os.path.join(mode_dir, "captions", f"{vid}_10min.json")
        cap_30sec_path = os.path.join(mode_dir, "captions", f"{vid}_30sec.json")
        out_dir = mode_dir

        if not os.path.exists(cap_10min_path):
            logger.warning(f"No 10min captions at {cap_10min_path}, skipping")
            continue

        caps_10min = json.load(open(cap_10min_path))
        logger.info(f"Loaded {len(caps_10min)} 10min captions")

        # Step 1: Extract activities (chunked to avoid token limits)
        step1_path = os.path.join(out_dir, "storyline_step1_activities.json")
        if args.resume and os.path.exists(step1_path):
            activities = json.load(open(step1_path))
            logger.info(f"Resuming step 1: {len(activities)} activities loaded")
        else:
            CHUNK_SIZE = 5  # segments per LLM call
            activities = []
            for i in range(0, len(caps_10min), CHUNK_SIZE):
                chunk = caps_10min[i:i + CHUNK_SIZE]
                segments_text = "\n\n".join(
                    f"[{format_sec_to_hms(c.get('start_sec', 0))}] {c.get('text', '')}"
                    for c in chunk
                )
                prompt = STEP1_PROMPT.format(segments=segments_text)
                logger.info(f"Step 1: chunk {i//CHUNK_SIZE+1} ({len(chunk)} segments)...")
                result = call_llm(client, args.model, prompt, max_tokens=8000)
                if isinstance(result, list):
                    activities.extend(result)
                    logger.info(f"  Found {len(result)} activities in chunk")
                else:
                    logger.warning(f"  Chunk {i//CHUNK_SIZE+1} failed")

            if activities:
                logger.info(f"Step 1 total: {len(activities)} activities")
                with open(step1_path, 'w') as f:
                    json.dump(activities, f, indent=2, ensure_ascii=False)
            else:
                logger.warning("  Failed to extract activities")
                activities = []

        # Step 2: Find storylines (chunk if too many activities)
        step2_path = os.path.join(out_dir, "storyline_step2_chains.json")
        if args.resume and os.path.exists(step2_path):
            storylines = json.load(open(step2_path))
            logger.info(f"Resuming step 2: {len(storylines)} storylines loaded")
        else:
            STEP2_CHUNK = 50  # max activities per LLM call
            if len(activities) <= STEP2_CHUNK:
                activities_text = ""
                for act in activities:
                    activities_text += f"- [{act.get('time_range', '')}] {act.get('name', '')}: {act.get('summary', '')}\n"
                    if act.get('key_entities'):
                        activities_text += f"  Entities: {', '.join(act['key_entities'])}\n"

                prompt = STEP2_PROMPT.format(activities=activities_text)
                logger.info(f"Step 2: Finding storylines ({len(activities)} activities)...")
                storylines = call_llm(client, args.model, prompt, max_tokens=12000)
                if not storylines:
                    logger.warning("  Failed to extract storylines")
                    storylines = []
                else:
                    logger.info(f"  Found {len(storylines)} storylines")
            else:
                # Split activities into chunks, find storylines per chunk, then merge
                storylines = []
                seen_names = set()
                for ci in range(0, len(activities), STEP2_CHUNK):
                    chunk = activities[ci:ci + STEP2_CHUNK]
                    activities_text = ""
                    for act in chunk:
                        activities_text += f"- [{act.get('time_range', '')}] {act.get('name', '')}: {act.get('summary', '')}\n"
                        if act.get('key_entities'):
                            activities_text += f"  Entities: {', '.join(act['key_entities'])}\n"

                    prompt = STEP2_PROMPT.format(activities=activities_text)
                    logger.info(f"Step 2: chunk {ci//STEP2_CHUNK+1} ({len(chunk)} activities)...")
                    result = call_llm(client, args.model, prompt, max_tokens=12000)
                    if isinstance(result, list):
                        for sl in result:
                            name = sl.get("name", "")
                            if name.lower() not in seen_names:
                                storylines.append(sl)
                                seen_names.add(name.lower())
                        logger.info(f"  Found {len(result)} storylines in chunk")
                    else:
                        logger.warning(f"  Step 2 chunk {ci//STEP2_CHUNK+1} failed")

                logger.info(f"Step 2 total: {len(storylines)} storylines (merged)")

            if storylines:
                with open(step2_path, 'w') as f:
                    json.dump(storylines, f, indent=2, ensure_ascii=False)
            else:
                storylines = []

        # Step 3: Enrich with 30sec captions
        step3_path = os.path.join(out_dir, "storyline_step3_enriched.json")
        if args.resume and os.path.exists(step3_path):
            enriched = json.load(open(step3_path))
            logger.info(f"Resuming step 3: {len(enriched)} enriched storylines loaded")
        else:
            if not os.path.exists(cap_30sec_path):
                logger.warning(f"No 30sec captions, skipping step 3")
                enriched = storylines
            else:
                import copy
                caps_30sec = json.load(open(cap_30sec_path))
                caps_30sec.sort(key=lambda x: x.get('start_sec', 0))

                import copy
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading

                enriched = copy.deepcopy(storylines)
                total_steps = sum(len(sl.get('steps', [])) for sl in enriched)

                def parse_hms(t):
                    parts = t.split(':')
                    if len(parts) == 3:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    return 0

                # Build all tasks: (sl_idx, step_idx, chain_name, start_time, end_time, matched_caps)
                tasks = []
                for sl_idx, sl in enumerate(enriched):
                    chain_name = sl.get('name', '')
                    for step_idx, step in enumerate(sl.get('steps', [])):
                        st = step.get('start_time', '')
                        et = step.get('end_time', '')
                        start_secs = parse_hms(st)
                        end_secs = parse_hms(et)

                        matched_caps = []
                        for c in caps_30sec:
                            cap_start = c.get('start_sec', 0)
                            if start_secs <= cap_start <= end_secs:
                                ts = format_sec_to_hms(cap_start)
                                matched_caps.append(f"[{ts}] {c.get('text', '')}")

                        if matched_caps:
                            tasks.append((sl_idx, step_idx, chain_name, st, et, matched_caps))

                logger.info(f"Step 3: Enriching {len(tasks)}/{total_steps} steps with 8 workers")
                processed = [0]
                lock = threading.Lock()

                def enrich_one(task):
                    sl_idx, step_idx, chain_name, st, et, matched_caps = task
                    captions_text = "\n\n".join(matched_caps)
                    prompt = STEP3_PROMPT.format(
                        chain_name=chain_name, start_time=st, end_time=et,
                        captions_text=captions_text)
                    result = call_llm(client, args.model, prompt, max_tokens=1000)
                    if result and isinstance(result, dict) and result.get('description'):
                        return sl_idx, step_idx, result['description'], len(matched_caps)
                    return sl_idx, step_idx, None, len(matched_caps)

                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(enrich_one, t): t for t in tasks}
                    for future in as_completed(futures):
                        sl_idx, step_idx, desc, n_caps = future.result()
                        with lock:
                            processed[0] += 1
                            chain_name = enriched[sl_idx]['name']
                            st = enriched[sl_idx]['steps'][step_idx].get('start_time', '')
                            if desc:
                                enriched[sl_idx]['steps'][step_idx]['description'] = desc
                                logger.info(f"  [{processed[0]}/{len(tasks)}] {chain_name} [{st}]: enriched ({n_caps} captions)")
                            else:
                                logger.info(f"  [{processed[0]}/{len(tasks)}] {chain_name} [{st}]: LLM failed")

                            if processed[0] % 5 == 0:
                                with open(step3_path, 'w') as f:
                                    json.dump(enriched, f, indent=2, ensure_ascii=False)

                with open(step3_path, 'w') as f:
                    json.dump(enriched, f, indent=2, ensure_ascii=False)

        total_steps = sum(len(sl.get('steps', [])) for sl in enriched)
        logger.info(f"Video {vid}: {len(enriched)} storylines, {total_steps} steps")
        logger.info(f"Saved to: {step3_path}")


if __name__ == "__main__":
    main()
