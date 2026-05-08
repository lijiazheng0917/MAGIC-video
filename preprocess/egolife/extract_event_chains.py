"""Extract cross-day event chains from 10min captions.

Step 1: Per-day extraction — identify major activities per day
Step 2: Cross-day merge — identify event chains that span multiple days
Step 3: Detail extraction — for each event-chain step, extract facts from 30sec captions
"""
import json, os, re, sys, argparse, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def call_llm(client, model, prompt, max_tokens=4000, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
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
            logger.warning(f"JSON parse error (attempt {attempt+1})")
            time.sleep(1)
        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None


STEP1_PROMPT = """You are analyzing a first-person daily life video. The narrator is A1_JAKE, living with housemates: Shure, Katrina, Tasha, Lucia, Alice, and sometimes visitors (Nicous, Choiszt, Violet, Jack, Liu Ting).

Below are 10-minute summary segments from {date}.

For each major activity that happened on this day, extract:
- name: a short descriptive name (e.g., "Making a cake", "Shopping at supermarket")
- time_range: approximate start and end time (e.g., "11:00:00-12:30:00")
- summary: 2-3 sentences describing what happened. Must include:
  * WHO specifically participated (by name, never "he/she/they/I/we")
  * WHO helped whom, WHO worked together
  * WHAT concrete actions were taken (not vague words like "coordinated", "handled", "managed")
  * WHAT was the outcome or decision made
- key_entities: important objects or places involved (do NOT include person names here — only objects and places)

Return a JSON list:
[
  {{
    "name": "activity name",
    "time_range": "HH:MM:SS-HH:MM:SS",
    "summary": "detailed description with person names, specific actions, and outcomes",
    "key_entities": ["object1", "place1"]
  }}
]

BAD summary: "A1_JAKE coordinated box stacking and workspace assembly planning with the group."
GOOD summary: "A1_JAKE and Shure carried boxes upstairs and assembled a display rack. Lucia asked about placement and helped hold the frame. Katrina suggested making the rack longer. The rack was installed with a whiteboard on top."

Focus on significant, goal-directed activities (cooking, shopping, meetings, performances, crafts), not trivial actions. Aim for 5-15 activities per day.

Segments:
{segments}
"""

STEP2_PROMPT = """Below are daily activities from a 7-day first-person video. The participants are: A1_JAKE, Shure, Katrina, Tasha, Lucia, Alice, and visitors Nicous, Choiszt.

Find activities that are related across different time periods. This includes:
- Different stages of the same project (e.g., "discussed cake plan" DAY1 → "bought ingredients" DAY3 → "baked the cake" DAY5)
- Repeated occurrences of the same activity (e.g., "brewed coffee" on DAY2, DAY3, DAY4)
- Ongoing work done intermittently (e.g., "room decoration" on DAY1 morning and DAY2 evening)
- Same type of errand repeated (e.g., "shopping at supermarket" on DAY1, DAY3, DAY5)

Only output events that have 2 or more related activities across different time periods. Ignore standalone activities. Be thorough and find as many as possible.

For each event, output:
- name: specific name
- steps: list of {{day, start_time, end_time, description}}
  * description must name specific people involved (never "he/she/they/I/we")
  * description must state concrete actions and outcomes, not vague summaries
  * description should explain how this step relates to the previous step (continuation, escalation, completion, etc.)
- key_entities: important objects and places only (do NOT include person names)

BAD description: "A1_JAKE handled coffee preparation while the group reviewed the shopping list."
GOOD description: "Shure brewed pour-over coffee for everyone in the courtyard. A1_JAKE served espresso shots. Lucia and Katrina discussed which coffee beans to buy for the next day."

Return JSON list.

Daily activities:
{daily_activities}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="A1_JAKE")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--data-dir", default="data/EgoLife")
    parser.add_argument("--output-dir", default="output/metadata/event_chains")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, args.subject)
    os.makedirs(out_dir, exist_ok=True)

    # Load captions
    caps_10min = json.load(open(os.path.join(
        args.data_dir, f"EgoLifeCap/{args.subject}/{args.subject}_10min.json")))
    # Group by day
    by_day_10min = {}
    for c in caps_10min:
        d = c.get('date', '')
        if d not in by_day_10min:
            by_day_10min[d] = []
        by_day_10min[d].append(c)

    # Setup LLM
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Step 1: Per-day activity extraction
    step1_path = os.path.join(out_dir, "step1_daily_activities.json")
    if args.resume and os.path.exists(step1_path):
        daily_activities = json.load(open(step1_path))
        logger.info(f"Resuming step 1: {len(daily_activities)} days loaded")
    else:
        daily_activities = {}

    def _format_ts(ts_str):
        """Convert HHMMSSFF like '11210000' to 'HH:MM:SS' like '11:21:00'."""
        if len(ts_str) >= 6:
            return f"{ts_str[:2]}:{ts_str[2:4]}:{ts_str[4:6]}"
        elif len(ts_str) >= 4:
            return f"{ts_str[:2]}:{ts_str[2:4]}"
        return ts_str

    for date in sorted(by_day_10min.keys()):
        if date in daily_activities:
            continue
        segments = by_day_10min[date]
        segments_text = "\n\n".join(
            f"[{_format_ts(c.get('start_time', ''))}] {c.get('text', '')}" for c in segments
        )
        prompt = STEP1_PROMPT.format(date=date, segments=segments_text)
        logger.info(f"Step 1: Extracting activities for {date} ({len(segments)} segments)...")
        result = call_llm(client, args.model, prompt, max_tokens=4000)
        if result:
            daily_activities[date] = result
            logger.info(f"  {date}: {len(result)} activities")
            with open(step1_path, 'w') as f:
                json.dump(daily_activities, f, indent=2, ensure_ascii=False)
        else:
            logger.warning(f"  {date}: failed to extract")

    logger.info(f"Step 1 done: {sum(len(v) for v in daily_activities.values())} total activities")

    # Step 2a: Per-day event grouping
    step2a_path = os.path.join(out_dir, "step2a_daily_events.json")
    if args.resume and os.path.exists(step2a_path):
        daily_events = json.load(open(step2a_path))
        logger.info(f"Resuming step 2a: {len(daily_events)} days loaded")
    else:
        daily_events = {}

    # Step 2: Find cross-time event chains from daily activities
    step2_path = os.path.join(out_dir, "step2_event_chains.json")
    if args.resume and os.path.exists(step2_path):
        event_chains = json.load(open(step2_path))
        logger.info(f"Resuming step 2: {len(event_chains)} event chains loaded")
    else:
        # Format all daily activities
        daily_text = ""
        for date in sorted(daily_activities.keys()):
            daily_text += f"\n=== {date} ===\n"
            for act in daily_activities[date]:
                daily_text += f"- [{act.get('time_range','')}] {act.get('name','')}: {act.get('summary','')}\n"
                if act.get('key_entities'):
                    daily_text += f"  Entities: {', '.join(act['key_entities'])}\n"

        prompt = STEP2_PROMPT.format(daily_activities=daily_text)
        logger.info(f"Step 2: Finding cross-time event chains... (input ~{len(prompt)//4} tokens)")
        event_chains = call_llm(client, args.model, prompt, max_tokens=16000)
        if event_chains:
            logger.info(f"  Round 1: found {len(event_chains)} event chains")
        else:
            logger.warning("  Round 1: failed to extract event chains")
            event_chains = []

        # Round 2: find missed event chains
        if event_chains:
            found_names = "\n".join(f"- {ec.get('name','')}" for ec in event_chains)
            round2_prompt = f"""Below are daily activities from a 7-day first-person video, and a list of cross-time event chains already found.

Your task: find ADDITIONAL cross-time event chains that were MISSED in the first round. Do not repeat any chain already found.

Look specifically for:
- Recurring daily routines (meals, cleaning, waking up, sleeping)
- Equipment/device management across days (cameras, computers, glasses, phones)
- Social activities that repeat (games, outings, group discussions)
- Food/cooking activities across days (hot pot, cake, coffee, barbecue)
- Any other related activities across different time periods

Already found chains (do NOT repeat these):
{found_names}

Daily activities:
{daily_text}

For each NEW event chain, output:
- name: specific name
- steps: list of {{day, start_time, end_time, description}}
  * description must name specific people involved (never "he/she/they/I/we")
  * description must state concrete actions and outcomes
- key_entities: important objects and places only (do NOT include person names)

Return JSON list. If no additional chains found, return [].
"""
            logger.info(f"Step 2 Round 2: finding missed event chains...")
            extra_chains = call_llm(client, args.model, round2_prompt, max_tokens=16000)
            if extra_chains and isinstance(extra_chains, list):
                # Deduplicate by name
                existing_names = {ec.get('name', '').lower() for ec in event_chains}
                new_chains = [ec for ec in extra_chains if ec.get('name', '').lower() not in existing_names]
                event_chains.extend(new_chains)
                logger.info(f"  Round 2: found {len(new_chains)} additional chains (total: {len(event_chains)})")
            else:
                logger.info("  Round 2: no additional chains found")

        with open(step2_path, 'w') as f:
            json.dump(event_chains, f, indent=2, ensure_ascii=False)

    # Step 3: Enrich each step with details from 30sec captions
    step3_path = os.path.join(out_dir, "step3_enriched_chains.json")
    if args.resume and os.path.exists(step3_path):
        enriched_chains = json.load(open(step3_path))
        logger.info(f"Resuming step 3: {len(enriched_chains)} enriched chains loaded")
    else:
        # Load 30sec captions
        caps_30sec = json.load(open(os.path.join(
            args.data_dir, f"EgoLifeCap/{args.subject}/{args.subject}_30sec.json")))

        # Group by day for fast lookup
        caps_by_day = {}
        for c in caps_30sec:
            d = c.get('date', '')
            if d not in caps_by_day:
                caps_by_day[d] = []
            caps_by_day[d].append(c)
        for d in caps_by_day:
            caps_by_day[d].sort(key=lambda x: int(x.get('start_time', '0')))

        def _time_to_secs(t_str):
            """Convert 'HH:MM:SS' to seconds."""
            parts = t_str.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        def _ts_to_secs(ts_str):
            """Convert HHMMSSFF like '11210000' to seconds."""
            s = str(ts_str).zfill(8)
            return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])

        def find_captions_in_range(day, start_time, end_time):
            """Find 30sec captions within a time range."""
            if day not in caps_by_day:
                return []
            start_secs = _time_to_secs(start_time)
            end_secs = _time_to_secs(end_time)
            result = []
            for c in caps_by_day[day]:
                cap_secs = _ts_to_secs(c['start_time'])
                if start_secs <= cap_secs <= end_secs:
                    ts_fmt = f"{cap_secs//3600:02d}:{(cap_secs%3600)//60:02d}:{cap_secs%60:02d}"
                    result.append(f"[{ts_fmt}] {c.get('text', '')}")
            return result

        STEP3_PROMPT = """You are analyzing a first-person daily life video. The narrator is A1_JAKE, living with housemates: Shure, Katrina, Tasha, Lucia, Alice, and sometimes visitors (Nicous, Choiszt).

This is one step of the event chain "{chain_name}".
Time: {day} {start_time} - {end_time}

Below are the 30-second episode captions from this time period. Extract ONLY information relevant to "{chain_name}". Ignore unrelated content.

Captions:
{captions_text}

Write 2-5 sentences describing what happened in this step, focusing on:
- WHO specifically participated (by name, never "I/he/she/they/we")
- WHO helped whom, WHO gave what to whom
- WHAT concrete actions were taken and what was the outcome
- WHAT decisions were made

Output a single JSON string (just the description text):
{{"description": "detailed description here"}}"""

        import copy
        enriched_chains = copy.deepcopy(event_chains)
        total_steps = sum(len(ec['steps']) for ec in enriched_chains)
        processed = 0

        for ec in enriched_chains:
            chain_name = ec['name']
            for step in ec['steps']:
                day = step.get('day', '')
                st = step.get('start_time', '')
                et = step.get('end_time', '')

                captions = find_captions_in_range(day, st, et)
                processed += 1

                if not captions:
                    logger.info(f"  [{processed}/{total_steps}] {chain_name} [{day} {st}]: no captions found, keeping original")
                    continue

                captions_text = "\n\n".join(captions)
                prompt = STEP3_PROMPT.format(
                    chain_name=chain_name, day=day, start_time=st, end_time=et,
                    captions_text=captions_text)

                result = call_llm(client, args.model, prompt, max_tokens=1000)
                if result and isinstance(result, dict) and result.get('description'):
                    step['description'] = result['description']
                    logger.info(f"  [{processed}/{total_steps}] {chain_name} [{day} {st}]: enriched ({len(captions)} captions)")
                else:
                    logger.info(f"  [{processed}/{total_steps}] {chain_name} [{day} {st}]: LLM failed, keeping original")

                # Incremental save every 10 steps
                if processed % 10 == 0:
                    with open(step3_path, 'w') as f:
                        json.dump(enriched_chains, f, indent=2, ensure_ascii=False)

        with open(step3_path, 'w') as f:
            json.dump(enriched_chains, f, indent=2, ensure_ascii=False)
        logger.info(f"Step 3 done: enriched {processed} steps")

    # Print summary
    total_steps = sum(len(s.get('steps',[])) for s in enriched_chains)
    cross_day = sum(1 for s in enriched_chains if len(set(st.get('day','') for st in s.get('steps',[]))) > 1)
    logger.info(f"Done! {len(enriched_chains)} event chains ({cross_day} cross-day), {total_steps} total steps")
    for ec in enriched_chains:
        days = sorted(set(st.get('day','') for st in ec.get('steps',[])))
        logger.info(f"  {ec.get('name','')}: {len(ec.get('steps',[]))} steps, days={days}, entities={ec.get('key_entities',[])}")


if __name__ == "__main__":
    main()
