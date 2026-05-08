"""EgoR1 chain-injection breakdown analyzer.

Pools UG and Chain result JSONs across seeds (one result dir per seed),
joins by (ID, split), classifies each question-run by Topic/Event markers
in the Chain run's retrieved_content, and prints a Table-4-style breakdown:
All / None / Topic only / Event only / Both / Any.

Example:
  python eval/analyze_egor1_breakdown.py \
    --ug-dirs output/eval/egor1_ug_r1,output/eval/egor1_ug_r2,output/eval/egor1_ug_r3 \
    --chain-dirs output/eval/egor1_chain_v3_r1,output/eval/egor1_chain_v3_r2,output/eval/egor1_chain_v3_r3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

MODEL_SUBDIR = "openai_gpt-oss-120b_qwen_qwen3.5-flash-02-23"
# Filenames vary across eval_egor1.py versions. Match by glob + "_cm_facts"
# suffix to distinguish Chain from UG.
FILE_GLOB = "egor1_eval_A1_JAKE_*.json"

TOPIC_PAT = re.compile(r"\[Topic:\s")
EVENT_PAT = re.compile(r"\[Event:\s")


def _resolve(dir_path: str, with_chain: bool) -> str:
    """Pick the JSON under dir_path; with_chain=True selects files whose
    name contains '_cm_' (the chain-mode suffix), False picks the others.

    Falls back to the only JSON available if the filter leaves no candidate,
    which helps when the CLI of eval_egor1.py was changed and suffixes shifted.
    """
    # Search both `dir_path/MODEL_SUBDIR/*.json` and `dir_path/*.json`.
    search_dirs = [os.path.join(dir_path, MODEL_SUBDIR), dir_path]
    all_json: list[str] = []
    for sd in search_dirs:
        if os.path.isdir(sd):
            all_json.extend(sorted(glob.glob(os.path.join(sd, FILE_GLOB))))
    if not all_json:
        raise FileNotFoundError(
            f"No {FILE_GLOB} found under {dir_path} "
            f"(also checked {os.path.join(dir_path, MODEL_SUBDIR)})"
        )
    filtered = [p for p in all_json
                if ("_cm_" in os.path.basename(p)) == with_chain]
    if not filtered:
        # Fall back to whatever is there. Usually this means the caller passed
        # a Chain dir as UG (or vice versa) — let the caller verify.
        filtered = all_json
    return filtered[0]


def _load(dir_path: str, with_chain: bool):
    path = _resolve(dir_path, with_chain)
    with open(path) as f:
        return json.load(f), path


def _classify_chain(entry) -> tuple[bool, bool]:
    has_t = has_e = False
    for r in entry.get("round_history", []) or []:
        rc = r.get("retrieved_content", "") or ""
        if TOPIC_PAT.search(rc):
            has_t = True
        if EVENT_PAT.search(rc):
            has_e = True
    return has_t, has_e


def analyze(ug_dirs: list[str], chain_dirs: list[str]) -> dict:
    """Return bucket stats comparing Chain to UG.

    Two modes:
      - Equal lengths (len(ug_dirs) == len(chain_dirs)): paired per-seed pool.
        Each (ID, split, seed) is one observation; UG and Chain are compared
        at matched seeds.
      - Asymmetric (typically many UG seeds vs 1 Chain seed): for each Chain
        seed, its per-question outcome is compared to the MEAN of all UG seeds
        on that question. Useful for single-run Chain screening against a
        stable 3-seed UG pool.

    Metric convention: UG and Chain columns store total "credit"; n stores
    total observation count; accuracy = sum(credit) / n. Since Ego-R1 uses
    binary evaluate, UG credit in asymmetric mode is fractional in {0, 1/N,
    2/N, ..., 1} where N is number of UG seeds.
    """
    buckets: dict = defaultdict(lambda: {"UG": 0.0, "Chain": 0.0, "n": 0})
    per_seed: list = []

    if len(ug_dirs) == len(chain_dirs):
        # Paired per-seed mode
        for ug_d, chain_d in zip(ug_dirs, chain_dirs):
            ug_data, ug_path = _load(ug_d, with_chain=False)
            ch_data, ch_path = _load(chain_d, with_chain=True)
            ug_map = {(e["ID"], e["split"]): e for e in ug_data}
            ch_map = {(e["ID"], e["split"]): e for e in ch_data}
            keys = sorted(set(ug_map) & set(ch_map))
            if not keys:
                raise RuntimeError(f"No overlap between {ug_path} and {ch_path}")

            seed_stat = {"ug_path": ug_path, "ch_path": ch_path,
                         "ug_correct": 0, "ch_correct": 0, "n": len(keys)}

            for k in keys:
                ug_e = ug_map[k]
                ch_e = ch_map[k]
                ht, he = _classify_chain(ch_e)
                if ht and he: b = "Both"
                elif ht:      b = "Topic only"
                elif he:      b = "Event only"
                else:         b = "None"

                ug_eval = int(ug_e.get("evaluate", 0))
                ch_eval = int(ch_e.get("evaluate", 0))
                seed_stat["ug_correct"] += ug_eval
                seed_stat["ch_correct"] += ch_eval

                buckets[b]["UG"] += ug_eval
                buckets[b]["Chain"] += ch_eval
                buckets[b]["n"] += 1
                buckets["All"]["UG"] += ug_eval
                buckets["All"]["Chain"] += ch_eval
                buckets["All"]["n"] += 1
                if b != "None":
                    buckets["Any"]["UG"] += ug_eval
                    buckets["Any"]["Chain"] += ch_eval
                    buckets["Any"]["n"] += 1
            per_seed.append(seed_stat)
    else:
        # Asymmetric mode: for each Chain seed, UG = mean(all UG seeds per key)
        ug_maps = []
        for ug_d in ug_dirs:
            ug_data, ug_path = _load(ug_d, with_chain=False)
            ug_maps.append(({(e["ID"], e["split"]): e for e in ug_data}, ug_path))

        for chain_d in chain_dirs:
            ch_data, ch_path = _load(chain_d, with_chain=True)
            ch_map = {(e["ID"], e["split"]): e for e in ch_data}

            common_keys = set(ch_map)
            for ug_map, _ in ug_maps:
                common_keys &= set(ug_map)
            keys = sorted(common_keys)
            if not keys:
                raise RuntimeError(f"No overlap between UG seeds and {ch_path}")

            seed_stat = {"ug_path": f"<mean of {len(ug_maps)} seeds>",
                         "ch_path": ch_path,
                         "ug_correct": 0.0, "ch_correct": 0, "n": len(keys)}

            for k in keys:
                ch_e = ch_map[k]
                ht, he = _classify_chain(ch_e)
                if ht and he: b = "Both"
                elif ht:      b = "Topic only"
                elif he:      b = "Event only"
                else:         b = "None"

                # Mean UG credit across UG seeds
                ug_vals = [int(um[k].get("evaluate", 0)) for um, _ in ug_maps]
                ug_mean = sum(ug_vals) / len(ug_vals)
                ch_eval = int(ch_e.get("evaluate", 0))

                seed_stat["ug_correct"] += ug_mean
                seed_stat["ch_correct"] += ch_eval

                buckets[b]["UG"] += ug_mean
                buckets[b]["Chain"] += ch_eval
                buckets[b]["n"] += 1
                buckets["All"]["UG"] += ug_mean
                buckets["All"]["Chain"] += ch_eval
                buckets["All"]["n"] += 1
                if b != "None":
                    buckets["Any"]["UG"] += ug_mean
                    buckets["Any"]["Chain"] += ch_eval
                    buckets["Any"]["n"] += 1
            per_seed.append(seed_stat)

    return {"buckets": dict(buckets), "per_seed": per_seed}


def _print_table(buckets: dict, title: str = "") -> None:
    order = ["All", "None", "Topic only", "Event only", "Both", "Any"]
    if title:
        print(f"\n{title}")
    print(f"{'Subset':18s} {'n':>5s} {'UG':>8s} {'Chain':>8s} {'Δ':>8s}")
    print("-" * 52)
    for b in order:
        d = buckets.get(b)
        if not d or d["n"] == 0:
            continue
        ug = 100 * d["UG"] / d["n"]
        ch = 100 * d["Chain"] / d["n"]
        print(f"{b:18s} {d['n']:>5d} {ug:>8.2f} {ch:>8.2f} {ch - ug:>+8.2f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ug-dirs", required=True,
                   help="Comma-separated list of UG result dirs (one per seed)")
    p.add_argument("--chain-dirs", required=True,
                   help="Comma-separated list of Chain result dirs (one per seed)")
    p.add_argument("--label", default="",
                   help="Optional title printed above the table")
    p.add_argument("--per-seed", action="store_true",
                   help="Also show per-seed accuracy summary")
    args = p.parse_args()

    ug_dirs = [d.strip() for d in args.ug_dirs.split(",") if d.strip()]
    chain_dirs = [d.strip() for d in args.chain_dirs.split(",") if d.strip()]

    res = analyze(ug_dirs, chain_dirs)

    if args.per_seed:
        print("Per-seed accuracy:")
        for i, s in enumerate(res["per_seed"], 1):
            ug = s["ug_correct"] / s["n"] * 100
            ch = s["ch_correct"] / s["n"] * 100
            print(f"  seed {i}: UG {s['ug_correct']}/{s['n']}={ug:.2f}%  "
                  f"Chain {s['ch_correct']}/{s['n']}={ch:.2f}%")

    _print_table(res["buckets"], title=args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
