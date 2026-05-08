"""Chain-injection breakdown analyzer (generic across EgoLifeQA, Ego-R1, MM-Lifelong).

For each chain run, classifies every (chain) record by which kind of NMC fact
fired in the round_history (`[Topic:` / `[Event:` markers), then prints a
Table-style breakdown of UG vs Chain accuracy across the buckets:
All / None / Topic only / Event only / Both / Any.

Two modes:
  - Paired per-seed (len(ug_dirs) == len(chain_dirs)): keys are matched
    one-to-one across seed dirs and the totals are summed.
  - Asymmetric (typically many UG seeds vs 1 Chain seed): each Chain
    record is compared to the MEAN of all UG seeds on that question.

Per-dataset adaptors handle the file glob, chain-mode suffix, primary key
and score field:

| Dataset       | file glob                            | chain suffix     | key                | score field      |
|---------------|--------------------------------------|------------------|--------------------|------------------|
| egolife       | `egolife_eval_A1_JAKE_*.json`        | `_cm_facts`      | `ID`               | `evaluate` bool  |
| egor1         | `egor1_eval_A1_JAKE_*.json`          | `_cm_`           | `(ID, split)`      | `evaluate` bool  |
| mmlifelong    | `mmlifelong_eval_unified.json`       | (single file)    | `index`            | `smoothed_score` |

Examples:

  # Ego-R1 — paired 3 chain seeds vs 3 UG seeds
  python eval/analyze_breakdown.py --dataset egor1 \\
    --ug-dirs   output/eval/egor1_sweep/UG_repro_r3,output/eval/egor1_sweep/UG_repro_r4,output/eval/egor1_sweep/UG_repro_r6 \\
    --chain-dirs output/eval/egor1_sweep/X3_final_r1,output/eval/egor1_sweep/X3_final_r3,output/eval/egor1_sweep/X3_final_r5

  # EgoLifeQA — single UG file vs 5 chain batches
  python eval/analyze_breakdown.py --dataset egolife \\
    --ug-files  output/final_eval2/egolife/ug/ug_500q.json \\
    --chain-files output/final_eval2/egolife/chain/chain_W4_batch1_102q.json,output/final_eval2/egolife/chain/chain_W4_batch2_100q.json,output/final_eval2/egolife/chain/chain_W4_batch3_100q.json,output/final_eval2/egolife/chain/chain_W4_batch4_99q.json,output/final_eval2/egolife/chain/chain_W4_batch5_99q.json

  # MM-Lifelong — paired single UG vs single Chain JSON
  python eval/analyze_breakdown.py --dataset mmlifelong \\
    --ug-files    output/final_eval2/mmlifelong/ug/ug_623q_postfix.json \\
    --chain-files output/final_eval2/mmlifelong/chain/chain_F5_623q_postfix.json
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

DATASETS = {
    "egolife": {
        "file_glob": "egolife_eval_A1_JAKE_*.json",
        "chain_marker": "_cm_facts",   # substring that distinguishes chain output
        "key_fields": ("ID",),
        "score_field": "evaluate",
        "score_kind": "bool",          # True/False
    },
    "egor1": {
        "file_glob": "egor1_eval_A1_JAKE_*.json",
        "chain_marker": "_cm_",
        "key_fields": ("ID", "split"),
        "score_field": "evaluate",
        "score_kind": "bool",
    },
    "mmlifelong": {
        "file_glob": "mmlifelong_eval_unified.json",
        "chain_marker": None,          # MML stores chain/UG in different output dirs, not by suffix
        "key_fields": ("index",),
        "score_field": "smoothed_score",
        "score_kind": "float",         # 0 / 0.5 / 1
    },
}

TOPIC_PAT = re.compile(r"\[Topic:\s")
EVENT_PAT = re.compile(r"\[Event:\s")


def _resolve_dir(dir_path: str, glob_pat: str, chain_marker, with_chain: bool) -> str:
    """Pick the JSON under dir_path/MODEL_SUBDIR or dir_path matching the glob,
    filtering by chain_marker presence/absence when chain_marker is not None."""
    search_dirs = [os.path.join(dir_path, MODEL_SUBDIR), dir_path]
    all_json: list[str] = []
    for sd in search_dirs:
        if os.path.isdir(sd):
            all_json.extend(sorted(glob.glob(os.path.join(sd, glob_pat))))
    if not all_json:
        raise FileNotFoundError(
            f"No {glob_pat} found under {dir_path} "
            f"(also checked {os.path.join(dir_path, MODEL_SUBDIR)})"
        )
    if chain_marker is None:
        return all_json[0]
    filtered = [p for p in all_json
                if (chain_marker in os.path.basename(p)) == with_chain]
    return (filtered or all_json)[0]


def _classify_chain(entry) -> tuple[bool, bool]:
    has_t = has_e = False
    for r in entry.get("round_history", []) or []:
        rc = r.get("retrieved_content", "") or ""
        if TOPIC_PAT.search(rc):
            has_t = True
        if EVENT_PAT.search(rc):
            has_e = True
    return has_t, has_e


def _key_of(entry, key_fields) -> tuple:
    return tuple(entry[k] for k in key_fields)


def _score_of(entry, score_field, score_kind) -> float:
    v = entry.get(score_field)
    if v is None:
        return 0.0
    if score_kind == "bool":
        return float(bool(v))
    return float(v)


def _load_json(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("results", [])


def analyze(ug_paths: list[str], chain_paths: list[str], cfg: dict) -> dict:
    """Return bucket stats + per-seed list."""
    key_fields = cfg["key_fields"]
    score_field = cfg["score_field"]
    score_kind = cfg["score_kind"]
    buckets: dict = defaultdict(lambda: {"UG": 0.0, "Chain": 0.0, "n": 0})
    per_seed: list = []

    if len(ug_paths) == len(chain_paths):
        # Paired per-seed mode
        for ug_p, ch_p in zip(ug_paths, chain_paths):
            ug_data = _load_json(ug_p)
            ch_data = _load_json(ch_p)
            ug_map = {_key_of(e, key_fields): e for e in ug_data
                      if e.get(score_field) is not None}
            ch_map = {_key_of(e, key_fields): e for e in ch_data
                      if e.get(score_field) is not None}
            keys = sorted(set(ug_map) & set(ch_map))
            if not keys:
                raise RuntimeError(f"No overlap between {ug_p} and {ch_p}")
            seed_stat = {"ug_path": ug_p, "ch_path": ch_p,
                         "ug_correct": 0.0, "ch_correct": 0.0, "n": len(keys)}
            for k in keys:
                ug_e = ug_map[k]; ch_e = ch_map[k]
                ht, he = _classify_chain(ch_e)
                if ht and he: b = "Both"
                elif ht: b = "Topic only"
                elif he: b = "Event only"
                else: b = "None"
                ug_v = _score_of(ug_e, score_field, score_kind)
                ch_v = _score_of(ch_e, score_field, score_kind)
                seed_stat["ug_correct"] += ug_v
                seed_stat["ch_correct"] += ch_v
                buckets[b]["UG"] += ug_v
                buckets[b]["Chain"] += ch_v
                buckets[b]["n"] += 1
                buckets["All"]["UG"] += ug_v
                buckets["All"]["Chain"] += ch_v
                buckets["All"]["n"] += 1
                if b != "None":
                    buckets["Any"]["UG"] += ug_v
                    buckets["Any"]["Chain"] += ch_v
                    buckets["Any"]["n"] += 1
            per_seed.append(seed_stat)
    else:
        # Asymmetric: each Chain seed compared to mean of all UG seeds on that question.
        ug_maps = []
        for ug_p in ug_paths:
            d = _load_json(ug_p)
            ug_maps.append(({_key_of(e, key_fields): e for e in d
                             if e.get(score_field) is not None}, ug_p))
        for ch_p in chain_paths:
            ch_data = _load_json(ch_p)
            ch_map = {_key_of(e, key_fields): e for e in ch_data
                      if e.get(score_field) is not None}
            common_keys = set(ch_map)
            for ug_map, _ in ug_maps:
                common_keys &= set(ug_map)
            keys = sorted(common_keys)
            if not keys:
                raise RuntimeError(f"No overlap between UG seeds and {ch_p}")
            seed_stat = {"ug_path": f"<mean of {len(ug_maps)} seeds>",
                         "ch_path": ch_p,
                         "ug_correct": 0.0, "ch_correct": 0.0, "n": len(keys)}
            for k in keys:
                ch_e = ch_map[k]
                ht, he = _classify_chain(ch_e)
                if ht and he: b = "Both"
                elif ht: b = "Topic only"
                elif he: b = "Event only"
                else: b = "None"
                ug_vals = [_score_of(um[k], score_field, score_kind) for um, _ in ug_maps]
                ug_mean = sum(ug_vals) / len(ug_vals)
                ch_v = _score_of(ch_e, score_field, score_kind)
                seed_stat["ug_correct"] += ug_mean
                seed_stat["ch_correct"] += ch_v
                buckets[b]["UG"] += ug_mean
                buckets[b]["Chain"] += ch_v
                buckets[b]["n"] += 1
                buckets["All"]["UG"] += ug_mean
                buckets["All"]["Chain"] += ch_v
                buckets["All"]["n"] += 1
                if b != "None":
                    buckets["Any"]["UG"] += ug_mean
                    buckets["Any"]["Chain"] += ch_v
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


def _resolve_inputs(args, cfg: dict) -> tuple[list[str], list[str]]:
    if args.ug_files and args.chain_files:
        return ([p.strip() for p in args.ug_files.split(",") if p.strip()],
                [p.strip() for p in args.chain_files.split(",") if p.strip()])
    if not (args.ug_dirs and args.chain_dirs):
        sys.exit("Must pass either --ug-files/--chain-files or --ug-dirs/--chain-dirs")
    ug_dirs = [d.strip() for d in args.ug_dirs.split(",") if d.strip()]
    chain_dirs = [d.strip() for d in args.chain_dirs.split(",") if d.strip()]
    ug_paths = [_resolve_dir(d, cfg["file_glob"], cfg["chain_marker"], with_chain=False)
                for d in ug_dirs]
    chain_paths = [_resolve_dir(d, cfg["file_glob"], cfg["chain_marker"], with_chain=True)
                   for d in chain_dirs]
    return ug_paths, chain_paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=list(DATASETS),
                   help="Dataset adaptor (egolife / egor1 / mmlifelong)")
    p.add_argument("--ug-dirs", default=None,
                   help="Comma-separated UG result directories (one per seed)")
    p.add_argument("--chain-dirs", default=None,
                   help="Comma-separated Chain result directories (one per seed)")
    p.add_argument("--ug-files", default=None,
                   help="Comma-separated UG JSON file paths (alternative to --ug-dirs)")
    p.add_argument("--chain-files", default=None,
                   help="Comma-separated Chain JSON file paths (alternative to --chain-dirs)")
    p.add_argument("--label", default="", help="Optional title above the table")
    p.add_argument("--per-seed", action="store_true",
                   help="Also show per-seed accuracy summary")
    args = p.parse_args()

    cfg = DATASETS[args.dataset]
    ug_paths, chain_paths = _resolve_inputs(args, cfg)
    res = analyze(ug_paths, chain_paths, cfg)

    if args.per_seed:
        print("Per-seed accuracy:")
        for i, s in enumerate(res["per_seed"], 1):
            ug = 100 * s["ug_correct"] / s["n"]
            ch = 100 * s["ch_correct"] / s["n"]
            print(f"  seed {i}: UG {s['ug_correct']:.1f}/{s['n']}={ug:.2f}%  "
                  f"Chain {s['ch_correct']:.1f}/{s['n']}={ch:.2f}%")

    title = args.label or f"{args.dataset} Chain vs UG breakdown"
    _print_table(res["buckets"], title=title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
