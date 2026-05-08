"""
UnifiedMemory: interface layer over MultimodalGraph + cross-modal PPR.

Replaces the three independent retrieve_from_* methods when
retrieval_backend == "unified_graph".
"""

import json
import logging
import os
import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from .graph import MultimodalGraph
from .nodes import GraphNode, NodeType
from .retrieval import cross_modal_retrieve
from ..utils import RetrievedItem

logger = logging.getLogger(__name__)

# Module-level lock for GPU embedding operations (shared across all instances)
_embed_lock = threading.Lock()


class UnifiedMemory:
    """
    Unified multimodal memory backed by a single heterogeneous knowledge graph.

    Usage
    -----
    mem = UnifiedMemory(embedding_model, graph_path)
    mem.load()                       # load serialised graph
    mem.index(until_time)            # build subgraph + igraph for the query time
    results = mem.retrieve(query)    # cross-modal PPR → (GraphNode, score) list
    items = mem.format_results(results, visual_memory)  # → List[RetrievedItem]
    """

    def __init__(
        self,
        embedding_model,
        graph_path: str,
        chain_mode: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        embedding_model        : EmbeddingModel
        graph_path             : Path to the serialised MultimodalGraph pkl file.
        chain_mode             : Reserved for future chain injection method, or None.
        """
        self.embedding_model = embedding_model
        self.graph_path      = graph_path

        self._full_graph:     Optional[MultimodalGraph] = None  # loaded once
        self._active_graph:   Optional[MultimodalGraph] = None  # time-filtered subgraph
        self._indexed_time    = 0  # int (EgoLife) or float (Video-MME)

        self._chain_mode             = chain_mode  # reserved for future chain injection
        self._event_chains             = []   # cross-time event chains

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self,
             event_facts_path: Optional[str] = None,
             topic_chain_facts_path: Optional[str] = None,
             event_chain_path: Optional[str] = None,
             **kwargs) -> None:
        """Load the serialised graph and chain facts from disk.

        Args:
            event_facts_path: Legacy event facts for episode metadata injection.
            topic_chain_facts_path: Topic chain who-did-what facts (topic_chains_filtered.json).
            event_chain_path: Cross-time event chains (step3_enriched_chains.json).
            **kwargs: Accepts and ignores legacy params.
        """
        logger.info(f"Loading unified graph from {self.graph_path}…")
        self._full_graph = MultimodalGraph.load(self.graph_path)
        logger.info(f"Graph loaded: {self._full_graph.stats()}")

        if event_facts_path and os.path.exists(event_facts_path):
            facts_data = json.load(open(event_facts_path))
            injected = 0
            for nid, node in self._full_graph.nodes.items():
                if node.node_type != NodeType.EPISODE:
                    continue
                date = node.metadata.get("date", "")
                ts_full = str(node.start_ts)
                ts_no_day = ts_full[1:] if len(ts_full) >= 2 else ts_full
                entry = (facts_data.get(f"{date}_{ts_no_day}")
                         or facts_data.get(f"{date}_{ts_full}")
                         or facts_data.get(ts_no_day)
                         or facts_data.get(ts_full))
                if entry and entry.get("fact"):
                    node.metadata["event_fact"] = entry["fact"]
                    injected += 1
            logger.info(f"Event facts injected: {injected} episodes")

        # Load topic chain facts
        self._topic_facts = {}   # entity → [{"time": ..., "fact": ...}]

        if topic_chain_facts_path and os.path.exists(topic_chain_facts_path):
            raw = json.load(open(topic_chain_facts_path))
            for entity, data in raw.items():
                if data.get("has_lifecycle") and data.get("facts"):
                    self._topic_facts[entity] = data["facts"]
            logger.info(f"Loaded topic chain facts: {len(self._topic_facts)} entities, "
                        f"{sum(len(f) for f in self._topic_facts.values())} total facts")

        # Load event chains
        if event_chain_path and os.path.exists(event_chain_path):
            raw = json.load(open(event_chain_path))
            def _parse_hms(t, as_seconds=False):
                """Parse 'HH:MM:SS' or 'HH:MM'. If as_seconds, return total seconds
                (MM-Lifelong / Video-MME float-second nodes); otherwise return
                DHHMMSSFF time-component (HH*1e6+MM*1e4+SS*100) for EgoLife nodes."""
                m = re.match(r'(\d{1,2}):(\d{2}):?(\d{2})?', t)
                if not m:
                    return 0
                h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
                if as_seconds:
                    return h * 3600 + mn * 60 + s
                return h * 1000000 + mn * 10000 + s * 100
            for ec in raw:
                # Detect format: EgoLife steps carry "DAY*"; livestream steps don't.
                # When no day prefix, node.start_ts is in float seconds, so step
                # times must also be parsed as seconds to allow comparison.
                steps_with_ts = []
                for step in ec.get("steps", []):
                    day = step.get("day", "")
                    dm = re.match(r'DAY(\d+)', day)
                    day_prefix = int(dm.group(1)) * 100000000 if dm else 0
                    as_seconds = (day_prefix == 0)
                    st = step.get("start_time", "")
                    et = step.get("end_time", "")
                    start_ts = day_prefix + _parse_hms(st, as_seconds=as_seconds)
                    end_ts = day_prefix + _parse_hms(et, as_seconds=as_seconds)
                    if start_ts == 0 and end_ts == 0:
                        continue
                    steps_with_ts.append({
                        "day": day,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "description": step.get("description", ""),
                    })
                if len(steps_with_ts) >= 2:
                    self._event_chains.append({
                        "name": ec.get("name", ""),
                        "key_entities": ec.get("key_entities", []),
                        "steps": steps_with_ts,
                    })
            logger.info(f"Loaded event chains: {len(self._event_chains)} chains, "
                        f"{sum(len(s['steps']) for s in self._event_chains)} total steps")

    # ------------------------------------------------------------------
    # Chain facts injection (chain_mode="facts")
    # ------------------------------------------------------------------

    def get_chain_facts_for_results(
        self,
        results: List[Tuple[GraphNode, float]],
        query_time=None,
        query_text: str = "",
        min_hits: int = 2,
        max_topic_chains: int = 3,
        max_event_chains: int = 2,
        embedding_model=None,
        topic_sim_threshold: float = 0.5,
        event_chain_sim_threshold: float = 0.5,
        event_chain_min_hits: int = 1,
        event_chain_granularities: Tuple[str, ...] = ("30sec", "3min"),
    ) -> List[dict]:
        """Find chain facts relevant to retrieval results.

        Matching:
          - Topic chains: entity keyword appears in episode caption text
            AND in the query text (query relevance filter)
          - Event chains: episode time falls within event's [start_time, end_time]

        For chains with >= min_hits episodes, collect all facts filtered by query_time.

        Returns list of {"time": str, "fact": str, "label": str} dicts,
        sorted by time.
        """
        if not self._topic_facts and not self._event_chains:
            return []

        # Robust pattern matching: case-insensitive, plural (s/es), hyphen/space variants
        def _robust_pattern(entity):
            """Build a regex that matches entity with plural, case, hyphen robustness."""
            e = entity.lower().strip()
            variants = [re.escape(e)]
            # Add hyphen/space variants
            if '-' in e:
                variants.append(re.escape(e.replace('-', ' ')))
                variants.append(re.escape(e.replace('-', '')))
            if ' ' in e:
                variants.append(re.escape(e.replace(' ', '-')))
                variants.append(re.escape(e.replace(' ', '')))
            pattern = '|'.join(variants)
            return re.compile(r'\b(?:' + pattern + r')(?:s|es)?\b', re.IGNORECASE)

        topic_patterns = {
            entity: _robust_pattern(entity)
            for entity in self._topic_facts
        }

        query_lower = query_text.lower() if query_text else ""

        # Topic chains: coarse filter — entity mentioned in retrieved 30sec episodes
        topic_hits = {}  # entity → hit count
        for node, _ in results:
            if node.node_type != NodeType.EPISODE:
                continue
            gran = node.metadata.get("granularity", "")
            if gran != "30sec":
                continue
            caption = (node.text or "").lower()
            for entity, pat in topic_patterns.items():
                if pat.search(caption):
                    topic_hits[entity] = topic_hits.get(entity, 0) + 1

        _topic_coarse_passed = {ent for ent, cnt in topic_hits.items() if cnt >= min_hits}

        # Event-chain matching: check which episodes fall into
        # any step of the same event chain (≥event_chain_min_hits step hits)
        # Then filter by query keyword matching against key_entities.
        # Granularities used for the coarse filter are configurable.
        _event_chain_grans = set(event_chain_granularities)
        event_chain_step_hits = {}  # ec_idx → set of step_indices
        for node, _ in results:
            if node.node_type != NodeType.EPISODE:
                continue
            gran = node.metadata.get("granularity", "")
            if gran not in _event_chain_grans:
                continue
            for ec_idx, ec in enumerate(self._event_chains):
                for step_idx, step in enumerate(ec["steps"]):
                    if step["start_ts"] <= node.start_ts <= step["end_ts"]:
                        if ec_idx not in event_chain_step_hits:
                            event_chain_step_hits[ec_idx] = set()
                        event_chain_step_hits[ec_idx].add(step_idx)

        # Filter: ≥1 step hit + query keyword matches event-chain key entities
        def _robust_entity_match(entity, text):
            """Match entity in text using robust pattern (case, plural, hyphen)."""
            return bool(_robust_pattern(entity).search(text))

        _stopwords = {'and', 'the', 'for', 'from', 'with', 'at', 'setup', 'coordination',
                      'management', 'handling', 'preparation', 'service', 'based'}

        # Event-chain matching: two-tier like topic
        # Tier 1: key_entities match query → always included (no topk limit)
        # Tier 2: no match → embedding candidate, fill remaining slots
        keyword_matched_ec = []      # always included
        _embedding_ec_candidates = set()  # candidates for embedding check
        if query_lower:
            for ec_idx, step_indices in event_chain_step_hits.items():
                if len(step_indices) < event_chain_min_hits:
                    continue
                ec = self._event_chains[ec_idx]
                matched = False
                for ent in ec.get("key_entities", []):
                    if _robust_entity_match(ent, query_lower):
                        matched = True
                        break
                if matched:
                    keyword_matched_ec.append((ec_idx, len(step_indices)))
                else:
                    _embedding_ec_candidates.add(ec_idx)

        relevant_event_chains = list(keyword_matched_ec)
        relevant_event_chains.sort(key=lambda x: -x[1])
        relevant_event_chains = relevant_event_chains[:max_event_chains]
        embedding_ec_slots = max(0, max_event_chains - len(relevant_event_chains))

        # Topic chains matching
        keyword_matched = []    # entities matched by query keyword
        _topic_embedding_set = set()  # candidates for embedding check
        for entity in _topic_coarse_passed:
            pat = topic_patterns[entity]
            if query_lower and pat.search(query_lower):
                keyword_matched.append((("topic", entity), topic_hits.get(entity, 0)))
            else:
                _topic_embedding_set.add(entity)

        relevant = list(keyword_matched)
        relevant.sort(key=lambda x: -x[1])
        relevant = relevant[:max_topic_chains]
        embedding_slots = max(0, max_topic_chains - len(relevant))

        # Bug fix: also keep going when only Tier-2 event-chain candidates remain,
        # otherwise embedding-tier-only event-chain matches are silently dropped.
        if (not relevant and not _topic_embedding_set
                and not relevant_event_chains and not _embedding_ec_candidates):
            return []

        # Parse query_time for filtering
        qt_day, qt_secs = 999, 999999
        if query_time and isinstance(query_time, (int, float)) and query_time > 1000 and query_time != float('inf'):
            s = str(int(query_time))
            if len(s) >= 7:
                qt_day = int(s[0])
                qt_secs = int(s[1:3]) * 3600 + int(s[3:5]) * 60 + int(s[5:7])

        # Helpers to parse timestamps in both formats:
        #   EgoLife: DHHMMSSFF int (e.g. 111094300 = DAY1 11:09:43.00)
        #   MM-Lifelong / Video-MME: float seconds within a single video (e.g. 630.0)
        def _ts_to_day_secs(ts):
            s = str(int(ts))
            if len(s) >= 7:
                return (int(s[0]),
                        int(s[1:3]) * 3600 + int(s[3:5]) * 60 + int(s[5:7]))
            return (0, int(ts))

        def _fact_time_to_day_secs(time_str):
            """Parse a fact/step `time` string. Returns (day, secs) or None."""
            if not time_str:
                return None
            m = re.match(r'DAY(\d+)\s+(\d{1,2}):(\d{2}):?(\d{2})?', time_str)
            if m:
                return (int(m.group(1)),
                        int(m.group(2)) * 3600 + int(m.group(3)) * 60
                        + (int(m.group(4)) if m.group(4) else 0))
            m2 = re.match(r'^(\d{1,2}):(\d{2}):?(\d{2})?', time_str.strip())
            if m2:
                return (0,
                        int(m2.group(1)) * 3600 + int(m2.group(2)) * 60
                        + (int(m2.group(3)) if m2.group(3) else 0))
            return None

        # Build set of time ranges covered by retrieved episodes (for dedup)
        _DEDUP_MARGIN = 60  # seconds — facts within 60s of an episode are "covered"
        covered_ranges = []
        for node, _ in results:
            if node.node_type != NodeType.EPISODE:
                continue
            st_parsed = _ts_to_day_secs(node.start_ts)
            et_parsed = _ts_to_day_secs(node.end_ts)
            if st_parsed and et_parsed:
                st_day, st_secs = st_parsed
                _, et_secs = et_parsed
                covered_ranges.append((st_day, st_secs - _DEDUP_MARGIN, et_secs + _DEDUP_MARGIN))

        def _is_covered(day, secs):
            for c_day, c_start, c_end in covered_ranges:
                if day == c_day and c_start <= secs <= c_end:
                    return True
            return False

        # Collect facts from relevant topic chains (skip if covered by episodes)
        all_facts = []
        skipped_covered = 0
        for (chain_type, chain_name), _ in relevant:
            facts = self._topic_facts.get(chain_name, [])
            for f in facts:
                time_str = f.get("time", "")
                parsed = _fact_time_to_day_secs(time_str)
                if parsed is not None:
                    f_day, f_secs = parsed
                    if (f_day, f_secs) > (qt_day, qt_secs):
                        continue
                    if _is_covered(f_day, f_secs):
                        skipped_covered += 1
                        continue
                fact_text = f.get("fact", "")
                if not fact_text:
                    continue
                all_facts.append({
                    "time": time_str,
                    "fact": fact_text,
                    "label": chain_name,
                    "chain_type": "topic",
                })

        # Collect step descriptions from relevant event chains (skip if covered)
        for ec_idx, n_steps in relevant_event_chains:
            ec = self._event_chains[ec_idx]
            for step in ec["steps"]:
                step_ts = step["start_ts"]
                s_parsed = _ts_to_day_secs(step_ts)
                if s_parsed is not None:
                    s_day, s_secs = s_parsed
                    if (s_day, s_secs) > (qt_day, qt_secs):
                        continue
                    if _is_covered(s_day, s_secs):
                        skipped_covered += 1
                        continue

                # Format time from step. EgoLife steps carry "DAY*" → step_ts is
                # DHHMMSSFF; MML/Video-MME steps have empty day → step_ts is total
                # seconds (post _parse_hms fix).
                day = step.get("day", "")
                st = step_ts
                et = step.get("end_ts", step_ts)
                if day:
                    d = int(st) // 100000000
                    remainder = int(st) % 100000000
                    h = remainder // 1000000
                    m = (remainder % 1000000) // 10000
                    s = (remainder % 10000) // 100
                    remainder2 = int(et) % 100000000
                    h2 = remainder2 // 1000000
                    m2 = (remainder2 % 1000000) // 10000
                    s2 = (remainder2 % 10000) // 100
                    time_str = f"{day} {h:02d}:{m:02d}:{s:02d}"
                    end_time_str = f"{h2:02d}:{m2:02d}:{s2:02d}"
                else:
                    total_s = int(st)
                    h, m, s = total_s // 3600, (total_s % 3600) // 60, total_s % 60
                    total_e = int(et)
                    h2, m2, s2 = total_e // 3600, (total_e % 3600) // 60, total_e % 60
                    time_str = f"{h:02d}:{m:02d}:{s:02d}"
                    end_time_str = f"{h2:02d}:{m2:02d}:{s2:02d}"

                desc = step.get("description", "")
                if desc:
                    all_facts.append({
                        "time": time_str,
                        "end_time": end_time_str,
                        "fact": desc,
                        "label": ec["name"],
                        "chain_type": "event",
                    })

        # Sort by time. Handles both formats:
        #   - EgoLife: "DAY1 11:09:43"
        #   - MM-Lifelong / Video-MME: "00:10:30" (relative seconds within a video)
        def sort_key(f):
            t = f.get("time", "")
            m = re.match(r'DAY(\d+)\s+(\d{1,2}):(\d{2}):?(\d{2})?', t)
            if m:
                day = int(m.group(1))
                h = int(m.group(2)); mn = int(m.group(3)); s = int(m.group(4) or 0)
                return day * 86400 + h * 3600 + mn * 60 + s
            m2 = re.match(r'(\d{1,2}):(\d{2}):?(\d{2})?', t)
            if m2:
                h = int(m2.group(1)); mn = int(m2.group(2)); s = int(m2.group(3) or 0)
                return h * 3600 + mn * 60 + s
            return 0
        all_facts.sort(key=sort_key)

        # Embedding filter for topic candidates: query vs time-filtered facts
        # Only fill remaining slots (embedding_slots) after keyword-matched topics.
        # NOTE: previous version read facts from all_facts, which only contained Tier-1
        # entities → Tier-2 candidates always had empty facts and were always dropped.
        # Fix: read facts directly from self._topic_facts, applying the same time filter
        # used for Tier-1, then append accepted Tier-2 entities' facts to all_facts.
        _TOPIC_SIM_THRESHOLD = topic_sim_threshold
        _embedding_keep_topics = set()  # topics that pass embedding check
        if _topic_embedding_set and embedding_model and query_text and embedding_slots > 0:
            # Collect time-filtered facts text per Tier-2 candidate topic
            # (read directly from source, not from all_facts).
            topic_candidate_facts = {}
            for entity in _topic_embedding_set:
                fact_texts = []
                for f in self._topic_facts.get(entity, []):
                    parsed = _fact_time_to_day_secs(f.get("time", ""))
                    if parsed is not None:
                        f_day, f_secs = parsed
                        if (f_day, f_secs) > (qt_day, qt_secs):
                            continue
                    fact_text = f.get("fact", "")
                    if fact_text:
                        fact_texts.append(fact_text)
                if fact_texts:
                    topic_candidate_facts[entity] = fact_texts

            if topic_candidate_facts:
                try:
                    labels = list(topic_candidate_facts.keys())
                    texts = [" ".join(topic_candidate_facts[lbl])[:2000] for lbl in labels]
                    all_emb_texts = [query_text] + texts
                    embeddings = embedding_model.encode_text(all_emb_texts)
                    query_emb = embeddings[0]

                    # Score all candidates, keep top embedding_slots that pass threshold
                    scored_candidates = []
                    for i, lbl in enumerate(labels):
                        t_emb = embeddings[i + 1]
                        cos_sim = float(np.dot(query_emb, t_emb) /
                                       (np.linalg.norm(query_emb) * np.linalg.norm(t_emb) + 1e-8))
                        if cos_sim >= _TOPIC_SIM_THRESHOLD:
                            scored_candidates.append((lbl, cos_sim))
                            logger.info(f"[chain] Topic embedding: {lbl[:30]} sim={cos_sim:.4f} [KEEP]")
                        else:
                            logger.info(f"[chain] Topic embedding: {lbl[:30]} sim={cos_sim:.4f} [DROP]")

                    # Sort by similarity, take top embedding_slots
                    scored_candidates.sort(key=lambda x: -x[1])
                    for lbl, sim in scored_candidates[:embedding_slots]:
                        _embedding_keep_topics.add(lbl)
                        relevant.append((("topic", lbl), topic_hits.get(lbl, 0)))
                        # Also append this entity's facts to all_facts so they are
                        # available downstream (with the same time + cover filter as Tier-1).
                        for f in self._topic_facts.get(lbl, []):
                            time_str = f.get("time", "")
                            parsed = _fact_time_to_day_secs(time_str)
                            if parsed is not None:
                                f_day, f_secs = parsed
                                if (f_day, f_secs) > (qt_day, qt_secs):
                                    continue
                                if _is_covered(f_day, f_secs):
                                    skipped_covered += 1
                                    continue
                            fact_text = f.get("fact", "")
                            if not fact_text:
                                continue
                            all_facts.append({
                                "time": time_str,
                                "fact": fact_text,
                                "label": lbl,
                                "chain_type": "topic",
                            })

                except Exception as e:
                    logger.warning(f"Topic embedding filter failed: {e}")

            # Also drop candidates with no facts (nothing to embed)
            no_facts_topics = _topic_embedding_set - set(topic_candidate_facts.keys())
            if no_facts_topics:
                for lbl in no_facts_topics:
                    logger.info(f"[chain] Topic embedding: {lbl[:30]} no facts after time filter [DROP]")

        # Build set of all accepted topic labels (keyword + embedding)
        _accepted_topics = set()
        for (_, entity), _ in keyword_matched:
            _accepted_topics.add(entity)
        _accepted_topics |= _embedding_keep_topics

        # Filter all_facts to only include accepted topics (drop rejected ones)
        all_facts = [f for f in all_facts
                     if f.get("chain_type") != "topic" or f.get("label", "") in _accepted_topics]

        # Event-chain Tier 2: embedding check for candidates, fill remaining slots
        _accepted_event_chain_names = set()
        for ec_idx, _ in keyword_matched_ec:
            _accepted_event_chain_names.add(self._event_chains[ec_idx]["name"])

        if _embedding_ec_candidates and embedding_model and query_text and embedding_ec_slots > 0:
            ec_idx_to_name = {idx: self._event_chains[idx]["name"] for idx in _embedding_ec_candidates}

            # NOTE: previous version read event-chain texts from all_facts (which only
            # contained Tier-1 chains), so Tier-2 candidates always had empty
            # text → all KEEP/DROP log lines never fired.
            # Fix: build candidate texts directly from self._event_chains step descriptions.
            candidate_facts_text = {}
            for ec_idx in _embedding_ec_candidates:
                ec_name = ec_idx_to_name[ec_idx]
                step_texts = [step.get("description", "") for step in self._event_chains[ec_idx]["steps"]
                              if step.get("description")]
                if step_texts:
                    candidate_facts_text[ec_name] = step_texts

            if candidate_facts_text:
                try:
                    labels = list(candidate_facts_text.keys())
                    texts = [" ".join(candidate_facts_text[lbl])[:2000] for lbl in labels]
                    all_texts = [query_text] + texts
                    embeddings = embedding_model.encode_text(all_texts)
                    query_emb = embeddings[0]

                    scored_ec = []
                    for i, lbl in enumerate(labels):
                        ec_emb = embeddings[i + 1]
                        cos_sim = float(np.dot(query_emb, ec_emb) /
                                       (np.linalg.norm(query_emb) * np.linalg.norm(ec_emb) + 1e-8))
                        if cos_sim >= event_chain_sim_threshold:
                            scored_ec.append((lbl, cos_sim))
                            logger.info(f"[chain] Event-chain embedding: {lbl[:30]} sim={cos_sim:.4f} [KEEP]")
                        else:
                            logger.info(f"[chain] Event-chain embedding: {lbl[:30]} sim={cos_sim:.4f} [DROP]")

                    # Sort by similarity, take top embedding_ec_slots
                    scored_ec.sort(key=lambda x: -x[1])
                    name_to_idx = {v: k for k, v in ec_idx_to_name.items()}
                    for lbl, sim in scored_ec[:embedding_ec_slots]:
                        _accepted_event_chain_names.add(lbl)
                        if lbl in name_to_idx:
                            ec_idx = name_to_idx[lbl]
                            step_count = len(event_chain_step_hits.get(ec_idx, set()))
                            relevant_event_chains.append((ec_idx, step_count))
                            # Also append this event chain's step facts to all_facts
                            # (with the same time + cover filter as Tier-1).
                            ec = self._event_chains[ec_idx]
                            for step in ec["steps"]:
                                step_ts = step["start_ts"]
                                s_parsed = _ts_to_day_secs(step_ts)
                                if s_parsed is not None:
                                    s_day, s_secs = s_parsed
                                    if (s_day, s_secs) > (qt_day, qt_secs):
                                        continue
                                    if _is_covered(s_day, s_secs):
                                        skipped_covered += 1
                                        continue
                                day = step.get("day", "")
                                st = step_ts
                                et = step.get("end_ts", step_ts)
                                if day:
                                    remainder = int(st) % 100000000
                                    h = remainder // 1000000
                                    mm = (remainder % 1000000) // 10000
                                    ss = (remainder % 10000) // 100
                                    remainder2 = int(et) % 100000000
                                    h2 = remainder2 // 1000000
                                    m2 = (remainder2 % 1000000) // 10000
                                    s2 = (remainder2 % 10000) // 100
                                    time_str = f"{day} {h:02d}:{mm:02d}:{ss:02d}"
                                    end_time_str = f"{h2:02d}:{m2:02d}:{s2:02d}"
                                else:
                                    total_s = int(st)
                                    h, mm, ss = total_s // 3600, (total_s % 3600) // 60, total_s % 60
                                    total_e = int(et)
                                    h2, m2, s2 = total_e // 3600, (total_e % 3600) // 60, total_e % 60
                                    time_str = f"{h:02d}:{mm:02d}:{ss:02d}"
                                    end_time_str = f"{h2:02d}:{m2:02d}:{s2:02d}"
                                desc = step.get("description", "")
                                if desc:
                                    all_facts.append({
                                        "time": time_str,
                                        "end_time": end_time_str,
                                        "fact": desc,
                                        "label": ec["name"],
                                        "chain_type": "event",
                                    })

                except Exception as e:
                    logger.warning(f"Event-chain embedding fallback failed: {e}")

        # Filter event-chain facts to only accepted chains
        all_facts = [f for f in all_facts
                     if f.get("chain_type") != "event" or f.get("label", "") in _accepted_event_chain_names]

        # Re-sort: Tier-2 facts were appended after the initial sort, so resort
        # the whole list to keep facts time-ordered downstream.
        all_facts.sort(key=sort_key)

        n_topic = len(relevant)
        n_event_chain = len(relevant_event_chains)
        # Accurate Tier-1 vs Tier-2 counts (the previous diff-based formula went
        # negative when keyword_matched got truncated by max_topic_chains).
        topic_tier2_count = sum(1 for (_, e), _ in relevant if e in _embedding_keep_topics)
        topic_tier1_count = n_topic - topic_tier2_count
        keyword_matched_ec_idx = {idx for idx, _ in keyword_matched_ec}
        ec_tier1_count = sum(1 for idx, _ in relevant_event_chains if idx in keyword_matched_ec_idx)
        ec_tier2_count = n_event_chain - ec_tier1_count
        facts_per_label = {}
        for f in all_facts:
            lbl = f.get("label", "")
            facts_per_label[lbl] = facts_per_label.get(lbl, 0) + 1
        topic_info = [f'{n}({topic_hits.get(n,0)} hits, {facts_per_label.get(n,0)}f)' for (_, n), _ in relevant[:5]]
        ec_info = [f'{self._event_chains[idx]["name"][:25]}({c} hits, {facts_per_label.get(self._event_chains[idx]["name"],0)}f)' for idx, c in relevant_event_chains[:5]]
        logger.info(f"[chain] {n_topic} topics ({topic_tier1_count} keyword, {topic_tier2_count} embedding), "
                    f"{n_event_chain} event chains ({ec_tier1_count} keyword, {ec_tier2_count} embedding), "
                    f"{len(all_facts)} facts (skipped {skipped_covered} covered) "
                    f"({', '.join(topic_info + ec_info)})")

        return all_facts

    # ------------------------------------------------------------------
    # Index (time-bounded subgraph)
    # ------------------------------------------------------------------

    def index(self, until_time) -> None:
        """
        Build (or re-build) the active subgraph for nodes observable at or
        before ``until_time``, then construct igraph for PPR.
        """
        if self._full_graph is None:
            raise RuntimeError("Call load() before index().")

        if self._indexed_time == until_time and self._active_graph is not None:
            logger.debug(f"Already indexed at {until_time}, skipping.")
            return

        logger.info(f"Building active subgraph until_time={until_time}…")
        sub = self._full_graph.filter_by_time(until_time)

        # Ensure all text embeddings are present (should already be pre-computed,
        # but re-compute any that are missing).
        # Lock protects GPU model from concurrent access by parallel workers.
        missing = [nid for nid, n in sub.nodes.items() if n.text_embedding is None]
        if missing:
            logger.warning(f"{len(missing)} nodes missing text embeddings; computing now…")
            texts = [sub.nodes[nid].text for nid in missing]
            with _embed_lock:
                embs = self.embedding_model.encode_text(texts)
            for nid, emb in zip(missing, embs):
                sub.nodes[nid].text_embedding = emb

        with _embed_lock:
            sub.build_igraph()

        self._active_graph = sub
        self._indexed_time = until_time
        logger.info(f"Active graph ready: {sub.stats()}")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        target_node_types: Optional[List[NodeType]] = None,
        alpha: float = 0.7,
        seed_top_k: int = 5,
        damping: float = 0.85,
        choices: Optional[dict] = None,
        bm25_weight: float = 0.3,
    ) -> List[Tuple[GraphNode, float]]:
        """
        Run cross-modal PPR and return top-k (GraphNode, score) pairs.

        target_node_types: restrict output to specific node types.
        damping: PPR damping factor (default 0.85).
        choices: answer choices dict for contrastive strategy.
        bm25_weight: weight for BM25 keyword seeds (0 to disable).
        """
        if self._active_graph is None:
            raise RuntimeError("Call index(until_time) before retrieve().")

        # Compute query text embedding
        with _embed_lock:
            q_emb = self.embedding_model.encode_text(query)
        if hasattr(q_emb, 'cpu'):
            q_emb = q_emb.cpu().numpy()
        if len(q_emb.shape) > 1:
            q_emb = q_emb[0]
        self._last_query_embedding = q_emb

        with _embed_lock:
            results = cross_modal_retrieve(
                graph             = self._active_graph,
                query             = query,
                embedding_model   = self.embedding_model,
                top_k             = top_k,
                target_node_types = target_node_types,
                alpha             = alpha,
                seed_top_k        = seed_top_k,
                damping           = damping,
                choices           = choices,
                bm25_weight       = bm25_weight,
            )

        return results

    # ------------------------------------------------------------------
    # Format results → RetrievedItem list
    # ------------------------------------------------------------------

    def format_results(
        self,
        results: List[Tuple[GraphNode, float]],
        visual_memory,                          # VisualMemory instance
        fps: float = 1.0,
        max_frames_per_clip: int = 16,
        max_total_frames: int = 64,
        _current_total_frames: int = 0,
    ) -> List[RetrievedItem]:
        """Convert (GraphNode, score) pairs to RetrievedItem objects compatible
        with WorldMemory._render_retrieved_items_for_qa().

        Node type mapping:
          EPISODE / ENTITY  → memory_type="episodic",  content=str
          SEMANTIC           → memory_type="semantic",  content=str
          VISUAL_CLIP        → memory_type="visual",    content=List[PIL.Image]
        """
        items: List[RetrievedItem] = []
        total_frames = _current_total_frames

        for node, _score in results:
            if node.node_type == NodeType.ENTITY:
                continue

            if node.node_type == NodeType.EPISODE:
                ep_item = RetrievedItem(
                    memory_type = "episodic",
                    content     = node.display_text(),
                    query       = "",
                    round_num   = 0,
                )
                ep_item._broadcast_id = getattr(node, '_broadcast_id', None)
                items.append(ep_item)

            elif node.node_type == NodeType.SEMANTIC:
                sem_item = RetrievedItem(
                    memory_type = "semantic",
                    content     = node.display_text(),
                    query       = "",
                    round_num   = 0,
                )
                sem_item._broadcast_id = getattr(node, '_broadcast_id', None)
                items.append(sem_item)

            elif node.node_type == NodeType.VISUAL_CLIP:
                # Don't add episodic item here — caption comes from EPISODE node.
                # If no EPISODE node for this time range, add caption for context.
                ep_id = f"ep_{node.start_ts}_{node.end_ts}_30sec"
                has_episode = any(
                    n.node_type == NodeType.EPISODE and n.start_ts == node.start_ts
                    and n.end_ts == node.end_ts
                    for n, _ in results
                )
                if not has_episode:
                    ep_node = self._full_graph.nodes.get(ep_id) if self._full_graph else None
                    if ep_node and ep_node.text:
                        vis_text = ep_node.display_text()
                    elif node.text:
                        vis_text = f"[{_fmt_ts(node.start_ts)} - {_fmt_ts(node.end_ts)}] {node.text}"
                    else:
                        vis_text = node.display_text()
                    items.append(RetrievedItem(
                        memory_type = "episodic",
                        content     = vis_text,
                        query       = "",
                        round_num   = 0,
                    ))
                remaining = max_total_frames - total_frames
                if remaining <= 0:
                    continue
                video_path = node.metadata.get("video_path")
                if not video_path:
                    continue
                clip_cap = min(max_frames_per_clip, remaining)
                start_sec = node.metadata.get("start_sec")
                end_sec = node.metadata.get("end_sec")
                frame_entries = visual_memory._extract_frames(
                    video_path = video_path,
                    fps        = fps,
                    max_frames = clip_cap,
                    start_sec  = start_sec,
                    end_sec    = end_sec,
                )
                pil_images: List[Image.Image] = [
                    fe.frame for fe in frame_entries if fe.frame is not None
                ]
                if pil_images:
                    total_frames += len(pil_images)
                    vis_item = RetrievedItem(
                        memory_type = "visual",
                        content     = pil_images,
                        query       = "",
                        round_num   = 0,
                    )
                    # Attach time info so round history can sort correctly
                    vis_item._start_ts = node.start_ts
                    items.append(vis_item)

        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Free GPU tensors held by the active graph."""
        if self._active_graph is not None:
            if self._active_graph._text_emb_matrix is not None:
                del self._active_graph._text_emb_matrix
                self._active_graph._text_emb_matrix = None
            if self._active_graph._vis_emb_matrix is not None:
                del self._active_graph._vis_emb_matrix
                self._active_graph._vis_emb_matrix = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def indexed_time(self):
        return self._indexed_time


class MultiGraphUnifiedMemory:
    """Wrapper that queries multiple per-broadcast UnifiedMemory instances
    and merges results by score. Used for cross-broadcast questions."""

    def __init__(self):
        self._memories: Dict[str, UnifiedMemory] = {}  # bid → UnifiedMemory
        # Merged chain data across all broadcasts
        self._topic_facts: Dict[str, list] = {}   # entity → merged facts
        self._event_chains: list = []                # merged event chains
        self._chain_mode: Optional[str] = None

    def add(self, broadcast_id: str, mem: UnifiedMemory) -> None:
        self._memories[broadcast_id] = mem
        # Merge chain data from this broadcast
        if mem._topic_facts:
            for entity, facts in mem._topic_facts.items():
                # Tag facts with broadcast_id for provenance
                tagged = []
                for f in facts:
                    fc = dict(f)
                    if "broadcast_id" not in fc:
                        fc["broadcast_id"] = broadcast_id
                    tagged.append(fc)
                if entity in self._topic_facts:
                    self._topic_facts[entity].extend(tagged)
                else:
                    self._topic_facts[entity] = list(tagged)
        if mem._event_chains:
            for ec in mem._event_chains:
                ec_copy = dict(ec)
                if "broadcast_id" not in ec_copy:
                    ec_copy["broadcast_id"] = broadcast_id
                self._event_chains.append(ec_copy)
        if mem._chain_mode:
            self._chain_mode = mem._chain_mode

    def has(self, broadcast_id: str) -> bool:
        return broadcast_id in self._memories

    @property
    def broadcast_ids(self) -> List[str]:
        return list(self._memories.keys())

    def load(self) -> None:
        for mem in self._memories.values():
            if mem._full_graph is None:
                mem.load()

    def index(self, until_time) -> None:
        for mem in self._memories.values():
            mem.index(until_time)

    def retrieve(self, query: str, **kwargs) -> List[Tuple[GraphNode, float]]:
        """Query each graph independently, merge results by score, return top-k.
        Tags each node with _broadcast_id for cross-video display."""
        top_k = kwargs.get("top_k", 10)
        all_results: List[Tuple[GraphNode, float]] = []
        for bid, mem in self._memories.items():
            try:
                results = mem.retrieve(query=query, **kwargs)
                for node, score in results:
                    node._broadcast_id = bid
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Retrieve failed for broadcast {bid}: {e}")
        # Sort by score descending, take top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def format_results(self, results, visual_memory, **kwargs) -> List["RetrievedItem"]:
        """Delegate to the first memory's format_results.
        Items inherit _broadcast_id from their source nodes."""
        first = next(iter(self._memories.values()))
        return first.format_results(results, visual_memory, **kwargs)

    def get_chain_facts_for_results(self, results, **kwargs):
        """Per-video chain injection: run each video's chain matching on its own results."""
        all_facts = []
        # Group results by broadcast_id
        bid_results = defaultdict(list)
        for node, score in results:
            bid = getattr(node, '_broadcast_id', None)
            if bid:
                bid_results[bid].append((node, score))

        # For each video, use its own chain data
        for bid, bid_res in bid_results.items():
            mem = self._memories.get(bid)
            if mem and (mem._topic_facts or mem._event_chains):
                facts = mem.get_chain_facts_for_results(bid_res, **kwargs)
                # Tag facts with broadcast_id
                for f in facts:
                    f["broadcast_id"] = bid
                all_facts.extend(facts)

        return all_facts

    def cleanup(self) -> None:
        for mem in self._memories.values():
            mem.cleanup()
        self._memories.clear()
        self._topic_facts.clear()
        self._event_chains.clear()
