"""
UnifiedMemory: interface layer over MultimodalGraph + cross-modal PPR.

Replaces the three independent retrieve_from_* methods when
retrieval_backend == "unified_graph".
"""

import logging
import os
import threading
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
        self._storylines             = []   # cross-time event chains

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self,
             event_facts_path: Optional[str] = None,
             topic_chain_facts_path: Optional[str] = None,
             storyline_path: Optional[str] = None,
             **kwargs) -> None:
        """Load the serialised graph and chain facts from disk.

        Args:
            event_facts_path: Legacy event facts for episode metadata injection.
            topic_chain_facts_path: Topic chain who-did-what facts (topic_chains_filtered.json).
            storyline_path: Cross-time event chains (step2_event_chains.json).
            **kwargs: Accepts and ignores legacy params.
        """
        logger.info(f"Loading unified graph from {self.graph_path}…")
        self._full_graph = MultimodalGraph.load(self.graph_path)
        logger.info(f"Graph loaded: {self._full_graph.stats()}")

        if event_facts_path and os.path.exists(event_facts_path):
            import json
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
            import json as _json3
            raw = _json3.load(open(topic_chain_facts_path))
            for entity, data in raw.items():
                if data.get("has_lifecycle") and data.get("facts"):
                    self._topic_facts[entity] = data["facts"]
            logger.info(f"Loaded topic chain facts: {len(self._topic_facts)} entities, "
                        f"{sum(len(f) for f in self._topic_facts.values())} total facts")

        # Load event chains (cross-time storylines)
        if storyline_path and os.path.exists(storyline_path):
            import json as _json5, re as _re3
            raw = _json5.load(open(storyline_path))
            def _parse_hms(t):
                m = _re3.match(r'(\d{1,2}):(\d{2}):?(\d{2})?', t)
                if m:
                    return (int(m.group(1)) * 1000000
                            + int(m.group(2)) * 10000
                            + int(m.group(3) or 0) * 100)
                return 0
            for sl in raw:
                # Convert step times to DHHMMSSFF for matching
                steps_with_ts = []
                for step in sl.get("steps", []):
                    day = step.get("day", "")
                    dm = _re3.match(r'DAY(\d+)', day)
                    day_prefix = int(dm.group(1)) * 100000000 if dm else 0
                    st = step.get("start_time", "")
                    et = step.get("end_time", "")
                    start_ts = day_prefix + _parse_hms(st)
                    end_ts = day_prefix + _parse_hms(et)
                    if start_ts == 0 and end_ts == 0:
                        continue
                    steps_with_ts.append({
                        "day": day,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "description": step.get("description", ""),
                    })
                if len(steps_with_ts) >= 2:
                    self._storylines.append({
                        "name": sl.get("name", ""),
                        "key_entities": sl.get("key_entities", []),
                        "steps": steps_with_ts,
                    })
            logger.info(f"Loaded storylines: {len(self._storylines)} event chains, "
                        f"{sum(len(s['steps']) for s in self._storylines)} total steps")

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
        storyline_sim_threshold: float = 0.5,
        keyword_bypass_topk: bool = True,
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
        import re

        if not self._topic_facts and not self._storylines:
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

        # Storyline matching: check if 30sec/3min episodes fall into
        # any step of the same storyline (≥1 step hit)
        # Then filter by query keyword matching against key_entities
        storyline_step_hits = {}  # storyline_idx → set of step_indices
        for node, _ in results:
            if node.node_type != NodeType.EPISODE:
                continue
            gran = node.metadata.get("granularity", "")
            if gran not in ("30sec", "3min"):
                continue
            for sl_idx, sl in enumerate(self._storylines):
                for step_idx, step in enumerate(sl["steps"]):
                    if step["start_ts"] <= node.start_ts <= step["end_ts"]:
                        if sl_idx not in storyline_step_hits:
                            storyline_step_hits[sl_idx] = set()
                        storyline_step_hits[sl_idx].add(step_idx)

        # Filter: ≥1 step hit + query keyword matches storyline entities
        def _robust_entity_match(entity, text):
            """Match entity in text using robust pattern (case, plural, hyphen)."""
            return bool(_robust_pattern(entity).search(text))

        _stopwords = {'and', 'the', 'for', 'from', 'with', 'at', 'setup', 'coordination',
                      'management', 'handling', 'preparation', 'service', 'based'}

        # Storyline matching: two-tier like topic
        # Tier 1: key_entities match query → always included (no topk limit)
        # Tier 2: no match → embedding candidate, fill remaining slots
        keyword_matched_sl = []      # always included
        _embedding_sl_candidates = set()  # candidates for embedding check
        if query_lower:
            for sl_idx, step_indices in storyline_step_hits.items():
                if len(step_indices) < 1:
                    continue
                sl = self._storylines[sl_idx]
                matched = False
                for ent in sl.get("key_entities", []):
                    if _robust_entity_match(ent, query_lower):
                        matched = True
                        break
                if matched:
                    keyword_matched_sl.append((sl_idx, len(step_indices)))
                else:
                    _embedding_sl_candidates.add(sl_idx)

        if keyword_bypass_topk:
            embedding_sl_slots = max(0, max_event_chains - len(keyword_matched_sl))
            relevant_storylines = list(keyword_matched_sl)
        else:
            relevant_storylines = list(keyword_matched_sl)
            relevant_storylines.sort(key=lambda x: -x[1])
            relevant_storylines = relevant_storylines[:max_event_chains]
            embedding_sl_slots = max(0, max_event_chains - len(relevant_storylines))

        # Topic chains matching
        keyword_matched = []    # entities matched by query keyword
        _topic_embedding_set = set()  # candidates for embedding check
        for entity in _topic_coarse_passed:
            pat = topic_patterns[entity]
            if query_lower and pat.search(query_lower):
                keyword_matched.append((("topic", entity), topic_hits.get(entity, 0)))
            else:
                _topic_embedding_set.add(entity)

        if keyword_bypass_topk:
            # New logic: keyword matches bypass topk, embedding fills remaining slots
            embedding_slots = max(0, max_topic_chains - len(keyword_matched))
            relevant = list(keyword_matched)
        else:
            # Old logic: keyword + embedding all compete for topk slots
            relevant = list(keyword_matched)
            relevant.sort(key=lambda x: -x[1])
            relevant = relevant[:max_topic_chains]
            embedding_slots = max(0, max_topic_chains - len(relevant))

        if not relevant and not _topic_embedding_set and not relevant_storylines:
            return []

        # Parse query_time for filtering
        qt_day, qt_secs = 999, 999999
        if query_time and isinstance(query_time, (int, float)) and query_time > 1000 and query_time != float('inf'):
            s = str(int(query_time))
            if len(s) >= 7:
                qt_day = int(s[0])
                qt_secs = int(s[1:3]) * 3600 + int(s[3:5]) * 60 + int(s[5:7])

        # Build set of time ranges covered by retrieved episodes (for dedup)
        # Convert to (day, start_secs, end_secs) with a small margin
        _DEDUP_MARGIN = 60  # seconds — facts within 60s of an episode are "covered"
        covered_ranges = []
        for node, _ in results:
            if node.node_type != NodeType.EPISODE:
                continue
            st_s = str(int(node.start_ts))
            et_s = str(int(node.end_ts))
            if len(st_s) >= 7 and len(et_s) >= 7:
                st_day = int(st_s[0])
                st_secs = int(st_s[1:3]) * 3600 + int(st_s[3:5]) * 60 + int(st_s[5:7])
                et_secs = int(et_s[1:3]) * 3600 + int(et_s[3:5]) * 60 + int(et_s[5:7])
                covered_ranges.append((st_day, st_secs - _DEDUP_MARGIN, et_secs + _DEDUP_MARGIN))

        def _is_covered(day, secs):
            """Check if a (day, secs) falls within any retrieved episode's time range."""
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
                m = re.match(r'DAY(\d+)\s+(\d{1,2}):(\d{2})', time_str)
                if m:
                    f_day = int(m.group(1))
                    f_secs = int(m.group(2)) * 3600 + int(m.group(3)) * 60
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

        # Collect step descriptions from relevant storylines (skip if covered)
        for sl_idx, n_steps in relevant_storylines:
            sl = self._storylines[sl_idx]
            for step in sl["steps"]:
                step_ts = step["start_ts"]
                s = str(int(step_ts))
                if len(s) >= 7:
                    s_day = int(s[0])
                    s_secs = int(s[1:3]) * 3600 + int(s[3:5]) * 60 + int(s[5:7])
                    if (s_day, s_secs) > (qt_day, qt_secs):
                        continue
                    if _is_covered(s_day, s_secs):
                        skipped_covered += 1
                        continue

                # Format time from step
                day = step.get("day", "")
                st = step_ts
                et = step.get("end_ts", step_ts)
                d = int(st) // 100000000
                remainder = int(st) % 100000000
                h = remainder // 1000000
                m = (remainder % 1000000) // 10000
                s = (remainder % 10000) // 100
                d2 = int(et) // 100000000
                remainder2 = int(et) % 100000000
                h2 = remainder2 // 1000000
                m2 = (remainder2 % 1000000) // 10000
                s2 = (remainder2 % 10000) // 100
                if day:
                    time_str = f"{day} {h:02d}:{m:02d}:{s:02d}"
                    end_time_str = f"{h2:02d}:{m2:02d}:{s2:02d}"
                else:
                    time_str = f"{h:02d}:{m:02d}:{s:02d}"
                    end_time_str = f"{h2:02d}:{m2:02d}:{s2:02d}"

                desc = step.get("description", "")
                if desc:
                    all_facts.append({
                        "time": time_str,
                        "end_time": end_time_str,
                        "fact": desc,
                        "label": sl["name"],
                        "chain_type": "event",
                    })

        # Sort by time
        def sort_key(f):
            import re
            m = re.match(r'DAY(\d+)\s+(\d{1,2}):(\d{2})', f["time"])
            if m:
                return int(m.group(1)) * 100000 + int(m.group(2)) * 3600 + int(m.group(3)) * 60
            return 0
        all_facts.sort(key=sort_key)

        # Embedding filter for topic candidates: query vs time-filtered facts
        # Only fill remaining slots (embedding_slots) after keyword-matched topics
        _TOPIC_SIM_THRESHOLD = topic_sim_threshold
        _embedding_keep_topics = set()  # topics that pass embedding check
        if _topic_embedding_set and embedding_model and query_text and embedding_slots > 0:
            import numpy as np
            # Collect time-filtered facts text per candidate topic
            topic_candidate_facts = {}
            for f in all_facts:
                lbl = f.get("label", "")
                if lbl in _topic_embedding_set and f.get("chain_type") == "topic":
                    if lbl not in topic_candidate_facts:
                        topic_candidate_facts[lbl] = []
                    topic_candidate_facts[lbl].append(f.get("fact", ""))

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

        # Storyline Tier 2: embedding check for candidates, fill remaining slots
        _accepted_storyline_names = set()
        for sl_idx, _ in keyword_matched_sl:
            _accepted_storyline_names.add(self._storylines[sl_idx]["name"])

        if _embedding_sl_candidates and embedding_model and query_text and embedding_sl_slots > 0:
            import numpy as np
            sl_idx_to_name = {idx: self._storylines[idx]["name"] for idx in _embedding_sl_candidates}
            candidate_names = set(sl_idx_to_name.values())

            candidate_facts_text = {}
            for f in all_facts:
                lbl = f.get("label", "")
                if lbl in candidate_names and f.get("chain_type") == "event":
                    if lbl not in candidate_facts_text:
                        candidate_facts_text[lbl] = []
                    candidate_facts_text[lbl].append(f.get("fact", ""))

            if candidate_facts_text:
                try:
                    labels = list(candidate_facts_text.keys())
                    texts = [" ".join(candidate_facts_text[lbl])[:2000] for lbl in labels]
                    all_texts = [query_text] + texts
                    embeddings = embedding_model.encode_text(all_texts)
                    query_emb = embeddings[0]

                    scored_sl = []
                    for i, lbl in enumerate(labels):
                        sl_emb = embeddings[i + 1]
                        cos_sim = float(np.dot(query_emb, sl_emb) /
                                       (np.linalg.norm(query_emb) * np.linalg.norm(sl_emb) + 1e-8))
                        if cos_sim >= storyline_sim_threshold:
                            scored_sl.append((lbl, cos_sim))
                            logger.info(f"[chain] Storyline embedding: {lbl[:30]} sim={cos_sim:.4f} [KEEP]")
                        else:
                            logger.info(f"[chain] Storyline embedding: {lbl[:30]} sim={cos_sim:.4f} [DROP]")

                    # Sort by similarity, take top embedding_sl_slots
                    scored_sl.sort(key=lambda x: -x[1])
                    name_to_idx = {v: k for k, v in sl_idx_to_name.items()}
                    for lbl, sim in scored_sl[:embedding_sl_slots]:
                        _accepted_storyline_names.add(lbl)
                        if lbl in name_to_idx:
                            sl_idx = name_to_idx[lbl]
                            step_count = len(storyline_step_hits.get(sl_idx, set()))
                            relevant_storylines.append((sl_idx, step_count))

                except Exception as e:
                    logger.warning(f"Storyline embedding fallback failed: {e}")

        # Filter storyline facts to only accepted storylines
        all_facts = [f for f in all_facts
                     if f.get("chain_type") != "event" or f.get("label", "") in _accepted_storyline_names]

        n_topic = len(relevant)
        n_storyline = len(relevant_storylines)
        facts_per_label = {}
        for f in all_facts:
            lbl = f.get("label", "")
            facts_per_label[lbl] = facts_per_label.get(lbl, 0) + 1
        topic_info = [f'{n}({topic_hits.get(n,0)} hits, {facts_per_label.get(n,0)}f)' for (_, n), _ in relevant[:5]]
        sl_info = [f'{self._storylines[idx]["name"][:25]}({c} hits, {facts_per_label.get(self._storylines[idx]["name"],0)}f)' for idx, c in relevant_storylines[:5]]
        logger.info(f"[chain] {n_topic} topics ({len(keyword_matched)} keyword, {n_topic-len(keyword_matched)} embedding), "
                    f"{n_storyline} storylines ({len(keyword_matched_sl)} keyword, {n_storyline-len(keyword_matched_sl)} embedding), "
                    f"{len(all_facts)} facts (skipped {skipped_covered} covered) "
                    f"({', '.join(topic_info + sl_info)})")

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
    # Query-matched fact retrieval
    # ------------------------------------------------------------------

    def match_facts(self, query: str, top_k: int = 5) -> str:
        """
        Match query against event fact summaries and chain key_facts.
        Returns a formatted string of top-K relevant facts to prepend to context.

        Only returns facts that are genuinely relevant to the query,
        avoiding the noise from attaching facts to every episode.
        """
        if self._active_graph is None:
            return ""

        import re
        query_lower = query.lower()
        query_words = set(re.findall(r'[a-z]+', query_lower))
        query_words -= {'the','a','an','is','was','were','are','in','on','at','to',
                        'and','of','for','with','from','what','who','when','where',
                        'how','did','do','does','which','that','this','i','my','me',
                        'we','our','last','time','first','about'}
        query_words = {w for w in query_words if len(w) >= 3}

        if not query_words:
            return ""

        # Score event facts by keyword overlap with query
        fact_scores = []  # (score, text)
        for nid, node in self._active_graph.nodes.items():
            if node.node_type != NodeType.EPISODE:
                continue
            fact = node.metadata.get("event_fact")
            if not fact or not fact.get("summary"):
                continue
            summary = fact["summary"].lower()
            overlap = len(query_words & set(re.findall(r'[a-z]+', summary)))
            if overlap >= 2:  # at least 2 query words match
                # Build display text
                parts = [f"[Event] {fact['summary']}"]
                participants = fact.get("participants", [])
                if participants:
                    roles = "; ".join(f"{p['name']}={p['role']}" for p in participants
                                      if isinstance(p, dict) and 'name' in p and 'role' in p)
                    if roles:
                        parts.append(f"  Roles: {roles}")
                fact_scores.append((overlap, "\n".join(parts)))

        # Score chain facts by keyword overlap
        chain_scores = []
        seen_chains = set()
        for nid, node in self._active_graph.nodes.items():
            if node.node_type != NodeType.EPISODE:
                continue
            for cf in node.metadata.get("chain_facts", []):
                topic = cf.get("topic", "")
                if topic in seen_chains:
                    continue
                summary = cf.get("summary", "").lower()
                key_facts = cf.get("key_facts", [])
                combined = summary + " " + " ".join(key_facts).lower()
                overlap = len(query_words & set(re.findall(r'[a-z]+', combined)))
                if overlap >= 2:
                    seen_chains.add(topic)
                    facts_str = "; ".join(key_facts[:3])
                    chain_scores.append((overlap, f"[Cross-episode: {topic}] {facts_str}"))

        # Take top-K by overlap score
        all_scored = sorted(fact_scores + chain_scores, key=lambda x: -x[0])
        top = all_scored[:top_k]

        if not top:
            return ""

        lines = [item[1] for item in top]
        return "\n".join(lines)

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
        structured_output: bool = False,
    ) -> List[RetrievedItem]:
        """
        Convert (GraphNode, score) pairs to RetrievedItem objects compatible
        with WorldMemory._render_retrieved_items_for_qa().

        Node type mapping:
          EPISODE / ENTITY  → memory_type="episodic",  content=str
          SEMANTIC           → memory_type="semantic",  content=str
          VISUAL_CLIP        → memory_type="visual",    content=List[PIL.Image]
        """
        if structured_output:
            return self._format_results_structured(
                results, visual_memory, fps, max_frames_per_clip,
                max_total_frames, _current_total_frames,
            )
        return self._format_results_default(
            results, visual_memory, fps, max_frames_per_clip,
            max_total_frames, _current_total_frames,
        )

    def _format_results_default(
        self, results, visual_memory, fps, max_frames_per_clip,
        max_total_frames, _current_total_frames,
    ) -> List[RetrievedItem]:
        """Format retrieval results into RetrievedItem list (PPR score order)."""
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

    def _format_results_structured(
        self, results, visual_memory, fps, max_frames_per_clip,
        max_total_frames, _current_total_frames,
    ) -> List[RetrievedItem]:
        """Structured format: episodes (by time) → visual → semantic (grouped).

        Gives the responder coherent narrative first, then visual evidence,
        then supplementary facts.
        """
        episode_nodes = []
        visual_nodes = []
        semantic_nodes = []

        for node, _score in results:
            if node.node_type == NodeType.ENTITY:
                continue
            elif node.node_type == NodeType.EPISODE:
                episode_nodes.append(node)
            elif node.node_type == NodeType.VISUAL_CLIP:
                visual_nodes.append(node)
            elif node.node_type == NodeType.SEMANTIC:
                semantic_nodes.append(node)

        # Sort episodes by time for coherent reading order
        episode_nodes.sort(key=lambda n: n.start_ts)

        items: List[RetrievedItem] = []
        total_frames = _current_total_frames

        # 1. Episode captions (coherent narrative)
        for node in episode_nodes:
            items.append(RetrievedItem(
                memory_type = "episodic",
                content     = node.display_text(),
                query       = "",
                round_num   = 0,
            ))

        # 2. Visual clips (caption + frames)
        for node in visual_nodes:
            items.append(RetrievedItem(
                memory_type = "episodic",
                content     = node.display_text(),
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
                items.append(RetrievedItem(
                    memory_type = "visual",
                    content     = pil_images,
                    query       = "",
                    round_num   = 0,
                ))

        # 3. Semantic triples (grouped as supplementary facts)
        if semantic_nodes:
            triple_texts = [node.display_text() for node in semantic_nodes]
            items.append(RetrievedItem(
                memory_type = "semantic",
                content     = "Related facts:\n" + "\n".join(triple_texts),
                query       = "",
                round_num   = 0,
            ))

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
        self._storylines: list = []                # merged storylines
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
        if mem._storylines:
            for sl in mem._storylines:
                sl_copy = dict(sl)
                if "broadcast_id" not in sl_copy:
                    sl_copy["broadcast_id"] = broadcast_id
                self._storylines.append(sl_copy)
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
        from collections import defaultdict
        bid_results = defaultdict(list)
        for node, score in results:
            bid = getattr(node, '_broadcast_id', None)
            if bid:
                bid_results[bid].append((node, score))

        # For each video, use its own chain data
        for bid, bid_res in bid_results.items():
            mem = self._memories.get(bid)
            if mem and (mem._topic_facts or mem._storylines):
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
        self._storylines.clear()
