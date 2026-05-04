"""
MultimodalGraph: builds and stores the unified heterogeneous graph.

Node types : EntityNode, EpisodeNode, SemanticNode, VisualClipNode
Edge types : mentioned_in, appeared_in, has_property,
             temporal_next, contains, co_clip
"""

import json
import logging
import pickle
from hashlib import md5
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig
import numpy as np
import torch
import torch.nn.functional as F

from .nodes import (
    GraphNode, GraphEdge, NodeType, EdgeType,
    caption_to_timestamps, make_entity_id, make_episode_id,
    make_semantic_id, make_visual_id, normalize_entity_name,
    ts_to_seconds,
)

logger = logging.getLogger(__name__)

# Gap threshold in seconds above which two consecutive events are NOT linked
# by a temporal_next edge (e.g. overnight sleep gap).
_TEMPORAL_NEXT_MAX_GAP_SEC = 6 * 3600  # 6 hours


class MultimodalGraph:
    """
    Unified multimodal knowledge graph built from WorldMM preprocessed outputs.

    Build once offline via ``build_unified_graph.py``, serialise with ``save()``,
    and load at eval time with ``MultimodalGraph.load()``.

    At eval time call ``filter_by_time(until_time)`` to get a time-bounded
    subgraph, then ``build_igraph()`` to prepare it for PPR queries.
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge]      = []

        # igraph objects (built by build_igraph or filter_by_time)
        self._ig:              Optional[ig.Graph]   = None
        self._node_id_to_idx:  Dict[str, int]       = {}
        self._idx_to_node_id:  Dict[int, str]       = {}
        self._edge_weights:    List[float]          = []

        # Cached tensors for fast similarity search (populated by build_igraph)
        self._all_node_ids:    List[str]            = []
        self._text_emb_matrix: Optional[torch.Tensor] = None   # (N, 2560)
        self._vis_node_ids:    List[str]            = []
        self._vis_emb_matrix:  Optional[torch.Tensor] = None   # (M, 3584)
        self._bm25 = None  # BM25Okapi index for keyword-based seed selection

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _add_node(self, node: GraphNode) -> None:
        if node.id not in self.nodes:
            self.nodes[node.id] = node
        else:
            # Merge: update end_ts (last seen) and frequency
            existing = self.nodes[node.id]
            existing.end_ts = max(existing.end_ts, node.end_ts)
            existing.metadata["frequency"] = existing.metadata.get("frequency", 1) + 1

    def _add_edge(self, src: str, dst: str, etype: EdgeType, weight: float = 1.0) -> None:
        if src in self.nodes and dst in self.nodes:
            self.edges.append(GraphEdge(src_id=src, dst_id=dst, edge_type=etype, weight=weight))

    # ------------------------------------------------------------------
    # Step A: Episode nodes from all-granularity caption JSONs
    # ------------------------------------------------------------------

    def add_episode_nodes(self, caption_files: Dict[str, str]) -> None:
        """
        Build EpisodeNodes from caption JSON files for all granularities.
        Also builds CONTAINS and TEMPORAL_NEXT edges.

        caption_files: dict mapping granularity → file path
            e.g. {"30sec": "…/A1_JAKE_30sec.json", "3min": "…/A1_JAKE_3min.json", …}
        """
        granularity_order = ["30sec", "3min", "10min", "1h"]
        # Store per-granularity lists for edge construction
        gran_nodes: Dict[str, List[GraphNode]] = {}

        for gran in granularity_order:
            path = caption_files.get(gran)
            if not path:
                continue
            with open(path, "r") as f:
                data = json.load(f)

            nodes_this_gran: List[GraphNode] = []
            for entry in data:
                start_ts, end_ts = caption_to_timestamps(entry)
                nid  = make_episode_id(start_ts, end_ts, gran)
                text = entry.get("text", "")
                meta = {
                    "granularity": gran,
                    "video_path":  entry.get("video_path"),
                }
                if "start_sec" in entry:
                    # Video-MME format
                    meta["start_sec"] = float(entry["start_sec"])
                    meta["end_sec"]   = float(entry["end_sec"])
                else:
                    # EgoLife format
                    meta["date"]       = entry.get("date", "")
                    meta["start_time"] = str(entry.get("start_time", ""))
                    meta["end_time"]   = str(entry.get("end_time", ""))
                node = GraphNode(
                    id        = nid,
                    node_type = NodeType.EPISODE,
                    text      = text,
                    start_ts  = start_ts,
                    end_ts    = end_ts,
                    metadata  = meta,
                )
                self._add_node(node)
                nodes_this_gran.append(node)

            # Sort by start_ts for temporal edges
            nodes_this_gran.sort(key=lambda n: n.start_ts)
            gran_nodes[gran] = nodes_this_gran
            logger.info(f"Added {len(nodes_this_gran)} EpisodeNodes ({gran})")

        # Build TEMPORAL_NEXT edges (same granularity, adjacent in time)
        for gran, nodes in gran_nodes.items():
            for i in range(len(nodes) - 1):
                cur, nxt = nodes[i], nodes[i + 1]
                gap = ts_to_seconds(nxt.start_ts) - ts_to_seconds(cur.end_ts)
                if 0 <= gap <= _TEMPORAL_NEXT_MAX_GAP_SEC:
                    w = 1.0 / (gap + 1)
                    self._add_edge(cur.id, nxt.id, EdgeType.TEMPORAL_NEXT, weight=w)
                    self._add_edge(nxt.id, cur.id, EdgeType.TEMPORAL_NEXT, weight=w)  # bidirectional

        # Build CONTAINS edges (coarse → fine, adjacent layers only)
        layer_pairs = [("3min", "30sec"), ("10min", "3min"), ("1h", "10min")]
        for coarse_gran, fine_gran in layer_pairs:
            coarse_nodes = gran_nodes.get(coarse_gran, [])
            fine_nodes   = gran_nodes.get(fine_gran,   [])
            for coarse in coarse_nodes:
                for fine in fine_nodes:
                    if coarse.start_ts <= fine.start_ts and fine.end_ts <= coarse.end_ts:
                        self._add_edge(coarse.id, fine.id, EdgeType.CONTAINS, weight=1.0)

        # Also build cross-layer CONTAINS (1h → 30sec) with lower weight
        for coarse in gran_nodes.get("1h", []):
            for fine in gran_nodes.get("30sec", []):
                if coarse.start_ts <= fine.start_ts and fine.end_ts <= coarse.end_ts:
                    self._add_edge(coarse.id, fine.id, EdgeType.CONTAINS, weight=0.5)

        logger.info("Episode nodes and structural edges built.")

    # ------------------------------------------------------------------
    # Step B: Visual clip nodes from 30sec captions + embeddings
    # ------------------------------------------------------------------

    def add_visual_nodes(
        self,
        caption_30sec: List[dict],
        visual_embeddings: Dict[str, np.ndarray],
    ) -> None:
        """
        Build VisualClipNodes from 30sec caption data and precomputed visual embeddings.
        Also builds CO_CLIP edges linking each VisualClipNode ↔ its EpisodeNode(30sec).

        caption_30sec:     list of dicts from caption JSON (EgoLife or Video-MME format)
        visual_embeddings: Dict[key, np.ndarray] from visual_embeddings.pkl
                           Key is video_path for EgoLife, or "video_path:start-end" for Video-MME.
        """
        added = 0
        for entry in caption_30sec:
            video_path = entry.get("video_path")
            if not video_path:
                continue

            start_ts, end_ts = caption_to_timestamps(entry)

            if "start_sec" in entry:
                # Video-MME format
                vid = make_visual_id(entry["start_sec"])
                emb_key = f"{video_path}:{entry['start_sec']}-{entry['end_sec']}"
                vis_emb = visual_embeddings.get(emb_key)
                meta = {
                    "video_path":  video_path,
                    "start_sec":   float(entry["start_sec"]),
                    "end_sec":     float(entry["end_sec"]),
                }
            else:
                # EgoLife format
                date       = entry.get("date", "")
                start_time = str(entry.get("start_time", ""))
                vid = make_visual_id(date, start_time)
                vis_emb = visual_embeddings.get(video_path)
                meta = {
                    "video_path":  video_path,
                    "date":        date,
                    "start_time":  start_time,
                    "end_time":    str(entry.get("end_time", "")),
                }

            node = GraphNode(
                id             = vid,
                node_type      = NodeType.VISUAL_CLIP,
                text           = entry.get("text", ""),
                start_ts       = start_ts,
                end_ts         = end_ts,
                visual_embedding = vis_emb,
                metadata       = meta,
            )
            self._add_node(node)
            added += 1

            # CO_CLIP: link to the matching EpisodeNode(30sec)
            ep_id = make_episode_id(start_ts, end_ts, "30sec")
            if ep_id in self.nodes:
                self._add_edge(vid,   ep_id, EdgeType.CO_CLIP, weight=1.0)
                self._add_edge(ep_id, vid,   EdgeType.CO_CLIP, weight=1.0)

        logger.info(f"Added {added} VisualClipNodes with CO_CLIP edges.")

    # ------------------------------------------------------------------
    # Step C: Entity nodes from NER results
    # ------------------------------------------------------------------

    def add_entity_nodes(
        self,
        openie_path: str,
        caption_30sec: List[dict],
        person: str = "I",
    ) -> None:
        """
        Build EntityNodes from NER results and add MENTIONED_IN edges
        (EntityNode → EpisodeNode(30sec)).
        Also derives APPEARED_IN edges via the CO_CLIP shortcut (v1 rule).

        openie_path:   path to openie_results_*.json
        caption_30sec: list of dicts from A1_JAKE_30sec.json (needed to map chunk-md5 → caption)
        person:        subject identifier used to map first-person 'I' (e.g. 'A1_JAKE')
        """
        with open(openie_path, "r") as f:
            openie_data = json.load(f)

        ner_results: Dict[str, List[str]] = openie_data.get("ner_results", {})

        # Build md5(caption_text) → EpisodeNode id mapping
        chunk_to_ep: Dict[str, str] = {}
        for entry in caption_30sec:
            text = entry.get("text", "")
            if not text:
                continue
            chunk_id = "chunk-" + md5(text.encode()).hexdigest()
            start_ts, end_ts = caption_to_timestamps(entry)
            ep_id = make_episode_id(start_ts, end_ts, "30sec")
            if ep_id in self.nodes:
                chunk_to_ep[chunk_id] = ep_id

        entity_first_seen: Dict[str, int] = {}
        entity_last_seen:  Dict[str, int] = {}

        for chunk_id, entities in ner_results.items():
            ep_id = chunk_to_ep.get(chunk_id)
            if ep_id is None:
                continue
            ep_node = self.nodes[ep_id]

            for name in entities:
                if not name or not name.strip():
                    continue
                ent_id = make_entity_id(name, person)
                norm   = normalize_entity_name(name, person)

                # Create or update EntityNode
                if ent_id not in self.nodes:
                    node = GraphNode(
                        id        = ent_id,
                        node_type = NodeType.ENTITY,
                        text      = name.strip(),
                        start_ts  = ep_node.start_ts,
                        end_ts    = ep_node.end_ts,
                        metadata  = {"frequency": 1, "normalized_name": norm},
                    )
                    self._add_node(node)
                    entity_first_seen[ent_id] = ep_node.start_ts
                    entity_last_seen[ent_id]  = ep_node.end_ts
                else:
                    # Update frequency and time bounds
                    ent_node = self.nodes[ent_id]
                    ent_node.metadata["frequency"] = ent_node.metadata.get("frequency", 1) + 1
                    ent_node.end_ts = max(ent_node.end_ts, ep_node.end_ts)
                    entity_last_seen[ent_id] = max(
                        entity_last_seen.get(ent_id, 0), ep_node.end_ts
                    )

                # MENTIONED_IN: EntityNode → EpisodeNode(30sec)
                self._add_edge(ent_id, ep_id, EdgeType.MENTIONED_IN, weight=1.0)

        # APPEARED_IN: derive from MENTIONED_IN + CO_CLIP (v1 rule)
        # Entity→Episode→VisualClip shortcut edge
        appeared_pairs: Set[Tuple[str, str]] = set()
        for edge in self.edges:
            if edge.edge_type == EdgeType.MENTIONED_IN:
                ent_id = edge.src_id
                ep_id  = edge.dst_id
                # Find VisualClipNode co-clipped with this episode
                for e2 in self.edges:
                    if e2.edge_type == EdgeType.CO_CLIP and e2.src_id == ep_id:
                        vis_id = e2.dst_id
                        key = (ent_id, vis_id)
                        if key not in appeared_pairs:
                            appeared_pairs.add(key)
                            self._add_edge(ent_id, vis_id, EdgeType.APPEARED_IN, weight=0.8)

        logger.info(
            f"Added {sum(1 for n in self.nodes.values() if n.node_type == NodeType.ENTITY)} "
            f"EntityNodes, {len(appeared_pairs)} APPEARED_IN edges."
        )

    # ------------------------------------------------------------------
    # Step D: Semantic nodes from consolidation results
    # ------------------------------------------------------------------

    def add_semantic_nodes(self, consolidation_path: str, person: str = "I") -> None:
        """
        Build SemanticNodes from semantic_consolidation_results_*.json.
        Also builds HAS_PROPERTY edges (EntityNode → SemanticNode).

        consolidation_path: path to semantic_consolidation_results_*.json
        """
        with open(consolidation_path, "r") as f:
            data = json.load(f)

        sem_count = 0
        for timestamp_str, content in data.items():
            timestamp = int(timestamp_str)
            triples   = content.get("consolidated_semantic_triples", [])

            for idx, triple in enumerate(triples):
                if len(triple) < 3:
                    continue
                subj, pred, obj = triple[0], triple[1], triple[2]
                sem_id   = make_semantic_id(timestamp, idx)
                sem_text = f"{subj} {pred} {obj}"

                node = GraphNode(
                    id        = sem_id,
                    node_type = NodeType.SEMANTIC,
                    text      = sem_text,
                    start_ts  = timestamp,
                    end_ts    = timestamp,
                    metadata  = {
                        "subject":   subj,
                        "predicate": pred,
                        "object":    obj,
                    },
                )
                self._add_node(node)
                sem_count += 1

                # HAS_PROPERTY: subject EntityNode → SemanticNode
                subj_ent_id = make_entity_id(subj, person)
                if subj_ent_id in self.nodes:
                    self._add_edge(subj_ent_id, sem_id, EdgeType.HAS_PROPERTY, weight=1.0)

                # HAS_PROPERTY: object EntityNode → SemanticNode (if object is known entity)
                obj_ent_id = make_entity_id(obj, person)
                if obj_ent_id in self.nodes:
                    self._add_edge(obj_ent_id, sem_id, EdgeType.HAS_PROPERTY, weight=0.5)

        logger.info(f"Added {sem_count} SemanticNodes with HAS_PROPERTY edges.")

    # ------------------------------------------------------------------
    # Step E: Compute text embeddings for all nodes
    # ------------------------------------------------------------------

    def compute_text_embeddings(self, embedding_model, batch_size: int = 256) -> None:
        """
        Batch-compute text embeddings for all nodes whose text_embedding is None.
        embedding_model: EmbeddingModel instance (encode_text method)
        """
        # Skip SemanticNodes at build time: the consolidation file has 623 cumulative
        # snapshots summing to 1.33M triples, but filter_by_time() will only keep the
        # latest snapshot (≤3821 triples). Those will be embedded on-demand in
        # UnifiedMemory.index() after filtering. Embedding all 1.33M here would OOM.
        to_compute = [
            (nid, node.text)
            for nid, node in self.nodes.items()
            if node.text_embedding is None and node.node_type != NodeType.SEMANTIC
        ]
        if not to_compute:
            logger.info("All text embeddings already computed.")
            return

        logger.info(f"Computing text embeddings for {len(to_compute)} nodes…")
        ids   = [t[0] for t in to_compute]
        texts = [t[1] for t in to_compute]

        for start in range(0, len(texts), batch_size):
            batch_ids   = ids[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            embs        = embedding_model.encode_text(batch_texts)  # np.ndarray (B, D)
            for nid, emb in zip(batch_ids, embs):
                self.nodes[nid].text_embedding = emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Text embeddings computed.")

    # ------------------------------------------------------------------
    # Step F: Build igraph (call after filter_by_time or on full graph)
    # ------------------------------------------------------------------

    def build_igraph(self) -> None:
        """
        Convert node/edge lists to an igraph Graph for PPR queries.
        Also caches text and visual embedding matrices.
        """
        node_ids = list(self.nodes.keys())
        n        = len(node_ids)

        self._node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        self._idx_to_node_id = {i: nid for i, nid in enumerate(node_ids)}
        self._all_node_ids   = node_ids

        # Build igraph
        g = ig.Graph(n=n, directed=False)
        g.vs["name"] = node_ids

        edge_list    = []
        edge_weights = []
        edge_types   = []
        seen_pairs: Set[Tuple[int, int]] = set()

        for edge in self.edges:
            si = self._node_id_to_idx.get(edge.src_id)
            di = self._node_id_to_idx.get(edge.dst_id)
            if si is None or di is None:
                continue
            # Deduplicate undirected pairs
            key = (min(si, di), max(si, di))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edge_list.append((si, di))
            edge_weights.append(edge.weight)
            edge_types.append(edge.edge_type.value)

        g.add_edges(edge_list)
        g.es["weight"] = edge_weights
        g.es["edge_type"] = edge_types
        self._ig           = g
        self._edge_weights = edge_weights

        # Cache embedding matrices for fast similarity search
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Probe text embedding dim from the first node that has one. We don't
        # know it ahead of time because users may swap the embedding model.
        emb_dim: Optional[int] = None
        for nid in node_ids:
            node = self.nodes[nid]
            if node.text_embedding is not None:
                emb_dim = int(node.text_embedding.shape[-1])
                break
        if emb_dim is None:
            emb_dim = 2560  # fall back to Qwen3-Embedding-4B default

        text_embs = []
        for nid in node_ids:
            node = self.nodes[nid]
            # VisualClipNodes share identical text with Episode(30sec) →
            # zero out to avoid duplicate text seeds (visual info handled
            # by _vis_emb_matrix).  EntityNodes are short names ("Shure")
            # with low sentence-level similarity to queries → zero out so
            # they don't waste seed slots; they still serve as PPR bridges.
            if node.node_type in (NodeType.VISUAL_CLIP, NodeType.ENTITY):
                text_embs.append(np.zeros(emb_dim, dtype=np.float32))
            elif node.text_embedding is not None:
                text_embs.append(node.text_embedding)
            else:
                text_embs.append(np.zeros(emb_dim, dtype=np.float32))

        self._text_emb_matrix = torch.tensor(
            np.stack(text_embs), dtype=torch.float32, device=device
        )

        vis_ids  = []
        vis_embs = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.node_type == NodeType.VISUAL_CLIP and node.visual_embedding is not None:
                vis_ids.append(nid)
                vis_embs.append(node.visual_embedding)

        self._vis_node_ids = vis_ids
        if vis_embs:
            self._vis_emb_matrix = torch.tensor(
                np.stack(vis_embs), dtype=torch.float32, device=device
            )
        else:
            self._vis_emb_matrix = None

        # ---- BM25 index for keyword-based seed selection ----
        try:
            from rank_bm25 import BM25Okapi
            corpus_tokens = []
            for nid in node_ids:
                node = self.nodes[nid]
                text = node.display_text() if hasattr(node, 'display_text') else node.text
                corpus_tokens.append(text.lower().split())
            self._bm25 = BM25Okapi(corpus_tokens)
        except ImportError:
            logger.warning("rank_bm25 not installed; BM25 seeding disabled.")
            self._bm25 = None

        logger.info(
            f"igraph built: {n} nodes, {len(edge_list)} edges, "
            f"{len(vis_ids)} visual nodes with embeddings."
        )

    # ------------------------------------------------------------------
    # Time filtering
    # ------------------------------------------------------------------

    def filter_by_time(self, until_time) -> "MultimodalGraph":
        """
        Return a new MultimodalGraph containing only nodes observable at or before
        ``until_time``, and only edges whose both endpoints are visible.

        Visibility rules:
          EpisodeNode / VisualClipNode : end_ts   <= until_time
          EntityNode                   : start_ts <= until_time  (seen at least once)
          SemanticNode                 : start_ts == latest consolidated snapshot ts <= until_time
                                         (each snapshot is cumulative; only the newest is needed)
        """
        import copy

        # Find the latest semantic snapshot timestamp <= until_time.
        # Semantic consolidation is cumulative: each snapshot supersedes all earlier ones.
        sem_ts_candidates = [
            node.start_ts
            for node in self.nodes.values()
            if node.node_type == NodeType.SEMANTIC and node.start_ts <= until_time
        ]
        latest_sem_ts = max(sem_ts_candidates) if sem_ts_candidates else None

        sub = MultimodalGraph()

        for nid, node in self.nodes.items():
            if node.node_type in (NodeType.EPISODE, NodeType.VISUAL_CLIP):
                visible = node.end_ts <= until_time
            elif node.node_type == NodeType.ENTITY:
                visible = node.start_ts <= until_time
            else:  # SEMANTIC
                visible = (latest_sem_ts is not None and node.start_ts == latest_sem_ts)

            if visible:
                sub.nodes[nid] = copy.copy(node)

        visible_ids = set(sub.nodes.keys())
        for edge in self.edges:
            if edge.src_id in visible_ids and edge.dst_id in visible_ids:
                sub.edges.append(edge)

        logger.debug(
            f"filter_by_time({until_time}): {len(sub.nodes)} nodes, {len(sub.edges)} edges."
        )
        return sub

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise nodes and edges to a pickle file (igraph is NOT saved)."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"nodes": self.nodes, "edges": self.edges}, f)
        logger.info(f"Graph saved to {path}  ({len(self.nodes)} nodes, {len(self.edges)} edges)")

    @classmethod
    def load(cls, path: str) -> "MultimodalGraph":
        """Load a serialised graph.  Call build_igraph() afterwards."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        g = cls()
        g.nodes = data["nodes"]
        g.edges = data["edges"]
        logger.info(f"Graph loaded from {path}  ({len(g.nodes)} nodes, {len(g.edges)} edges)")
        return g

    # ------------------------------------------------------------------
    # Quick stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        type_counts = {}
        for node in self.nodes.values():
            k = node.node_type.value
            type_counts[k] = type_counts.get(k, 0) + 1
        edge_counts = {}
        for edge in self.edges:
            k = edge.edge_type.value
            edge_counts[k] = edge_counts.get(k, 0) + 1
        return {"nodes": type_counts, "edges": edge_counts}
