"""
Cross-modal Personalized PageRank retrieval over the unified multimodal graph.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .nodes import GraphNode, NodeType
from .graph import MultimodalGraph

logger = logging.getLogger(__name__)


# ======================================================================
# Reset vector (PPR seed) construction
# ======================================================================

def build_reset_vector(
    graph: MultimodalGraph,
    query: str,
    embedding_model,
    alpha: float = 0.7,
    seed_top_k: int = 5,
    bm25_weight: float = 0.3,
) -> np.ndarray:
    """
    Build the PPR reset vector by combining seed channels:
      - Text seeds  (alpha):       top-K nodes by cosine similarity to encode_text(query)
      - BM25 seeds  (bm25_weight): top-K nodes by BM25 keyword matching
      - Visual seeds (1-alpha):    top-K VisualClipNodes by cosine similarity to encode_vis_query(query)

    Returns a 1-D numpy array of shape (N,) normalised to sum=1.
    """
    n = len(graph._all_node_ids)
    reset = np.zeros(n, dtype=np.float32)
    device = graph._text_emb_matrix.device if graph._text_emb_matrix is not None else "cpu"

    # Pre-compute text similarities
    text_sims = None
    if graph._text_emb_matrix is not None:
        q_text = embedding_model.encode_text(query)
        if len(q_text.shape) == 1:
            q_text = q_text[None, :]
        q_text_t = torch.tensor(q_text, dtype=torch.float32, device=device)
        text_sims = F.cosine_similarity(
            q_text_t, graph._text_emb_matrix, dim=1
        ).cpu().numpy()

    # ---- Text seeds ----
    if text_sims is not None and alpha > 0:
        k = min(seed_top_k, n)
        top_idx = np.argsort(text_sims)[-k:]
        for idx in top_idx:
            val = float(text_sims[idx])
            if val > 0:
                reset[idx] += alpha * val

    # ---- BM25 keyword seeds ----
    if graph._bm25 is not None and bm25_weight > 0:
        query_tokens = query.lower().split()
        bm25_scores = graph._bm25.get_scores(query_tokens)
        bm25_scores = np.clip(bm25_scores, 0, None)
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores = bm25_scores / bm25_max
            k_bm25 = min(seed_top_k, n)
            top_bm25_idx = np.argsort(bm25_scores)[-k_bm25:]
            for idx in top_bm25_idx:
                val = float(bm25_scores[idx])
                if val > 0:
                    reset[idx] += bm25_weight * val

    # ---- Visual seeds ----
    if graph._vis_emb_matrix is not None and alpha < 1.0 and len(graph._vis_node_ids) > 0:
        q_vis = embedding_model.encode_vis_query(query)
        if len(q_vis.shape) == 1:
            q_vis = q_vis[None, :]
        q_vis_t = torch.tensor(q_vis, dtype=torch.float32, device=device)
        vis_sims = F.cosine_similarity(
            q_vis_t, graph._vis_emb_matrix, dim=1
        ).cpu().numpy()

        k_vis = min(seed_top_k, len(graph._vis_node_ids))
        top_vis_local = np.argsort(vis_sims)[-k_vis:]
        for local_idx in top_vis_local:
            vis_nid = graph._vis_node_ids[local_idx]
            g_idx = graph._node_id_to_idx.get(vis_nid)
            if g_idx is None:
                continue
            val = float(vis_sims[local_idx])
            if val > 0:
                reset[g_idx] += (1.0 - alpha) * val

    # ---- Normalise ----
    total = reset.sum()
    if total > 0:
        reset /= total
    else:
        reset = np.ones(n, dtype=np.float32) / n

    return reset


# ======================================================================
# Core PPR helpers
# ======================================================================

def _compute_text_sims(graph, query, embedding_model):
    """Compute per-node cosine similarity to query text, shifted to [0,1]."""
    if graph._text_emb_matrix is None:
        return None
    device = graph._text_emb_matrix.device
    q_text = embedding_model.encode_text(query)
    if len(q_text.shape) == 1:
        q_text = q_text[None, :]
    q_text_t = torch.tensor(q_text, dtype=torch.float32, device=device)
    sims = F.cosine_similarity(q_text_t, graph._text_emb_matrix, dim=1).cpu().numpy()
    return (sims + 1.0) / 2.0


def _run_ppr(graph, reset, damping=0.85, edge_weight_attr="weight"):
    """Run igraph PPR and return raw score array."""
    return graph._ig.personalized_pagerank(
        directed       = False,
        damping        = damping,
        reset          = reset.tolist(),
        weights        = edge_weight_attr,
        implementation = "prpack",
    )


# ======================================================================
# Multiscale filtering
# ======================================================================

_GRANULARITY_QUOTA = {
    "30sec": 8,
    "3min":  4,
    "10min": 3,
    "1h":    0,
}

_NODE_TYPE_QUOTA = {
    NodeType.SEMANTIC: 5,
    NodeType.ENTITY: 3,
}


def _multiscale_filter(
    results: List[Tuple[GraphNode, float]],
    top_k: int,
) -> List[Tuple[GraphNode, float]]:
    """Apply per-granularity and per-node-type quota caps to PPR results."""
    out: List[Tuple[GraphNode, float]] = []
    gran_counts: dict = {}
    ntype_counts: dict = {}

    for node, score in results:
        if len(out) >= top_k:
            break

        if node.node_type == NodeType.EPISODE:
            gran = node.metadata.get("granularity", "30sec")
            quota = _GRANULARITY_QUOTA.get(gran, 8)
            count = gran_counts.get(gran, 0)
            if count >= quota:
                continue
            gran_counts[gran] = count + 1

        elif node.node_type in _NODE_TYPE_QUOTA:
            quota = _NODE_TYPE_QUOTA[node.node_type]
            count = ntype_counts.get(node.node_type, 0)
            if count >= quota:
                continue
            ntype_counts[node.node_type] = count + 1

        out.append((node, score))

    return out


# ======================================================================
# Baseline retrieval (igraph PPR)
# ======================================================================

def _baseline_retrieve(
    graph, query, embedding_model, top_k, target_node_types,
    alpha, seed_top_k, damping,
    edge_weight_attr="weight", bm25_weight=0.3,
) -> List[Tuple[GraphNode, float]]:
    """Baseline: PPR x cosine reranking."""
    reset = build_reset_vector(
        graph, query, embedding_model, alpha, seed_top_k,
        bm25_weight=bm25_weight,
    )

    ppr_scores = _run_ppr(graph, reset, damping, edge_weight_attr)
    text_sims = _compute_text_sims(graph, query, embedding_model)

    results: List[Tuple[GraphNode, float]] = []
    for idx, score in enumerate(ppr_scores):
        nid = graph._idx_to_node_id[idx]
        node = graph.nodes[nid]
        if target_node_types is None or node.node_type in target_node_types:
            sim = float(text_sims[idx]) if text_sims is not None else 1.0
            hybrid = float(score) * sim
            results.append((node, hybrid))

    results.sort(key=lambda x: -x[1])
    return _multiscale_filter(results, top_k)


# ======================================================================
# Main entry point
# ======================================================================

def cross_modal_retrieve(
    graph: MultimodalGraph,
    query: str,
    embedding_model,
    top_k: int = 10,
    target_node_types: Optional[List[NodeType]] = None,
    alpha: float = 0.7,
    seed_top_k: int = 5,
    damping: float = 0.85,
    choices: Optional[Dict[str, str]] = None,
    bm25_weight: float = 0.3,
    **kwargs,
) -> List[Tuple[GraphNode, float]]:
    """
    Run Personalised PageRank on the unified graph and return the top-k nodes.
    """
    if graph._ig is None:
        raise RuntimeError("Call graph.build_igraph() before retrieval.")

    return _baseline_retrieve(
        graph, query, embedding_model, top_k, target_node_types,
        alpha, seed_top_k, damping, "weight", bm25_weight,
    )
