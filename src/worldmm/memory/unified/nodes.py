"""
Node and edge dataclass definitions for the Unified Multimodal Graph.

Supports two timestamp formats:
  - EgoLife: integer DHHMMSSFF (e.g. 111094300 = DAY1 11:09:43)
  - Video-MME: float seconds (e.g. 30.0 = 0:00:30)

Per-graph consistency: each graph uses one format throughout.  The two
formats never mix within a single graph.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
import numpy as np


class NodeType(Enum):
    ENTITY      = "entity"
    EPISODE     = "episode"
    SEMANTIC    = "semantic"
    VISUAL_CLIP = "visual_clip"


class EdgeType(Enum):
    MENTIONED_IN  = "mentioned_in"   # EntityNode  → EpisodeNode
    APPEARED_IN   = "appeared_in"    # EntityNode  → VisualClipNode
    HAS_PROPERTY  = "has_property"   # EntityNode  → SemanticNode
    TEMPORAL_NEXT = "temporal_next"  # EpisodeNode → EpisodeNode (same granularity, adjacent)
    CONTAINS      = "contains"       # EpisodeNode(coarse) → EpisodeNode(fine)
    CO_CLIP       = "co_clip"        # EpisodeNode(30sec) ↔ VisualClipNode


@dataclass
class GraphNode:
    """
    A node in the unified multimodal graph.

    Timestamp format depends on the dataset:
      - EgoLife:   integer DHHMMSSFF  (e.g. 111094300)
      - Video-MME: float seconds      (e.g. 30.0)
    """
    id:        str
    node_type: NodeType
    text:      str          # Natural-language description; used to compute text_embedding

    # Time bounds.  Type is int (EgoLife DHHMMSSFF) or float (Video-MME seconds).
    # EntityNode:    start_ts = first_seen_ts,  end_ts = last_seen_ts
    # SemanticNode:  start_ts = end_ts = consolidation_timestamp
    # Episode/Visual: start_ts, end_ts from caption JSON
    start_ts: Union[int, float] = 0
    end_ts:   Union[int, float] = 0

    # Embeddings (populated lazily during graph.compute_text_embeddings / load_visual_embeddings)
    text_embedding:   Optional[np.ndarray] = None  # shape (2560,) – all node types
    visual_embedding: Optional[np.ndarray] = None  # shape (3584,) – VisualClipNode only

    # Arbitrary metadata (video_path, granularity, predicate, frequency, …)
    metadata: dict = field(default_factory=dict)

    def display_text(self) -> str:
        """Human-readable label used when feeding retrieved nodes to the LLM."""
        ts = _fmt_ts(self.start_ts)
        te = _fmt_ts(self.end_ts)
        if self.node_type == NodeType.EPISODE:
            gran = self.metadata.get("granularity", "")
            return f"[{ts} - {te}] ({gran}) {self.text}"
        elif self.node_type == NodeType.SEMANTIC:
            subj = self.metadata.get("subject", "")
            pred = self.metadata.get("predicate", "")
            obj  = self.metadata.get("object", "")
            return f"({subj}, {pred}, {obj})"
        elif self.node_type == NodeType.ENTITY:
            return f"Entity: {self.text}"
        elif self.node_type == NodeType.VISUAL_CLIP:
            return f"[{ts} - {te}]"
        return self.text


@dataclass
class GraphEdge:
    src_id:    str
    dst_id:    str
    edge_type: EdgeType
    weight:    float = 1.0


# ---------------------------------------------------------------------------
# Helper: timestamp formatting
# ---------------------------------------------------------------------------

def _is_seconds_ts(ts) -> bool:
    """Return True if *ts* is in seconds format (float, or int < 1_000_000)."""
    if isinstance(ts, float):
        return True
    return isinstance(ts, int) and ts < 1_000_000


def _fmt_ts(ts) -> str:
    """Format a timestamp for human-readable display.

    Handles both EgoLife DHHMMSSFF ints and Video-MME float seconds.
    """
    if _is_seconds_ts(ts):
        total_sec = int(ts)
        hh = total_sec // 3600
        mm = (total_sec % 3600) // 60
        ss = total_sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    # EgoLife DHHMMSSFF
    s = str(ts)
    if len(s) < 7:
        return str(ts)
    day     = s[0]
    rest    = s[1:].zfill(8)
    hh, mm, ss = rest[0:2], rest[2:4], rest[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


# ---------------------------------------------------------------------------
# Helper: entity normalisation
# ---------------------------------------------------------------------------

def normalize_entity_name(name: str, person: str = "I") -> str:
    """
    Normalize an entity name to a stable node ID suffix.

    'I' / 'i' (first-person) is mapped to the subject person id.
    """
    stripped = str(name).strip()
    if stripped in ("I", "i", "me", "Me", "my", "My"):
        return person.lower().replace("/", "_").replace("-", "_")
    return stripped.lower().replace(" ", "_").replace("'", "").replace("-", "_")


# ---------------------------------------------------------------------------
# Node ID constructors
# ---------------------------------------------------------------------------

def make_entity_id(name: str, person: str = "I") -> str:
    return f"ent_{normalize_entity_name(name, person)}"


def make_episode_id(start_ts, end_ts, granularity: str) -> str:
    return f"ep_{start_ts}_{end_ts}_{granularity}"


def make_semantic_id(timestamp, index: int) -> str:
    return f"sem_{timestamp}_{index}"


def make_visual_id(date_or_sec, start_time: Optional[str] = None) -> str:
    """Make visual node ID.

    Two calling conventions:
      - EgoLife:   make_visual_id("DAY1", "11094300")
      - Video-MME: make_visual_id(30.0)   (start seconds, no start_time arg)
    """
    if start_time is not None:
        return f"vis_{date_or_sec}_{start_time}"
    # Video-MME: use integer seconds for a clean ID
    return f"vis_{int(date_or_sec)}"


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------

def caption_to_timestamps(entry: dict) -> tuple:
    """Convert a caption JSON entry to (start_ts, end_ts).

    For Video-MME (entry has ``start_sec``): returns ``(float, float)`` in seconds.
    For EgoLife  (entry has ``date/start_time/end_time``): returns ``(int, int)``
    in DHHMMSSFF format (unchanged from the original code path).
    """
    if "start_sec" in entry:
        # Video-MME format
        return float(entry["start_sec"]), float(entry["end_sec"])
    # EgoLife format (unchanged)
    date = str(entry.get("date", "DAY1"))
    day  = date.replace("DAY", "").replace("Day", "").strip()
    start = str(entry.get("start_time", "0")).zfill(8)
    end   = str(entry.get("end_time",   "0")).zfill(8)
    return int(day + start), int(day + end)


def ts_to_seconds(ts) -> float:
    """Convert a timestamp to total seconds.

    - float values: returned as-is (already seconds).
    - int < 1_000_000: treated as seconds.
    - int >= 1_000_000: parsed as DHHMMSSFF → seconds since DAY1 00:00:00.
    """
    if isinstance(ts, float):
        return ts
    if ts < 1_000_000:
        return float(ts)
    # DHHMMSSFF
    s    = str(ts).zfill(9)
    day  = int(s[0])
    rest = s[1:].zfill(8)
    hh   = int(rest[0:2])
    mm   = int(rest[2:4])
    ss   = int(rest[4:6])
    return float((day - 1) * 86400 + hh * 3600 + mm * 60 + ss)
