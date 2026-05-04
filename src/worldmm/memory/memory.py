"""
WorldMemory: Unified memory system integrating episodic, semantic, and visual memories
with iterative reasoning for long-term video reasoning.
"""

import copy
import gc
import json
import logging
import re
from string import Template
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# Time-aware retrieval: use LLM to extract timestamps from search queries
# ---------------------------------------------------------------------------

_time_extract_llm = None  # lazy-initialized


def _get_time_extract_llm():
    """Get or create a lightweight LLM for time extraction."""
    global _time_extract_llm
    if _time_extract_llm is None:
        from ..llm.openrouter import OpenRouterModel
        _time_extract_llm = OpenRouterModel(model_name="qwen/qwen3.5-flash-02-23")
    return _time_extract_llm


_TIME_EXTRACT_TEMPLATE = (
    'Extract the specific video timestamp or time range from the query below.\n'
    'If the query mentions a specific time (e.g., "at 13:13", "from 35:52 to 36:41", '
    '"between 37:30 and 38:05", "around the 45 minute mark"), return a JSON object '
    'with start and end in seconds.\n'
    'If no specific time is mentioned, return null.\n\n'
    'For point references (e.g., "at 13:13"), use a 30-second window around it.\n\n'
    'Examples:\n'
    '- "What happens from 35:52-36:41?" -> {"start": 2152, "end": 2201}\n'
    '- "What animal are they hunting at 13:13?" -> {"start": 763, "end": 823}\n'
    '- "Who scored the winning goal?" -> null\n'
    '- "What happens between 1:05:30 and 1:06:00?" -> {"start": 3930, "end": 3960}\n'
    '- "What occurs around the 45 minute mark?" -> {"start": 2670, "end": 2730}\n\n'
    'Query: $query\n\n'
    'Return ONLY the JSON object or null, nothing else.'
)


def parse_time_reference(query: str) -> Optional[Tuple[float, float]]:
    """Use LLM to extract a (start_sec, end_sec) time range from a search query.

    Returns None if no time reference is found.
    """
    try:
        llm = _get_time_extract_llm()
        prompt = Template(_TIME_EXTRACT_TEMPLATE).substitute(query=query)
        response = llm.generate(prompt)
        if not response or response.strip().lower() == "null":
            return None
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict) and "start" in parsed and "end" in parsed:
            return (float(parsed["start"]), float(parsed["end"]))
    except Exception:
        pass
    return None

from ..llm import LLMModel, PromptTemplateManager
from ..embedding import EmbeddingModel

from .episodic import EpisodicMemory, CaptionEntry
from .semantic import SemanticMemory, SemanticTripleEntry
from .visual import VisualMemory
from .utils import (
    MemorySearchOutput,
    ReasoningOutput,
    RetrievedItem,
    QAResult,
    transform_timestamp,
)
from .unified import UnifiedMemory
from .unified.nodes import NodeType

logger = logging.getLogger(__name__)


class WorldMemory:
    """
    Unified memory system for WorldMM that integrates episodic, semantic, 
    and visual memories with iterative reasoning.
    
    The system implements a multi-round retrieval process:
    1. Given a query, the reasoning agent decides whether to search or answer
    2. If searching, it selects a memory type and forms a search query
    3. Retrieved context is accumulated across rounds
    4. When the agent decides to answer, the QA model uses all accumulated context
    
    Memory Types:
    - Episodic: Specific events/actions using HippoRAG for retrieval
    - Semantic: Entity/relationship knowledge using PPR graph retrieval  
    - Visual: Scene/setting snapshots using embedding similarity
    
    Attributes:
        episodic_memory: EpisodicMemory instance
        semantic_memory: SemanticMemory instance
        visual_memory: VisualMemory instance
        retriever_llm_model: LLM for retrieval operations (NER, OpenIE)
        respond_llm_model: LLM for iterative reasoning and generating answers
        prompt_template_manager: Manager for prompt templates
        max_rounds: Maximum retrieval rounds
        max_errors: Maximum errors before forcing answer
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        retriever_llm_model: LLMModel,
        respond_llm_model: Optional[LLMModel] = None,
        prompt_template_manager: Optional[PromptTemplateManager] = None,
        episodic_granularities: Optional[List[str]] = None,
        max_rounds: int = 5,
        max_errors: int = 5,
        retrieval_backend: str = "unified_graph",
        graph_path: Optional[str] = None,
        cache_dir: str = ".cache/episodic_memory",
        event_facts_path: Optional[str] = None,
        chain_mode: Optional[str] = None,
        topic_chain_facts_path: Optional[str] = None,
        storyline_path: Optional[str] = None,
    ):
        """
        Initialize WorldMemory with all memory subsystems.

        Args:
            embedding_model: Embedding model for all memory types
            retriever_llm_model: LLM for retrieval operations (NER, OpenIE)
            respond_llm_model: LLM for iterative reasoning and generating answers (defaults to retriever_llm_model)
            prompt_template_manager: Manager for prompt templates (creates default if None)
            episodic_granularities: Granularity levels for episodic memory
            max_rounds: Maximum retrieval rounds before forcing answer
            max_errors: Maximum errors before forcing answer
            retrieval_backend: "unified_graph" (default) or "independent"
            graph_path: Path to serialised MultimodalGraph pkl (required when retrieval_backend="unified_graph")
        """
        self.embedding_model = embedding_model
        self.retriever_llm_model = retriever_llm_model
        self.respond_llm_model = respond_llm_model or retriever_llm_model
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
        self.max_rounds = max_rounds
        self.max_errors = max_errors

        # Initialize memory subsystems (always kept for preprocess compatibility)
        self.episodic_memory = EpisodicMemory(
            embedding_model=embedding_model,
            llm_model=retriever_llm_model,
            prompt_template_manager=self.prompt_template_manager,
            granularities=episodic_granularities,
            cache_dir=cache_dir,
        )

        self.semantic_memory = SemanticMemory(embedding_model=embedding_model)

        self.visual_memory = VisualMemory(embedding_model=embedding_model)

        # Track indexed time
        self.indexed_time: int = 0

        # Retrieval configuration
        self.episodic_top_k: int = 3
        self.semantic_top_k: int = 10
        self.visual_top_k: int = 3
        self.frames_per_clip: int = 16

        # Unified graph backend (optional)
        self.retrieval_backend = retrieval_backend
        self.bm25_weight: float = 0.3              # set via eval.py
        self.damping: float = 0.85                 # PPR damping factor
        self.reasoning_version: str = "egolife"    # template suffix: egolife / mmlifelong
        self.qa_template_name: str = "qa_egolife"  # QA template: "qa_egolife" (MCQ) or "qa_mmlifelong" (open-ended)
        self.seed_top_k: int = 5                   # number of seed nodes per channel for PPR
        self.retrieval_top_k: int = 0              # 0 = auto (episodic+semantic+visual), >0 = override
        self._chain_mode = chain_mode
        self.debug: bool = False  # verbose debug logging
        self.unified_memory: Optional[UnifiedMemory] = None
        if retrieval_backend == "unified_graph":
            if not graph_path:
                raise ValueError("graph_path is required when retrieval_backend='unified_graph'")
            self.unified_memory = UnifiedMemory(
                embedding_model, graph_path, chain_mode=chain_mode,
            )
            self.unified_memory.load(
                event_facts_path=event_facts_path,
                topic_chain_facts_path=topic_chain_facts_path,
                storyline_path=storyline_path,
            )

    def load_episodic_captions(
        self,
        caption_files: Optional[Dict[str, str]] = None,
        caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Load episodic captions from files or data.
        
        Args:
            caption_files: Dict mapping granularity -> JSON file path
            caption_data: Dict mapping granularity -> list of caption dicts
        """
        if caption_files:
            self.episodic_memory.load_captions_from_files(caption_files)
        if caption_data:
            self.episodic_memory.load_captions_from_data(caption_data)
    
    def load_semantic_triples(
        self,
        file_path: Optional[str] = None,
        data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Load semantic triples from file or data.
        
        Args:
            file_path: Path to JSON file with semantic triples
            data: In-memory dict with semantic triples
        """
        if file_path:
            self.semantic_memory.load_triples_from_file(file_path)
        if data:
            self.semantic_memory.load_triples_from_data(data)
    
    def load_visual_clips(
        self,
        embeddings_path: Optional[str] = None,
        clips_path: Optional[str] = None,
        clips_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Load visual clips and embeddings.
        
        Args:
            embeddings_path: Path to pickle file with precomputed embeddings
            clips_path: Path to JSON file with clip metadata
            clips_data: In-memory list of clip metadata dicts
        """
        if embeddings_path:
            self.visual_memory.load_embeddings_from_file(embeddings_path)
        if clips_path:
            self.visual_memory.load_clips_from_file(clips_path)
        if clips_data:
            self.visual_memory.load_clips_from_data(clips_data)
    
    def retrieve_by_time(
        self, start_sec: float, end_sec: float,
    ) -> str:
        """Retrieve captions that overlap with a time range.

        Used for temporal grounding questions where the query contains an
        explicit timestamp (e.g., "What happens from 35:52-36:41?").

        Returns formatted text of matching captions (30sec + 3min).
        """
        results = []
        for gran in ["30sec", "3min"]:
            for cap in self.episodic_memory.captions.get(gran, []):
                try:
                    cap_start = float(cap.start_time)
                    cap_end = float(cap.end_time)
                except (ValueError, TypeError):
                    continue

                # Check overlap
                if cap_end > start_sec and cap_start < end_sec and cap.text.strip():
                    mm_s, ss_s = divmod(int(cap_start), 60)
                    mm_e, ss_e = divmod(int(cap_end), 60)
                    results.append(
                        f"[{mm_s:02d}:{ss_s:02d} - {mm_e:02d}:{ss_e:02d}] ({gran})\n{cap.text}"
                    )

        if results:
            return "\n\n".join(results)
        return ""

    def index(self, until_time) -> None:
        """
        Index all memory types up to the specified timestamp.

        This should be called before any retrieval to ensure memories
        are indexed up to the query time.

        Args:
            until_time: Timestamp (int DHHMMSSFF for EgoLife, or float seconds/inf for Video-MME)
        """
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
            return

        # transform_timestamp expects EgoLife DHHMMSSFF string; skip for float timestamps
        ts_display = (
            transform_timestamp(str(until_time))
            if isinstance(until_time, int) and until_time >= 1_000_000
            else str(until_time)
        )
        logger.info(f"Indexing all memories up to {ts_display}")

        if self.retrieval_backend == "unified_graph":
            # Unified graph backend does not use episodic/semantic/visual indexes;
            # skipping them avoids HippoRAG cache conflicts when running in parallel
            # with an original-backend job.
            self.unified_memory.index(until_time)
        else:
            self.episodic_memory.index(until_time)
            self.semantic_memory.index(until_time)
            self.visual_memory.index(until_time)

        self.indexed_time = until_time
        logger.info(f"Indexing complete for all memory types")
    
    def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
        """
        Parse the reasoning agent's JSON response.
        
        Args:
            response: JSON string from the reasoning LLM
            
        Returns:
            ReasoningOutput with decision and optional memory selection
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            decision = data.get("decision", "answer").lower()
            reason = data.get("reason")
            
            selected_memory = None
            if decision == "search" and "selected_memory" in data:
                mem_data = data["selected_memory"]
                selected_memory = MemorySearchOutput(
                    memory_type=mem_data.get("memory_type", "").lower(),
                    search_query=mem_data.get("search_query", ""),
                )
            
            return ReasoningOutput(
                decision=decision,
                selected_memory=selected_memory,
                reason=reason,
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse reasoning response: {e}")
            # Default to answer if parsing fails
            return ReasoningOutput(decision="answer")
    
    def _format_round_history(self, rounds: List[Dict[str, Any]]):
        """
        Format the round history for the reasoning prompt.

        Args:
            rounds: List of round information dicts

        Returns:
            str (UG mode) or list of multimodal content dicts (chain mode)
        """
        if not rounds:
            return "[]"

        # Check if any round has multimodal content
        has_multimodal = any(isinstance(r.get('retrieved_content'), list) for r in rounds)

        if not has_multimodal:
            # Pure text mode (UG)
            lines = []
            for r in rounds:
                round_str = f"""### Round {r['round_num']}
Decision: {r['decision']}
Memory: {r['memory_type']}
Search Query: {r['search_query']}
Retrieved:
{r['retrieved_content']}"""
                lines.append(round_str)
            return "\n\n".join(lines)
        else:
            # Multimodal mode (chain)
            content_parts = []
            for r in rounds:
                header = f"""### Round {r['round_num']}
Decision: {r['decision']}
Memory: {r['memory_type']}
Search Query: {r['search_query']}
Retrieved:
"""
                content_parts.append({"type": "text", "text": header})
                rc = r.get('retrieved_content', '')
                if isinstance(rc, list):
                    content_parts.extend(rc)
                else:
                    content_parts.append({"type": "text", "text": str(rc)})
                content_parts.append({"type": "text", "text": "\n\n"})
            return content_parts
    
    MAX_QA_FRAMES: int = 64  # global frame cap for QA context (both backends)

    def _render_retrieved_items_for_qa(
        self,
        retrieved_items: List[RetrievedItem]
    ) -> List[Dict[str, Any]]:
        """
        Render retrieved items for the QA prompt.

        Args:
            retrieved_items: List of RetrievedItem objects

        Returns:
            List of message content dicts for the LLM
        """
        messages = []
        total_frames = 0
        for item in retrieved_items:
            if item.memory_type in ("episodic", "semantic"):
                messages.append({"type": "text", "text": item.content})
            elif item.memory_type == "visual":
                if isinstance(item.content, list):
                    for img in item.content:
                        if total_frames >= self.MAX_QA_FRAMES:
                            break
                        if isinstance(img, Image.Image):
                            messages.append({"type": "image", "image": img})
                            total_frames += 1
                        elif isinstance(img, dict) and "image" in img:
                            messages.append({"type": "image", "image": img["image"]})
                            total_frames += 1
        return messages

    def _render_retrieved_items_interleaved(
        self,
        retrieved_items: List[RetrievedItem]
    ) -> List[Dict[str, Any]]:
        """
        Render retrieved items interleaved by time (for chain_mode="facts").
        Episode text, chain facts, and visual frames are sorted by time together,
        so the LLM sees a chronological narrative with images inline.
        """
        def _extract_time_key(text):
            # EgoLife: [DAY1 11:09:43 - ...]
            m = re.match(r'\[DAY(\d+)\s+(\d{1,2}):(\d{2})', text)
            if m:
                return int(m.group(1)) * 100000 + int(m.group(2)) * 3600 + int(m.group(3)) * 60
            # MM-Lifelong / Video-MME: [00:05:30 - ...]
            m = re.match(r'\[(\d{1,2}):(\d{2}):(\d{2})', text)
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            return 0

        # Group items: text items get a time key, visual items sort after their episode
        timed_items = []  # (time_key, item)
        for item in retrieved_items:
            if item.memory_type in ("episodic", "semantic", "topic_chain", "event_chain"):
                tk = _extract_time_key(item.content) if isinstance(item.content, str) else 0
                timed_items.append((tk, item))
            elif item.memory_type == "visual":
                start_ts = getattr(item, '_start_ts', None)
                if start_ts is not None:
                    ts = int(start_ts)
                    if ts > 100000000:
                        # EgoLife DHHMMSSFF
                        d = ts // 100000000
                        r = ts % 100000000
                        secs = (r // 1000000) * 3600 + ((r % 1000000) // 10000) * 60
                        tk = d * 100000 + secs + 0.5
                    else:
                        # MM-Lifelong float seconds
                        tk = float(start_ts) + 0.5
                else:
                    tk = timed_items[-1][0] + 0.5 if timed_items else 0.5
                timed_items.append((tk, item))

        # Sort by time, but keep semantic at the end
        semantic_items = [(tk, item) for tk, item in timed_items if item.memory_type == "semantic"]
        non_semantic = [(tk, item) for tk, item in timed_items if item.memory_type != "semantic"]

        # Check if this is cross-broadcast (items have _broadcast_id)
        broadcast_ids = set()
        for _, item in non_semantic:
            bid = getattr(item, '_broadcast_id', None)
            if bid:
                broadcast_ids.add(bid)

        def _render_item(item, messages, total_frames):
            if item.memory_type == "episodic":
                messages.append({"type": "text", "text": f"[Retrieved episode] {item.content}"})
            elif item.memory_type == "topic_chain":
                label = getattr(item, 'label', '')
                messages.append({"type": "text", "text": f"[Topic: {label}] {item.content}"})
            elif item.memory_type == "event_chain":
                label = getattr(item, 'label', '')
                messages.append({"type": "text", "text": f"[Event: {label}] {item.content}"})
            elif item.memory_type == "visual":
                messages.append({"type": "text", "text": "[Visual frames of above episode]"})
                if isinstance(item.content, list):
                    for img in item.content:
                        if total_frames >= self.MAX_QA_FRAMES:
                            break
                        if isinstance(img, Image.Image):
                            messages.append({"type": "image", "image": img})
                            total_frames += 1
                        elif isinstance(img, dict) and "image" in img:
                            messages.append({"type": "image", "image": img["image"]})
                            total_frames += 1
            return total_frames

        messages = []
        total_frames = 0

        if len(broadcast_ids) > 1:
            # Cross-broadcast: group everything by video, sorted by video ID (= time order)
            # Include semantic triples in each video's section
            all_items = non_semantic + semantic_items
            # Also collect broadcast_ids from semantic items
            all_bids = set(broadcast_ids)
            for _, item in semantic_items:
                bid = getattr(item, '_broadcast_id', None)
                if bid:
                    all_bids.add(bid)

            for bid in sorted(all_bids, key=lambda x: int(x)):
                bid_non_sem = [(tk, item) for tk, item in non_semantic
                               if getattr(item, '_broadcast_id', None) == bid]
                bid_sem = [(tk, item) for tk, item in semantic_items
                           if getattr(item, '_broadcast_id', None) == bid]
                bid_non_sem.sort(key=lambda x: x[0])
                if bid_non_sem or bid_sem:
                    messages.append({"type": "text", "text": f"\n[Video {bid}]"})
                    for _, item in bid_non_sem:
                        total_frames = _render_item(item, messages, total_frames)
                    if bid_sem:
                        for _, item in bid_sem:
                            messages.append({"type": "text", "text": f"[Retrieved semantic] {item.content}"})

            # Items without broadcast_id
            no_bid_ns = [(tk, item) for tk, item in non_semantic
                         if getattr(item, '_broadcast_id', None) is None]
            no_bid_sem = [(tk, item) for tk, item in semantic_items
                          if getattr(item, '_broadcast_id', None) is None]
            if no_bid_ns or no_bid_sem:
                no_bid_ns.sort(key=lambda x: x[0])
                for _, item in no_bid_ns:
                    total_frames = _render_item(item, messages, total_frames)
                for _, item in no_bid_sem:
                    messages.append({"type": "text", "text": f"[Retrieved semantic] {item.content}"})
        else:
            # Single broadcast: sort by time as before, semantic at end
            non_semantic.sort(key=lambda x: x[0])
            for _, item in non_semantic:
                total_frames = _render_item(item, messages, total_frames)
            if semantic_items:
                messages.append({"type": "text", "text": "\n[Semantic Knowledge]"})
                for _, item in semantic_items:
                    messages.append({"type": "text", "text": f"[Retrieved semantic] {item.content}"})

        return messages
    
    def retrieve_from_episodic(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
    ) -> Tuple[str, Set[str]]:
        """
        Retrieve from episodic memory.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            retrieved_set: Set of already retrieved items to avoid duplicates
            
        Returns:
            Tuple of (formatted content string, updated retrieved set)
        """
        top_k = top_k or self.episodic_top_k
        retrieved_set = retrieved_set or set()
        
        # Retrieve from episodic memory
        result = self.episodic_memory.retrieve(
            query=query,
            final_top_k=top_k * 2,  # Get extra to filter duplicates
            as_context=False,
        )
        
        if not result:
            return "", retrieved_set
        
        # Result is List[CaptionEntry] when as_context=False
        if isinstance(result, str):
            return result, retrieved_set
        
        # Filter out already retrieved items
        new_items: List[CaptionEntry] = []
        for entry in result:
            if entry.text not in retrieved_set:
                new_items.append(entry)
                retrieved_set.add(entry.text)
            if len(new_items) >= top_k:
                break
        
        # Format as context string
        content = self.episodic_memory.retrieve_captions_as_str(new_items)
        return content, retrieved_set
    
    def retrieve_from_semantic(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
    ) -> Tuple[str, Set[str]]:
        """
        Retrieve from semantic memory.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            retrieved_set: Set of already retrieved items to avoid duplicates
            
        Returns:
            Tuple of (formatted content string, updated retrieved set)
        """
        top_k = top_k or self.semantic_top_k
        retrieved_set = retrieved_set or set()
        
        # Retrieve from semantic memory
        result = self.semantic_memory.retrieve(
            query=query,
            top_k=top_k * 2,  # Get extra to filter duplicates
            as_context=False,
        )
        
        if not result:
            return "", retrieved_set
        
        # Result is List[SemanticTripleEntry] when as_context=False
        if isinstance(result, str):
            return result, retrieved_set
        
        # Filter out already retrieved items
        new_items: List[SemanticTripleEntry] = []
        for entry in result:
            if entry.id not in retrieved_set:
                new_items.append(entry)
                retrieved_set.add(entry.id)
            if len(new_items) >= top_k:
                break
        
        # Format as context string
        content = self.semantic_memory.retrieve_triples_as_str(new_items)
        return content, retrieved_set
    
    def retrieve_from_visual(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, List[Any]], Set[str]]:
        """
        Retrieve from visual memory.
        
        Args:
            query: Search query (text or time range)
            top_k: Number of clips to retrieve
            retrieved_set: Set of already retrieved items to avoid duplicates
            
        Returns:
            Tuple of (content dict with images, updated retrieved set)
        """
        top_k = top_k or self.visual_top_k
        retrieved_set = retrieved_set or set()
        
        # Retrieve from visual memory
        result = self.visual_memory.retrieve(
            query=query,
            top_k=top_k,
            as_context=True,
        )
        
        if not result:
            return {}, retrieved_set
        
        # Result should be Dict[str, List[Image]] when as_context=True
        if isinstance(result, dict):
            # Track retrieved clips by their display keys
            for key in result.keys():
                retrieved_set.add(key)
            return result, retrieved_set
        
        # Fallback for unexpected return type
        return {}, retrieved_set
    
    def answer(
        self,
        query: str,
        choices: Optional[Dict[str, str]] = None,
        until_time=None,
    ) -> "QAResult":
        """Route to the appropriate backend."""
        if self.retrieval_backend == "unified_graph":
            return self._answer_unified(query, choices, until_time)
        return self._answer_original(query, choices, until_time)

    def _answer_original(
        self,
        query: str,
        choices: Optional[Dict[str, str]] = None,
        until_time=None,
    ) -> QAResult:
        """
        Answer a question using iterative memory retrieval.
        
        This is the main entry point for the WorldMM pipeline:
        1. Index memories up to the query time
        2. Iteratively retrieve from memories based on reasoning
        3. Answer the question using accumulated context
        
        Args:
            query: The question to answer
            choices: Optional dict of answer choices (e.g., {"A": "...", "B": "..."})
            until_time: Timestamp to index up to (uses current indexed time if None)
            
        Returns:
            QAResult with the answer and retrieval history
        """
        # Index if needed
        if until_time and until_time > self.indexed_time:
            self.index(until_time)
        
        # Format query with choices if provided
        # Format current time for the LLM
        current_time_str = ""
        if until_time and isinstance(until_time, (int, float)) and until_time > 1000 and until_time != float("inf"):
            s = str(int(until_time))
            if len(s) >= 7:
                day = int(s[0])
                hh = int(s[1:3])
                mm = int(s[3:5])
                ss = int(s[5:7])
                current_time_str = f"DAY{day} {hh:02d}:{mm:02d}:{ss:02d}"

        full_query = f"Query: {query}"
        if current_time_str:
            full_query += f"\nCurrent time: {current_time_str}"
        if choices:
            choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
            full_query += f"\nChoices: {choices_str}"
        
        # Initialize retrieval state
        retrieved_set: Set[str] = set()
        retrieved_items: List[RetrievedItem] = []
        round_history: List[Dict[str, Any]] = []
        
        # Get reasoning prompt template (independent backend uses upstream template)
        reasoning_prompt = self.prompt_template_manager.render("memory_reasoning")
        
        round_num = 0
        err_count = 0
        
        while round_num < self.max_rounds and err_count < self.max_errors:
            round_num += 1
            logger.info(f"Reasoning round {round_num}")
            
            # Build the user message for reasoning
            history_str = self._format_round_history(round_history)
            
            user_content = f"""{full_query}

Round History:
{history_str}

Task:
Step 1: Decide whether to "search" or "answer".
Step 2 (only if search): Pick one memory type (episodic/semantic/visual) and form a search query."""
            
            # Get reasoning decision
            reasoning_messages = copy.deepcopy(reasoning_prompt)
            reasoning_messages.append({
                "role": "user",
                "content": user_content,
            })
            
            try:
                response = self.respond_llm_model.generate(reasoning_messages)
                reasoning_output = self._parse_reasoning_response(response)
            except Exception as e:
                logger.error(f"Reasoning failed: {e}")
                err_count += 1
                continue
            
            logger.info(f"Decision: {reasoning_output.decision}")
            
            # Handle decision
            if reasoning_output.decision == "answer":
                break
            
            if reasoning_output.decision == "search":
                if not reasoning_output.selected_memory:
                    logger.warning("Search decision but no memory selected")
                    err_count += 1
                    continue
                
                memory_type = reasoning_output.selected_memory.memory_type
                search_query = reasoning_output.selected_memory.search_query

                logger.info(f"Searching {memory_type}: {search_query}")

                # Check for temporal reference in query — bypass semantic
                # retrieval and directly return captions at the specified time
                time_ref = parse_time_reference(search_query)
                content = ""
                images = None

                if time_ref:
                    t_start, t_end = time_ref
                    logger.info(f"Temporal retrieval: {t_start:.0f}s - {t_end:.0f}s")
                    content = self.retrieve_by_time(t_start, t_end)
                    memory_type = "episodic"  # treat as episodic for round history

                elif memory_type == "episodic":
                    content, retrieved_set = self.retrieve_from_episodic(
                        search_query,
                        retrieved_set=retrieved_set
                    )
                    
                elif memory_type == "semantic":
                    content, retrieved_set = self.retrieve_from_semantic(
                        search_query,
                        retrieved_set=retrieved_set
                    )
                    
                elif memory_type == "visual":
                    images, retrieved_set = self.retrieve_from_visual(
                        search_query,
                        retrieved_set=retrieved_set
                    )
                    # Format visual content for round history
                    if images:
                        content = f"[{len(sum(images.values(), []))} images from {len(images)} clips]"
                        # Flatten images for retrieved items
                        all_images = []
                        for clip_images in images.values():
                            all_images.extend(clip_images)
                        retrieved_items.append(RetrievedItem(
                            memory_type="visual",
                            content=all_images,
                            query=search_query,
                            round_num=round_num,
                        ))
                else:
                    logger.warning(f"Unknown memory type: {memory_type}")
                    err_count += 1
                    continue
                
                # Log retrieved content for analysis
                if content:
                    # Truncate for logging: show first 500 chars
                    log_content = content[:500].replace('\n', '\n  ')
                    logger.info(f"Retrieved ({memory_type}):\n  {log_content}")
                else:
                    logger.info(f"Retrieved ({memory_type}): [empty]")

                # Add to retrieved items (text memories)
                if memory_type in ("episodic", "semantic") and content:
                    retrieved_items.append(RetrievedItem(
                        memory_type=memory_type,
                        content=content,
                        query=search_query,
                        round_num=round_num,
                    ))

                # Add to round history
                round_history.append({
                    "round_num": round_num,
                    "decision": "search",
                    "memory_type": memory_type,
                    "search_query": search_query,
                    "retrieved_content": content if content else "[No results]",
                })
        
        # Generate final answer
        logger.info("Generating answer from accumulated context")
        
        try:
            qa_prompt = self.prompt_template_manager.render(self.qa_template_name)
        except Exception as e:
            logger.error(f"Failed to load {self.qa_template_name} template: {e}")
            raise

        # Build QA message with all retrieved context
        qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
        qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
        
        if choices:
            qa_content.append({
                "type": "text", 
                "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
            })
        
        qa_messages = copy.deepcopy(qa_prompt)
        qa_messages.append({
            "role": "user",
            "content": qa_content,
        })
        
        try:
            answer = self.respond_llm_model.generate(qa_messages)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = None

        # Fallback: retry with text-only context (drop images to reduce input)
        if not answer:
            logger.warning("First QA attempt returned None; retrying with text-only context")
            text_only_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
            for item in retrieved_items:
                if item.memory_type in ("episodic", "semantic") and isinstance(item.content, str):
                    text_only_content.append({"type": "text", "text": item.content})
            if choices:
                text_only_content.append({
                    "type": "text",
                    "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
                })
            qa_messages_retry = copy.deepcopy(qa_prompt)
            qa_messages_retry.append({"role": "user", "content": text_only_content})
            try:
                answer = self.respond_llm_model.generate(qa_messages_retry)
            except Exception as e:
                logger.error(f"Text-only QA retry also failed: {e}")
                answer = None

        if not answer:
            answer = "Unable to generate answer"

        return QAResult(
            question=query,
            answer=answer,
            retrieved_items=retrieved_items,
            round_history=round_history,
            num_rounds=round_num,
        )
    
    # ------------------------------------------------------------------
    # Mix chain facts into retrieval results by time
    # ------------------------------------------------------------------

    def _mix_facts_by_time(self, text_parts, chain_facts):
        """Mix chain facts into episode text parts, sorted by time.

        Each text_part is an episode description starting with [DAYx HH:MM...].
        Each chain_fact has {"time": "DAYx HH:MM", "fact": "...", "label": "..."}.
        Merge them all, sort by time.
        """
        def _extract_time_key(text):
            """Extract time key from text containing [DAYx HH:MM...] or [HH:MM:SS...]."""
            # EgoLife: [DAY1 11:09:43 - ...] or [Retrieved episode] [DAY1 11:09:43 - ...]
            m = re.search(r'\[DAY(\d+)\s+(\d{1,2}):(\d{2})', text)
            if m:
                return int(m.group(1)) * 100000 + int(m.group(2)) * 3600 + int(m.group(3)) * 60
            # MM-Lifelong: [00:05:30 - ...] or [Retrieved episode] [00:05:30 - ...]
            m = re.search(r'\[(\d{1,2}):(\d{2}):(\d{2})', text)
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            return 0

        # Convert chain facts to text lines with labels
        fact_lines = []
        for f in chain_facts:
            ct = f.get("chain_type", "topic")
            label = f.get("label", "")
            if ct == "event":
                end_time = f.get("end_time", "")
                tag = f"[Event: {label}]"
                line = f"{tag} [{f['time']} - {end_time}] {f['fact']}"
            else:
                tag = f"[Topic: {label}]"
                line = f"{tag} [{f['time']}] {f['fact']}"
            fact_lines.append(line)

        # Merge all lines and sort by time (semantic triples at the end)
        def _sort_key(text):
            if '[Retrieved semantic]' in text:
                return (1, 0)  # semantic at the end
            return (0, _extract_time_key(text))

        all_lines = text_parts + fact_lines
        all_lines.sort(key=_sort_key)

        return all_lines

    # Unified-graph backend
    # ------------------------------------------------------------------

    def _parse_unified_reasoning(self, response: str) -> tuple:
        """
        Parse the unified reasoning agent's JSON response.
        Returns (decision: str, search_query: str).
        decision is 'search' or 'answer'.
        """
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            data = json.loads(json_match.group() if json_match else response)
            decision     = data.get("decision", "answer").lower()
            search_query = data.get("search_query", "")
            return decision, search_query
        except Exception as e:
            logger.warning(f"Failed to parse unified reasoning response: {e}")
            return "answer", ""

    def _answer_unified(
        self,
        query: str,
        choices: Optional[Dict[str, str]] = None,
        until_time=None,
    ) -> QAResult:
        """
        Answer a question using the unified multimodal graph backend.
        The agent only decides 'search' or 'answer' without selecting a memory type.
        A single cross-modal PPR call covers all modalities.
        """
        if self.unified_memory is None:
            raise RuntimeError("unified_memory is not initialised.")

        # Index if needed
        if until_time and until_time > self.indexed_time:
            self.index(until_time)

        # Format current time for the LLM
        current_time_str = ""
        if until_time and isinstance(until_time, (int, float)) and until_time > 1000 and until_time != float("inf"):
            s = str(int(until_time))
            if len(s) >= 7:
                day = int(s[0])
                hh = int(s[1:3])
                mm = int(s[3:5])
                ss = int(s[5:7])
                current_time_str = f"DAY{day} {hh:02d}:{mm:02d}:{ss:02d}"

        full_query = f"Query: {query}"
        if current_time_str:
            full_query += f"\nCurrent time: {current_time_str}"
        if choices:
            choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
            full_query += f"\nChoices: {choices_str}"

        retrieved_items: List[RetrievedItem] = []
        retrieved_node_ids: Set[str] = set()
        round_history: List[Dict[str, Any]] = []
        total_visual_frames: int = 0          # cross-round frame budget tracker
        MAX_TOTAL_FRAMES: int = self.MAX_QA_FRAMES  # respect no-images setting

        template_name = f"memory_reasoning_unified_{self.reasoning_version}"
        reasoning_prompt = self.prompt_template_manager.render(template_name)

        round_num  = 0
        err_count  = 0
        injected_chain_facts = set()  # cross-round dedup for chain facts

        while round_num < self.max_rounds and err_count < self.max_errors:
            round_num += 1
            logger.info(f"[unified] Reasoning round {round_num}")

            history_result = self._format_round_history(round_history)

            if isinstance(history_result, str):
                # UG mode: pure text
                user_content = (
                    f"{full_query}\n\nRound History:\n{history_result}\n\n"
                    "Task:\nDecide whether to 'search' (retrieve more evidence) "
                    "or 'answer' (sufficient evidence accumulated)."
                )
            else:
                # Chain mode: multimodal content list
                user_content = [
                    {"type": "text", "text": f"{full_query}\n\nRound History:\n"},
                    *history_result,
                    {"type": "text", "text": "\nTask:\nDecide whether to 'search' (retrieve more evidence) "
                     "or 'answer' (sufficient evidence accumulated)."},
                ]

            reasoning_messages = copy.deepcopy(reasoning_prompt)
            reasoning_messages.append({"role": "user", "content": user_content})

            if self.debug:
                # Log reasoning input (full, no truncation)
                if isinstance(user_content, str):
                    logger.info(f"[DEBUG] Round {round_num} reasoning input:\n{user_content}")
                else:
                    text_parts = [p.get('text','') for p in user_content if p.get('type') == 'text']
                    img_count = sum(1 for p in user_content if isinstance(p, dict) and p.get('type') in ('image', 'image_url'))
                    logger.info(f"[DEBUG] Round {round_num} reasoning input ({img_count} images):\n{''.join(text_parts)}")

            try:
                response = self.respond_llm_model.generate(reasoning_messages)
                decision, search_query = self._parse_unified_reasoning(response)
                if self.debug:
                    logger.info(f"[DEBUG] Round {round_num} reasoning response: {response[:200]}")
            except Exception as e:
                logger.error(f"[unified] Reasoning failed: {e}")
                err_count += 1
                continue

            logger.info(f"[unified] Decision: {decision}")

            if decision == "answer":
                break

            if decision == "search":
                if not search_query:
                    logger.warning("[unified] Empty search_query, skipping.")
                    err_count += 1
                    continue

                logger.info(f"[unified] Searching: {search_query}")

                # Check for temporal reference — directly return captions
                time_ref = parse_time_reference(search_query)
                if time_ref:
                    t_start, t_end = time_ref
                    logger.info(f"[unified] Temporal retrieval: {t_start:.0f}s - {t_end:.0f}s")
                    content = self.retrieve_by_time(t_start, t_end)
                    if content:
                        retrieved_items.append(RetrievedItem(
                            memory_type="episodic",
                            content=content,
                            query=search_query,
                            round_num=round_num,
                        ))
                    round_history.append({
                        "round_num":        round_num,
                        "decision":         "search",
                        "memory_type":      "temporal",
                        "search_query":     search_query,
                        "retrieved_content": content if content else "[No results]",
                    })
                    continue

                try:
                    effective_top_k = self.retrieval_top_k if self.retrieval_top_k > 0 \
                        else max(self.episodic_top_k + self.semantic_top_k + self.visual_top_k, 10)
                    raw_results = self.unified_memory.retrieve(
                        query    = search_query,
                        top_k    = effective_top_k,
                        target_node_types = [NodeType.EPISODE, NodeType.SEMANTIC, NodeType.VISUAL_CLIP],
                        seed_top_k = self.seed_top_k,
                        damping  = self.damping,
                        choices  = choices,
                        bm25_weight = self.bm25_weight,
                    )
                except Exception as e:
                    logger.error(f"[unified] Retrieval failed: {e}")
                    err_count += 1
                    continue

                # Deduplicate by node id
                new_results = [
                    (n, s) for n, s in raw_results
                    if n.id not in retrieved_node_ids
                ]
                for n, _ in new_results:
                    retrieved_node_ids.add(n.id)

                if not new_results:
                    logger.info("[unified] Retrieved: [No new results]")
                    round_history.append({
                        "round_num":        round_num,
                        "decision":         "search",
                        "memory_type":      "unified",
                        "search_query":     search_query,
                        "retrieved_content": "[No new results]",
                    })
                    continue

                # Format results → RetrievedItem (reuses _render_retrieved_items_for_qa)
                new_items = self.unified_memory.format_results(
                    new_results, self.visual_memory,
                    max_frames_per_clip=self.frames_per_clip,
                    max_total_frames=MAX_TOTAL_FRAMES,
                    _current_total_frames=total_visual_frames,
                )
                # Stamp round number and query
                for item in new_items:
                    item.query     = search_query
                    item.round_num = round_num
                retrieved_items.extend(new_items)

                # Update cross-round frame counter
                round_img_count = sum(
                    len(item.content) if isinstance(item.content, list) else 0
                    for item in new_items
                )
                total_visual_frames += round_img_count

                # Build round history summary (images shown as placeholder)
                # Sort by time: episodes/visual first (by time), semantic last
                def _time_sort_key(item):
                    if item.memory_type == "semantic":
                        return (1, 0)  # semantic at the end
                    if isinstance(item.content, str):
                        # EgoLife: [DAY1 11:09:43 - ...]
                        m = re.match(r'\[DAY(\d+)\s+(\d{1,2}):(\d{2})', item.content)
                        if m:
                            return (0, int(m.group(1)) * 100000 + int(m.group(2)) * 3600 + int(m.group(3)) * 60)
                        # MM-Lifelong / Video-MME: [00:05:30 - ...]
                        m = re.match(r'\[(\d{1,2}):(\d{2}):(\d{2})', item.content)
                        if m:
                            return (0, int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
                    return (0, 0)

                # Build timed_items: text items by time, visual items after episode
                # Build a map: episodic time_key → exists, for visual attachment
                ep_time_keys = {}
                for item in new_items:
                    if isinstance(item.content, str) and item.memory_type == "episodic":
                        tk = _time_sort_key(item)
                        # Extract time range string for matching
                        ep_time_keys[tk] = True

                timed_items = []
                for item in new_items:
                    if isinstance(item.content, str):
                        tk = _time_sort_key(item)
                        timed_items.append((tk, item))
                    elif item.memory_type == "visual":
                        # Use start_ts attached by format_results to compute time_key
                        start_ts = getattr(item, '_start_ts', None)
                        if start_ts is not None:
                            # Convert start_ts to time_key matching _time_sort_key format
                            ts = int(start_ts)
                            if ts > 100000000:
                                # EgoLife DHHMMSSFF format
                                d = ts // 100000000
                                r = ts % 100000000
                                secs = (r // 1000000) * 3600 + ((r % 1000000) // 10000) * 60
                                timed_items.append(((0, d * 100000 + secs + 0.5), item))
                            else:
                                # MM-Lifelong float seconds
                                timed_items.append(((0, float(start_ts) + 0.5), item))
                        else:
                            prev = timed_items[-1][0] if timed_items else (0, 0)
                            timed_items.append(((prev[0], prev[1] + 0.5), item))

                # Sort: non-semantic by time, semantic at end
                timed_items.sort(key=lambda x: x[0])

                def _format_item_text(item):
                    """Format a single item as text for round history."""
                    if item.memory_type == "visual":
                        n_frames = len(item.content) if isinstance(item.content, list) else 0
                        if n_frames:
                            start_ts = getattr(item, '_start_ts', None)
                            if start_ts is not None:
                                ts_int = int(start_ts)
                                if ts_int > 100000000:
                                    r = ts_int % 100000000
                                    h, m, s = r // 1000000, (r % 1000000) // 10000, (r % 10000) // 100
                                    ts_str = f"DAY{ts_int // 100000000} {h:02d}:{m:02d}:{s:02d}"
                                else:
                                    total = int(float(start_ts))
                                    h, m, s = total // 3600, (total % 3600) // 60, total % 60
                                    ts_str = f"{h:02d}:{m:02d}:{s:02d}"
                                return f"[{ts_str}] [{n_frames} visual frames of above episode]"
                            return f"[{n_frames} visual frames of above episode]"
                        return None
                    elif item.memory_type == "episodic":
                        return f"[Retrieved episode] {item.content}"
                    elif item.memory_type == "semantic":
                        return f"[Retrieved semantic] {item.content}"
                    else:
                        return item.content

                # Check if cross-broadcast
                _bids = set(getattr(item, '_broadcast_id', None) for _, item in timed_items)
                _bids.discard(None)

                text_parts = []
                if len(_bids) > 1:
                    # Cross-broadcast: group by video ID
                    for bid in sorted(_bids, key=lambda x: int(x)):
                        bid_items = [(tk, item) for tk, item in timed_items
                                     if getattr(item, '_broadcast_id', None) == bid]
                        bid_items.sort(key=lambda x: x[0])
                        if bid_items:
                            text_parts.append(f"[Video {bid}]")
                            for _, item in bid_items:
                                t = _format_item_text(item)
                                if t:
                                    text_parts.append(t)
                else:
                    # Single broadcast
                    for _, item in timed_items:
                        t = _format_item_text(item)
                        if t:
                            text_parts.append(t)
                img_count = round_img_count

                # chain_mode="facts": inject chain facts mixed with episodes by time
                if self._chain_mode == "facts" and self.unified_memory:
                    chain_facts = self.unified_memory.get_chain_facts_for_results(
                        new_results, query_time=until_time,
                        query_text=query,
                        embedding_model=self.embedding_model,
                        min_hits=getattr(self, 'chain_min_hits', 2),
                        max_topic_chains=getattr(self, 'chain_max_topics', 3),
                        max_event_chains=getattr(self, 'chain_max_events', 2),
                        topic_sim_threshold=getattr(self, 'chain_topic_sim', 0.5),
                        storyline_sim_threshold=getattr(self, 'chain_storyline_sim', 0.5),
                        storyline_min_hits=getattr(self, 'chain_storyline_min_hits', 1),
                        storyline_granularities=getattr(self, 'chain_storyline_granularities', ("30sec", "3min")),
                    )
                    # Cross-round dedup
                    if chain_facts:
                        chain_facts = [
                            cf for cf in chain_facts
                            if (cf.get("label",""), cf.get("time",""), cf.get("fact","")) not in injected_chain_facts
                        ]
                    if chain_facts:
                        for cf in chain_facts:
                            injected_chain_facts.add((cf.get("label",""), cf.get("time",""), cf.get("fact","")))
                    if chain_facts:
                        # Mix facts into text_parts by time
                        if len(_bids) > 1:
                            # Cross-broadcast: rebuild per-video sections with facts mixed in
                            new_text_parts: List[str] = []
                            current_bid: Optional[str] = None
                            current_section: List[str] = []
                            for line in text_parts:
                                if line.startswith("[Video "):
                                    # Flush previous section with its facts
                                    if current_bid is not None and current_section:
                                        bid_facts = [cf for cf in chain_facts
                                                     if cf.get("broadcast_id") == current_bid]
                                        if bid_facts:
                                            current_section = self._mix_facts_by_time(
                                                current_section, bid_facts)
                                        new_text_parts.extend(current_section)
                                    new_text_parts.append(line)
                                    m = re.match(r'\[Video (\d+)\]', line)
                                    current_bid = m.group(1) if m else None
                                    current_section = []
                                else:
                                    current_section.append(line)
                            # Flush last section
                            if current_bid is not None and current_section:
                                bid_facts = [cf for cf in chain_facts
                                             if cf.get("broadcast_id") == current_bid]
                                if bid_facts:
                                    current_section = self._mix_facts_by_time(
                                        current_section, bid_facts)
                                new_text_parts.extend(current_section)
                            text_parts = new_text_parts
                        else:
                            text_parts = self._mix_facts_by_time(text_parts, chain_facts)
                        # Also add chain facts as RetrievedItems for final QA
                        for cf in chain_facts:
                            ct = cf.get("chain_type", "topic")
                            if ct == "event":
                                content = f"[{cf['time']} - {cf.get('end_time', '')}] {cf['fact']}"
                            else:
                                content = f"[{cf['time']}] {cf['fact']}"
                            chain_item = RetrievedItem(
                                memory_type=f"{ct}_chain",
                                content=content,
                                query=search_query,
                                round_num=round_num,
                                label=cf.get("label", ""),
                            )
                            chain_item._broadcast_id = cf.get("broadcast_id")
                            retrieved_items.append(chain_item)

                summary = "\n".join(text_parts)
                if img_count:
                    summary += f"\n[{img_count} image frames retrieved]"

                logger.info(f"[unified] Retrieved:\n{summary or '[No text results]'}")

                # Build round history content (text-only; chain facts already in summary)
                retrieved_content = summary or "[No text results]"

                round_history.append({
                    "round_num":        round_num,
                    "decision":         "search",
                    "memory_type":      "unified",
                    "search_query":     search_query,
                    "retrieved_content": retrieved_content,
                })
            else:
                logger.warning(f"[unified] Unknown decision: {decision}")
                err_count += 1

        # Generate final answer
        logger.info("[unified] Generating answer from accumulated context")
        qa_template = self.qa_template_name
        if self._chain_mode == "facts":
            chain_qa_template = qa_template + "_chain"
            try:
                qa_prompt = self.prompt_template_manager.render(chain_qa_template)
                logger.info(f"[unified] Using chain QA template: {chain_qa_template}")
            except Exception:
                logger.info(f"[unified] Chain QA template not found, falling back to {qa_template}")
                qa_prompt = self.prompt_template_manager.render(qa_template)
        else:
            qa_prompt = self.prompt_template_manager.render(qa_template)

        qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
        qa_content.extend(self._render_retrieved_items_interleaved(retrieved_items))
        if choices:
            qa_content.append({
                "type": "text",
                "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
            })

        qa_messages = copy.deepcopy(qa_prompt)
        qa_messages.append({"role": "user", "content": qa_content})

        if self.debug:
            # Log final QA context
            text_parts = [p.get('text','') for p in qa_content if isinstance(p, dict) and p.get('type') == 'text']
            img_count = sum(1 for p in qa_content if isinstance(p, dict) and p.get('type') in ('image', 'image_url'))
            full_text = '\n'.join(text_parts)
            logger.info(f"[DEBUG] Final QA context ({len(text_parts)} text parts, {img_count} images):\n{full_text}")

        try:
            answer = self.respond_llm_model.generate(qa_messages)
            if self.debug:
                logger.info(f"[DEBUG] LLM raw answer: {answer}")
        except Exception as e:
            logger.error(f"[unified] Answer generation failed: {e}")
            answer = None

        # Fallback: retry with text-only context (drop images to shorten input)
        if not answer:
            logger.warning("[unified] First QA attempt returned None; retrying with text-only context")
            text_only_items = [it for it in retrieved_items if isinstance(it.content, str)]
            qa_content_short = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
            qa_content_short.extend(self._render_retrieved_items_for_qa(text_only_items))
            if choices:
                qa_content_short.append({
                    "type": "text",
                    "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
                })
            qa_messages_short = copy.deepcopy(qa_prompt)
            qa_messages_short.append({"role": "user", "content": qa_content_short})
            try:
                answer = self.respond_llm_model.generate(qa_messages_short)
            except Exception as e:
                logger.error(f"[unified] Text-only QA retry also failed: {e}")
                answer = None

        if not answer:
            answer = "Unable to generate answer"

        return QAResult(
            question       = query,
            answer         = answer,
            retrieved_items = retrieved_items,
            round_history   = round_history,
            num_rounds      = round_num,
        )

    def reset(self) -> None:
        """Clear all loaded data and indexes for processing a new video/subject."""
        self.reset_index()
        self.cleanup()
        # Clear episodic data
        self.episodic_memory.captions = {g: [] for g in self.episodic_memory.granularities}
        self.episodic_memory.caption_id_to_entry.clear()
        self.episodic_memory.text_to_entry.clear()
        # Clear semantic data
        self.semantic_memory.triple_id_to_entry.clear()
        self.semantic_memory.timestamp_to_triples.clear()
        self.semantic_memory.available_timestamps.clear()
        # Clear visual data
        self.visual_memory.clips.clear()
        self.visual_memory.clip_id_to_entry.clear()
        self.visual_memory.video_path_to_embedding.clear()
        # Clear unified graph
        if self.unified_memory:
            self.unified_memory._full_graph = None
            self.unified_memory._active_graph = None
            self.unified_memory._indexed_time = 0
        logger.info("All memory data and indices reset")

    def reset_index(self) -> None:
        """Reset all indexed state across all memory types."""
        self.episodic_memory.reset_index()
        self.semantic_memory.reset_index()
        self.visual_memory.reset_index()
        self.indexed_time = 0
        logger.info("All memory indices reset")
    
    def cleanup(self) -> None:
        """Release GPU memory, HippoRAG data, and other resources."""
        self.semantic_memory.cleanup()
        self.visual_memory.cleanup()
        if self.unified_memory is not None:
            self.unified_memory.cleanup()
        # Release HippoRAG instances (embedding stores, faiss indexes, graphs)
        if hasattr(self, 'episodic_memory') and hasattr(self.episodic_memory, 'hipporag'):
            self.episodic_memory.hipporag.clear()
        # Force garbage collection to free memory before next video
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Memory cleanup complete")
    
    def get_indexed_time(self) -> str:
        """Get the current indexed time as human-readable string."""
        return transform_timestamp(str(self.indexed_time))
    
    def set_retrieval_top_k(
        self,
        episodic: Optional[int] = None,
        semantic: Optional[int] = None,
        visual: Optional[int] = None,
    ) -> None:
        """
        Configure the number of items to retrieve from each memory type.
        
        Args:
            episodic: Top-k for episodic memory
            semantic: Top-k for semantic memory
            visual: Top-k for visual memory
        """
        if episodic is not None:
            self.episodic_top_k = episodic
        if semantic is not None:
            self.semantic_top_k = semantic
        if visual is not None:
            self.visual_top_k = visual
