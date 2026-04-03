from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .dataset import MemoryBenchmarkDataset, history_from_round_ids
from .retrieval import select_round_ids_for_qa


# ---------------------------------------------------------------------------
# Token estimation helpers for context-window truncation
# ---------------------------------------------------------------------------

# Rough chars-per-token ratio (conservative for English + mixed content)
_CHARS_PER_TOKEN = 4

# Estimated token cost per image, following Mem-Gallery (predefined token cost)
_IMAGE_TOKEN_COST = 765


def _estimate_turn_tokens(turn: Dict[str, Any]) -> int:
    """Estimate token count for a single history turn (text + images)."""
    text = str(turn.get("text", ""))
    text_tokens = max(1, len(text) // _CHARS_PER_TOKEN)
    image_tokens = len(turn.get("images", []) or []) * _IMAGE_TOKEN_COST
    return text_tokens + image_tokens


def _truncate_history(history: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    """Truncate history from the front (keep most recent turns) to fit within max_tokens.

    This follows the standard Full Memory approach in Mem-Gallery:
    include all memory and truncate according to the context token limit.
    Truncation removes the oldest turns first (FIFO-style).
    """
    if max_tokens <= 0:
        return history

    # Walk backwards (most recent first) and accumulate token budget
    cumulative = 0
    cutoff_idx = len(history)
    for i in range(len(history) - 1, -1, -1):
        cumulative += _estimate_turn_tokens(history[i])
        if cumulative > max_tokens:
            cutoff_idx = i + 1
            break
    else:
        # Everything fits
        return history

    return history[cutoff_idx:]


class HistoryMethod(ABC):
    name = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.runtime_info: Dict[str, Any] = {}

    @abstractmethod
    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class FullContextMethod(HistoryMethod):
    name = "full_context"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build history from ALL sessions, truncated to fit context window.

        Following Mem-Gallery (Bei et al., 2025): include all memory information
        as context and truncate according to the context token limit.  Oldest
        turns are removed first (keep most recent).

        Config keys:
          context_token_limit: int  (default 128000, matching GPT-4.1 family)
        """
        history: List[Dict[str, Any]] = []
        for sid in dataset.session_order():
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds))

        max_tokens = int(self.config.get("context_token_limit", 128_000))
        # Reserve tokens for system prompt (~500) + question (~200) + answer generation
        reserved = int(self.config.get("reserved_tokens", 1_000))
        return _truncate_history(history, max_tokens - reserved)


class TargetSessionContextMethod(HistoryMethod):
    name = "target_session_context"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds))
        return history


class _RetrievalHistoryMethod(HistoryMethod):
    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_round_ids = select_round_ids_for_qa(dataset, qa, self.config, runtime_info=self.runtime_info)
        if not selected_round_ids:
            return []

        history: List[Dict[str, Any]] = []
        allowed_round_ids = set(selected_round_ids)
        for sid in dataset.session_order():
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, allowed_round_ids))
        return history


class LexicalRAGMethod(_RetrievalHistoryMethod):
    name = "lexical_rag"


class SemanticRAGMethod(_RetrievalHistoryMethod):
    name = "semantic_rag"


class SemanticRAGMultimodalMethod(_RetrievalHistoryMethod):
    name = "semantic_rag_multimodal"


class M2AAgentMethod(HistoryMethod):
    name = "m2a"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._system: Optional[Any] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._system is not None and self._dataset_key == dataset_id:
            return

        from .m2a import M2ASystem

        self._system = M2ASystem(self.config)
        sessions = dataset.session_order()
        print(f"[M2A] Building memory from {len(sessions)} session(s)...")
        self._system.process_all_sessions(dataset)
        self._dataset_key = dataset_id
        print(f"[M2A] Memory ready: {self._system.num_memories} semantic memories stored.")

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._system is not None
        return self._system.answer_question(question)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


def get_method(method_name: str, config: Optional[Dict[str, Any]] = None) -> HistoryMethod:
    registry = {
        FullContextMethod.name: FullContextMethod,
        TargetSessionContextMethod.name: TargetSessionContextMethod,
        LexicalRAGMethod.name: LexicalRAGMethod,
        SemanticRAGMethod.name: SemanticRAGMethod,
        SemanticRAGMultimodalMethod.name: SemanticRAGMultimodalMethod,
        M2AAgentMethod.name: M2AAgentMethod,
    }
    cls = registry.get(method_name)
    if cls is None:
        raise ValueError(f"Unsupported method: {method_name!r}")
    return cls(config=config)
