from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .dataset import MemoryBenchmarkDataset, history_from_round_ids
from .retrieval import select_round_ids_for_qa


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
        history: List[Dict[str, Any]] = []
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds))
        return history


class _ClueRoundFallbackMethod(HistoryMethod):
    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        target_sessions = set(qa.get("session_id", []))
        clue_rounds = set(qa.get("clue", []) or [])
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, clue_rounds))
        return history


class HybridRAGMethod(HistoryMethod):
    name = "hybrid_rag"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_round_ids = select_round_ids_for_qa(dataset, qa, self.config)
        if not selected_round_ids:
            return _ClueRoundFallbackMethod().build_history(dataset, qa)

        history: List[Dict[str, Any]] = []
        allowed_round_ids = set(selected_round_ids)
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, allowed_round_ids))
        return history


class M2AAgentMethod(HistoryMethod):
    """
    Faithful M2A (Multimodal Memory Agent) implementation.

    Two-phase protocol (mirrors official eval_wrapper.py):
      1. Chat phase  — process all sessions sequentially to build semantic memory
                       (lazy-initialized on first call, once per dataset instance)
      2. Question phase — query memory to answer each QA item

    This method exposes answer(dataset, qa, question) → str so that the runner
    bypasses the build_history + router.answer pipeline and calls the M2A system
    end-to-end (agentic inference).
    """

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
        """End-to-end agentic inference. Called by runner instead of build_history."""
        self._ensure_initialized(dataset)
        assert self._system is not None
        return self._system.answer_question(question)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Not used when runner detects answer() — included for interface completeness.
        return []


def get_method(method_name: str, config: Optional[Dict[str, Any]] = None) -> HistoryMethod:
    registry = {
        FullContextMethod.name: FullContextMethod,
        HybridRAGMethod.name: HybridRAGMethod,
        M2AAgentMethod.name: M2AAgentMethod,
    }
    cls = registry.get(method_name)
    if cls is None:
        raise ValueError(f"Unsupported method: {method_name!r}")
    return cls(config=config)
