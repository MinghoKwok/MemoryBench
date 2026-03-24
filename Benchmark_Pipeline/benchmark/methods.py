from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .dataset import MemoryBenchmarkDataset, history_from_round_ids
from .retrieval import (
    get_last_m2a_capabilities,
    select_round_ids_for_qa,
    select_round_ids_for_qa_m2a_full,
    select_round_ids_for_qa_m2a_lite,
)


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


class M2ALiteMethod(HistoryMethod):
    name = "m2a_lite"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_round_ids = select_round_ids_for_qa_m2a_lite(dataset, qa, self.config)
        if not selected_round_ids:
            return HybridRAGMethod(config=self.config).build_history(dataset, qa)

        history: List[Dict[str, Any]] = []
        allowed_round_ids = set(selected_round_ids)
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, allowed_round_ids))
        return history


class M2AFullMethod(HistoryMethod):
    name = "m2a_full"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_round_ids = select_round_ids_for_qa_m2a_full(dataset, qa, self.config)
        self.runtime_info = get_last_m2a_capabilities()
        if not selected_round_ids:
            return M2ALiteMethod(config=self.config).build_history(dataset, qa)

        history: List[Dict[str, Any]] = []
        allowed_round_ids = set(selected_round_ids)
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, allowed_round_ids))
        return history


class M2AFullTunedMethod(HistoryMethod):
    """M2A Full with tuned parameters aligned with M2A Lite + image retrieval."""
    name = "m2a_full_tuned"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected_round_ids = select_round_ids_for_qa_m2a_full(dataset, qa, self.config)
        self.runtime_info = get_last_m2a_capabilities()
        if not selected_round_ids:
            return M2ALiteMethod(config=self.config).build_history(dataset, qa)

        history: List[Dict[str, Any]] = []
        allowed_round_ids = set(selected_round_ids)
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, allowed_round_ids))
        return history


class M2ATfidfMethod(HistoryMethod):
    """M2A method using TF-IDF only (baseline without dense embeddings)."""
    name = "m2a_tfidf"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Force TF-IDF mode by setting use_dense_embeddings to False
        config = dict(self.config)
        config["use_dense_embeddings"] = False
        config["use_image_retrieval"] = False

        selected_round_ids = select_round_ids_for_qa_m2a_full(dataset, qa, config)
        self.runtime_info = get_last_m2a_capabilities()
        if not selected_round_ids:
            return M2ALiteMethod(config=config).build_history(dataset, qa)

        history: List[Dict[str, Any]] = []
        allowed_round_ids = set(selected_round_ids)
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, allowed_round_ids))
        return history


def get_method(method_name: str, config: Optional[Dict[str, Any]] = None) -> HistoryMethod:
    registry = {
        FullContextMethod.name: FullContextMethod,
        HybridRAGMethod.name: HybridRAGMethod,
        M2ALiteMethod.name: M2ALiteMethod,
        M2AFullMethod.name: M2AFullMethod,
        M2AFullTunedMethod.name: M2AFullTunedMethod,
        M2ATfidfMethod.name: M2ATfidfMethod,
    }
    cls = registry.get(method_name)
    if cls is None:
        raise ValueError(f"Unsupported method: {method_name}")
    return cls(config=config)
