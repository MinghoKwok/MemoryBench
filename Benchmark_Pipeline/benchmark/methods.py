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
        """Build history from ALL sessions to stress context window."""
        history: List[Dict[str, Any]] = []
        for sid in dataset.session_order():
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds))
        return history


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
