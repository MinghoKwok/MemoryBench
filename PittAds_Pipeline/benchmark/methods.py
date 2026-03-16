from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .dataset import PittAdsDataset, history_from_round_ids


class HistoryMethod(ABC):
    name = "base"

    @abstractmethod
    def build_history(self, dataset: PittAdsDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class FullContextMethod(HistoryMethod):
    name = "full_context"

    def build_history(self, dataset: PittAdsDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds))
        return history


class ClueOnlyMethod(HistoryMethod):
    name = "clue_only"

    def build_history(self, dataset: PittAdsDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        target_sessions = set(qa.get("session_id", []))
        clue_rounds = set(qa.get("clue", []) or [])
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(history_from_round_ids(dataset.get_session(sid), dataset.rounds, clue_rounds))
        return history


def get_method(method_name: str) -> HistoryMethod:
    registry = {
        FullContextMethod.name: FullContextMethod,
        ClueOnlyMethod.name: ClueOnlyMethod,
    }
    cls = registry.get(method_name)
    if cls is None:
        raise ValueError(f"Unsupported method: {method_name}")
    return cls()
