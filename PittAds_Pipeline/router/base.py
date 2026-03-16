from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRouter(ABC):
    @abstractmethod
    def answer(self, history_messages: List[Dict[str, Any]], question: str) -> str:
        """
        Generate an answer given multimodal history messages and a question.

        history_messages format:
        [
          {
            "role": "user" | "assistant",
            "text": "...",
            "images": ["/abs/path/a.jpg", ...]   # optional
          },
          ...
        ]
        """
        raise NotImplementedError
