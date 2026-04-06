from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRouter(ABC):
    @abstractmethod
    def answer(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate an answer given multimodal history messages and a question.

        Args:
            history_messages: List of message dicts with role, text, images.
            question: The question to answer.
            system_prompt: If provided, overrides self.system_prompt for this call.

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
