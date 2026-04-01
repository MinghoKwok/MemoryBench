from typing import Any, Dict, List

from .base import BaseRouter
from .http_utils import encode_image_data_url, post_json, require_api_key


class OpenAIAPIRouter(BaseRouter):
    def __init__(
        self,
        model: str,
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        max_new_tokens: int = 128,
        timeout: int = 90,
        system_prompt: str = "",
    ) -> None:
        self.model = model
        self.api_key = require_api_key(api_key=api_key, api_key_env=api_key_env)
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt

    def _to_messages(self, history_messages: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        for msg in history_messages:
            content: List[Dict[str, Any]] = []
            text = str(msg.get("text", "")).strip()
            if text:
                content.append({"type": "text", "text": text})
            for image_path in msg.get("images", []) or []:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_data_url(image_path), "detail": "low"},
                    }
                )
            if content:
                messages.append({"role": msg.get("role", "user"), "content": content})

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Answer based on the conversation and images above. "
                            "Be concise and factual.\n"
                            f"Question: {question}"
                        ),
                    }
                ],
            }
        )
        return messages

    def answer(self, history_messages: List[Dict[str, Any]], question: str) -> str:
        payload = {
            "model": self.model,
            "messages": self._to_messages(history_messages, question),
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        response = post_json(
            url=f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout=self.timeout,
        )
        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise RuntimeError(f"Unexpected OpenAI response shape: {response}") from exc
