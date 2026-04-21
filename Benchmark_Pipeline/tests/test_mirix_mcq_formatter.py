import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Benchmark_Pipeline.benchmark.mirix.official import _StrictMCQFormatter


class _FakeOpenAIResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeOpenAIClient:
    def __init__(self, *, api_key: str, base_url: str, timeout: int) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeOpenAIResponse('{"answer":"B"}')


class _FakeGeminiModels:
    def __init__(self) -> None:
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text='{"answer":"C"}')


class _FakeGeminiClient:
    def __init__(self, *, api_key: str, http_options) -> None:
        self.api_key = api_key
        self.http_options = http_options
        self.models = _FakeGeminiModels()


class MirixStrictMCQFormatterTests(unittest.TestCase):
    def test_openai_formatter_uses_existing_chat_completion_path(self):
        fake_client = _FakeOpenAIClient(api_key="openai-key", base_url="https://api.openai.com/v1", timeout=45)
        with patch("openai.OpenAI", return_value=fake_client):
            formatter = _StrictMCQFormatter(
                {
                    "_model_cfg": {
                        "provider": "openai_api",
                        "model": "gpt-4.1-nano",
                        "api_key": "openai-key",
                        "timeout": 45,
                    }
                }
            )
            answer, prompt = formatter.format_answer(
                "Which card was shown?",
                {"options": {"A": "Spade", "B": "Heart", "C": "Club"}},
                "The answer is Heart.",
            )

        self.assertEqual(answer, "B")
        self.assertIn("Select exactly one option letter from: A, B, C.", prompt)
        self.assertEqual(len(fake_client.calls), 1)
        self.assertEqual(fake_client.calls[0]["temperature"], 0.0)
        self.assertEqual(fake_client.calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(
            fake_client.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"],
            ["A", "B", "C"],
        )

    def test_gemini_formatter_uses_generate_content_with_json_schema(self):
        fake_client = None

        def _build_client(**kwargs):
            nonlocal fake_client
            fake_client = _FakeGeminiClient(**kwargs)
            return fake_client

        with patch("google.genai.Client", side_effect=_build_client):
            formatter = _StrictMCQFormatter(
                {
                    "_model_cfg": {
                        "provider": "gemini_api",
                        "model": "gemini-2.5-flash-lite",
                        "api_key": "gemini-key",
                        "base_url": "https://generativelanguage.googleapis.com/v1beta",
                        "timeout": 33,
                    }
                }
            )
            answer, prompt = formatter.format_answer(
                "Which color was the brand logo?",
                {"options": {"A": "Red", "B": "Green", "C": "Blue"}},
                "The draft answer points to blue.",
            )

        assert fake_client is not None
        self.assertEqual(answer, "C")
        self.assertIn("Mirix draft answer", prompt)
        self.assertEqual(fake_client.api_key, "gemini-key")
        self.assertEqual(fake_client.http_options.base_url, "https://generativelanguage.googleapis.com/v1beta")
        self.assertEqual(fake_client.http_options.timeout, 33)
        self.assertEqual(len(fake_client.models.calls), 1)
        call = fake_client.models.calls[0]
        self.assertEqual(call["model"], "gemini-2.5-flash-lite")
        self.assertEqual(call["contents"], prompt)
        self.assertEqual(call["config"].response_mime_type, "application/json")
        self.assertEqual(call["config"].temperature, 0.0)
        self.assertEqual(call["config"].response_schema["properties"]["answer"]["enum"], ["A", "B", "C"])

    def test_gemini_formatter_rejects_invalid_option(self):
        fake_client = _FakeGeminiClient(api_key="gemini-key", http_options=None)
        fake_client.models.generate_content = lambda **kwargs: SimpleNamespace(text=json.dumps({"answer": "Z"}))
        with patch("google.genai.Client", return_value=fake_client):
            formatter = _StrictMCQFormatter(
                {
                    "_model_cfg": {
                        "provider": "gemini_api",
                        "model": "gemini-2.5-flash-lite",
                        "api_key": "gemini-key",
                    }
                }
            )
            with self.assertRaisesRegex(ValueError, "Invalid Mirix MCQ formatter answer"):
                formatter.format_answer(
                    "Which color was the brand logo?",
                    {"options": {"A": "Red", "B": "Green", "C": "Blue"}},
                    "The draft answer points to blue.",
                )


if __name__ == "__main__":
    unittest.main()
