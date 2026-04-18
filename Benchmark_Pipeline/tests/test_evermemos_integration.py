import asyncio
import sys
import types
import unittest
from pathlib import Path

bson_stub = types.ModuleType("bson")
bson_objectid_stub = types.ModuleType("bson.objectid")


class _ObjectId:
    def __str__(self):
        return "stub-object-id"


bson_stub.ObjectId = _ObjectId
bson_objectid_stub.ObjectId = _ObjectId
sys.modules.setdefault("bson", bson_stub)
sys.modules.setdefault("bson.objectid", bson_objectid_stub)

from Benchmark_Pipeline.benchmark.evermemos import (
    EverMemOSMethod,
    _parse_session_base,
    _raw_question_text,
    _question_with_image_caption,
    _round_messages,
)
from evaluation.src.adapters.evermemos.stage1_memcells_extraction import (
    _normalize_memcell_participants,
)
from evaluation.src.adapters.evermemos_adapter import _doc_context_text
from evaluation.src.adapters.evermemos.config import ExperimentConfig
from evaluation.src.adapters.evermemos.stage4_response import locomo_response
from benchmark_runtime.models import MemCell


class _FakeLLMProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def generate(self, prompt: str, temperature: float = 0):
        self.prompts.append(prompt)
        if not self._responses:
            return ""
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EverMemOSIntegrationTests(unittest.TestCase):
    def test_round_messages_preserve_roles_and_image_caption_on_user_turn(self):
        payload = {
            "user": "Here is the frame.",
            "assistant": "I see it.",
            "raw": {
                "input_image": ["a.jpg"],
                "image_caption": ["A green dinosaur next to a tree."],
            },
        }

        messages = _round_messages(payload, "user (Hannah)", "assistant")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn("Here is the frame.", messages[0]["content"])
        self.assertIn("image_caption: A green dinosaur next to a tree.", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "I see it.")

    def test_non_iso_session_dates_preserve_order_via_synthetic_offsets(self):
        first = _parse_session_base("Ep1 — Cave and Eggs", 0)
        second = _parse_session_base("Ep2 — Forest", 1)
        self.assertLess(first, second)
        self.assertEqual((second - first).days, 1)

    def test_mcq_uses_caption_augmented_query(self):
        qa = {
            "question": "What suit is shown?",
            "question_image": ["q1.jpg"],
            "image_caption": ["A blue card showing a queen of hearts."],
            "options": {"A": "Spade", "B": "Heart"},
        }
        query = _question_with_image_caption(qa, _raw_question_text(qa, "fallback"))
        self.assertIn("image_caption: A blue card showing a queen of hearts.", query)
        self.assertNotIn("Answer with ONLY", query)

        method = EverMemOSMethod(config={})
        captured = {}

        def fake_answer_mcq(query_text, context, options):
            captured["query"] = query_text
            captured["context"] = context
            captured["options"] = options
            return "B"

        method._answer_mcq = fake_answer_mcq  # type: ignore[method-assign]
        method._ensure_initialized = lambda dataset: None  # type: ignore[method-assign]
        method._flush_debug = lambda dataset: None  # type: ignore[method-assign]

        class _Adapter:
            async def search(self, **kwargs):
                return type(
                    "SearchResult",
                    (),
                    {
                        "results": [],
                        "retrieval_metadata": {"formatted_context": "ctx"},
                    },
                )()

            def _convert_config_to_experiment_config(self):
                return ExperimentConfig()

        method._adapter = _Adapter()
        method._answer_llm = object()
        method._conversation = type("Conversation", (), {"conversation_id": "0"})()
        method._index_metadata = {}
        method._load_components = lambda: {"stage4_response": None}  # type: ignore[method-assign]

        prediction = method.answer(type("Dataset", (), {})(), qa, "What suit is shown?")
        self.assertEqual(prediction, "B")
        self.assertEqual(captured["query"], query)

    def test_open_answer_parses_final_answer_and_falls_back_to_last_line(self):
        config = ExperimentConfig()
        config.max_retries = 1

        provider = _FakeLLMProvider(
            [
                "## STEP 1\n...\nFINAL ANSWER: The card is a heart.",
            ]
        )
        result = asyncio.run(locomo_response(provider, "ctx", "question", config))
        self.assertEqual(result, "The card is a heart.")

        provider = _FakeLLMProvider(
            [
                "## STEP 1\n...\nThe concise answer without marker.",
            ]
        )
        result = asyncio.run(locomo_response(provider, "ctx", "question", config))
        self.assertEqual(result, "The concise answer without marker.")

    def test_open_answer_no_unboundlocal_on_total_failure(self):
        config = ExperimentConfig()
        config.max_retries = 2
        provider = _FakeLLMProvider([RuntimeError("boom"), RuntimeError("boom again")])
        result = asyncio.run(locomo_response(provider, "ctx", "question", config))
        self.assertIn("ERROR:", result)

    def test_runtime_and_debug_dirs_do_not_use_output_evermemos_tree(self):
        method = EverMemOSMethod(config={})
        dataset = type(
            "Dataset",
            (),
            {
                "data": {"task_name": "Card Playlog Test"},
                "dialog_json_path": Path("/tmp/Card_Playlog_Test.json"),
            },
        )()
        runtime_text = str(method._runtime_dir(dataset))
        debug_text = str(method._debug_dir(dataset))
        self.assertIn("/Benchmark_Pipeline/runs/card_playlog_test/evermemos/", runtime_text)
        self.assertEqual(runtime_text, debug_text)
        self.assertNotIn("/Benchmark_Pipeline/output/card_playlog_test/evermemos/", debug_text)

    def test_participant_recovery_from_original_data(self):
        memcell = MemCell(
            user_id_list=["benchmark_user", "benchmark_assistant"],
            original_data=[
                {
                    "message": {
                        "sender_id": "benchmark_user",
                        "sender_name": "user",
                        "content": "hello",
                        "timestamp": "2025-01-01T00:00:00+00:00",
                        "role": "user",
                    }
                },
                {
                    "message": {
                        "sender_id": "benchmark_assistant",
                        "sender_name": "assistant",
                        "content": "hi",
                        "timestamp": "2025-01-01T00:00:30+00:00",
                        "role": "assistant",
                    }
                },
            ],
            timestamp=__import__("datetime").datetime.fromisoformat("2025-01-01T00:00:30+00:00"),
        )
        _normalize_memcell_participants(memcell)
        self.assertEqual(memcell.participants, ["benchmark_user", "benchmark_assistant"])
        self.assertEqual(memcell.sender_ids, ["benchmark_user", "benchmark_assistant"])

    def test_context_text_does_not_duplicate_subject_for_original_data_fallback(self):
        doc = {
            "original_data": [
                {
                    "message": {
                        "sender_name": "user (Card_Playlog_Test)",
                        "content": "Deck initialized and shuffled.\nInitial card is red-4-number.",
                    }
                }
            ]
        }
        text = _doc_context_text(doc)
        self.assertNotIn("Deck initialized and shuffled.: user (Card_Playlog_Test):", text)
        self.assertIn("user (Card_Playlog_Test): Deck initialized and shuffled.", text)


if __name__ == "__main__":
    unittest.main()
