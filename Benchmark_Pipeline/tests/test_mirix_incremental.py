from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Benchmark_Pipeline.benchmark.dataset import MemoryBenchmarkDataset
from Benchmark_Pipeline.benchmark.mirix import (
    MIRIXMemoryEntry,
    MIRIXMemoryStore,
    MIRIXMemoryType,
    RoundEvidence,
    _extract_json_captions,
    _normalize_confidence,
    apply_memory_updates,
    build_round_evidence,
    extract_memories_incremental_llm,
)


class MIRIXIncrementalTests(unittest.TestCase):
    def _make_dataset(self) -> MemoryBenchmarkDataset:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        dialog_dir = root / "data" / "dialog"
        image_dir = root / "data" / "image" / "toy"
        dialog_dir.mkdir(parents=True)
        image_dir.mkdir(parents=True)
        (image_dir / "img1.png").write_bytes(b"png")

        payload = {
            "character_profile": {
                "name": "Taylor",
                "persona_summary": "Tracks household changes.",
            },
            "multi_session_dialogues": [
                {
                    "session_id": "S1",
                    "date": "2025-01-01",
                    "dialogues": [
                        {
                            "round": "S1:1",
                            "user": "I bought a red lamp yesterday.",
                            "assistant": "Noted. The new red lamp is important.",
                        },
                        {
                            "round": "S1:2",
                            "user": "Here is Option A.",
                            "assistant": "Looks compact and easy to clean.",
                            "image_id": ["IMG1"],
                            "input_image": ["../image/toy/img1.png"],
                            "image_caption": ["blue sofa with wooden legs"],
                        },
                    ],
                }
            ],
            "qas": [],
        }
        dialog_path = dialog_dir / "toy.json"
        dialog_path.write_text(json.dumps(payload), encoding="utf-8")
        return MemoryBenchmarkDataset(dialog_path, root / "data" / "image")

    def tearDown(self) -> None:
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()

    def test_build_round_evidence_includes_captions(self) -> None:
        dataset = self._make_dataset()
        evidences = build_round_evidence(dataset)
        self.assertEqual(len(evidences), 2)
        self.assertEqual(evidences[0].image_captions, [])
        self.assertEqual(evidences[1].image_captions, ["blue sofa with wooden legs"])

    def test_build_round_evidence_direct_image_ingestion_can_ignore_dataset_captions(self) -> None:
        dataset = self._make_dataset()
        evidences = build_round_evidence(
            dataset,
            llm_config={
                "direct_image_ingestion": True,
                "use_dataset_image_captions": False,
                "model": "test-model",
            },
        )
        self.assertEqual(evidences[1].image_captions, [])

    def test_build_round_evidence_direct_image_ingestion_can_keep_dataset_captions(self) -> None:
        dataset = self._make_dataset()
        evidences = build_round_evidence(
            dataset,
            llm_config={
                "direct_image_ingestion": True,
                "use_dataset_image_captions": True,
                "model": "test-model",
            },
        )
        self.assertEqual(evidences[1].image_captions, ["blue sofa with wooden legs"])
        self.assertEqual(_extract_json_captions(dataset.data["multi_session_dialogues"][0]["dialogues"][1]), ["blue sofa with wooden legs"])

    def test_apply_memory_updates_merges_by_update_key(self) -> None:
        existing = MIRIXMemoryEntry(
            memory_id="episodic_0001",
            memory_type=MIRIXMemoryType.EPISODIC.value,
            content="old summary old details",
            summary="old summary",
            details="old details",
            source_round_ids=["S1:1"],
            source_session_id="S1",
            order_index=1,
            occurred_at="2025-01-01",
            metadata={"update_key": "item:red_lamp"},
        )
        store = MIRIXMemoryStore([existing])
        counters = {"id": 1, "order": 1}
        newest = RoundEvidence(
            session_id="S1",
            round_id="S1:2",
            date="2025-01-02",
            user_text="The lamp is now in the hallway.",
            assistant_text="Noted.",
            image_paths=[],
            image_captions=[],
        )
        apply_memory_updates(
            store,
            [
                {
                    "memory_type": "episodic",
                    "update_action": "merge",
                    "update_key": "item:red_lamp",
                    "summary": "Red lamp moved to hallway.",
                    "details": "The red lamp was moved from the living room to the hallway.",
                    "source_round_ids": ["S1:2"],
                    "confidence": 0.9,
                }
            ],
            newest,
            counters,
        )
        self.assertEqual(len(store.memories), 1)
        updated = store.memories[0]
        self.assertIn("S1:2", updated.source_round_ids)
        self.assertEqual(updated.summary, "Red lamp moved to hallway.")
        self.assertIn("hallway", updated.details)

    def test_incremental_llm_supports_text_and_image_episodic(self) -> None:
        dataset = self._make_dataset()
        meta_responses = [
            {"memory_types": ["episodic"], "focus": {"episodic": "store the lamp purchase"}},
            {"memory_types": ["episodic"], "focus": {"episodic": "store the sofa option shown"}},
        ]
        agent_responses = [
            {
                "memory_type": "episodic",
                "update_action": "insert",
                "update_key": "item:red_lamp",
                "summary": "User bought a red lamp.",
                "details": "The user said they bought a red lamp yesterday.",
                "source_round_ids": ["S1:1"],
                "occurred_at": "2025-01-01",
                "confidence": 0.95,
            },
            {
                "memory_type": "episodic",
                "update_action": "insert",
                "update_key": "option:sofa_a",
                "summary": "Option A sofa: blue sofa with wooden legs; compact and easy to clean.",
                "details": "Image shows a blue sofa with wooden legs. Assistant says it looks compact and easy to clean.",
                "caption": "blue sofa with wooden legs",
                "source_round_ids": ["S1:2"],
                "occurred_at": "2025-01-01",
                "confidence": 0.88,
            },
        ]
        with patch("Benchmark_Pipeline.benchmark.mirix.memory._call_meta_memory_agent", side_effect=meta_responses), patch(
            "Benchmark_Pipeline.benchmark.mirix.memory._call_type_memory_agent",
            side_effect=[[agent_responses[0]], [agent_responses[1]]],
        ):
            store = extract_memories_incremental_llm(
                dataset,
                {"model": "test-model", "history_size": 1, "direct_image_ingestion": True},
            )
        episodic = [m for m in store.memories if m.memory_type == MIRIXMemoryType.EPISODIC.value]
        self.assertEqual(len(episodic), 2)
        self.assertTrue(any("red lamp" in m.summary.lower() for m in episodic))
        self.assertTrue(any("blue sofa" in m.summary.lower() for m in episodic))

    def test_apply_memory_updates_prefers_target_memory_id(self) -> None:
        existing = MIRIXMemoryEntry(
            memory_id="semantic_0002",
            memory_type=MIRIXMemoryType.SEMANTIC.value,
            content="old summary old details",
            summary="Old semantic summary",
            details="Old semantic details",
            source_round_ids=["S1:1"],
            source_session_id="S1",
            order_index=2,
            occurred_at="2025-01-01",
            metadata={"update_key": "semantic:option_a"},
        )
        store = MIRIXMemoryStore([existing])
        counters = {"id": 2, "order": 2}
        newest = RoundEvidence(
            session_id="S1",
            round_id="S1:2",
            date="2025-01-02",
            user_text="Actually option A is the washable one.",
            assistant_text="That helps.",
            image_paths=[],
            image_captions=[],
        )
        apply_memory_updates(
            store,
            [{
                "memory_type": "semantic",
                "update_action": "merge",
                "target_memory_id": "semantic_0002",
                "update_key": "semantic:option_a_material",
                "summary": "Option A is washable.",
                "details": "Option A was clarified as washable.",
                "source_round_ids": ["S1:2"],
                "confidence": "high",
            }],
            newest,
            counters,
        )
        self.assertEqual(len(store.memories), 1)
        self.assertEqual(store.memories[0].summary, "Option A is washable.")

    def test_normalize_confidence_accepts_labels(self) -> None:
        self.assertEqual(_normalize_confidence("high"), 0.85)
        self.assertEqual(_normalize_confidence("medium"), 0.6)
        self.assertEqual(_normalize_confidence("low"), 0.35)


if __name__ == "__main__":
    unittest.main()
