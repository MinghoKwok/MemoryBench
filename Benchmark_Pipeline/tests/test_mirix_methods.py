from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Benchmark_Pipeline.benchmark.dataset import MemoryBenchmarkDataset
from Benchmark_Pipeline.benchmark.methods import get_method
from Benchmark_Pipeline.benchmark.mirix import MIRIXMemoryEntry, MIRIXMemoryStore, MIRIXMemoryType


class MIRIXMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        dialog_dir = root / "data" / "dialog"
        image_dir = root / "data" / "image" / "toy"
        cache_dir = root / "data" / "mirix_cache"
        dialog_dir.mkdir(parents=True)
        image_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        (image_dir / "img1.png").write_bytes(b"png")

        payload = {
            "character_profile": {"name": "Taylor"},
            "multi_session_dialogues": [
                {
                    "session_id": "S1",
                    "date": "2025-01-01",
                    "dialogues": [
                        {
                            "round": "S1:1",
                            "user": "Show me the blue sofa.",
                            "assistant": "Here it is.",
                            "image_id": ["IMG1"],
                            "input_image": ["../image/toy/img1.png"],
                            "image_caption": ["blue sofa with wooden legs"],
                        }
                    ],
                }
            ],
            "qas": [
                {
                    "point": [["X1"], ["Y1"]],
                    "question": "What color was the sofa?",
                    "answer": "blue",
                    "session_id": ["S1"],
                    "clue": ["S1:1"],
                }
            ],
        }
        dialog_path = dialog_dir / "toy.json"
        dialog_path.write_text(json.dumps(payload), encoding="utf-8")
        self.dataset = MemoryBenchmarkDataset(dialog_path, root / "data" / "image")

        store = MIRIXMemoryStore(
            [
                MIRIXMemoryEntry(
                    memory_id="core_0001",
                    memory_type=MIRIXMemoryType.CORE.value,
                    content="Name: Taylor",
                    summary="Name: Taylor",
                    details="Name: Taylor",
                    metadata={"update_key": "core:profile"},
                ),
                MIRIXMemoryEntry(
                    memory_id="episodic_0002",
                    memory_type=MIRIXMemoryType.EPISODIC.value,
                    content="Blue sofa shown in image.",
                    summary="Blue sofa shown in image.",
                    details="The retrieved image shows a blue sofa with wooden legs.",
                    source_round_ids=["S1:1"],
                    source_session_id="S1",
                    image_paths=[str(image_dir / "img1.png")],
                    timestamp="2025-01-01",
                    order_index=2,
                    occurred_at="2025-01-01",
                    caption="blue sofa with wooden legs",
                    metadata={"update_key": "item:blue_sofa"},
                ),
            ],
            extraction_mode="incremental_llm",
            extraction_model="test-model",
        )
        store.save(cache_dir / "toy_mirix_memories.json")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_method_loads_mirix_original(self) -> None:
        method = get_method("mirix_original", config={"memory_cache_dir": "mirix_cache"})
        self.assertEqual(method.name, "mirix_original")

    def test_mirix_original_augments_system_prompt(self) -> None:
        method = get_method(
            "mirix_original",
            config={
                "memory_cache_dir": "mirix_cache",
                "use_dense_embeddings": False,
                "use_image_retrieval": False,
            },
        )
        qa = self.dataset.qas[0]
        prompt = method.get_system_prompt("Base prompt", self.dataset, qa)
        self.assertIn("<core_memory>", prompt)
        self.assertIn("<episodic_memory>", prompt)

    def test_mirix_siglip_builds_image_history(self) -> None:
        method = get_method(
            "mirix_siglip",
            config={
                "memory_cache_dir": "mirix_cache",
                "use_dense_embeddings": False,
                "use_image_retrieval": True,
                "image_limit": 2,
            },
        )
        qa = self.dataset.qas[0]
        with patch("Benchmark_Pipeline.benchmark.mirix.memory._image_rank_entries", return_value=[]):
            history = method.build_history(self.dataset, qa)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertTrue(history[0]["images"])


if __name__ == "__main__":
    unittest.main()
