from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..dataset import MemoryBenchmarkDataset
from ..embeddings import ImageEmbedder, TextEmbedder, get_image_embedder, get_text_embedder
from ..methods import HistoryMethod
from .memory import (
    MIRIX_CACHE_SCHEMA_VERSION,
    MIRIXMemoryStore,
    collect_images_from_retrieved,
    extract_memories_incremental_llm,
    format_retrieved_original,
    retrieve_mirix_memories,
)

logger = logging.getLogger(__name__)


class MIRIXOriginalMethod(HistoryMethod):
    name = "mirix_original"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._memory_store: Optional[MIRIXMemoryStore] = None
        self._text_embedder: Optional[TextEmbedder] = None
        self._retrieval_cache_key: Optional[Tuple[Any, ...]] = None
        self._retrieval_cache_value: Optional[Tuple[Dict[str, List[Any]], int, Dict[str, Any]]] = None

    def _resolve_cache_path(self, dataset: MemoryBenchmarkDataset) -> Path:
        cache_dir = self.config.get("memory_cache_dir", "mirix_cache")
        base = dataset.dialog_json_path.parent.parent / cache_dir
        scenario_name = dataset.dialog_json_path.stem
        return base / f"{scenario_name}_mirix_memories.json"

    def _build_store(self, dataset: MemoryBenchmarkDataset, extraction_mode: str) -> MIRIXMemoryStore:
        llm_cfg = dict(self.config.get("incremental_extraction", {}))
        llm_cfg.update(
            {
                "model": llm_cfg.get("model", self.config.get("extraction_model", "gpt-4.1-mini")),
                "api_key": llm_cfg.get("api_key", self.config.get("api_key", "")),
                "api_key_env": llm_cfg.get("api_key_env", self.config.get("api_key_env", "OPENAI_API_KEY")),
                "base_url": llm_cfg.get("base_url", self.config.get("base_url", "https://api.openai.com/v1")),
                "timeout": llm_cfg.get("timeout", self.config.get("timeout", 120)),
            }
        )
        if extraction_mode == "incremental_llm":
            return extract_memories_incremental_llm(dataset, llm_cfg)
        raise ValueError(f"Unsupported MIRIX extraction mode: {extraction_mode}")

    def _ensure_store(self, dataset: MemoryBenchmarkDataset) -> MIRIXMemoryStore:
        if self._memory_store is not None:
            return self._memory_store

        cache_path = self._resolve_cache_path(dataset)
        extraction_mode = str(self.config.get("extraction_mode", "incremental_llm"))
        if cache_path.exists():
            loaded_store = MIRIXMemoryStore.load(cache_path)
            if (
                loaded_store.schema_version >= MIRIX_CACHE_SCHEMA_VERSION
                and loaded_store.extraction_mode == extraction_mode
            ):
                self._memory_store = loaded_store
                return self._memory_store
            logger.warning(
                "MIRIX cache at %s is incompatible (schema=%s mode=%s); rebuilding %s cache",
                cache_path,
                loaded_store.schema_version,
                loaded_store.extraction_mode,
                extraction_mode,
            )
        else:
            logger.warning("MIRIX cache not found at %s; extracting %s memories", cache_path, extraction_mode)

        self._memory_store = self._build_store(dataset, extraction_mode)
        self._memory_store.save(cache_path)
        return self._memory_store

    def _get_text_embedder(self) -> Optional[TextEmbedder]:
        if self._text_embedder is None and self.config.get("use_dense_embeddings", True):
            self._text_embedder = get_text_embedder(
                self.config.get("text_embedding_model", "text-embedding-3-small")
            )
        return self._text_embedder

    def _qa_cache_key(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            str(dataset.dialog_json_path),
            qa.get("point", ""),
            qa.get("question", ""),
            tuple(qa.get("session_id", []) or []),
            tuple(qa.get("clue", []) or []),
        )

    def _retrieve(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> Tuple[Dict[str, List[Any]], int]:
        cache_key = self._qa_cache_key(dataset, qa)
        if self._retrieval_cache_key == cache_key and self._retrieval_cache_value is not None:
            retrieved, total, runtime_info = self._retrieval_cache_value
            self.runtime_info = dict(runtime_info)
            return retrieved, total

        store = self._ensure_store(dataset)
        text_embedder = self._get_text_embedder()
        retrieved = retrieve_mirix_memories(
            store,
            qa.get("question", ""),
            self.config,
            text_embedder=text_embedder,
        )
        total = sum(len(values) for values in retrieved.values())
        type_counts = {key: len(values) for key, values in retrieved.items() if values}
        self.runtime_info = {
            "retrieved_counts": type_counts,
            "total_retrieved": total,
            "extraction_mode": store.extraction_mode,
            "extraction_model": store.extraction_model,
            "memory_cache_schema_version": getattr(store, "schema_version", None),
            "text_embedding_model": self.config.get("text_embedding_model", ""),
            "text_embedding_available": bool(text_embedder and text_embedder.is_available),
        }
        self._retrieval_cache_key = cache_key
        self._retrieval_cache_value = (retrieved, total, dict(self.runtime_info))
        return retrieved, total

    def get_system_prompt(self, base_prompt: str, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> str:
        retrieved, total = self._retrieve(dataset, qa)
        if total == 0:
            return base_prompt
        return base_prompt + "\n\n" + format_retrieved_original(retrieved, question=qa.get("question", ""))

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class MIRIXSigLIPMethod(MIRIXOriginalMethod):
    name = "mirix_siglip"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._image_embedder: Optional[ImageEmbedder] = None

    def _get_image_embedder(self) -> Optional[ImageEmbedder]:
        if self._image_embedder is None and self.config.get("use_image_retrieval", True):
            self._image_embedder = get_image_embedder(
                self.config.get("image_embedding_model", "google/siglip2-so400m-patch14-384")
            )
        return self._image_embedder

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        retrieved, total = self._retrieve(dataset, qa)
        images = collect_images_from_retrieved(retrieved)

        from .memory import _image_rank_entries

        store = self._ensure_store(dataset)
        image_embedder = self._get_image_embedder()
        image_limit = int(self.config.get("image_limit", 5))
        image_ranked_entries = _image_rank_entries(
            qa.get("question", ""),
            store.all_entries(),
            image_embedder,
            limit=image_limit,
        )
        image_ranked_paths: List[str] = []
        for entry in image_ranked_entries:
            for image_path in entry.image_paths:
                image_ranked_paths.append(image_path)
                if image_path not in images:
                    images.append(image_path)

        self.runtime_info.update(
            {
                "total_images": len(images),
                "image_ranked_entries": len(image_ranked_entries),
                "image_ranked_paths": len(image_ranked_paths),
                "image_embedding_model": self.config.get("image_embedding_model", ""),
                "image_embedding_available": bool(image_embedder and image_embedder.is_available),
            }
        )
        self._retrieval_cache_value = (retrieved, total, dict(self.runtime_info))

        if not images:
            return []

        return [
            {
                "role": "user",
                "text": "The following images were retrieved from memory as relevant to the question:",
                "images": images,
            },
            {
                "role": "assistant",
                "text": "I have reviewed the retrieved images.",
                "images": [],
            },
        ]


def get_mirix_method(method_name: str, config: Optional[Dict[str, Any]] = None) -> HistoryMethod:
    registry = {
        MIRIXOriginalMethod.name: MIRIXOriginalMethod,
        MIRIXSigLIPMethod.name: MIRIXSigLIPMethod,
    }
    cls = registry.get(method_name)
    if cls is None:
        raise ValueError(f"Unsupported method: {method_name!r}")
    return cls(config=config)
