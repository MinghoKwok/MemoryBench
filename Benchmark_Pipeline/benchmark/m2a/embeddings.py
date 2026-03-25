"""
Embedding wrappers for M2A. Faithful to official agent/embeddings/.

TextEmbedder      : all-MiniLM-L6-v2 via sentence-transformers (local, 384-dim)
MultimodalEmbedder: siglip2-base-patch16-384 via vLLM OpenAI-compatible API (768-dim)
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional


class TextEmbedder:
    """
    Local sentence-transformers text embedder.
    Model: all-MiniLM-L6-v2 (384-dim). Faithful to official M2A TextEmbedding.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_name)

    @property
    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def embed_query(self, text: str) -> List[float]:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()  # type: ignore

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load()
        return self._model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore


class MultimodalEmbedder:
    """
    SigLIP2 embedder via vLLM's OpenAI-compatible embedding API.
    Model : siglip2-base-patch16-384 (768-dim)
    URL   : http://localhost:8050/v1  (matches official default)
    Faithful to official M2A MultimodalEmbedder.
    """

    DEFAULT_MODEL = "siglip2-base-patch16-384"
    DEFAULT_BASE_URL = "http://localhost:8050/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "dummy",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = None

    def _load(self) -> None:
        if self._client is None:
            import openai  # type: ignore
            self._client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)

    @property
    def is_available(self) -> bool:
        try:
            self._load()
            self._client.models.list()  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def embed_text(self, text: str) -> List[float]:
        """Embed text for cross-modal text→image search."""
        self._load()
        resp = self._client.embeddings.create(model=self._model, input=text)  # type: ignore[union-attr]
        return resp.data[0].embedding

    def embed_image(self, image_path: str) -> List[float]:
        """Embed image for visual similarity / image→image search."""
        self._load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        b64 = base64.b64encode(path.read_bytes()).decode()
        ext = path.suffix.lstrip(".").lower() or "jpeg"
        data_url = f"data:image/{ext};base64,{b64}"
        resp = self._client.embeddings.create(model=self._model, input=data_url)  # type: ignore[union-attr]
        return resp.data[0].embedding
