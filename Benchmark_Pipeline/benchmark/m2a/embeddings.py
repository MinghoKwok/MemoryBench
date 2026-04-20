"""
Embedding wrappers for M2A. Faithful to official agent/embeddings/.

TextEmbedder       : all-MiniLM-L6-v2 via sentence-transformers (local, 384-dim)
MultimodalEmbedder : siglip2-base-patch16-384 via vLLM OpenAI-compatible API (768-dim)
LocalCLIPEmbedder  : openai/clip-vit-base-patch32 via transformers (local fallback, 512-dim)
"""
from __future__ import annotations

import base64
import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union
from urllib import error as urllib_error
from urllib import request as urllib_request




@contextmanager
def _sanitized_hf_token_env():
    # NOTE: not thread-safe — modifies os.environ globally.
    original = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if original is not None and any(ord(ch) > 127 for ch in original):
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        try:
            yield
        finally:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = original
    else:
        yield


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

            cache_folder = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
            with _sanitized_hf_token_env():
                self._model = SentenceTransformer(self._model_name, cache_folder=cache_folder)

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
    EMPTY_CHAT_TEMPLATE = "{% for message in messages %}{% endfor %}"
    # SigLIP2 text encoder context is 64 tokens.  Approximate at ~3.5
    # chars/token and leave 4-token headroom for special tokens.
    DEFAULT_MAX_TEXT_TOKENS = 64

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "dummy",
        max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_text_tokens = max_text_tokens
        self._client = None

    def _load(self) -> None:
        if self._client is None:
            import openai  # type: ignore
            self._client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)

    def _post_embeddings(self, payload: dict) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        req = urllib_request.Request(
            f"{self._base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM embeddings request failed with HTTP {exc.code}: {detail}"
            ) from exc

        parsed = json.loads(body)
        if "error" in parsed:
            raise RuntimeError(f"vLLM embeddings request failed: {parsed['error']}")
        return parsed["data"][0]["embedding"]

    @property
    def is_available(self) -> bool:
        try:
            self._load()
            self._client.models.list()  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within the model's token limit.

        Uses a conservative character heuristic (~3.5 chars/token for
        SentencePiece) minus headroom for special tokens.
        """
        max_chars = int(self._max_text_tokens * 3.5) - 14  # ~4 special tokens
        if len(text) <= max_chars:
            return text
        # Truncate on a word boundary when possible.
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            truncated = truncated[:last_space]
        warnings.warn(
            f"[MultimodalEmbedder] Text truncated from {len(text)} to "
            f"{len(truncated)} chars to fit {self._max_text_tokens}-token limit"
        )
        return truncated

    def embed_text(self, text: str) -> List[float]:
        """Embed text for cross-modal text→image search."""
        self._load()
        text = self._truncate_text(text)
        return self._post_embeddings({"model": self._model, "input": text})

    def embed_image(self, image_path: str) -> List[float]:
        """Embed image for visual similarity / image→image search."""
        self._load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        b64 = base64.b64encode(path.read_bytes()).decode()
        ext = path.suffix.lstrip(".").lower() or "jpeg"
        data_url = f"data:image/{ext};base64,{b64}"
        return self._post_embeddings(
            {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ],
                    }
                ],
                # SigLIP image embeddings require an empty text prompt.
                "chat_template": self.EMPTY_CHAT_TEMPLATE,
                "add_special_tokens": False,
            }
        )


class LocalCLIPEmbedder:
    """
    Local CLIP embedder via transformers. Fallback when vLLM is unavailable.
    Model: openai/clip-vit-base-patch32 (512-dim)
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
            with _sanitized_hf_token_env():
                self._processor = CLIPProcessor.from_pretrained(self._model_name, cache_dir=cache_dir)
                self._model = CLIPModel.from_pretrained(self._model_name, cache_dir=cache_dir)

            # Use GPU if available
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            self._model = self._model.to(self._device)
            self._model.eval()

    @property
    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as e:
            warnings.warn(f"LocalCLIPEmbedder not available: {e}")
            return False

    def embed_text(self, text: str) -> List[float]:
        """Embed text for cross-modal text→image search."""
        import torch

        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        # Only pass text-related keys to get_text_features
        text_inputs = {k: v.to(self._device) for k, v in inputs.items()
                       if k in ("input_ids", "attention_mask")}

        with torch.no_grad():
            text_features = self._model.get_text_features(**text_inputs)
            # Handle both tensor and BaseModelOutput returns
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().flatten().tolist()

    def embed_image(self, image_path: str) -> List[float]:
        """Embed image for visual similarity / image→image search."""
        import torch
        from PIL import Image  # type: ignore

        self._load()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        # Only pass image-related keys to get_image_features
        image_inputs = {k: v.to(self._device) for k, v in inputs.items()
                        if k in ("pixel_values",)}

        with torch.no_grad():
            image_features = self._model.get_image_features(**image_inputs)
            # Handle both tensor and BaseModelOutput returns
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten().tolist()


def get_multimodal_embedder(
    vllm_model: str = "siglip2-base-patch16-384",
    vllm_url: str = "http://localhost:8050/v1",
    vllm_api_key: str = "dummy",
    clip_model: str = "openai/clip-vit-base-patch32",
) -> Optional[Union[MultimodalEmbedder, LocalCLIPEmbedder]]:
    """
    Factory function: try vLLM first, fallback to local CLIP.
    Returns None if neither is available.
    """
    # Try vLLM SigLIP2 first
    vllm_embedder = MultimodalEmbedder(vllm_model, vllm_url, vllm_api_key)
    if vllm_embedder.is_available:
        print("[M2A] Using vLLM SigLIP2 for image embeddings")
        return vllm_embedder

    # Fallback to local CLIP
    warnings.warn(
        f"[M2A] vLLM server at {vllm_url} not available. "
        "Falling back to local CLIP for image embeddings."
    )
    clip_embedder = LocalCLIPEmbedder(clip_model)
    if clip_embedder.is_available:
        print("[M2A] Using local CLIP for image embeddings")
        return clip_embedder

    # Neither available
    warnings.warn(
        "[M2A] Neither vLLM nor local CLIP available. "
        "Image retrieval will be DISABLED."
    )
    return None
