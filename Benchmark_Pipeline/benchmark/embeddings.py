"""
Embedding module for M2A retrieval.

Provides text embeddings via sentence-transformers and image embeddings via CLIP,
implementing the dual-layer hybrid memory approach from the original M2A paper.

If sentence-transformers or CLIP cannot be loaded (e.g., due to environment issues),
the module provides a graceful fallback that disables dense embeddings.
"""

import hashlib
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Lazy imports for optional dependencies
_sentence_transformer_model = None
_clip_model = None
_clip_processor = None
_device = None
_torch_available = None
_sentence_transformers_available = None


def _check_torch_available() -> bool:
    """Check if torch is available and working."""
    global _torch_available
    if _torch_available is None:
        try:
            import torch
            # Try a simple operation to verify torch works
            _ = torch.zeros(1)
            _torch_available = True
        except Exception as e:
            warnings.warn(f"PyTorch not available or has issues: {e}. Dense embeddings will be disabled.")
            _torch_available = False
    return _torch_available


def _get_device() -> str:
    """Get the best available device."""
    global _device
    if _device is None:
        if not _check_torch_available():
            _device = "cpu"
            return _device
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device


def _get_text_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load the sentence-transformers model."""
    global _sentence_transformer_model, _sentence_transformers_available

    if _sentence_transformers_available is False:
        return None

    if _sentence_transformer_model is None:
        if not _check_torch_available():
            _sentence_transformers_available = False
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer_model = SentenceTransformer(model_name, device=_get_device())
            _sentence_transformers_available = True
        except ImportError as e:
            warnings.warn(
                f"sentence-transformers not available: {e}. "
                "Dense embeddings will be disabled. "
                "Install with: pip install sentence-transformers"
            )
            _sentence_transformers_available = False
            return None
        except Exception as e:
            warnings.warn(f"Failed to load sentence-transformers model: {e}. Dense embeddings will be disabled.")
            _sentence_transformers_available = False
            return None
    return _sentence_transformer_model


_clip_available = None


def _get_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """Lazily load the CLIP model for image embeddings."""
    global _clip_model, _clip_processor, _clip_available

    if _clip_available is False:
        return None, None

    if _clip_model is None:
        if not _check_torch_available():
            _clip_available = False
            return None, None
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            _clip_processor = CLIPProcessor.from_pretrained(model_name)
            _clip_model = CLIPModel.from_pretrained(model_name)
            _clip_model = _clip_model.to(_get_device())
            _clip_model.eval()
            _clip_available = True
        except ImportError as e:
            warnings.warn(
                f"transformers/CLIP not available: {e}. "
                "Image embeddings will be disabled. "
                "Install with: pip install transformers"
            )
            _clip_available = False
            return None, None
        except Exception as e:
            warnings.warn(f"Failed to load CLIP model: {e}. Image embeddings will be disabled.")
            _clip_available = False
            return None, None
    return _clip_model, _clip_processor


class EmbeddingCache:
    """Simple in-memory cache for embeddings with optional disk persistence."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, text: str, model_name: str) -> str:
        """Generate a hash key for caching."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache."""
        key = self._hash_key(text, model_name)
        if key in self.memory_cache:
            return self.memory_cache[key]
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                embedding = np.load(cache_file)
                self.memory_cache[key] = embedding
                return embedding
        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._hash_key(text, model_name)
        self.memory_cache[key] = embedding
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.npy"
            np.save(cache_file, embedding)


# Global cache instance
_embedding_cache = EmbeddingCache()


def set_cache_dir(cache_dir: Union[str, Path]) -> None:
    """Set the cache directory for embeddings."""
    global _embedding_cache
    _embedding_cache = EmbeddingCache(Path(cache_dir))


class TextEmbedder:
    """Text embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_cache: bool = True):
        self.model_name = model_name
        self.use_cache = use_cache
        self._model = None
        self._model_loaded = False
        self._model_unavailable = False

    @property
    def model(self):
        if self._model_unavailable:
            return None
        if self._model is None:
            self._model = _get_text_model(self.model_name)
            if self._model is None:
                self._model_unavailable = True
            else:
                self._model_loaded = True
        return self._model

    @property
    def is_available(self) -> bool:
        """Check if the embedder is available and working."""
        if self._model_unavailable:
            return False
        if not self._model_loaded:
            _ = self.model  # Trigger loading
        return self._model_loaded and not self._model_unavailable

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if not self.is_available:
            return 384  # Default dimension for all-MiniLM-L6-v2
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text string. Returns None if model unavailable."""
        if not self.is_available:
            return None

        if self.use_cache:
            cached = _embedding_cache.get(text, self.model_name)
            if cached is not None:
                return cached

        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        if self.use_cache:
            _embedding_cache.set(text, self.model_name, embedding)

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """Embed a batch of texts efficiently. Returns None if model unavailable."""
        if not texts:
            return np.array([])

        if not self.is_available:
            return None

        # Check cache for all texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.use_cache:
            for i, text in enumerate(texts):
                cached = _embedding_cache.get(text, self.model_name)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Embed uncached texts
        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False
            )

            for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                results[idx] = embedding
                if self.use_cache:
                    _embedding_cache.set(text, self.model_name, embedding)

        return np.array(results)


class ImageEmbedder:
    """Image embedding using CLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", use_cache: bool = True):
        self.model_name = model_name
        self.use_cache = use_cache
        self._model = None
        self._processor = None
        self._model_loaded = False
        self._model_unavailable = False

    def _load_model(self):
        if self._model_unavailable:
            return False
        if self._model is None:
            self._model, self._processor = _get_clip_model(self.model_name)
            if self._model is None:
                self._model_unavailable = True
                return False
            self._model_loaded = True
        return True

    @property
    def is_available(self) -> bool:
        """Check if the embedder is available and working."""
        if self._model_unavailable:
            return False
        if not self._model_loaded:
            self._load_model()
        return self._model_loaded and not self._model_unavailable

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (512 for CLIP ViT-B/32)."""
        return 512

    def _extract_feature_tensor(self, outputs):
        """Normalize transformer outputs to a tensor before L2 normalization."""
        if outputs is None:
            return None
        if hasattr(outputs, "norm"):
            return outputs
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            value = getattr(outputs, attr, None)
            if value is not None:
                return value
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state[:, 0, :]
        return None

    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        """Embed a single image from file path."""
        if not os.path.exists(image_path):
            return None

        if not self.is_available:
            return None

        cache_key = f"image:{image_path}"
        if self.use_cache:
            cached = _embedding_cache.get(cache_key, self.model_name)
            if cached is not None:
                return cached

        try:
            from PIL import Image
            import torch

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(_get_device()) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self._extract_feature_tensor(self._model.get_image_features(**inputs))
                if image_features is None:
                    raise ValueError("CLIP image features could not be extracted as a tensor")
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embedding = image_features.cpu().numpy().flatten()

            if self.use_cache:
                _embedding_cache.set(cache_key, self.model_name, embedding)

            return embedding
        except Exception as e:
            warnings.warn(f"Failed to embed image {image_path}: {e}")
            return None

    def embed_text_for_image_search(self, text: str) -> Optional[np.ndarray]:
        """Embed text for cross-modal image search using CLIP text encoder."""
        if not self.is_available:
            return None

        cache_key = f"clip_text:{text}"
        if self.use_cache:
            cached = _embedding_cache.get(cache_key, self.model_name)
            if cached is not None:
                return cached

        try:
            import torch

            inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(_get_device()) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = self._extract_feature_tensor(self._model.get_text_features(**inputs))
                if text_features is None:
                    raise ValueError("CLIP text features could not be extracted as a tensor")
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            embedding = text_features.cpu().numpy().flatten()

            if self.use_cache:
                _embedding_cache.set(cache_key, self.model_name, embedding)

            return embedding
        except Exception as e:
            warnings.warn(f"Failed to embed text for image search: {e}")
            return None

    def embed_images_batch(self, image_paths: List[str], batch_size: int = 16) -> List[Optional[np.ndarray]]:
        """Embed a batch of images."""
        results = []
        for path in image_paths:
            results.append(self.embed_image(path))
        return results


class HybridEmbedder:
    """Combined text and image embedder for M2A dual-layer memory."""

    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        image_model: str = "openai/clip-vit-base-patch32",
        use_cache: bool = True
    ):
        self.text_embedder = TextEmbedder(text_model, use_cache)
        self.image_embedder = ImageEmbedder(image_model, use_cache)

    def embed_memory(
        self,
        text: str,
        image_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Embed a memory entry with both text and image components.

        Returns a dict with:
        - text_embedding: numpy array of text embedding
        - image_embeddings: list of image embeddings (if images provided)
        - combined_embedding: weighted average if both modalities present
        """
        result = {
            "text_embedding": self.text_embedder.embed(text),
            "image_embeddings": [],
            "combined_embedding": None
        }

        if image_paths:
            for path in image_paths:
                img_emb = self.image_embedder.embed_image(path)
                if img_emb is not None:
                    result["image_embeddings"].append(img_emb)

        # Combined embedding: use text embedding as primary
        result["combined_embedding"] = result["text_embedding"]

        return result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and multiple candidates."""
    if query is None or candidates is None or len(candidates) == 0:
        return np.array([])

    # Ensure query is 1D
    query = query.flatten()

    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(candidates))
    query_normalized = query / query_norm

    # Normalize candidates
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    candidate_norms = np.where(candidate_norms == 0, 1, candidate_norms)
    candidates_normalized = candidates / candidate_norms

    # Compute similarities
    similarities = np.dot(candidates_normalized, query_normalized)
    return similarities


# Singleton instances for convenience
_text_embedder: Optional[TextEmbedder] = None
_image_embedder: Optional[ImageEmbedder] = None
_hybrid_embedder: Optional[HybridEmbedder] = None


def get_text_embedder(model_name: str = "all-MiniLM-L6-v2") -> TextEmbedder:
    """Get or create a text embedder instance."""
    global _text_embedder
    if _text_embedder is None or _text_embedder.model_name != model_name:
        _text_embedder = TextEmbedder(model_name)
    return _text_embedder


def get_image_embedder(model_name: str = "openai/clip-vit-base-patch32") -> ImageEmbedder:
    """Get or create an image embedder instance."""
    global _image_embedder
    if _image_embedder is None or _image_embedder.model_name != model_name:
        _image_embedder = ImageEmbedder(model_name)
    return _image_embedder


def get_hybrid_embedder(
    text_model: str = "all-MiniLM-L6-v2",
    image_model: str = "openai/clip-vit-base-patch32"
) -> HybridEmbedder:
    """Get or create a hybrid embedder instance."""
    global _hybrid_embedder
    if _hybrid_embedder is None:
        _hybrid_embedder = HybridEmbedder(text_model, image_model)
    return _hybrid_embedder
