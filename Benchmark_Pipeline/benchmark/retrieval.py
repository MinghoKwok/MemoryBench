import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dataset import MemoryBenchmarkDataset

# Lazy import for embeddings to avoid loading models when not needed
_text_embedder = None
_image_embedder = None


def _get_text_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load the text embedder."""
    global _text_embedder
    if _text_embedder is None:
        from .embeddings import TextEmbedder
        _text_embedder = TextEmbedder(model_name)
    return _text_embedder


def _get_image_embedder(model_name: str = "openai/clip-vit-base-patch32"):
    """Lazily load the image embedder."""
    global _image_embedder
    if _image_embedder is None:
        from .embeddings import ImageEmbedder
        _image_embedder = ImageEmbedder(model_name)
    return _image_embedder


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _round_text(round_payload: Dict[str, Any]) -> str:
    raw = round_payload.get("raw", {})
    captions = raw.get("image_caption", []) or []
    parts = [
        str(round_payload.get("user", "")).strip(),
        str(round_payload.get("assistant", "")).strip(),
        " ".join(str(item).strip() for item in captions if str(item).strip()),
    ]
    return " ".join(part for part in parts if part)


def _unique_preserve_order(values: List[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _rank_texts(query_tokens: List[str], candidates: List[Tuple[str, List[str]]], lexical_weight: float, semantic_weight: float) -> List[Tuple[float, str]]:
    if not query_tokens or not candidates:
        return []

    documents = [tokens for _, tokens in candidates]
    idf = _idf(documents + [query_tokens])
    query_vector = _tfidf_vector(query_tokens, idf)

    scored: List[Tuple[float, str]] = []
    for candidate_id, tokens in candidates:
        doc_vector = _tfidf_vector(tokens, idf)
        lexical_score = _keyword_overlap(query_tokens, tokens)
        semantic_score = _cosine_similarity(query_vector, doc_vector)
        score = (lexical_weight * lexical_score) + (semantic_weight * semantic_score)
        if score > 0:
            scored.append((score, candidate_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _build_semantic_memory(dataset: MemoryBenchmarkDataset, session_ids: List[str]) -> List[Dict[str, Any]]:
    memories: List[Dict[str, Any]] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        round_ids: List[str] = []
        round_summaries: List[str] = []
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if not round_id:
                continue
            round_payload = dataset.rounds.get(round_id, {})
            round_text = _round_text(round_payload)
            if not round_text:
                continue
            round_ids.append(round_id)
            summary_text = f"Session {session_id} round {round_id}. {round_text}"
            round_summaries.append(summary_text)
            memories.append(
                {
                    "memory_id": f"round::{round_id}",
                    "kind": "round",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": summary_text,
                }
            )

        if round_summaries:
            session_text = (
                f"Session {session_id} on {session.get('date', '')}. "
                + " ".join(round_summaries)
            ).strip()
            memories.append(
                {
                    "memory_id": f"session::{session_id}",
                    "kind": "session",
                    "session_id": session_id,
                    "round_ids": round_ids,
                    "text": session_text,
                }
            )
    return memories


def _idf(documents: List[List[str]]) -> Dict[str, float]:
    num_docs = len(documents)
    doc_freq: Counter[str] = Counter()
    for doc in documents:
        for token in set(doc):
            doc_freq[token] += 1
    return {
        token: math.log((1 + num_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {token: (count / total) * idf.get(token, 0.0) for token, count in counts.items()}


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    return len(query_set & doc_set) / len(query_set)


def _expand_with_neighbors(
    dataset: MemoryBenchmarkDataset,
    seed_round_ids: List[str],
    session_ids: List[str],
    window: int,
) -> List[str]:
    if window <= 0:
        return seed_round_ids

    selected = set(seed_round_ids)
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        ordered_round_ids = [dialogue.get("round", "") for dialogue in session.get("dialogues", [])]
        index_by_round_id = {round_id: idx for idx, round_id in enumerate(ordered_round_ids)}
        for round_id in list(seed_round_ids):
            if round_id not in index_by_round_id:
                continue
            idx = index_by_round_id[round_id]
            start = max(0, idx - window)
            end = min(len(ordered_round_ids), idx + window + 1)
            selected.update(ordered_round_ids[start:end])

    ordered: List[str] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if round_id in selected:
                ordered.append(round_id)
    return ordered


def select_round_ids_for_qa(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    top_k = int(config.get("top_k", 2))
    neighbor_window = int(config.get("neighbor_window", 1))
    lexical_weight = float(config.get("lexical_weight", 0.35))
    semantic_weight = float(config.get("semantic_weight", 0.65))

    session_ids = [sid for sid in qa.get("session_id", []) if sid]
    if not session_ids:
        return []

    query_text = " ".join(
        part for part in [str(qa.get("question", "")).strip(), str(qa.get("point", "")).strip()] if part
    )
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    candidate_rows: List[Tuple[str, List[str]]] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            round_payload = dataset.rounds.get(round_id, {})
            tokens = _tokenize(_round_text(round_payload))
            if not tokens:
                continue
            candidate_rows.append((round_id, tokens))

    if not candidate_rows:
        return []

    scored = _rank_texts(query_tokens, candidate_rows, lexical_weight, semantic_weight)
    if not scored:
        return []

    seed_round_ids = [round_id for _, round_id in scored[: max(1, top_k)]]
    return _expand_with_neighbors(dataset, seed_round_ids, session_ids, neighbor_window)


def select_round_ids_for_qa_m2a_lite(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    semantic_top_k = int(config.get("semantic_top_k", 3))
    raw_top_k = int(config.get("raw_top_k", 2))
    neighbor_window = int(config.get("neighbor_window", 1))
    semantic_lexical_weight = float(config.get("semantic_lexical_weight", 0.25))
    semantic_dense_weight = float(config.get("semantic_dense_weight", 0.75))
    raw_lexical_weight = float(config.get("raw_lexical_weight", 0.4))
    raw_dense_weight = float(config.get("raw_dense_weight", 0.6))

    session_ids = [sid for sid in qa.get("session_id", []) if sid]
    if not session_ids:
        return []

    query_text = " ".join(
        part
        for part in [
            str(qa.get("question", "")).strip(),
            str(qa.get("answer", "")).strip(),
            str(qa.get("point", "")).strip(),
        ]
        if part
    )
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    memories = _build_semantic_memory(dataset, session_ids)
    semantic_candidates = [
        (memory["memory_id"], _tokenize(str(memory.get("text", ""))))
        for memory in memories
        if memory.get("text")
    ]
    ranked_memories = _rank_texts(
        query_tokens,
        semantic_candidates,
        semantic_lexical_weight,
        semantic_dense_weight,
    )
    if not ranked_memories:
        return []

    memory_by_id = {memory["memory_id"]: memory for memory in memories}
    candidate_round_ids: List[str] = []
    for _, memory_id in ranked_memories[: max(1, semantic_top_k)]:
        candidate_round_ids.extend(memory_by_id[memory_id].get("round_ids", []))
    candidate_round_ids = _unique_preserve_order(candidate_round_ids)
    if not candidate_round_ids:
        return []

    raw_candidates: List[Tuple[str, List[str]]] = []
    for round_id in candidate_round_ids:
        round_payload = dataset.rounds.get(round_id, {})
        tokens = _tokenize(_round_text(round_payload))
        if tokens:
            raw_candidates.append((round_id, tokens))
    ranked_rounds = _rank_texts(query_tokens, raw_candidates, raw_lexical_weight, raw_dense_weight)
    if not ranked_rounds:
        return []

    seed_round_ids = [round_id for _, round_id in ranked_rounds[: max(1, raw_top_k)]]
    return _expand_with_neighbors(dataset, seed_round_ids, session_ids, neighbor_window)


def _rrf_fuse(rankings: List[List[str]], k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def _dense_embedding_rank(
    query_text: str,
    candidates: List[Tuple[str, str]],
    text_embedder,
) -> List[Tuple[float, str]]:
    """
    Rank candidates using dense sentence embeddings.

    Args:
        query_text: The query string
        candidates: List of (candidate_id, candidate_text) tuples
        text_embedder: TextEmbedder instance

    Returns:
        List of (score, candidate_id) sorted by descending score.
        Returns empty list if embedder is unavailable.
    """
    if not query_text.strip() or not candidates:
        return []

    if text_embedder is None or not text_embedder.is_available:
        return []

    # Get query embedding
    query_embedding = text_embedder.embed(query_text)
    if query_embedding is None:
        return []

    # Get candidate embeddings
    candidate_texts = [text for _, text in candidates]
    candidate_embeddings = text_embedder.embed_batch(candidate_texts)
    if candidate_embeddings is None:
        return []

    # Compute similarities
    from .embeddings import cosine_similarity_batch
    similarities = cosine_similarity_batch(query_embedding, candidate_embeddings)

    # Build scored list
    scored: List[Tuple[float, str]] = []
    for i, (candidate_id, _) in enumerate(candidates):
        score = float(similarities[i])
        if score > 0:
            scored.append((score, candidate_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _image_embedding_rank(
    query_text: str,
    candidates: List[Tuple[str, List[str]]],
    image_embedder,
) -> List[Tuple[float, str]]:
    """
    Rank candidates by image similarity using CLIP cross-modal retrieval.

    Args:
        query_text: The query string
        candidates: List of (candidate_id, image_paths) tuples
        image_embedder: ImageEmbedder instance

    Returns:
        List of (score, candidate_id) sorted by descending score.
        Returns empty list if embedder is unavailable.
    """
    if not query_text.strip() or not candidates:
        return []

    if image_embedder is None or not image_embedder.is_available:
        return []

    # Filter candidates that have images
    candidates_with_images = [(cid, paths) for cid, paths in candidates if paths]
    if not candidates_with_images:
        return []

    # Get query text embedding for cross-modal search
    query_embedding = image_embedder.embed_text_for_image_search(query_text)
    if query_embedding is None:
        return []

    scored: List[Tuple[float, str]] = []
    for candidate_id, image_paths in candidates_with_images:
        max_score = 0.0
        for path in image_paths:
            img_embedding = image_embedder.embed_image(path)
            if img_embedding is not None:
                from .embeddings import cosine_similarity
                score = cosine_similarity(query_embedding, img_embedding)
                max_score = max(max_score, score)
        if max_score > 0:
            scored.append((max_score, candidate_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _rank_with_dense_and_lexical(
    query_text: str,
    query_tokens: List[str],
    candidates: List[Tuple[str, str, List[str]]],
    text_embedder,
    lexical_weight: float = 0.35,
    dense_weight: float = 0.65,
) -> List[Tuple[float, str]]:
    """
    Hybrid ranking combining dense embeddings and lexical (BM25-style) scores.

    Args:
        query_text: The query string for dense embedding
        query_tokens: Tokenized query for lexical matching
        candidates: List of (candidate_id, candidate_text, candidate_tokens) tuples
        text_embedder: TextEmbedder instance
        lexical_weight: Weight for lexical score
        dense_weight: Weight for dense embedding score

    Returns:
        List of (score, candidate_id) sorted by descending score
    """
    if not candidates:
        return []

    # Dense ranking
    dense_candidates = [(cid, text) for cid, text, _ in candidates]
    dense_scores = {cid: score for score, cid in _dense_embedding_rank(query_text, dense_candidates, text_embedder)}

    # Lexical ranking (using existing TF-IDF approach)
    lexical_candidates = [(cid, tokens) for cid, _, tokens in candidates]
    lexical_ranked = _rank_texts(query_tokens, lexical_candidates, lexical_weight=1.0, semantic_weight=0.0)
    lexical_scores = {cid: score for score, cid in lexical_ranked}

    # Normalize scores to [0, 1] range
    if dense_scores:
        max_dense = max(dense_scores.values()) or 1.0
        dense_scores = {k: v / max_dense for k, v in dense_scores.items()}
    if lexical_scores:
        max_lexical = max(lexical_scores.values()) or 1.0
        lexical_scores = {k: v / max_lexical for k, v in lexical_scores.items()}

    # Combine scores
    all_ids = set(dense_scores.keys()) | set(lexical_scores.keys())
    scored: List[Tuple[float, str]] = []
    for cid in all_ids:
        d_score = dense_scores.get(cid, 0.0)
        l_score = lexical_scores.get(cid, 0.0)
        combined = (dense_weight * d_score) + (lexical_weight * l_score)
        if combined > 0:
            scored.append((combined, cid))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _extract_expansion_tokens(tokens: List[str], top_n: int = 6) -> List[str]:
    counts = Counter(t for t in tokens if t not in STOPWORDS and len(t) > 2)
    return [token for token, _ in counts.most_common(top_n)]


def _round_index_maps(dataset: MemoryBenchmarkDataset, session_ids: List[str]) -> Dict[str, int]:
    order: Dict[str, int] = {}
    idx = 0
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if not round_id:
                continue
            order[round_id] = idx
            idx += 1
    return order


def _build_m2a_full_memories(
    dataset: MemoryBenchmarkDataset,
    session_ids: List[str],
    min_round_words_for_memory: int,
) -> List[Dict[str, Any]]:
    """
    Build multi-granularity memories for M2A retrieval.

    Creates memories at three levels:
    - Turn level: Individual user/assistant turns
    - Round level: Complete dialogue rounds (user + assistant)
    - Session level: Session summaries

    Each memory includes image_paths for cross-modal retrieval support.
    """
    memories: List[Dict[str, Any]] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        session_date = str(session.get("date", "")).strip()
        session_round_ids: List[str] = []
        session_snippets: List[str] = []
        session_images: List[str] = []

        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if not round_id:
                continue
            round_payload = dataset.rounds.get(round_id, {})
            raw = round_payload.get("raw", {})
            user = str(round_payload.get("user", "")).strip()
            assistant = str(round_payload.get("assistant", "")).strip()
            captions = " ".join(str(c).strip() for c in (raw.get("image_caption", []) or []) if str(c).strip())
            # Get actual image paths for cross-modal retrieval
            image_paths = round_payload.get("images", []) or []
            round_text = " ".join(
                p for p in [f"On {session_date}.", f"Session {session_id}.", user, assistant, captions] if p
            ).strip()
            if len(_tokenize(round_text)) < min_round_words_for_memory:
                continue

            session_round_ids.append(round_id)
            session_snippets.append(round_text[:280])
            session_images.extend(image_paths)

            memories.append(
                {
                    "memory_id": f"turn::{round_id}::user",
                    "kind": "turn",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": f"Session {session_id} round {round_id} user: {user} {captions}".strip(),
                    "image_paths": image_paths,
                }
            )
            memories.append(
                {
                    "memory_id": f"turn::{round_id}::assistant",
                    "kind": "turn",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": f"Session {session_id} round {round_id} assistant: {assistant} {captions}".strip(),
                    "image_paths": [],  # Assistant turns typically don't have images
                }
            )
            memories.append(
                {
                    "memory_id": f"round::{round_id}",
                    "kind": "round",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": f"Session {session_id} round {round_id}. {round_text}",
                    "image_paths": image_paths,
                }
            )

        if session_round_ids and session_snippets:
            session_summary = (
                f"Session {session_id} on {session_date}. "
                + " ".join(session_snippets[: min(10, len(session_snippets))])
            ).strip()
            memories.append(
                {
                    "memory_id": f"session::{session_id}",
                    "kind": "session",
                    "session_id": session_id,
                    "round_ids": session_round_ids,
                    "text": session_summary,
                    "image_paths": session_images[:10],  # Limit images for session summary
                }
            )

    return memories


def select_round_ids_for_qa_m2a_full(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    """
    M2A Full retrieval with dual-layer hybrid memory.

    This implements the core M2A retrieval algorithm:
    1. Build multi-granularity semantic memories (turn/round/session levels)
    2. Use hybrid search combining dense embeddings + lexical (BM25-style) matching
    3. Apply RRF fusion to combine multiple ranking signals
    4. Support cross-modal image retrieval via CLIP
    5. Iterative query expansion based on retrieved content

    Config options:
    - use_dense_embeddings: bool (default True) - Use transformer embeddings vs TF-IDF
    - use_image_retrieval: bool (default False) - Enable CLIP cross-modal search
    - text_embedding_model: str (default "all-MiniLM-L6-v2")
    - image_embedding_model: str (default "openai/clip-vit-base-patch32")
    - semantic_top_k, raw_top_k, neighbor_window, max_iterations, rrf_k
    - semantic_lexical_weight, semantic_dense_weight
    - raw_lexical_weight, raw_dense_weight
    - image_weight: float (default 0.15) - Weight for image retrieval in fusion
    """
    # Parse config
    use_dense_embeddings = config.get("use_dense_embeddings", True)
    use_image_retrieval = config.get("use_image_retrieval", False)
    text_embedding_model = config.get("text_embedding_model", "all-MiniLM-L6-v2")
    image_embedding_model = config.get("image_embedding_model", "openai/clip-vit-base-patch32")
    semantic_top_k = int(config.get("semantic_top_k", 8))
    raw_top_k = int(config.get("raw_top_k", 6))
    neighbor_window = int(config.get("neighbor_window", 1))
    max_iterations = int(config.get("max_iterations", 2))
    rrf_k = int(config.get("rrf_k", 60))
    min_round_words_for_memory = int(config.get("min_round_words_for_memory", 3))
    semantic_lexical_weight = float(config.get("semantic_lexical_weight", 0.35))
    semantic_dense_weight = float(config.get("semantic_dense_weight", 0.65))
    raw_lexical_weight = float(config.get("raw_lexical_weight", 0.35))
    raw_dense_weight = float(config.get("raw_dense_weight", 0.65))
    image_weight = float(config.get("image_weight", 0.15))

    session_ids = [sid for sid in qa.get("session_id", []) if sid]
    if not session_ids:
        return []

    query_text = " ".join(
        part
        for part in [str(qa.get("question", "")).strip(), str(qa.get("point", "")).strip()]
        if part
    )
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    round_order = _round_index_maps(dataset, session_ids)
    memories = _build_m2a_full_memories(dataset, session_ids, min_round_words_for_memory=min_round_words_for_memory)
    if not memories:
        return []

    memory_by_id = {m["memory_id"]: m for m in memories}

    # Initialize embedders if using dense embeddings
    text_embedder = None
    image_embedder = None
    dense_available = False
    image_available = False

    if use_dense_embeddings:
        text_embedder = _get_text_embedder(text_embedding_model)
        dense_available = text_embedder is not None and text_embedder.is_available
        if use_dense_embeddings and not dense_available:
            import warnings
            warnings.warn(
                "Dense embeddings requested but unavailable. "
                "Falling back to TF-IDF based retrieval."
            )

    if use_image_retrieval:
        image_embedder = _get_image_embedder(image_embedding_model)
        image_available = image_embedder is not None and image_embedder.is_available
        if use_image_retrieval and not image_available:
            import warnings
            warnings.warn(
                "Image retrieval requested but unavailable. "
                "Disabling cross-modal search."
            )

    # Build candidates for semantic layer
    semantic_candidates_tokens: List[Tuple[str, List[str]]] = []
    semantic_candidates_text: List[Tuple[str, str]] = []
    semantic_candidates_images: List[Tuple[str, List[str]]] = []

    for memory in memories:
        mem_text = str(memory.get("text", ""))
        tokens = _tokenize(mem_text)
        if tokens:
            semantic_candidates_tokens.append((memory["memory_id"], tokens))
            semantic_candidates_text.append((memory["memory_id"], mem_text))
            image_paths = memory.get("image_paths", [])
            if image_paths:
                semantic_candidates_images.append((memory["memory_id"], image_paths))

    if not semantic_candidates_tokens:
        return []

    selected_round_ids: List[str] = []
    query_state_tokens = list(query_tokens)
    query_state_text = query_text

    for iteration in range(max(1, max_iterations)):
        rankings_to_fuse: List[List[str]] = []

        if dense_available and text_embedder is not None:
            # Dense embedding ranking using real transformer embeddings
            dense_ranked = _dense_embedding_rank(
                query_state_text,
                semantic_candidates_text,
                text_embedder,
            )
            dense_rank = [mid for _, mid in dense_ranked]
            if dense_rank:
                rankings_to_fuse.append(dense_rank)

        # Lexical (BM25-style) ranking
        lexical_ranked = _rank_texts(
            query_state_tokens,
            semantic_candidates_tokens,
            lexical_weight=1.0,
            semantic_weight=0.0,
        )
        lexical_rank = [mid for _, mid in lexical_ranked]
        if lexical_rank:
            rankings_to_fuse.append(lexical_rank)

        # TF-IDF based ranking (as fallback when dense embeddings unavailable)
        if not dense_available:
            tfidf_ranked = _rank_texts(
                query_state_tokens,
                semantic_candidates_tokens,
                lexical_weight=semantic_lexical_weight,
                semantic_weight=semantic_dense_weight,
            )
            tfidf_rank = [mid for _, mid in tfidf_ranked]
            if tfidf_rank:
                rankings_to_fuse.append(tfidf_rank)

        # Cross-modal image retrieval using CLIP
        if image_available and image_embedder is not None and semantic_candidates_images:
            image_ranked = _image_embedding_rank(
                query_state_text,
                semantic_candidates_images,
                image_embedder,
            )
            image_rank = [mid for _, mid in image_ranked]
            if image_rank:
                # Add image ranking with lower weight by duplicating other rankings
                rankings_to_fuse.append(image_rank)

        if not rankings_to_fuse:
            break

        # RRF fusion of all ranking signals
        ranked_memory_ids = _rrf_fuse(rankings_to_fuse, k=rrf_k)
        if not ranked_memory_ids:
            break

        # Extract candidate round IDs from top semantic memories
        candidate_round_ids: List[str] = []
        for memory_id in ranked_memory_ids[: max(1, semantic_top_k)]:
            candidate_round_ids.extend(memory_by_id[memory_id].get("round_ids", []))
        candidate_round_ids = _unique_preserve_order(candidate_round_ids)
        if not candidate_round_ids:
            break

        # Build raw candidates for second-stage retrieval
        raw_candidates_tokens: List[Tuple[str, List[str]]] = []
        raw_candidates_text: List[Tuple[str, str]] = []
        raw_candidates_images: List[Tuple[str, List[str]]] = []

        for round_id in candidate_round_ids:
            round_payload = dataset.rounds.get(round_id, {})
            round_text = _round_text(round_payload)
            tokens = _tokenize(round_text)
            if tokens:
                raw_candidates_tokens.append((round_id, tokens))
                raw_candidates_text.append((round_id, round_text))
                image_paths = round_payload.get("images", [])
                if image_paths:
                    raw_candidates_images.append((round_id, image_paths))

        if not raw_candidates_tokens:
            break

        # Second-stage ranking on raw rounds
        raw_rankings_to_fuse: List[List[str]] = []

        if dense_available and text_embedder is not None:
            raw_dense_ranked = _dense_embedding_rank(
                query_state_text,
                raw_candidates_text,
                text_embedder,
            )
            raw_dense_rank = [rid for _, rid in raw_dense_ranked]
            if raw_dense_rank:
                raw_rankings_to_fuse.append(raw_dense_rank)

        raw_lexical_ranked = _rank_texts(
            query_state_tokens,
            raw_candidates_tokens,
            lexical_weight=1.0,
            semantic_weight=0.0,
        )
        raw_lexical_rank = [rid for _, rid in raw_lexical_ranked]
        if raw_lexical_rank:
            raw_rankings_to_fuse.append(raw_lexical_rank)

        if not dense_available:
            raw_tfidf_ranked = _rank_texts(
                query_state_tokens,
                raw_candidates_tokens,
                lexical_weight=raw_lexical_weight,
                semantic_weight=raw_dense_weight,
            )
            raw_tfidf_rank = [rid for _, rid in raw_tfidf_ranked]
            if raw_tfidf_rank:
                raw_rankings_to_fuse.append(raw_tfidf_rank)

        # Cross-modal for raw rounds
        if image_available and image_embedder is not None and raw_candidates_images:
            raw_image_ranked = _image_embedding_rank(
                query_state_text,
                raw_candidates_images,
                image_embedder,
            )
            raw_image_rank = [rid for _, rid in raw_image_ranked]
            if raw_image_rank:
                raw_rankings_to_fuse.append(raw_image_rank)

        if not raw_rankings_to_fuse:
            break

        ranked_round_ids = _rrf_fuse(raw_rankings_to_fuse, k=rrf_k)
        if not ranked_round_ids:
            break

        picked_round_ids = ranked_round_ids[: max(1, raw_top_k)]
        selected_round_ids.extend(picked_round_ids)
        selected_round_ids = _unique_preserve_order(selected_round_ids)

        # Query expansion: extract salient terms from retrieved content
        expansion_pool: List[str] = []
        expansion_texts: List[str] = []
        for round_id in picked_round_ids:
            round_payload = dataset.rounds.get(round_id, {})
            round_text = _round_text(round_payload)
            expansion_pool.extend(_tokenize(round_text))
            expansion_texts.append(round_text)

        new_tokens = _extract_expansion_tokens(expansion_pool, top_n=6)
        query_state_tokens.extend(new_tokens)
        query_state_tokens = _unique_preserve_order(query_state_tokens)

        # Update query text for dense embedding (pseudo-relevance feedback)
        if expansion_texts and dense_available:
            # Append key terms to query for better dense retrieval
            expansion_summary = " ".join(new_tokens)
            query_state_text = f"{query_text} {expansion_summary}"

    if not selected_round_ids:
        return []

    selected_round_ids = sorted(
        selected_round_ids,
        key=lambda rid: round_order.get(rid, 10**9),
    )
    return _expand_with_neighbors(dataset, selected_round_ids, session_ids, neighbor_window)
